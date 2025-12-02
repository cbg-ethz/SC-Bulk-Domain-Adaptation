import os
os.environ["SCIPY_ARRAY_API"] = "1"  # requested env var

import argparse
import warnings
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import wandb
from morgoth import MORGOTH
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import StandardScaler

from data_utils import (
    convert_to_ensembl,
    drop_all_nan_and_deduplicate,
    initialize_symbol_map,
    intersect_genes,
    normalize_cpm_log1p_if_counts,
)
from training_utils import calculate_all_metrics, set_seed

# ----------------------- Script Configuration -----------------------
SEED = 42
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "processed"))
SYMBOL_ENSEMBL_MAP = os.path.join(DATA_DIR, "..", "reference", "symbol_ensembl_map.txt")
WANDB_PROJECT = "hyper_tuning_v5"
DEFAULT_TOP_GENES = 100
DEFAULT_THRESHOLD = 0.5
TEMP_ROOT = os.path.join(DATA_DIR, "..", "morgoth_temp")

set_seed(SEED)


def _prepare_data(drug: str, target_tag: str, all_files: List[str], data_dir: str) -> Optional[Dict[str, pd.DataFrame]]:
    """Load, preprocess, align, and split source/target data using the hyper_tuning pipeline."""
    X_source_files = [f for f in all_files if f.startswith("X_") and "bulk" in f]
    y_source_files = [f for f in all_files if f.startswith("y_") and "bulk" in f]
    if not X_source_files or not y_source_files:
        warnings.warn(f"[{drug}] Missing source files. Skipping.")
        return None

    X_source_raw = pd.read_csv(os.path.join(data_dir, X_source_files[0]), index_col=0)
    y_source_raw = pd.read_csv(os.path.join(data_dir, y_source_files[0]), index_col=0)["viability"]
    y_source = 1 - y_source_raw
    y_source_bin = (y_source >= 0.5).astype(int)

    X_source = convert_to_ensembl(X_source_raw.copy())
    X_source = drop_all_nan_and_deduplicate(X_source)
    X_source = normalize_cpm_log1p_if_counts(X_source, "X_source")

    X_target_files = [f for f in all_files if f.startswith("X_") and drug in f and target_tag in f]
    y_target_files = [f for f in all_files if f.startswith("y_") and drug in f and target_tag in f]
    if not X_target_files or not y_target_files:
        warnings.warn(f'[{drug}] Missing target files for dataset {target_tag}. Skipping.')
        return None

    X_target_raw = pd.read_csv(os.path.join(data_dir, X_target_files[0]), index_col=0)
    y_target_raw = pd.read_csv(os.path.join(data_dir, y_target_files[0]), index_col=0)
    y_target_series = y_target_raw.iloc[:, 0]

    X_target = convert_to_ensembl(X_target_raw.copy())
    X_target = drop_all_nan_and_deduplicate(X_target)
    X_target = normalize_cpm_log1p_if_counts(X_target, "X_target")

    (X_source, X_target), common_genes = intersect_genes(X_source, X_target)
    if not common_genes:
        print("No common genes found. Skipping this target pair.")
        return None

    X_source_train, X_source_test, y_source_train, y_source_test = train_test_split(
        X_source, y_source, test_size=0.2, random_state=SEED, stratify=y_source_bin
    )
    y_source_train_bin = y_source_train >= 0.5

    X_source_train, X_source_val, y_source_train, y_source_val = train_test_split(
        X_source_train, y_source_train, test_size=0.2, random_state=SEED, stratify=y_source_train_bin
    )

    X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
        X_target, y_target_series, test_size=0.2, random_state=SEED, stratify=y_target_series
    )

    source_scaler = StandardScaler()
    X_source_train = pd.DataFrame(
        source_scaler.fit_transform(X_source_train), index=X_source_train.index, columns=X_source_train.columns
    )
    X_source_val = pd.DataFrame(
        source_scaler.transform(X_source_val), index=X_source_val.index, columns=X_source_val.columns
    )
    X_source_test = pd.DataFrame(
        source_scaler.transform(X_source_test), index=X_source_test.index, columns=X_source_test.columns
    )

    expected_cols = list(source_scaler.feature_names_in_)
    X_target_train = pd.DataFrame(
        source_scaler.transform(X_target_train.reindex(columns=expected_cols)),
        index=X_target_train.index,
        columns=expected_cols,
    )
    X_target_test = pd.DataFrame(
        source_scaler.transform(X_target_test.reindex(columns=expected_cols)),
        index=X_target_test.index,
        columns=expected_cols,
    )

    return {
        "x_train_source": X_source_train,
        "y_train_source": y_source_train,
        "x_val_source": X_source_val,
        "y_val_source": y_source_val,
        "x_test_source": X_source_test,
        "y_test_source": y_source_test,
        "x_train_target": X_target_train,
        "y_train_target": y_target_train,
        "x_test_target": X_target_test,
        "y_test_target": y_target_test,
    }


def _select_top_genes(X: pd.DataFrame, y: pd.Series, top_n: int) -> List[str]:
    """Return the top_n genes ranked by absolute correlation with the response."""
    correlations = X.corrwith(y)
    return correlations.abs().nlargest(top_n).index.tolist()


def _build_y_df(y: pd.Series) -> pd.DataFrame:
    """Create the MORGOTH-compatible target dataframe with continuous and binary labels."""
    return pd.DataFrame({"y": y.values, "y_binary": (y.values >= 0.5).astype(int)}, index=y.index)


def _extract_probabilities(raw_preds: Iterable) -> np.ndarray:
    """Extract probability column from MORGOTH predictions."""
    probs: List[float] = []
    for item in raw_preds:
        if isinstance(item, (list, tuple, np.ndarray)) and len(item) > 1:
            probs.append(float(item[1]))
        else:
            flattened = np.asarray(item).ravel()
            probs.append(float(flattened[0]))
    return np.asarray(probs, dtype=float)


def _tune_threshold_from_validation(y_true: Iterable, probs: Iterable) -> float:
    """Tune decision threshold on validation set using MCC over 0.05..0.95."""
    y_bin = (np.asarray(list(y_true)).ravel() >= 0.5).astype(int)
    probs_arr = np.asarray(list(probs), dtype=float).ravel()
    best_t, best_mcc = 0.5, -1.0

    for t in [i / 20.0 for i in range(1, 20)]:
        preds = (probs_arr >= t).astype(int)
        # Skip degenerate predictions that break MCC
        if len(np.unique(preds)) < 2:
            mcc = -1.0
        else:
            mcc = matthews_corrcoef(y_bin, preds)
        if mcc > best_mcc:
            best_mcc, best_t = mcc, t

    if wandb.run is not None:
        wandb.log({"decision_threshold": best_t, "decision_threshold_mcc": best_mcc})

    return best_t


def _run_morgoth(
    data: Dict[str, pd.DataFrame],
    drug: str,
    target_tag: str,
    top_genes: int,
    threshold: float,
    tune_threshold: bool,
) -> Dict[str, Dict[str, float]]:
    """Train and evaluate MORGOTH on the provided splits."""
    os.makedirs(TEMP_ROOT, exist_ok=True)
    run_dir = os.path.join(TEMP_ROOT, f"{drug}_{target_tag}")
    os.makedirs(run_dir, exist_ok=True)

    # MORGOTH does not require a dedicated validation split; train on train+val for more data.
    x_train_full = pd.concat([data["x_train_source"], data["x_val_source"]])
    y_train_full = pd.concat([data["y_train_source"], data["y_val_source"]])

    top_gene_names = _select_top_genes(x_train_full, y_train_full, top_genes)

    def subset(df: pd.DataFrame) -> pd.DataFrame:
        return df[top_gene_names]

    x_train = subset(x_train_full)
    x_val = subset(data["x_val_source"])
    x_test_source = subset(data["x_test_source"])
    x_test_target = subset(data["x_test_target"])

    y_train = _build_y_df(y_train_full)
    y_val = _build_y_df(data["y_val_source"])
    y_test_source = _build_y_df(data["y_test_source"])
    y_test_target = _build_y_df(data["y_test_target"])

    time_file = os.path.join(run_dir, "ElapsedTimeFitting.txt")
    sample_info_file = os.path.join(run_dir, "Additional_Sample_Information.txt")
    leaf_assignment_file_train = os.path.join(run_dir, "Training_Set_LeafAssignment.txt")
    feature_imp_output_file = os.path.join(run_dir, "Feature_Importance.txt")
    silhouette_score_file = os.path.join(run_dir, "Silhouette_Score.txt")
    silhouette_score_train_file = os.path.join(run_dir, "Silhouette_Score_Train.txt")
    cluster_assignment_file = os.path.join(run_dir, "Cluster_Assignment.txt")

    model = MORGOTH(
        X_train=x_train,
        y_train=y_train,
        sample_names_train=x_train.index,
        criterion_class="gini",
        criterion_reg="mse",
        min_number_of_samples_per_leaf=10,
        number_of_trees_in_forest=500,
        analysis_name=f"{drug}_{target_tag}",
        number_of_features_per_split="sqrt",
        class_names=[0, 1],
        output_format="multioutput",
        threshold=[threshold],
        time_file=time_file,
        sample_weights_included="simple",
        random_state=SEED,
        max_depth=20,
        impact_classification=0.5,
        sample_info_file=sample_info_file,
        leaf_assignment_file_train=leaf_assignment_file_train,
        feature_imp_output_file=feature_imp_output_file,
        tree_weights=False,
        silhouette_score_file=silhouette_score_file,
        distance_measure="",
        cluster_assignment_file=cluster_assignment_file,
        draw_graph=False,
        graph_path="",
        silhouette_score_train_file=silhouette_score_train_file,
    )

    model.fit()

    val_probs = _extract_probabilities(model.predict(X_test=x_val))
    tuned_threshold = _tune_threshold_from_validation(y_val["y"].values, val_probs) if tune_threshold else threshold
    threshold_for_metrics = tuned_threshold if tune_threshold else threshold

    source_probs = _extract_probabilities(model.predict(X_test=x_test_source))
    target_probs = _extract_probabilities(model.predict(X_test=x_test_target))

    return {
        "source_val": calculate_all_metrics(y_val["y"].values, val_probs, threshold=threshold_for_metrics),
        "source_test": calculate_all_metrics(
            y_test_source["y"].values, source_probs, threshold=threshold_for_metrics
        ),
        "target_test": calculate_all_metrics(
            y_test_target["y"].values, target_probs, threshold=threshold_for_metrics
        ),
        "top_genes_used": {"count": len(top_gene_names)},
        "threshold": {"decision_threshold": threshold_for_metrics, "tuned": tune_threshold},
    }


def main():
    parser = argparse.ArgumentParser(description="Run a single MORGOTH evaluation over specified drugs/targets.")
    parser.add_argument(
        "--drugs",
        nargs="+",
        default=[
            "Etoposide",
            "Erlotinib",
            "Vorinostat",
            "Gefitinib",
            "Afatinib",
            "Sorafenib",
            "Ibrutinib",
            "Olaparib",
            "Docetaxel",
            "Paclitaxel",
            "Cisplatin",
        ],
        help="Drugs to evaluate.",
    )
    parser.add_argument("--top-genes", type=int, default=DEFAULT_TOP_GENES, help="Number of genes selected by correlation.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Classification threshold passed to MORGOTH and metric calculation.",
    )
    parser.add_argument(
        "--disable-threshold-tuning",
        dest="tune_threshold",
        action="store_false",
        default=True,
        help="Disable decision-threshold tuning (enabled by default).",
    )
    cli_args = parser.parse_args()

    initialize_symbol_map(SYMBOL_ENSEMBL_MAP)

    target_file_names = {
        "Cisplatin": ["GSE138267", "GSE117872_HN120", "GSE117872_HN137"],
        "Paclitaxel": ["GSE163836_FCIBC02", "GSE131984"],
        "Ibrutinib": ["GSE111014_CLL"],
        "Olaparib": ["GSE228382"],
        "Vorinostat": ["SCC47"],
        "Etoposide": ["GSE149383_PC9"],
        "Erlotinib": ["GSE149383_PC9"],
        "Docetaxel": ["GSE140440_DU145", "GSE140440_PC3"],
        "Sorafenib": ["GSE175716_HCC", "SCC47"],
        "Gefitinib": ["GSE162045_PC9", "GSE202234_H1975", "GSE202234_PC9", "JHU006", "GSE112274_PC9"],
        "Afatinib": ["GSE228154_LT", "SCC47"],
    }

    for drug in cli_args.drugs:
        print(f'\n{"="*25} Processing drug: {drug} {"="*25}')
        all_files = [f for f in os.listdir(DATA_DIR) if drug in f and f.endswith(".csv")]

        for target_tag in target_file_names.get(drug, []):
            print(f"\n--- Preparing data for target: {target_tag} ---")
            data = _prepare_data(drug, target_tag, all_files, DATA_DIR)
            if data is None:
                print(f"Skipping {drug} with target {target_tag} due to data issues.")
                continue

            config = {
                "model": "MORGOTH",
                "drug": drug,
                "target": target_tag,
                "top_genes": cli_args.top_genes,
                "threshold": cli_args.threshold,
                "tune_threshold": cli_args.tune_threshold,
                "seed": SEED,
            }

            with wandb.init(
                project=WANDB_PROJECT,
                group=f"{drug}_{target_tag}_MORGOTH",
                job_type="single_run",
                config=config,
                reinit=True,
            ) as run:
                try:
                    results = _run_morgoth(
                        data=data,
                        drug=drug,
                        target_tag=target_tag,
                        top_genes=cli_args.top_genes,
                        threshold=cli_args.threshold,
                        tune_threshold=cli_args.tune_threshold,
                    )
                except Exception as exc:  # pragma: no cover - defensive logging
                    warning_msg = f"MORGOTH failed: {exc}"
                    print(warning_msg)
                    if run is not None:
                        run.log({"error": warning_msg})
                    continue

                log_payload = {
                    "drug": drug,
                    "target": target_tag,
                    "model_name": "MORGOTH",
                }
                for split, metrics in results.items():
                    if not isinstance(metrics, dict):
                        continue
                    for metric, value in metrics.items():
                        log_payload[f"{split}_{metric}"] = value

                if run is not None:
                    run.log(log_payload)

                print(f"Logged results for {drug}::{target_tag}")


if __name__ == "__main__":
    main()
