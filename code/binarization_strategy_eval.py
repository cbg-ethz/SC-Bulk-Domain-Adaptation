import os
os.environ["SCIPY_ARRAY_API"] = "1"  # requested env var

import argparse
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_utils import (
    convert_to_ensembl,
    drop_all_nan_and_deduplicate,
    initialize_symbol_map,
    intersect_genes,
    normalize_cpm_log1p_if_counts,
)
from training_utils import run_catboost_benchmark, set_seed

# ----------------------- Script Configuration -----------------------
SEED = 42
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "processed"))
SYMBOL_ENSEMBL_MAP = os.path.join(DATA_DIR, "..", "reference", "symbol_ensembl_map.txt")
WANDB_PROJECT = "threshold_test"

set_seed(SEED)


# ----------------------- Helpers -----------------------
def _compute_binarization_threshold(
    y_source: pd.Series, strategy: str, min_minor_fraction: float = 0.25, base_threshold: float = 0.5
) -> float:
    """Pick a binarization threshold. If classes at base are unbalanced, shift until minority >= min_minor_fraction."""
    strategy_norm = strategy.lower()
    if strategy_norm in {"fixed", "fixed_0.5", "default"}:
        return base_threshold
    if strategy_norm in {"balanced", "balanced_minority", "balanced_0.1", "balanced_0.2"}:
        frac_above = (y_source >= base_threshold).mean()
        frac_below = 1.0 - frac_above
        if min(frac_above, frac_below) >= min_minor_fraction:
            return base_threshold
        if frac_below < min_minor_fraction:
            candidate = float(y_source.quantile(min_minor_fraction))
        else:
            candidate = float(y_source.quantile(1.0 - min_minor_fraction))
        if np.isnan(candidate):
            return base_threshold
        return candidate
    raise ValueError(f"Unknown binarization strategy: {strategy}")


def _binarize_series(y: pd.Series, threshold: float) -> pd.Series:
    """Binarize labels at the given threshold."""
    return (y >= threshold).astype(int)


def _minority_fraction(y_bin: pd.Series) -> float:
    """Return the fraction occupied by the minority class."""
    counts = y_bin.value_counts()
    if counts.empty:
        return 0.0
    return counts.min() / counts.sum()


# ----------------------- Data Preparation -----------------------
def _prepare_data(
    drug: str,
    target_tag: str,
    all_files: List[str],
    data_dir: str,
    strategy: str,
    min_minor_fraction: float,
) -> Optional[Dict[str, object]]:
    """Load, preprocess, and split data, then binarize using the requested strategy."""
    X_source_files = [f for f in all_files if f.startswith("X_") and "bulk" in f]
    y_source_files = [f for f in all_files if f.startswith("y_") and "bulk" in f]
    if not X_source_files or not y_source_files:
        warnings.warn(f"[{drug}] Missing source files. Skipping.")
        return None

    X_source_raw = pd.read_csv(os.path.join(data_dir, X_source_files[0]), index_col=0)
    y_source_raw = pd.read_csv(os.path.join(data_dir, y_source_files[0]), index_col=0)["viability"]
    y_source = 1 - y_source_raw

    threshold = _compute_binarization_threshold(y_source, strategy, min_minor_fraction=min_minor_fraction)
    y_source_bin = _binarize_series(y_source, threshold)

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

    X_source_train, X_source_test, y_source_train_bin, y_source_test_bin = train_test_split(
        X_source, y_source_bin, test_size=0.2, random_state=SEED, stratify=y_source_bin
    )
    X_source_train, X_source_val, y_source_train_bin, y_source_val_bin = train_test_split(
        X_source_train, y_source_train_bin, test_size=0.2, random_state=SEED, stratify=y_source_train_bin
    )

    y_target_bin = _binarize_series(y_target_series, threshold)
    stratify_target = y_target_bin if y_target_bin.nunique() > 1 else None
    if stratify_target is None:
        warnings.warn(f"[{drug}::{target_tag}] Target labels single-class after binarization. Using unstratified split.")

    X_target_train, X_target_test, y_target_train_bin, y_target_test_bin = train_test_split(
        X_target, y_target_bin, test_size=0.2, random_state=SEED, stratify=stratify_target
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
        "y_train_source": y_source_train_bin,
        "x_val_source": X_source_val,
        "y_val_source": y_source_val_bin,
        "x_test_source": X_source_test,
        "y_test_source": y_source_test_bin,
        "x_train_target": X_target_train,
        "y_train_target": y_target_train_bin,
        "x_test_target": X_target_test,
        "y_test_target": y_target_test_bin,
        "X_target_independent": None,
        "y_target_independent": None,
        "binarization_threshold": threshold,
        "binarization_strategy": strategy,
        "source_class_counts": y_source_bin.value_counts().to_dict(),
        "target_class_counts": y_target_bin.value_counts().to_dict(),
        "source_minority_fraction": _minority_fraction(y_source_bin),
        "target_minority_fraction": _minority_fraction(y_target_bin),
    }


# ----------------------- Main Execution -----------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate CatBoost source models with alternative binarization thresholds.")
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
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["balanced_minority"],
        help="Binarization strategies to test (e.g., fixed_0.5, balanced_minority).",
    )
    parser.add_argument(
        "--minority-fraction",
        type=float,
        default=0.25,
        help="Minimum fraction required for the minority class when using balanced_minority.",
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
            for strategy in cli_args.strategies:
                data = _prepare_data(
                    drug,
                    target_tag,
                    all_files,
                    DATA_DIR,
                    strategy=strategy,
                    min_minor_fraction=cli_args.minority_fraction,
                )
                if data is None:
                    print(f"Skipping {drug}::{target_tag} for strategy {strategy} due to data issues.")
                    continue

                config = {
                    "model": "CatBoost_source_only",
                    "drug": drug,
                    "target": target_tag,
                    "strategy": strategy,
                    "seed": SEED,
                    "binarization_threshold": data["binarization_threshold"],
                    "minority_fraction_required": cli_args.minority_fraction,
                    "source_minority_fraction": data["source_minority_fraction"],
                    "target_minority_fraction": data["target_minority_fraction"],
                }

                with wandb.init(
                    project=WANDB_PROJECT,
                    group=f"{drug}_{target_tag}_CatBoost_thresholds",
                    job_type=strategy,
                    config=config,
                    reinit=True,
                ) as run:
                    try:
                        results = run_catboost_benchmark(
                            model_type="CatBoost_source_only",
                            x_train_source=data["x_train_source"],
                            y_train_source=data["y_train_source"],
                            x_val_source=data["x_val_source"],
                            y_val_source=data["y_val_source"],
                            x_test_source=data["x_test_source"],
                            y_test_source=data["y_test_source"],
                            x_train_target=data["x_train_target"],
                            y_train_target=data["y_train_target"],
                            x_test_target=data["x_test_target"],
                            y_test_target=data["y_test_target"],
                            X_target_independent=data.get("X_target_independent"),
                            y_target_independent=data.get("y_target_independent"),
                            seed=SEED,
                        )
                    except Exception as exc:  # pragma: no cover - defensive logging
                        warning_msg = f"CatBoost failed: {exc}"
                        print(warning_msg)
                        if run is not None:
                            run.log({"error": warning_msg})
                        continue

                    log_payload = {
                        "drug": drug,
                        "target": target_tag,
                        "model_name": "CatBoost_source_only",
                        "strategy": strategy,
                        "binarization_threshold": data["binarization_threshold"],
                        "source_minority_fraction": data["source_minority_fraction"],
                        "target_minority_fraction": data["target_minority_fraction"],
                    }
                    for label, count in data.get("source_class_counts", {}).items():
                        log_payload[f"source_class_{label}"] = int(count)
                    for label, count in data.get("target_class_counts", {}).items():
                        log_payload[f"target_class_{label}"] = int(count)

                    for split, metrics in results.items():
                        if not isinstance(metrics, dict):
                            continue
                        for metric, value in metrics.items():
                            log_payload[f"{split}_{metric}"] = value

                    if run is not None:
                        run.log(log_payload)

                    print(f"Logged results for {drug}::{target_tag} with strategy {strategy}")


if __name__ == "__main__":
    main()
