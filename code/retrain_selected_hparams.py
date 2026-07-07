# retrain_selected_hparams.py — Retrains each framework with its selected best hyperparameters
# (from the W&B sweep) and logs final benchmark results.

import os

os.environ["SCIPY_ARRAY_API"] = "1"

import argparse
import warnings
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import training_utils
from data_utils import (
    convert_to_ensembl,
    drop_all_nan_and_deduplicate,
    initialize_symbol_map,
    intersect_genes,
    normalize_cpm_log1p_if_counts,
)
from training_utils import (
    run_catboost_benchmark,
    run_catboost_fewshot_baseline,
    run_scad_benchmark,
    run_scdeal_benchmark,
    run_scatd_benchmark,
    run_ssda4drug_benchmark,
    set_seed,
)


SEED = 42
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "processed"))
SYMBOL_ENSEMBL_MAP = os.path.join(DATA_DIR, "..", "reference", "symbol_ensembl_map.txt")
WANDB_PROJECT = "target_train_vs_test_retrain"
DEFAULT_CSV_PATH = "/cluster/home/mibohl/work/master_thesis/paper/plots/wandb.csv"

# Keep deep models on GPU, but force CatBoost variants onto CPU to avoid CUDA OOMs.
training_utils._CATBOOST_TASK_TYPE = "CPU"


BASELINE_PARAMS: Dict[str, Dict[str, object]] = {
    "SCAD": {
        "lr": 0.0005,
        "dropout": 0.5,
        "h_dim": 1024,
        "predictor_z_dim": 128,
        "mbS": 8,
        "mbT": 8,
        "lam1": 1,
        "binarize_source": True,
    },
    "SSDA4Drug": {
        "lr": 0.001,
        "dropout": 0.3,
        "encoder_h_dims": "512,256",
        "predictor_h_dims": "64,32",
        "batch_size": 128,
        "binarize_source": True,
    },
    "scDEAL": {
        "lr": 0.01,
        "dropout": 0.3,
        "bulk_h_dims": "512,256",
        "predictor_h_dims": "64,32",
        "bottleneck": 128,
        "mmd_weight": 0.25,
        "binarize_source": True,
    },
    "scATD": {
        "lr": 0.001,
        "mmd_weight": 1,
        "batch_size": 128,
        "weight_decay": 0.001,
        "binarize_source": True,
    },
    "CatBoost_source_only": {
        "depth": 6,
        "lr": 0.03,
        "l2_leaf_reg": 3,
        "border_count": 254,
    },
    "CatBoost_target_only": {
        "depth": 6,
        "lr": 0.03,
        "l2_leaf_reg": 3,
        "border_count": 254,
    },
    "CatBoost_combined": {
        "depth": 6,
        "lr": 0.03,
        "l2_leaf_reg": 3,
        "border_count": 254,
    },
}


MODEL_PARAM_KEYS: Dict[str, List[str]] = {
    "SCAD": ["lr", "dropout", "h_dim", "predictor_z_dim", "mbS", "mbT", "lam1", "binarize_source"],
    "SSDA4Drug": [
        "lr",
        "dropout",
        "encoder_h_dims",
        "predictor_h_dims",
        "batch_size",
        "lambda_ent",
        "use_epsilon",
        "epsilon",
        "binarize_source",
    ],
    "scDEAL": [
        "lr",
        "dropout",
        "bulk_h_dims",
        "predictor_h_dims",
        "bottleneck",
        "mmd_weight",
        "binarize_source",
    ],
    "scATD": ["lr", "batch_size", "mmd_weight", "weight_decay", "binarize_source"],
    "CatBoost_source_only": ["depth", "lr", "l2_leaf_reg", "border_count"],
    "CatBoost_target_only": ["depth", "lr", "l2_leaf_reg", "border_count"],
    "CatBoost_combined": ["depth", "lr", "l2_leaf_reg", "border_count"],
    "CatBoost_fs": [],
}


METRIC_COLUMNS = {
    "source_val_mcc",
    "source_val_auc",
    "source_val_auprc",
    "source_test_accuracy",
    "source_test_auc",
    "source_test_auprc",
    "source_test_f1_score",
    "source_test_mcc",
    "source_test_precision",
    "source_test_recall_sensitivity",
    "source_test_specificity",
    "source_val_accuracy",
    "source_val_f1_score",
    "source_val_precision",
    "source_val_recall_sensitivity",
    "source_val_specificity",
    "target_test_accuracy",
    "target_test_auc",
    "target_test_auprc",
    "target_test_f1_score",
    "target_test_mcc",
    "target_test_precision",
    "target_test_recall_sensitivity",
    "target_test_specificity",
    "decision_threshold",
    "decision_threshold_mcc",
    "epoch",
    "_runtime",
    "_timestamp",
}


@dataclass
class ScadArgs:
    drug: str
    epochs: int = 100
    data_split: str = "test"
    log_normalized: bool = True
    plot_umap: bool = False
    mode: str = "SCAD"
    lr: float = 0.0005
    mbS: int = 8
    mbT: int = 8
    dropout: float = 0.5
    lam1: float = 1.0
    balancing_strategy: str = "weighted"
    h_dim: int = 1024
    predictor_z_dim: int = 128
    tune_threshold: bool = True
    binarize_source: bool = True
    patience: int = 10


@dataclass
class ScDealArgs:
    drug: str
    result_path: str = "scDEAL_temp/"
    model_path: str = "scDEAL_temp/"
    bulk_model: str = "scDEAL_temp/"
    sc_model: str = "scDEAL_temp/"
    epochs_bulk: int = 50
    epochs: int = 100
    patience: int = 10
    batch_size: int = 32
    bulk_h_dims: str = "512,256"
    predictor_h_dims: str = "64,32"
    dimreduce: str = "DAE"
    freeze_pretrain: int = 0
    pretrain: str = "True"
    regularization_weight: float = 1.0
    data_split: str = "test"
    log_normalized: bool = True
    mode: str = "scDEAL"
    lr_DA: float = 0.01
    dropout_sc: float = 0.3
    bottleneck: int = 128
    mmd_weight: float = 0.25
    balancing_strategy: str = "weighted"
    lr_bulk: float = 0.01
    lr_sc: float = 0.01
    dropout_bulk: float = 0.3
    tune_threshold: bool = True
    binarize_source: bool = True
    lr: float = 0.01
    dropout: float = 0.3


@dataclass
class Ssda4DrugArgs:
    drug: str
    epochs: int = 100
    batch_size: int = 128
    n_shot: int = 3
    encoder: str = "DAE"
    method: str = "adv"
    data_split: str = "test"
    mode: str = "SSDA4Drug"
    log_normalized: bool = True
    patience: int = 10
    epsilon: float = 0.0
    lr: float = 0.001
    dropout: float = 0.3
    encoder_h_dims: str = "512,256"
    predictor_h_dims: str = "64,32"
    lambda_ent: float = 0.1
    balancing_strategy: str = "weighted"
    tune_threshold: bool = True
    binarize_source: bool = True


@dataclass
class ScAtdArgs:
    drug: str
    epochs: int = 100
    epochs_classifier: int = 50
    patience: int = 10
    data_split: str = "test"
    log_normalized: bool = True
    mode: str = "scATD"
    lr: float = 1e-3
    batch_size: int = 128
    weight_decay: float = 1e-3
    mmd_weight: float = 1.0
    balancing_strategy: str = "weighted"
    z_dim: int = 421
    hidden_dim_layer0: int = 1664
    hidden_dim_layer_out_Z: int = 359
    pretrained_model_path: str = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "frameworks",
            "scATD",
            "pretrained_models",
            "checkpoint_fold1_epoch_30.pth",
        )
    )
    tune_threshold: bool = True
    binarize_source: bool = True


def _normalize_value(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    if isinstance(value, (int, np.integer)):
        return int(numeric)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and "." not in stripped and "e" not in stripped.lower():
            return int(numeric)
    return float(numeric)


def _coerce_config_value(value: object) -> object:
    normalized = _normalize_value(value)
    if normalized is None:
        return None
    if isinstance(normalized, float) and normalized.is_integer():
        return int(normalized)
    return normalized


def matches_baseline_params(row: pd.Series, model_name: str) -> bool:
    if model_name == "CatBoost_fs":
        return True
    initial_params = BASELINE_PARAMS.get(model_name)
    if initial_params is None:
        return False
    row_dict = row.to_dict()
    for param_name, baseline_value in initial_params.items():
        if param_name not in row_dict:
            return False
        if _normalize_value(row_dict[param_name]) != _normalize_value(baseline_value):
            return False
    return True


def compute_selection_masks(group: pd.DataFrame, model_name: str) -> Dict[str, pd.Series]:
    baseline_mask = group.apply(lambda row: matches_baseline_params(row, model_name), axis=1)

    if group["source_val_mcc"].notna().any():
        max_mcc = group["source_val_mcc"].max()
        tier1 = group["source_val_mcc"].eq(max_mcc)
        tie_score = group["source_val_auc"].fillna(0.0) + group["source_val_auprc"].fillna(0.0)
        fair_mask = tier1 & tie_score.eq(tie_score[tier1].max())
    else:
        fair_mask = pd.Series(False, index=group.index)

    if group["target_test_auc"].notna().any():
        leak_mask = group["target_test_auc"].eq(group["target_test_auc"].max())
    else:
        leak_mask = pd.Series(False, index=group.index)

    if model_name == "CatBoost_fs" and not baseline_mask.any():
        baseline_mask = pd.Series(True, index=group.index)

    return {
        "baseline": baseline_mask,
        "fairly_tuned": fair_mask,
        "leak_tuned": leak_mask,
    }


def _warn_if_tied(selection_name: str, model_name: str, drug: str, target: str, count: int) -> None:
    if count <= 1:
        return
    warnings.warn(
        f"{selection_name}: found {count} tied configs for {model_name} on {drug}::{target}. "
        "Logging all tied configs as separate runs.",
        RuntimeWarning,
    )


def _build_param_dict(row: pd.Series, model_name: str) -> Dict[str, object]:
    params: Dict[str, object] = {}
    for key in MODEL_PARAM_KEYS.get(model_name, []):
        if key not in row.index:
            continue
        value = row[key]
        coerced = _coerce_config_value(value)
        if coerced is not None:
            params[key] = coerced
    if model_name.startswith("CatBoost") and "lr" in params:
        params["learning_rate"] = params.pop("lr")
    if model_name == "SSDA4Drug":
        use_epsilon = bool(params.get("use_epsilon", False))
        if not use_epsilon:
            params["epsilon"] = 0.0
    return params


def prepare_data_for_target(drug: str, target_tag: str, data_dir: str) -> Optional[Dict[str, object]]:
    all_files = [f for f in os.listdir(data_dir) if drug in f and f.endswith(".csv")]

    x_source_files = [f for f in all_files if f.startswith("X_") and "bulk" in f]
    y_source_files = [f for f in all_files if f.startswith("y_") and "bulk" in f]
    if not x_source_files or not y_source_files:
        warnings.warn(f"[{drug}] Missing source bulk files. Skipping.", RuntimeWarning)
        return None

    x_source_raw = pd.read_csv(os.path.join(data_dir, x_source_files[0]), index_col=0)
    y_source_raw = pd.read_csv(os.path.join(data_dir, y_source_files[0]), index_col=0)["viability"]
    y_source = 1 - y_source_raw
    y_source_bin = (y_source >= 0.5).astype(int)

    x_source = convert_to_ensembl(x_source_raw.copy())
    x_source = drop_all_nan_and_deduplicate(x_source)
    x_source = normalize_cpm_log1p_if_counts(x_source, "X_source")

    x_target_files = [f for f in all_files if f.startswith("X_") and target_tag in f]
    y_target_files = [f for f in all_files if f.startswith("y_") and target_tag in f]
    if not x_target_files or not y_target_files:
        warnings.warn(
            f"[{drug}] Missing target files for dataset '{target_tag}'. Skipping.",
            RuntimeWarning,
        )
        return None

    x_target_raw = pd.read_csv(os.path.join(data_dir, x_target_files[0]), index_col=0)
    y_target_raw = pd.read_csv(os.path.join(data_dir, y_target_files[0]), index_col=0)
    y_target = y_target_raw.iloc[:, 0]

    x_target = convert_to_ensembl(x_target_raw.copy())
    x_target = drop_all_nan_and_deduplicate(x_target)
    x_target = normalize_cpm_log1p_if_counts(x_target, "X_target")

    (x_source, x_target), common_genes = intersect_genes(x_source, x_target)
    if not common_genes:
        warnings.warn(f"[{drug}] No overlapping genes for target '{target_tag}'. Skipping.", RuntimeWarning)
        return None

    x_source_train, x_source_test, y_source_train, y_source_test = train_test_split(
        x_source,
        y_source,
        test_size=0.2,
        random_state=SEED,
        stratify=y_source_bin,
    )
    y_source_train_bin = (y_source_train >= 0.5).astype(int)
    x_source_train, x_source_val, y_source_train, y_source_val = train_test_split(
        x_source_train,
        y_source_train,
        test_size=0.2,
        random_state=SEED,
        stratify=y_source_train_bin,
    )

    x_target_train, x_target_test, y_target_train, y_target_test = train_test_split(
        x_target,
        y_target,
        test_size=0.2,
        random_state=SEED,
        stratify=y_target,
    )

    scaler = StandardScaler()
    x_source_train = pd.DataFrame(
        scaler.fit_transform(x_source_train),
        index=x_source_train.index,
        columns=x_source_train.columns,
    )
    x_source_val = pd.DataFrame(
        scaler.transform(x_source_val),
        index=x_source_val.index,
        columns=x_source_val.columns,
    )
    x_source_test = pd.DataFrame(
        scaler.transform(x_source_test),
        index=x_source_test.index,
        columns=x_source_test.columns,
    )

    feature_order = list(scaler.feature_names_in_)
    x_target_train = pd.DataFrame(
        scaler.transform(x_target_train.reindex(columns=feature_order)),
        index=x_target_train.index,
        columns=feature_order,
    )
    x_target_test = pd.DataFrame(
        scaler.transform(x_target_test.reindex(columns=feature_order)),
        index=x_target_test.index,
        columns=feature_order,
    )

    return {
        "x_train_source": x_source_train,
        "y_train_source": y_source_train,
        "x_val_source": x_source_val,
        "y_val_source": y_source_val,
        "x_test_source": x_source_test,
        "y_test_source": y_source_test,
        "x_train_target": x_target_train,
        "y_train_target": y_target_train,
        "x_test_target": x_target_test,
        "y_test_target": y_target_test,
        "X_target_independent": x_target_train,
        "y_target_independent": y_target_train,
        "target_file": target_tag,
        "independent_target_file": f"{target_tag}_train",
    }


def _default_args_for_model(model_name: str, drug: str) -> Optional[Dict[str, object]]:
    if model_name == "SCAD":
        return asdict(ScadArgs(drug=drug))
    if model_name == "scDEAL":
        return asdict(ScDealArgs(drug=drug))
    if model_name == "SSDA4Drug":
        return asdict(Ssda4DrugArgs(drug=drug))
    if model_name == "scATD":
        return asdict(ScAtdArgs(drug=drug))
    if model_name in {"CatBoost_source_only", "CatBoost_target_only", "CatBoost_combined"}:
        return {
            "drug": drug,
            "data_split": "test",
            "mode": model_name,
            "tune_threshold": True,
        }
    if model_name == "CatBoost_fs":
        return {}
    return None


def retrain_and_evaluate(
    model_name: str,
    drug: str,
    target: str,
    params: Dict[str, object],
    data: Dict[str, object],
) -> Dict[str, object]:
    if model_name == "CatBoost_fs":
        results = run_catboost_fewshot_baseline(
            data["x_val_source"],
            data["y_val_source"],
            data["x_test_source"],
            data["y_test_source"],
            data["x_train_target"],
            data["y_train_target"],
            data["x_test_target"],
            data["y_test_target"],
            X_target_independent=data["X_target_independent"],
            y_target_independent=data["y_target_independent"],
            seed=SEED,
        )
    elif model_name in {"CatBoost_source_only", "CatBoost_target_only", "CatBoost_combined"}:
        args = _default_args_for_model(model_name, drug) or {}
        results = run_catboost_benchmark(
            model_type=model_name,
            **args,
            **data,
            catboost_params=params,
            seed=SEED,
        )
    else:
        args = _default_args_for_model(model_name, drug)
        if args is None:
            raise ValueError(f"Unsupported model: {model_name}")
        args.update(params)
        benchmark_map = {
            "SCAD": run_scad_benchmark,
            "scDEAL": run_scdeal_benchmark,
            "SSDA4Drug": run_ssda4drug_benchmark,
            "scATD": run_scatd_benchmark,
        }
        results = benchmark_map[model_name](args, **data, seed=SEED)

    target_train_metrics = results.get("independent_target_test", {})
    if target_train_metrics:
        results["target_train"] = target_train_metrics
    else:
        results["target_train"] = {}
    return results


def flatten_results(results: Dict[str, object]) -> Dict[str, object]:
    flat: Dict[str, object] = {}
    for split_name, split_metrics in results.items():
        if isinstance(split_metrics, dict):
            if split_name == "independent_target_test":
                continue
            for metric_name, value in split_metrics.items():
                flat[f"{split_name}_{metric_name}"] = value
        else:
            flat[split_name] = split_metrics
    return flat


def clean_wandb_config(config: Dict[str, object]) -> Dict[str, object]:
    cleaned: Dict[str, object] = {}
    for key, value in config.items():
        if isinstance(value, np.generic):
            cleaned[key] = value.item()
        elif isinstance(value, pd.Timestamp):
            cleaned[key] = value.isoformat()
        elif pd.isna(value):
            continue
        else:
            cleaned[key] = value
    return cleaned


def iter_selected_configs(df: pd.DataFrame) -> Iterable[Dict[str, object]]:
    for (drug, target, model_name), group in df.groupby(["drug", "target", "model_name"], sort=False):
        group = group.copy()
        masks = compute_selection_masks(group, model_name)
        for selection_name, mask in masks.items():
            selected = group.loc[mask].copy()
            if selected.empty:
                warnings.warn(
                    f"No rows matched selection '{selection_name}' for {model_name} on {drug}::{target}.",
                    RuntimeWarning,
                )
                continue
            _warn_if_tied(selection_name, model_name, drug, target, len(selected))
            for tie_rank, (_, row) in enumerate(selected.iterrows(), start=1):
                yield {
                    "drug": drug,
                    "target": target,
                    "model_name": model_name,
                    "selection_type": selection_name,
                    "tie_count": len(selected),
                    "tie_rank": tie_rank,
                    "row": row,
                }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain baseline, fairly tuned, and leak-tuned configs from plots/wandb.csv "
            "and log target-train and target-test metrics to Weights & Biases."
        )
    )
    parser.add_argument("--csv_path", default=DEFAULT_CSV_PATH, help="Path to the exported W&B CSV.")
    args = parser.parse_args()

    set_seed(SEED)
    initialize_symbol_map(SYMBOL_ENSEMBL_MAP)

    df = pd.read_csv(args.csv_path)
    required_cols = {"drug", "target", "model_name", "source_val_mcc", "source_val_auc", "source_val_auprc", "target_test_auc"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"wandb.csv is missing required columns: {sorted(missing_cols)}")

    data_cache: Dict[tuple[str, str], Optional[Dict[str, object]]] = {}

    for item in iter_selected_configs(df):
        drug = str(item["drug"])
        target = str(item["target"])
        model_name = str(item["model_name"])
        selection_type = str(item["selection_type"])
        tie_count = int(item["tie_count"])
        tie_rank = int(item["tie_rank"])
        row = item["row"]

        cache_key = (drug, target)
        if cache_key not in data_cache:
            data_cache[cache_key] = prepare_data_for_target(drug, target, DATA_DIR)
        data = data_cache[cache_key]
        if data is None:
            continue

        selected_params = _build_param_dict(row, model_name)
        print(
            f"Retraining {model_name} on {drug}::{target} with selection={selection_type} "
            f"(tie {tie_rank}/{tie_count})"
        )

        run_config = clean_wandb_config(
            {
                "seed": SEED,
                "drug": drug,
                "target": target,
                "model_name": model_name,
                "selection_type": selection_type,
                "tie_count": tie_count,
                "tie_rank": tie_rank,
                "csv_row_name": row.get("name"),
                "csv_timestamp": row.get("_timestamp"),
                **selected_params,
            }
        )

        run_name = (
            f"{model_name}_{drug}_{target}_{selection_type}_"
            f"tie{tie_rank}of{tie_count}"
        )

        with wandb.init(
            project=WANDB_PROJECT,
            group=f"{drug}_{target}_{model_name}",
            job_type="retrain_selected_config",
            name=run_name,
            config=run_config,
            reinit=True,
        ) as run:
            try:
                results = retrain_and_evaluate(model_name, drug, target, selected_params, data)
            except Exception as exc:
                warning_msg = (
                    f"Retraining failed for {model_name} on {drug}::{target} "
                    f"({selection_type}, tie {tie_rank}/{tie_count}): {exc}"
                )
                warnings.warn(warning_msg, RuntimeWarning)
                run.log({"error": warning_msg})
                continue

            run.log(flatten_results(results))


if __name__ == "__main__":
    main()
