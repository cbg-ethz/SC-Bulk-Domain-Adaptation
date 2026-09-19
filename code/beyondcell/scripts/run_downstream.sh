#!/bin/bash
# Reproducible post-compute chain: aggregate -> metrics -> figures.
#
# Assumes the per-pair BeyondCell runs already exist under results/per_pair
# (SSc) and, for the PSc-fallback drugs, results/per_pair_psc. Those heavy
# steps run on compute nodes via scripts/submit_array.sh; this chain is light
# (reads the saved bcs_matrix.rds + CSVs) and is safe to run interactively.
#
# Usage (from project root):  scripts/run_downstream.sh

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

source /cluster/home/mesteban/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/work/bewi/members/mesteban/miniconda3/envs/beyondcell

echo "[$(date)] 02_aggregate: SSc + PSc -> summary_combined.csv"
Rscript R/02_aggregate.R

echo "[$(date)] 03_classification_metrics: AUROC / AUPRC / MCC"
Rscript R/03_classification_metrics.R

echo "[$(date)] 04_figures: benchmark figure"
Rscript R/04_figures.R

echo "[$(date)] capturing sessionInfo -> results/sessionInfo.txt"
Rscript -e 'writeLines(capture.output(sessionInfo()), "results/sessionInfo.txt")'

echo "[$(date)] downstream complete."
