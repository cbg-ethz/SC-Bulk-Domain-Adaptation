#!/bin/bash
# Run a single (drug, dataset) pair locally without SLURM. Useful for testing.
#
# Usage:
#     scripts/run_pair_local.sh Cisplatin GSE117872_HN120

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <Drug> <Dataset>" >&2
  exit 1
fi

DRUG="$1"
DATASET="$2"

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

source /cluster/home/mesteban/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/work/bewi/members/mesteban/miniconda3/envs/beyondcell

Rscript R/01_run_pair.R --drug "$DRUG" --dataset "$DATASET"
