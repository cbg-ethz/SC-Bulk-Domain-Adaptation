#!/bin/bash
#SBATCH --job-name=beyondcell
#SBATCH --output=logs/beyondcell_%A_%a.out
#SBATCH --error=logs/beyondcell_%A_%a.err
#SBATCH --partition=normal.24h
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=24G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-34%8

## SLURM array job: one task per (drug, dataset) pair.
##
## Submit the default SSc pass from the project root:
##     sbatch scripts/submit_array.sh
##
## To run only a subset, override the array range, e.g.
##     sbatch --array=1-3 scripts/submit_array.sh
##
## The collection, pairs file and output root are configurable via environment
## variables passed with `--export`, so the *same* script drives both passes:
##
##   SSc (default):
##     sbatch scripts/submit_array.sh
##   PSc fallback pass (drugs absent from SSc — Cisplatin, Palbociclib):
##     sbatch --array=1-4 \
##       --export=ALL,COLLECTION=PSc,PAIRS_FILE=scripts/target_pairs_psc.tsv,OUT_ROOT=results/per_pair_psc \
##       scripts/submit_array.sh
##
## The total task count must match the number of data lines in the pairs file
## (header excluded). Set --array accordingly.

set -euo pipefail

## When SLURM executes this script it copies it to /var/spool/slurm/.../scripts,
## so $BASH_SOURCE no longer points at the project. Use $SLURM_SUBMIT_DIR
## (the directory `sbatch` was invoked from) as the project root.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

## Configurable pass parameters (defaults reproduce the original SSc run).
COLLECTION="${COLLECTION:-SSc}"
OUT_ROOT="${OUT_ROOT:-results/per_pair}"
PAIRS_FILE="${PAIRS_FILE:-scripts/target_pairs.tsv}"
## Allow a relative PAIRS_FILE/OUT_ROOT (resolved against the project root).
case "$PAIRS_FILE" in /*) ;; *) PAIRS_FILE="$PROJECT_ROOT/$PAIRS_FILE" ;; esac

## Pick the (drug, dataset) row for this array task (1-based, skip header).
LINE_NO=$((SLURM_ARRAY_TASK_ID + 1))
IFS=$'\t' read -r DRUG DATASET < <(sed -n "${LINE_NO}p" "$PAIRS_FILE")

if [[ -z "${DRUG:-}" || -z "${DATASET:-}" ]]; then
  echo "ERROR: empty (drug, dataset) at line $LINE_NO of $PAIRS_FILE" >&2
  exit 1
fi

echo "[$(date)] task ${SLURM_ARRAY_TASK_ID}: drug=$DRUG dataset=$DATASET collection=$COLLECTION out_root=$OUT_ROOT"

## Activate the project conda env.
source /cluster/home/mesteban/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/work/bewi/members/mesteban/miniconda3/envs/beyondcell

Rscript R/01_run_pair.R --drug "$DRUG" --dataset "$DATASET" \
  --collection "$COLLECTION" --out_root "$OUT_ROOT"
