[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-green.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-2026.02.24.707713-b31b1b?logo=biorxiv&logoColor=white&labelColor=gray)](https://www.biorxiv.org/content/10.64898/2026.02.24.707713)

# Domain Adaptation Benchmark Results

This repository contains the code to reproduce the results of the benchmark paper

M. Bohl, M. Esteban-Medina, N. Beerenwinkel, and K. Lenhof, [*Domain-adaptation deep learning models do not outperform simple baseline models in single-cell anti-cancer drug sensitivity prediction*](https://www.biorxiv.org/content/10.64898/2026.02.24.707713v1), [**bioRxiv**](https://www.biorxiv.org/content/10.64898/2026.02.24.707713v1) (2026).


## Setup

To reproduce the results, you need to have conda installed.
You can create the conda environment with the following command:

```bash
conda env create -f environment.yaml
```

This will create a conda environment named `benchmark` with all the necessary packages.
Additionally, you need a Weights & Biases (wandb) account. The secltion below explains a minimal configuration needed to run the `code/hyper_tuning.py` and `code/independent_evaluation.py` script.

Furthermore, you need to download and unzip the datasets used in the benchmark paper from [Zenodo](https://zenodo.org/records/17868777) into `datasets/processed/`.

### Weights & Biases Setup
- The hyperparameter tuning script logs the results to Weights & Biases (wandb). 
- Create or use an existing wandb account at https://wandb.ai and copy your API key (Profile → Settings → API Keys).
- Activate the `benchmark` environment and authenticate by running `wandb login <your_api_key>` (or set `WANDB_API_KEY=<your_api_key>` in the shell before starting the script).
- After authentication, `code/hyper_tuning.py` will create a run group per drug/target/model and store the local cache under `code/wandb/` automatically, no further setup is required.

## Hyperparameter Tuning

To reproduce the hyperparameter tuning, you can run the `hyper_tuning.py` script.
The script takes the following arguments:
- `--drugs`: A list of drugs to process.
- `--n_trials`: The number of Optuna trials.
- `--model`: The name of the model to tune.

For example, to run the hyperparameter tuning for the SCAD model on the drug Afatinib with 10 hyperparameter combinations (Optuna trials), you can run the following command:

```bash
bash -c "conda activate benchmark && python code/hyper_tuning.py --drugs Afatinib --n_trials 10 --model SCAD"
```

## Directory Structure

- `code/`: Contains the source code for the models and experiments.
- `datasets/`: Contains the datasets used in the paper.

## Code Overview

The `code/` folder gathers data processing helpers, experiment orchestration scripts, and the individual implementation of each transfer learning framework.

- `code/data_utils.py`: Shared data processing helpers for the harmonization of datasets, gene gene vocabularies (including scATD-specific alignment), etc. Also builds PyTorch dataloaders such as `CombinedDataLoader` for paired source/target batches and `create_shot_dataloaders` for semi-supervised setups.
- `code/training_utils.py`: Shared training helpers and baselines. It sets global seeds, defines callbacks (e.g., delayed early stopping), computes model metrics, and wraps framework-specific runners (`run_scad_benchmark`, `run_scdeal_benchmark`, etc.) alongside classical baselines such as CatBoost and RandomForest.
- `code/hyper_tuning.py`: Runs Optuna sweeps per drug/domain and logs trials to Weights & Biases. It standardizes preprocessing, constructs framework argument objects, and dispatches to the runners above.
- `code/independent_evaluation.py`: Repeats the preprocessing pipeline for held-out target datasets and launches framework benchmarks/few-shot baselines with consistent defaults, enabling cross-dataset comparisons.
- `code/frameworks/`: Houses the Lightning implementations of each domain adaptation method:
  - `SCAD/`: Domain-adversarial Lightning module that couples a shared encoder, response predictor, and gradient-reversal discriminator with a tunable weight lambda.
  - `scATD/`: Wraps a pre-trained Dist-VAE encoder and classifier head. `setup` loads checkpoints, aligning gene vocabularies by padding/truncation; fine-tuning alternates between frozen-classifier warm-up and optional encoder unfreezing, optimizing cross-entropy plus an RBF MMD penalty via manual optimization.
  - `scDeal/`: Implements the three-stage scDEAL workflow. Autoencoder/predictor pretraining is followed by a DaNN domain adaptation step with BCE, MMD, and Louvain-cluster similarity regularizers, orchestrated through manual optimization. Utilities in `scDEAL_utils.py` construct target KNN graphs and Louvain assignments.
  - `SSDA4Drug/`: Lightning module that implements, SSDA4Drug, with a shared encoder and classifier to which adversarial perturbations can be applied optionally. Training mixes supervised cross-entropy (source + few-shot target) with alternating entropy minimization and maximization on unlabeled target batches via `utils.adentropy`.

## Adding a New Benchmark Method

To add another method to the benchmark:

1. Add a new framework folder under `code/frameworks/<YourMethod>/` with a `main.py` entrypoint plus the model/data modules it needs (following the existing framework folders).
2. Implement a `run_<yourmethod>_benchmark(...)` function in `code/training_utils.py` that trains/evaluates the method and returns metrics in the same structure as existing runners.
3. Register the method in `code/hyper_tuning.py` by adding:
   - an argument/default config container (`<YourMethod>Args`), and
   - dispatch logic so Optuna calls your new benchmark runner.
4. Register the method in `code/independent_evaluation.py` so it can be evaluated with the same preprocessing and independent-target setup.
5. Reuse `code/data_utils.py` preprocessing helpers (gene mapping, normalization, splits) to keep comparisons fair and consistent across methods.

## Downloading Data
- To reproduce the results, you will need to download and unzip the processed datasesets used in the paper into  `datasets/processed/`. The datasets can be downloaded from [Zenodo](https://zenodo.org/records/17868777).
- To run scATD, the pre-trained model weights (file checkpoint_fold1_epoch_30.pth) need to be downloaded from figshare (https://figshare.com/articles/software/scATD/27908847) and placed in `code/frameworks/scATD/pretrained_models/`.
- If the original URL of the pretrained scATD model doesn't work, we provide a copy of the model weights in the repository on [Zenodo](https://zenodo.org/records/17868777).

    
