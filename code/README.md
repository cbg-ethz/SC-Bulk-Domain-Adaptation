# Code Directory Overview

This folder contains data preprocessing utilities, benchmark runners, hyperparameter search scripts, and framework-specific implementations for transfer learning in drug response prediction.

## Top-level files

- `data_utils.py`: Shared data pipeline utilities (gene-ID conversion, normalization, split loading, dataloader creation, and source/target dataset preparation).
- `training_utils.py`: Shared training/evaluation logic for all benchmarks (metrics, threshold tuning, logging, and per-framework run functions).
- `hyper_tuning.py`: Main Optuna/W&B hyperparameter tuning script for SCAD, scDEAL, SSDA4Drug, scATD, and CatBoost baselines.
- `independent_evaluation.py`: Evaluates tuned models on independent (hold-out) target datasets


## Framework implementations

### `frameworks/SCAD`
- `frameworks/SCAD/main.py`: CLI entrypoint to train/evaluate SCAD on custom source/target CSV inputs.
- `frameworks/SCAD/modules.py`: Core SCAD building blocks (encoder, predictor, gradient-reversal, discriminator).
- `frameworks/SCAD/lightning_datamodule.py`: PyTorch Lightning data module for SCAD loaders and source/target batching.
- `frameworks/SCAD/lightning_module.py`: PyTorch Lightning training logic for SCAD domain-adversarial optimization.
- `frameworks/SCAD/callbacks.py`: Optional callbacks (e.g., latent-space UMAP plotting during validation).

### `frameworks/SSDA4Drug`
- `frameworks/SSDA4Drug/main.py`: CLI entrypoint to train/evaluate SSDA4Drug with configurable semi-supervised settings.
- `frameworks/SSDA4Drug/modules.py`: SSDA4Drug model components (encoders, predictor heads, autoencoder, GRL helpers).
- `frameworks/SSDA4Drug/lightning_datamodule.py`: Data module that creates source loaders and n-shot labeled/unlabeled target loaders.
- `frameworks/SSDA4Drug/lightning_module.py`: Lightning module implementing supervised + entropy + optional adversarial (FGM-style) training.

### `frameworks/scATD`
- `frameworks/scATD/main.py`: CLI entrypoint to run scATD with pretrained checkpoint loading and fine-tuning settings.
- `frameworks/scATD/modules.py`: Core scATD models (residual VAE, classifier wrapper, MMD loss) and checkpoint adaptation logic.
- `frameworks/scATD/lightning_datamodule.py`: Data module for scATD split loading, vocabulary alignment, and combined source/target training batches.
- `frameworks/scATD/lightning_module.py`: Two-phase scATD Lightning training (classifier warm-up then domain-adaptive fine-tuning with MMD).
- `frameworks/scATD/pretrained_models/checkpoint_fold1_epoch_30.pth`: Pretrained Dist-VAE checkpoint used to initialize scATD.

### `frameworks/scDeal`
- `frameworks/scDeal/main.py`: CLI entrypoint to train/evaluate scDEAL on custom source/target inputs.
- `frameworks/scDeal/modules.py`: scDEAL neural network components (autoencoders, predictors, and related model blocks).
- `frameworks/scDeal/lightning_datamodule.py`: Data module for scDEAL pretraining/adaptation loaders and combined domain batching.
- `frameworks/scDeal/lightning_module.py`: Lightning training workflow for scDEAL (pretraining stages + domain adaptation).
- `frameworks/scDeal/scDEAL_utils.py`: Utility functions carried from scDEAL workflows (feature selection, plotting, graph/clustering helpers).
- `frameworks/scDeal/DaNN/mmd.py`: Original DaNN MMD implementations used by scDEAL adaptation.
- `frameworks/scDeal/DaNN/loss.py`: Additional classic domain adaptation losses (DAN/JAN variants, entropy helper).