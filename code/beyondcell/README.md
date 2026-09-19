# BeyondCell drug-response validation

Apply [BeyondCell](https://github.com/cnio-bu/beyondcell) (Fustero-Torre et al., *Genome Med* 2021) to a panel of single-cell RNA-seq datasets where each cell has a **binary sensitivity label** for a specific drug, and check whether BeyondCell's **Beyondcell Score (BCS)** for that drug discriminates sensitive vs resistant cells.

## What "running BeyondCell" means here

For every `(drug, dataset)` pair listed in [`scripts/target_pairs.tsv`](scripts/target_pairs.tsv):

1. Load the per-cell expression matrix (`data/X_<drug>_<dataset>.csv`) and the per-cell binary response labels (`data/y_<drug>_<dataset>.csv`).
2. Harmonize gene identifiers to **HGNC symbols** (some datasets ship in ENSEMBL, one is `hg19_`-prefixed) using `data/symbol_ensembl_map.txt`.
3. Build a Seurat object, run standard downstream (`log1p` → scale → PCA → cluster → UMAP). See *Normalization* below.
4. Load the **SSc** (drug sensitivity) collection and call `bcScore()` → produces a drug × cell BCS matrix.
5. Extract the BCS row for the target drug (mean across all matching SSc signatures).
6. Validate against the `y` label: AUROC, Wilcoxon p, mean BCS in sensitive vs resistant, and the per-cell rank of the target drug among the whole SSc collection.

The expectation, if BeyondCell is well-calibrated for these data, is that the drug used in the original experiment ranks high among the drugs prioritized for the cells labeled sensitive.

## Normalization decision

The input CSVs are per-cell library size normalized (not raw counts). So the pipeline:

- **does not** call `Seurat::NormalizeData()` — that would re-divide by per-cell sum, which is already constant in these data;
- **does** apply `log1p()` to populate the `data` slot — both BeyondCell's internal ranking and Seurat's `ScaleData → RunPCA → clustering → UMAP` expect log-transformed expression. Skipping log1p and going straight to PCA on CPM gives degenerate components because variance grows with mean.

## Layout

```
beyondcell/
├── requirements.txt               dependency list (R + BeyondCell + Seurat)
├── Makefile                       pipeline entry points (see "Reproducible pipeline")
├── R/
│   ├── 00_utils.R                 I/O, gene-ID harmonization, Seurat builder, file resolver
│   ├── 01_run_pair.R              run one (drug, dataset) pair end-to-end (seeded)
│   ├── 02_aggregate.R             merge per-pair rows → summary{,_psc,_combined}.csv
│   ├── 03_classification_metrics.R  AUROC / AUPRC / MCC from saved BCS + labels
│   ├── 04_figures.R               regenerate the manuscript benchmark figure
│   └── 05_mcc_comparison.py       MCC comparison utilities
└── scripts/
    ├── target_pairs.tsv           the 27 (drug, dataset) pairs for the SSc pass
    ├── target_pairs_psc.tsv       the 4 PSc-fallback pairs (Cisplatin, Palbociclib)
    ├── submit_array.sh            SLURM array (COLLECTION/OUT_ROOT/PAIRS_FILE configurable)
    ├── run_downstream.sh          aggregate → metrics → figures → sessionInfo (light)
    └── run_pair_local.sh          single-pair runner, no SLURM (for testing)
```

## Setup

Install dependencies listed in `requirements.txt` into an R environment with BeyondCell and Seurat.

## Run

Smoke-test on a small pair before launching the whole sweep:

```bash
scripts/run_pair_local.sh Cisplatin GSE117872_HN120
```

## Reproducible pipeline

The heavy per-pair scoring runs on SLURM compute nodes; the aggregation, metrics
and figures are light and run from the saved per-pair outputs. The `Makefile`
wraps every step:

```bash
make ssc          # submit the 34-pair SSc pass          (SLURM array)
make psc          # submit the 4-pair PSc fallback pass   (SLURM array)
make downstream   # aggregate → metrics → figures → sessionInfo (after the sweeps finish)
make figures      # just re-render the figure from existing metrics
```

`make downstream` (equivalently `scripts/run_downstream.sh`) chains:

1. `R/02_aggregate.R` — rolls per-pair rows into `results/summary.csv` (SSc),
   `results/summary_psc.csv` (PSc), and `results/summary_combined.csv` (the
   manuscript view: SSc primary, PSc fallback for Cisplatin / Palbociclib).
2. `R/03_classification_metrics.R` — recomputes AUROC / AUPRC / MCC per pair
   from the saved `bcs_matrix.rds` + per-cell labels →
   `results/classification_metrics.csv`.
3. `R/04_figures.R` — regenerates `results/figures/beyondcell_benchmark.{pdf,png}`.
4. `sessionInfo()` → `results/sessionInfo.txt` for the exact package versions.

`R/01_run_pair.R` calls `set.seed(1234)` before the stochastic Seurat steps.
The BeyondCell Score and every reported metric depend only on the deterministic
`log1p`-CPM `data` slot, so AUROC/MCC are seed-independent; the seed only pins
the per-pair UMAP diagnostic plots.

### PSc fallback pass

Cisplatin and Palbociclib are absent from SSc but present in the larger PSc
(LINCS L1000) collection. They are scored by the *same* `submit_array.sh` with
overrides:

```bash
sbatch --array=1-4 \
  --export=ALL,COLLECTION=PSc,PAIRS_FILE=scripts/target_pairs_psc.tsv,OUT_ROOT=results/per_pair_psc \
  scripts/submit_array.sh
```

Alectinib is in neither collection and stays unscoreable.

### Upstream data reorganization (important)

The `data/` symlink points at a collaborator's processed-data directory, which
was reorganized after the first sweep:

- **6 datasets were renamed** by dropping a trailing tag (`GSE228154_LT` →
  `GSE228154`, `GSE149214_D11_PC9` → `GSE149214`, `GSE111014_CLL` →
  `GSE111014`, `GSE223003_Kuramochi` → `GSE223003`, `GSE163836_FCIBC02` →
  `GSE163836`, `GSE175716_HCC` → `GSE175716`). `resolve_data_file()` in
  `00_utils.R` handles this by stripping trailing `_<tag>` segments, so the
  pipeline still runs against the renamed files without recomputing.

## Drug × dataset panel

Columns are dataset IDs; rows are drugs used to generate the per-cell sensitivity labels for that dataset.

| Drug | Datasets |
|---|---|
| Cisplatin   | GSE138267, GSE117872_HN120, GSE117872_HN137 |
| Paclitaxel  | GSE163836_FCIBC02, GSE131984 |
| Ibrutinib   | GSE111014_CLL |
| Dabrafenib  | GSE164614 |
| Olaparib    | GSE223003_Kuramochi |
| Vorinostat  | JHU006 |
| Etoposide   | GSE149383_PC9 |
| Erlotinib   | GSE149383_PC9, GSE149214_D11_PC9 |
| Docetaxel   | GSE140440_DU145, GSE140440_PC3 |
| Sorafenib   | GSE175716_HCC, SCC47 |
| Gefitinib   | GSE162045_PC9, GSE202234_H1975, GSE202234_PC9, JHU006, GSE112274_PC9 |
| Afatinib    | GSE228154_LT, SCC47 |
| Alectinib   | GSE223779 |
| Crizotinib  | GSE223779 |
| Gemcitabine | GSE186960 |
| SN-38       | GSE174376 |

27 pairs total.

## Caveats

- **Drug coverage in SSc.** BeyondCell's SSc collection (610 signatures incl. pathways, 581 drug-only) is built from CTRP/GDSC sensitivity profiles and does not include every compound. Coverage for this panel, verified against `inf$preferred.drug.names` + `inf$drugs`:

  | Drug | In SSc? | Notes |
  |---|---|---|
  | **Cisplatin** | no | no entry under cisplatin or any common synonym (`CIS-PT`, `DDP`, …). Only platinum entry is `OXALIPLATIN`. **Affects 3 pairs** (GSE138267, GSE117872_HN120/HN137). |
  | **Alectinib** | no | not in collection. **Affects 1 pair** (GSE223779). |
  | All other 14 drugs | yes | at least one pure-monotherapy SSc signature available. |

  Pairs with no SSc coverage write `drug_in_collection = FALSE` and skip the SSc validation step; the full BCS matrix is still saved. **Cisplatin and Palbociclib are then recovered via the PSc fallback pass** (see *Reproducible pipeline → PSc fallback pass*), which is how they get a score in `summary_combined.csv` and the benchmark figure. Alectinib is in neither collection and remains unscoreable.

- **Signature ID vs drug name lookup.** SSc row names are cryptic IDs (`sig-20879`); we map drug → IDs via `gs_obj@info$preferred.drug.names` + `gs_obj@info$drugs`. Every match is dedup'd by ID.

- **Combo confound.** Many drugs in SSc appear in combination signatures (`VORINOSTAT:CARBOPLATIN`, etc.). We report metrics two ways per pair: across **all** matched signatures (mono + combo) and across **pure monotherapy** only (`mono_*` columns in `summary.csv`). A meaningfully large mono-vs-all gap is the cleanest hint that combos are confounding the score.
- **Label coding.** `y_*.csv` files use `1 = sensitive, 0 = resistant` based on the cell-id `_sen_`/`_res_` convention in some datasets — the AUROC reported here assumes that orientation.
- **Per-dataset clustering.** Each pair is clustered independently with `resolution = 0.6`. Therapeutic-cluster analysis (`bcUMAP`, `bcRanks`) is not produced in this first sweep; can be added later.
