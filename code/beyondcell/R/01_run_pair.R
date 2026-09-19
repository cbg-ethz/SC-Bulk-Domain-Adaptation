## Per-pair BeyondCell run.
##
## Usage (from project root):
##   Rscript R/01_run_pair.R --drug Cisplatin --dataset GSE117872_HN120
##
## For each (drug, dataset) pair this script:
##   1. loads expression (X) + labels (y) from ./data,
##   2. harmonizes gene IDs to HGNC,
##   3. builds a Seurat object, log-transforms, scales, PCA, clusters, UMAP,
##   4. loads the SSc collection and runs bcScore,
##   5. extracts the BCS row for the target drug,
##   6. computes validation metrics against the y labels,
##   7. writes results under ./results/per_pair/<drug>_<dataset>/.

suppressPackageStartupMessages({
  library(optparse)
  library(beyondcell)
  library(Seurat)
  library(data.table)
  library(ggplot2)
  library(patchwork)
  library(pROC)
  library(jsonlite)
})

source("R/00_utils.R")

## Reproducibility: fixes the stochastic Seurat steps (PCA/UMAP/clustering).
## The BeyondCell Score and all validation metrics depend only on the
## deterministic log1p-CPM `data` slot, so the reported AUROC/MCC are
## seed-independent; this only pins the per-pair UMAP diagnostic plots.
set.seed(1234)

opt <- parse_args(OptionParser(option_list = list(
  make_option("--drug",       type = "character"),
  make_option("--dataset",    type = "character"),
  make_option("--out_root",   type = "character",
              default = file.path("results", "per_pair")),
  make_option("--resolution", type = "double", default = 0.6),
  make_option("--collection", type = "character", default = "SSc",
              help = "SSc or PSc")
)))

stopifnot(!is.null(opt$drug), !is.null(opt$dataset))

pair_id <- sprintf("%s_%s", opt$drug, opt$dataset)
out_dir <- file.path(opt$out_root, pair_id)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

log_line <- function(...) cat(format(Sys.time()), "|", pair_id, "|", ..., "\n")

## ---- 1. Load --------------------------------------------------------------
log_line("loading X + y")
mat    <- load_expression(opt$drug, opt$dataset)
labels <- load_labels(opt$drug, opt$dataset)

log_line(sprintf("X: %d genes x %d cells (flavor=%s); y: n=%d (sens=%d)",
                 nrow(mat), ncol(mat), attr(mat, "gene_flavor"),
                 length(labels), sum(labels == 1)))

## ---- 2. Seurat object + downstream ---------------------------------------
log_line("building Seurat object (log1p, no NormalizeData — CPM input)")
seu <- build_seurat_cpm(mat, project = pair_id)

common <- intersect(colnames(seu), names(labels))
seu$response <- labels[common][match(colnames(seu), common)]

log_line("scale / PCA / cluster / UMAP")
seu <- seurat_downstream(seu, cluster_resolution = opt$resolution)

## ---- 3. BeyondCell scoring -----------------------------------------------
log_line(sprintf("loading BeyondCell collection: %s", opt$collection))
gs_obj <- switch(opt$collection,
                 SSc = GetCollection(SSc, include.pathways = TRUE),
                 PSc = GetCollection(PSc, include.pathways = TRUE),
                 stop("Unknown collection"))

log_line("bcScore")
bc <- bcScore(seu, gs_obj, expr.thres = 0.1)

bcs_mat <- bc@normalized
saveRDS(bc,      file.path(out_dir, "bc_object.rds"))
saveRDS(bcs_mat, file.path(out_dir, "bcs_matrix.rds"))

## ---- 4. Locate the target drug in the collection -------------------------
## Signature row names are cryptic IDs (sig-XXXXX); resolve drug → IDs via
## collection metadata (preferred.drug.names + drugs).
match <- find_drug_signatures(gs_obj, opt$drug)
hits  <- intersect(match$ids, rownames(bcs_mat))
log_line(sprintf("drug='%s' → %d unique signature IDs (pure-mono: %d); %d of those present in bcs_mat",
                 opt$drug, length(match$ids), match$n_pure_mono, length(hits)))

summary_row <- list(
  drug                  = opt$drug,
  dataset               = opt$dataset,
  n_cells               = ncol(seu),
  n_genes               = nrow(seu),
  gene_flavor           = attr(mat, "gene_flavor"),
  collection            = opt$collection,
  n_signatures          = nrow(bcs_mat),
  target_signature_hits = length(hits),
  target_pure_mono      = match$n_pure_mono,
  matched_signatures    = paste(hits, collapse = ";"),
  matched_pref_names    = paste(match$matched_pref_names, collapse = ";"),
  drug_in_collection    = length(hits) > 0L
)

## ---- 5. Validation against per-cell sensitivity labels -------------------
## Also compute a "pure monotherapy" track: SSc has combo signatures like
## "VORINOSTAT:CARBOPLATIN" that confound a single-drug validation. By
## reporting both numbers per pair we can see whether the confound matters.
compute_metrics <- function(bcs_rows, prefix) {
  if (!length(bcs_rows)) return(setNames(list(NA, NA, NA, NA, NA, NA),
    paste0(prefix, c("auroc","wilcox_p","mean_bcs_sen","mean_bcs_res",
                     "mean_rank_sen","mean_rank_res"))))
  drug_bcs <- colMeans(bcs_mat[bcs_rows, , drop = FALSE], na.rm = TRUE)
  resp <- seu$response[names(drug_bcs)]
  valid <- !is.na(resp) & !is.na(drug_bcs)
  d  <- drug_bcs[valid]; yv <- resp[valid]
  if (length(unique(yv)) != 2L || length(d) < 10L) return(NULL)
  roc_obj <- pROC::roc(yv, d, quiet = TRUE, direction = "<")
  ## Per-cell rank of this drug score against all collection rows, 1 = top.
  full  <- rbind(target = drug_bcs,
                 bcs_mat[setdiff(rownames(bcs_mat), bcs_rows), , drop = FALSE])
  ranks <- apply(-full, 2L, rank, ties.method = "average")
  list(
    drug_bcs        = drug_bcs,
    auroc           = as.numeric(pROC::auc(roc_obj)),
    wilcox_p        = wilcox.test(d[yv == 1], d[yv == 0])$p.value,
    mean_bcs_sen    = mean(d[yv == 1]),
    mean_bcs_res    = mean(d[yv == 0]),
    mean_rank_sen   = mean(ranks["target", names(d)[yv == 1]]),
    mean_rank_res   = mean(ranks["target", names(d)[yv == 0]])
  )
}

if (length(hits) > 0L) {
  ## All matched signatures (mono + combos).
  m_all <- compute_metrics(hits, "")
  ## Pure monotherapy subset: signatures whose preferred name has no ":".
  pref_to_id <- function(p) {
    inf <- gs_obj@info
    unique(inf$IDs[toupper(unlist(inf$preferred.drug.names)) %in% toupper(p)])
  }
  mono_pref <- match$matched_pref_names[!grepl(":", match$matched_pref_names, fixed = TRUE)]
  mono_ids  <- intersect(pref_to_id(mono_pref), rownames(bcs_mat))
  m_mono    <- compute_metrics(mono_ids, "mono_")

  if (!is.null(m_all)) {
    summary_row$auroc                = m_all$auroc
    summary_row$wilcox_p             = m_all$wilcox_p
    summary_row$mean_bcs_sen         = m_all$mean_bcs_sen
    summary_row$mean_bcs_res         = m_all$mean_bcs_res
    summary_row$mean_rank_target_sen = m_all$mean_rank_sen
    summary_row$mean_rank_target_res = m_all$mean_rank_res
    drug_bcs <- m_all$drug_bcs
    d  <- drug_bcs[!is.na(seu$response[names(drug_bcs)]) & !is.na(drug_bcs)]
    yv <- seu$response[names(d)]

  }
  if (!is.null(m_mono)) {
    summary_row$mono_n_sigs              = length(mono_ids)
    summary_row$mono_auroc               = m_mono$auroc
    summary_row$mono_wilcox_p            = m_mono$wilcox_p
    summary_row$mono_mean_bcs_sen        = m_mono$mean_bcs_sen
    summary_row$mono_mean_bcs_res        = m_mono$mean_bcs_res
    summary_row$mono_mean_rank_sen       = m_mono$mean_rank_sen
    summary_row$mono_mean_rank_res       = m_mono$mean_rank_res
  } else {
    summary_row$mono_n_sigs = length(mono_ids)
  }

  ## ---- 6. Plots ---------------------------------------------------------
  log_line("plotting")
  seu$drug_bcs <- drug_bcs[colnames(seu)]
  p_umap_clust <- DimPlot(seu, label = TRUE) +
    ggtitle(pair_id, subtitle = "Seurat clusters")
  p_umap_resp <- DimPlot(seu, group.by = "response") +
    ggtitle(NULL, "sensitivity label (1 = sensitive)")
  p_umap_bcs <- FeaturePlot(seu, "drug_bcs") +
    ggtitle(NULL, sprintf("mean BCS — %s (SSc)", opt$drug))

  ggsave(file.path(out_dir, "umap.png"),
         (p_umap_clust | p_umap_resp | p_umap_bcs) +
           plot_layout(guides = "collect"),
         width = 15, height = 5, dpi = 150)

  if (length(unique(yv)) == 2L) {
    df <- data.frame(bcs = d,
                     response = factor(yv, levels = c(0, 1),
                                       labels = c("resistant", "sensitive")))
    p_box <- ggplot(df, aes(response, bcs, fill = response)) +
      geom_violin(trim = FALSE, alpha = .5) +
      geom_boxplot(width = .15, outlier.shape = NA) +
      theme_minimal() +
      labs(title = pair_id, y = sprintf("BCS — %s", opt$drug), x = NULL)
    ggsave(file.path(out_dir, "target_bcs_violin.png"), p_box,
           width = 5, height = 5, dpi = 150)
  }
} else {
  log_line("target drug not in collection — skipping validation + plots")
}

## ---- 7. Persist ----------------------------------------------------------
write.csv(as.data.frame(summary_row),
          file.path(out_dir, "summary_row.csv"), row.names = FALSE)
writeLines(toJSON(summary_row, auto_unbox = TRUE, pretty = TRUE),
           file.path(out_dir, "summary_row.json"))

log_line("done")
