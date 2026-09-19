## Per-pair classification metrics: AUROC, AUPRC, MCC.
##
## Reads each pair's saved BCS matrix (results/<bcs_root>/<pair>/bcs_matrix.rds)
## plus the sensitivity labels (data/y_<drug>_<dataset>.csv) and recomputes a
## consistent classification-metric triple against the per-cell labels.
##
## Driven by results/summary_combined.csv (produced by R/02_aggregate.R), so a
## pair scored via the PSc fallback (Cisplatin, Palbociclib) is picked up from
## results/per_pair_psc/ automatically via that table's `bcs_root` column.
##
## Score definition matches R/01_run_pair.R exactly:
##   score_cell = mean BCS over the *all matched* signature rows (mono + combo),
##   orientation: higher BCS -> predict "sensitive" (y = 1), as in the AUROC
##   already stored in results/summary.csv (pROC direction = "<").
##
## Metrics:
##   AUROC  - pROC::auc, reproduces summary.csv$auroc (sanity check).
##   AUPRC  - average precision (area under precision-recall), positive = sensitive.
##            Compare against `prevalence` (the random-classifier AUPRC).
##   MCC    - Matthews correlation at the PREVALENCE-MATCHED operating point:
##            the k highest-BCS cells are called sensitive, k = #sensitive cells.
##            Orientation is fixed (higher BCS -> sensitive), so anti-correlated
##            pairs get a NEGATIVE MCC instead of being silently re-oriented.
##
## Run from project root:  Rscript R/03_classification_metrics.R

suppressPackageStartupMessages({
  library(data.table)
  library(pROC)
})
source("R/00_utils.R")

## average precision = sum_{positives} precision@rank / P  (sklearn-style, no interp)
ap_auprc <- function(score, y) {
  P <- sum(y == 1L)
  if (P == 0L) return(NA_real_)
  o     <- order(score, decreasing = TRUE)
  ys    <- y[o]
  cumTP <- cumsum(ys)
  prec  <- cumTP / seq_along(ys)
  sum(prec[ys == 1L]) / P
}

## MCC at the prevalence-matched cut: top-k by score = predicted sensitive, k = #pos
mcc_prevalence <- function(score, y) {
  P <- sum(y == 1L); N <- sum(y == 0L)
  if (P == 0L || N == 0L) return(NA_real_)
  pred <- as.integer(rank(-score, ties.method = "first") <= P)
  TP <- sum(pred == 1L & y == 1L); FP <- sum(pred == 1L & y == 0L)
  FN <- sum(pred == 0L & y == 1L); TN <- sum(pred == 0L & y == 0L)
  denom <- sqrt(as.numeric(TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
  if (denom == 0) return(0)
  ((as.numeric(TP) * TN) - (as.numeric(FP) * FN)) / denom
}

## Prefer the combined table (SSc + PSc fallback); fall back to SSc-only.
summ_path <- if (file.exists("results/summary_combined.csv"))
  "results/summary_combined.csv" else "results/summary.csv"
summ <- fread(summ_path)
if (is.null(summ$bcs_root)) summ[, bcs_root := file.path("results", "per_pair")]
if (is.null(summ$collection)) summ[, collection := "SSc"]

res <- rbindlist(lapply(seq_len(nrow(summ)), function(i) {
  r   <- summ[i]
  pid <- sprintf("%s_%s", r$drug, r$dataset)
  base <- data.table(drug = r$drug, dataset = r$dataset, collection = r$collection,
                     n_eval = NA_integer_, n_sens = NA_integer_, prevalence = NA_real_,
                     auroc = NA_real_, auprc = NA_real_, auprc_baseline = NA_real_,
                     mcc = NA_real_, note = "")
  if (!isTRUE(as.logical(r$drug_in_collection))) {
    base$note <- "drug not in any collection (SSc/PSc)"
    return(base)
  }
  bcs  <- readRDS(file.path(r$bcs_root, pid, "bcs_matrix.rds"))
  hits <- intersect(strsplit(r$matched_signatures, ";")[[1]], rownames(bcs))
  score <- colMeans(bcs[hits, , drop = FALSE], na.rm = TRUE)
  ## Labels come from the upstream data dir; a handful of datasets were removed
  ## there after the original sweep (not part of the final study panel) — skip
  ## those with a note instead of crashing the whole table.
  labs  <- tryCatch(load_labels(r$drug, r$dataset), error = function(e) NULL)
  if (is.null(labs)) {
    base$note <- "source labels removed upstream (excluded from panel)"
    return(base)
  }
  common <- intersect(names(score), names(labs))
  d  <- score[common]
  yv <- as.integer(labs[common])
  ok <- !is.na(d) & !is.na(yv)
  d  <- d[ok]; yv <- yv[ok]
  if (length(unique(yv)) != 2L || length(d) < 10L) {
    base$note <- "too few cells / single class"
    return(base)
  }
  base$n_eval         <- length(d)
  base$n_sens         <- sum(yv == 1L)
  base$prevalence     <- mean(yv == 1L)
  base$auroc          <- as.numeric(pROC::auc(pROC::roc(yv, d, quiet = TRUE, direction = "<")))
  base$auprc          <- ap_auprc(d, yv)
  base$auprc_baseline <- mean(yv == 1L)
  base$mcc            <- mcc_prevalence(d, yv)
  base
}))

setorder(res, drug, dataset)
out <- file.path("results", "classification_metrics.csv")
fwrite(res, out)
cat("wrote", out, "with", nrow(res), "rows\n\n")

## Console view: reproduce-check against summary.csv$auroc + rounded table
chk <- merge(res[, .(drug, dataset, auroc_new = auroc)],
             summ[, .(drug, dataset, auroc_old = auroc)],
             by = c("drug", "dataset"), all.x = TRUE)
chk[, d := abs(auroc_new - auroc_old)]
cat("max |AUROC_new - AUROC_summary| =",
    format(max(chk$d, na.rm = TRUE), digits = 3), "(should be ~0)\n\n")

pr <- res[, .(drug, dataset, coll = collection, n = n_eval,
              prev = round(prevalence, 2),
              AUROC = round(auroc, 3), AUPRC = round(auprc, 3),
              MCC = round(mcc, 3), note)]
print(pr, nrows = 100)
