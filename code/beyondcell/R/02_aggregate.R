## Aggregate per-pair summary rows into top-level tables.
## Run from project root: Rscript R/02_aggregate.R
##
## Produces three tables under results/:
##   summary.csv          - one row per pair from the SSc pass (results/per_pair).
##   summary_psc.csv       - one row per pair from the PSc fallback pass
##                           (results/per_pair_psc), if that pass was run.
##   summary_combined.csv  - the manuscript view: one row per (drug, dataset),
##                           taking the SSc result when the drug is in SSc and
##                           falling back to the PSc result otherwise (this is
##                           how Cisplatin and Palbociclib get a score). An
##                           extra `pass` / `bcs_root` column records which pass
##                           each row came from so downstream steps can locate
##                           that pair's bcs_matrix.rds.

suppressPackageStartupMessages({
  library(data.table)
})

aggregate_pass <- function(in_root) {
  if (!dir.exists(in_root)) return(NULL)
  rows <- list.files(in_root, pattern = "^summary_row\\.csv$",
                     recursive = TRUE, full.names = TRUE)
  if (!length(rows)) return(NULL)
  dt <- rbindlist(lapply(rows, fread), use.names = TRUE, fill = TRUE)
  setorder(dt, drug, dataset)
  dt[]
}

ssc_root <- file.path("results", "per_pair")
psc_root <- file.path("results", "per_pair_psc")

ssc <- aggregate_pass(ssc_root)
if (is.null(ssc)) stop("No summary_row.csv files found under ", ssc_root)
fwrite(ssc, file.path("results", "summary.csv"))
cat("wrote results/summary.csv with", nrow(ssc), "rows\n")

psc <- aggregate_pass(psc_root)
if (!is.null(psc)) {
  fwrite(psc, file.path("results", "summary_psc.csv"))
  cat("wrote results/summary_psc.csv with", nrow(psc), "rows\n")
}

## ---- Combined manuscript table -------------------------------------------
## SSc is primary; PSc is only a fallback for drugs absent from SSc.
as_bool <- function(x) isTRUE(as.logical(x))

ssc[, pass := "SSc"][, bcs_root := ssc_root]
if (!is.null(psc)) psc[, pass := "PSc"][, bcs_root := psc_root]

keys <- unique(rbind(ssc[, .(drug, dataset)],
                     if (!is.null(psc)) psc[, .(drug, dataset)]))
setorder(keys, drug, dataset)

combined <- rbindlist(lapply(seq_len(nrow(keys)), function(i) {
  d <- keys$drug[i]; ds <- keys$dataset[i]
  s <- ssc[drug == d & dataset == ds]
  p <- if (!is.null(psc)) psc[drug == d & dataset == ds] else psc[0]
  ## Prefer the pass that actually placed the drug in its collection.
  if (nrow(s) && as_bool(s$drug_in_collection[1])) return(s[1])
  if (nrow(p) && as_bool(p$drug_in_collection[1])) return(p[1])
  if (nrow(s)) return(s[1])
  p[1]
}), use.names = TRUE, fill = TRUE)

setorder(combined, drug, dataset)
fwrite(combined, file.path("results", "summary_combined.csv"))
cat("wrote results/summary_combined.csv with", nrow(combined), "rows\n\n")

print(combined[, .(drug, dataset, n_cells, collection, pass,
                   drug_in_collection, auroc)])
