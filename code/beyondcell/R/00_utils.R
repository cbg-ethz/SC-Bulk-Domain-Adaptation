## Shared helpers for BeyondCell per-pair analysis.
##
## All scripts assume the working directory is the project root (the
## SLURM and local-run wrappers `cd` there before invoking Rscript).
##
## Conventions for the (X, y) CSVs in ./data:
##   X_<Drug>_<Dataset>.csv : rows = cells, cols = genes, values = CPM-like
##                            (continuous, already per-cell library-size
##                            normalized — NOT raw counts).
##   y_<Drug>_<Dataset>.csv : 2 cols (cell-id, response). response is 0/1.
##
## Gene identifiers are heterogeneous across datasets:
##   - ENSEMBL (ENSG…)         → mapped to HGNC via symbol_ensembl_map.txt
##   - HGNC symbols            → used as-is
##   - hg19-prefixed symbols   → "hg19_" prefix stripped, then used as-is
## BeyondCell signatures are defined in HGNC symbols, so the matrix has to
## arrive in HGNC space.

## BeyondCell 2.2 was written against Seurat v4 and accesses the `counts`
## slot directly. Seurat 5 splits assays into layers (`Assay5`), which breaks
## that path. Force every Seurat object created here to use the v3 `Assay`
## class. The option must be set *before* the package loads.
options(Seurat.object.assay.version = "v3")

suppressPackageStartupMessages({
  library(data.table)
  library(Matrix)
  library(Seurat)
})

DATA_DIR     <- "data"
ENS2SYM_FILE <- file.path(DATA_DIR, "symbol_ensembl_map.txt")

## Resolve a data file for a (drug, dataset) pair.
##
## The upstream processed-data directory (mibohl's) has been reorganized since
## the first sweep: several datasets were renamed by dropping a trailing tag
## (e.g. GSE228154_LT -> GSE228154, GSE149214_D11_PC9 -> GSE149214,
## GSE111014_CLL -> GSE111014). Our per-pair result folders and summaries still
## carry the original tagged dataset IDs. To keep the pipeline runnable against
## the current data without recomputing, resolve the file by trying the exact
## name first and then progressively stripping trailing `_<tag>` segments.
## Returns the path, or NULL if nothing matches (dataset removed upstream).
resolve_data_file <- function(prefix, drug, dataset) {
  ds <- dataset
  repeat {
    fp <- file.path(DATA_DIR, sprintf("%s_%s_%s.csv", prefix, drug, ds))
    if (file.exists(fp)) return(fp)
    if (!grepl("_", ds)) return(NULL)        # no more suffix to strip
    ds <- sub("_[^_]+$", "", ds)             # drop the last _token
  }
}

.ens2sym_cache <- NULL

load_ens2sym <- function() {
  if (!is.null(.ens2sym_cache)) return(.ens2sym_cache)
  m <- fread(ENS2SYM_FILE)
  setnames(m, c("ensembl", "synonym", "symbol"))
  m <- unique(m[, .(ensembl, symbol)])
  m <- m[!is.na(symbol) & symbol != "" & !is.na(ensembl) & ensembl != ""]
  m <- m[, .(symbol = symbol[1]), by = ensembl]
  .ens2sym_cache <<- m
  m
}

detect_gene_flavor <- function(gene_ids) {
  head_ids <- head(gene_ids, 50)
  if (mean(grepl("^ENSG\\d+", head_ids)) > 0.5) return("ensembl")
  if (mean(grepl("^hg19_", head_ids)) > 0.5)    return("hg19_prefixed")
  "symbol"
}

harmonize_gene_ids <- function(gene_ids) {
  flavor <- detect_gene_flavor(gene_ids)
  if (flavor == "ensembl") {
    map <- load_ens2sym()
    ## Some datasets ship versioned ENSEMBL IDs (`ENSG00000241860.6`). The
    ## map carries unversioned IDs only, so strip the `.<n>` suffix first.
    gene_ids_unversioned <- sub("\\.\\d+$", "", gene_ids)
    sym <- map$symbol[match(gene_ids_unversioned, map$ensembl)]
    keep <- which(!is.na(sym) & sym != "")
    list(symbols = sym[keep], keep = keep, flavor = flavor)
  } else if (flavor == "hg19_prefixed") {
    sym <- sub("^hg19_", "", gene_ids)
    keep <- seq_along(sym)
    list(symbols = sym, keep = keep, flavor = flavor)
  } else {
    list(symbols = gene_ids, keep = seq_along(gene_ids), flavor = flavor)
  }
}

#' Load X_<drug>_<dataset>.csv as a (gene × cell) dgCMatrix in HGNC space.
load_expression <- function(drug, dataset) {
  fp <- resolve_data_file("X", drug, dataset)
  if (is.null(fp)) stop(sprintf("Expression file not found for %s / %s", drug, dataset))
  dt <- fread(fp)
  cell_ids <- as.character(dt[[1]])
  dt[, (1L) := NULL]
  ## A handful of files have one duplicated cell barcode (CreateSeuratObject
  ## would auto-suffix and that breaks the response-label join). Drop the
  ## duplicate occurrences, keep the first.
  if (anyDuplicated(cell_ids)) {
    keep_cell <- !duplicated(cell_ids)
    cell_ids  <- cell_ids[keep_cell]
    dt        <- dt[keep_cell]
  }
  gene_ids <- colnames(dt)

  harm <- harmonize_gene_ids(gene_ids)
  m <- as.matrix(dt[, harm$keep, with = FALSE])
  rownames(m) <- cell_ids
  colnames(m) <- harm$symbols

  if (anyDuplicated(colnames(m))) {
    m <- t(rowsum(t(m), group = colnames(m), reorder = FALSE))
  }

  m <- t(m)                              # (gene × cell)
  m <- as(m, "CsparseMatrix")
  attr(m, "gene_flavor") <- harm$flavor
  m
}

load_labels <- function(drug, dataset) {
  fp <- resolve_data_file("y", drug, dataset)
  if (is.null(fp)) stop(sprintf("Label file not found for %s / %s", drug, dataset))
  dt <- fread(fp)
  setnames(dt, c("cell", "response"))
  setNames(as.integer(dt$response), dt$cell)
}

#' Build a Seurat object from CPM data. log1p directly populates the `data`
#' slot — we skip NormalizeData() because input is already library-size
#' normalized.
build_seurat_cpm <- function(mat, project = "beyondcell") {
  seu <- CreateSeuratObject(counts = mat, project = project,
                            min.cells = 3, min.features = 200)
  ## Belt-and-braces: if the global option above didn't take effect (e.g.
  ## Seurat already loaded), coerce explicitly to v3.
  if (inherits(seu[[DefaultAssay(seu)]], "Assay5")) {
    seu[[DefaultAssay(seu)]] <- as(seu[[DefaultAssay(seu)]], "Assay")
  }
  SetAssayData(seu, slot = "data",
               new.data = log1p(GetAssayData(seu, slot = "counts")))
}

#' Map a drug query string to the signature IDs in a BeyondCell collection.
#'
#' SSc/PSc rows in the BCS matrix are keyed by cryptic signature IDs
#' (e.g. "sig-20879"), not drug names. The drug → ID mapping lives in
#' `collection@info$preferred.drug.names` and `collection@info$drugs`
#' (both aligned with `collection@info$IDs`, one row per alias).
#'
#' Returns a list with:
#'   - ids:                unique signature IDs that match the query
#'   - n_pure_mono:        of those, how many are monotherapy (no ":" in pref name)
#'   - matched_pref_names: the preferred drug names hit (unique)
find_drug_signatures <- function(collection, drug_query) {
  inf  <- collection@info
  q    <- toupper(drug_query)
  pref <- toupper(unlist(inf$preferred.drug.names))
  drg  <- toupper(unlist(inf$drugs))
  hit  <- grepl(q, pref, fixed = TRUE) | grepl(q, drg, fixed = TRUE)
  ids  <- unique(inf$IDs[hit])
  pn   <- unique(inf$preferred.drug.names[hit])
  list(
    ids                 = ids,
    n_pure_mono         = sum(!grepl(":", pn, fixed = TRUE)),
    matched_pref_names  = pn
  )
}

seurat_downstream <- function(seu,
                              n_variable = 2000,
                              n_pcs = 30,
                              cluster_resolution = 0.6) {
  seu <- FindVariableFeatures(seu, nfeatures = n_variable, verbose = FALSE)
  seu <- ScaleData(seu, verbose = FALSE)
  npc <- min(n_pcs, ncol(seu) - 1L)
  seu <- RunPCA(seu, npcs = npc, verbose = FALSE)
  seu <- FindNeighbors(seu, dims = seq_len(npc), verbose = FALSE)
  seu <- FindClusters(seu, resolution = cluster_resolution, verbose = FALSE)
  seu <- RunUMAP(seu, dims = seq_len(npc), verbose = FALSE)
  seu
}
