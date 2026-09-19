## Manuscript figures for the BeyondCell benchmark.
##
## Reproducibly regenerates the BeyondCell benchmark figure from
## results/classification_metrics.csv (written by R/03_classification_metrics.R).
## This REPLACES the previously hand-authored results/beyondcell_benchmark.svg,
## which was not reproducible and predated the PSc-fallback results.
##
## Outputs (results/figures/):
##   beyondcell_benchmark.pdf  - vector, for the manuscript.
##   beyondcell_benchmark.png  - raster preview (300 dpi).
##
## Run from project root: Rscript R/04_figures.R

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(patchwork)
})

fig_dir <- file.path("results", "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

m <- fread("results/classification_metrics.csv")
if (is.null(m$collection)) m[, collection := "SSc"]

## Evaluable = a classification metric was computable (both classes, >=10 cells).
ev  <- m[!is.na(auroc)]
na  <- m[is.na(auroc)]
setorder(ev, -auroc)

## PSc-fallback pairs get a marked label; everything else is SSc.
ev[, is_psc := collection == "PSc"]
ev[, label := sprintf("%s · %s%s", drug, dataset, ifelse(is_psc, " *", ""))]
ev[, label := factor(label, levels = rev(label))]   # highest AUROC on top

## ---- Headline numbers (also printed to console) --------------------------
stats <- list(
  n_eval     = nrow(ev),
  mean_auroc = mean(ev$auroc),
  frac_auroc = mean(ev$auroc > 0.5),
  n_auroc    = sum(ev$auroc > 0.5),
  mean_mcc   = mean(ev$mcc, na.rm = TRUE),
  n_mcc      = sum(ev$mcc > 0, na.rm = TRUE)
)
subtitle <- sprintf(
  "%d evaluable drug–dataset pairs  ·  mean AUROC %.2f  ·  %d/%d AUROC>0.5  ·  mean MCC %+.2f  ·  %d/%d MCC>0",
  stats$n_eval, stats$mean_auroc, stats$n_auroc, stats$n_eval,
  stats$mean_mcc, stats$n_mcc, stats$n_eval)

base_theme <- theme_minimal(base_size = 11) +
  theme(panel.grid.major.y = element_blank(),
        panel.grid.minor    = element_blank(),
        axis.title.y        = element_blank(),
        plot.title.position = "plot")

## ---- Panel A: AUROC (chance = 0.5) ---------------------------------------
pA <- ggplot(ev, aes(auroc, label, fill = auroc)) +
  geom_vline(xintercept = 0.5, linetype = "dashed", colour = "grey55") +
  geom_col(width = 0.72) +
  ## Value label always just past the bar tip (outside), so short bars near 0
  ## don't collide with the y-axis labels.
  geom_text(aes(label = sprintf("%.2f", auroc)), hjust = -0.2, size = 3) +
  scale_fill_gradient2(low = "#2166ac", mid = "#f7f7f7", high = "#6a3d9a",
                       midpoint = 0.5, limits = c(0, 1), guide = "none") +
  scale_x_continuous(limits = c(0, 1.12), breaks = seq(0, 1, 0.25),
                     expand = expansion(mult = c(0, 0))) +
  labs(x = "AUROC", title = "AUROC (higher BeyondCell Score → sensitive)") +
  base_theme

## ---- Panel B: MCC (chance = 0) -------------------------------------------
pB <- ggplot(ev, aes(mcc, label, fill = mcc)) +
  geom_vline(xintercept = 0, colour = "grey40") +
  geom_col(width = 0.72) +
  geom_text(aes(label = sprintf("%+.2f", mcc),
                hjust = ifelse(mcc >= 0, -0.15, 1.15)),
            size = 3) +
  scale_fill_gradient2(low = "#2166ac", mid = "#f7f7f7", high = "#b2182b",
                       midpoint = 0, limits = c(-1, 1), guide = "none") +
  scale_x_continuous(limits = c(-1.05, 1.05), breaks = seq(-1, 1, 0.5),
                     expand = expansion(mult = c(0, 0))) +
  labs(x = "MCC (prevalence-matched)",
       title = "Matthews correlation") +
  base_theme +
  theme(axis.text.y = element_blank())

## ---- Caption: unscoreable + excluded pairs + PSc note --------------------
if (is.null(na$note)) na[, note := ""]
excl  <- na[grepl("removed upstream", note)]
unsc  <- na[!grepl("removed upstream", note)]
pair_lbl <- function(d) paste(sprintf("%s · %s", d$drug, d$dataset), collapse = "; ")
unsc_txt <- if (nrow(unsc))
  paste0("\nNot evaluable (no BeyondCell signature in SSc/PSc, or single-class labels): ",
         pair_lbl(unsc), ".") else ""
excl_txt <- if (nrow(excl))
  paste0("\nExcluded — source data removed upstream (not in final study panel): ",
         pair_lbl(excl), ".") else ""
psc_txt <- if (any(ev$is_psc))
  " * scored via the PSc fallback collection (drug absent from SSc)." else ""
caption <- paste0(
  "BeyondCell Score vs. ground-truth sensitivity labels, per drug–dataset pair, sorted by AUROC. ",
  "AUROC 0.5 = chance; MCC 0 = no better than chance, <0 = anti-correlated.",
  psc_txt, unsc_txt, excl_txt)

fig <- (pA | pB) +
  plot_layout(widths = c(1, 1)) +
  plot_annotation(
    title    = "BeyondCell benchmark: drug–response classification",
    subtitle = subtitle,
    caption  = caption,
    theme = theme(
      plot.title    = element_text(face = "bold", size = 15),
      plot.subtitle = element_text(size = 10, colour = "grey30"),
      plot.caption  = element_text(size = 8.5, colour = "grey35", hjust = 0)))

h <- max(4.5, 0.32 * nrow(ev) + 1.6)
ggsave(file.path(fig_dir, "beyondcell_benchmark.pdf"), fig,
       width = 11, height = h, device = cairo_pdf)
ggsave(file.path(fig_dir, "beyondcell_benchmark.png"), fig,
       width = 11, height = h, dpi = 300)

cat("wrote", file.path(fig_dir, "beyondcell_benchmark.{pdf,png}"),
    "\n  ", stats$n_eval, "evaluable pairs; mean AUROC",
    round(stats$mean_auroc, 3), "; mean MCC", round(stats$mean_mcc, 3), "\n")
if (nrow(na)) cat("  not evaluable:", nrow(na), "pairs\n")
