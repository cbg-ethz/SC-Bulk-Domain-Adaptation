# Reproducible BeyondCell benchmark pipeline.
#
# The heavy per-pair scoring runs on SLURM compute nodes (submit targets);
# the light aggregation/metrics/figure chain runs locally (downstream target).
#
#   make ssc         # submit the 34-pair SSc pass   (SLURM array)
#   make psc         # submit the 4-pair PSc fallback (SLURM array)
#   make downstream  # aggregate -> metrics -> figures -> sessionInfo (local)
#   make figures     # just re-render the figure from existing metrics
#
# `downstream` requires the per-pair runs to have finished first.

RSCRIPT := /cluster/work/bewi/members/mesteban/miniconda3/envs/beyondcell/bin/Rscript

.PHONY: ssc psc downstream aggregate metrics figures sessioninfo clean-figures

ssc:
	sbatch scripts/submit_array.sh

psc:
	sbatch --array=1-4 \
	  --export=ALL,COLLECTION=PSc,PAIRS_FILE=scripts/target_pairs_psc.tsv,OUT_ROOT=results/per_pair_psc \
	  scripts/submit_array.sh

downstream:
	scripts/run_downstream.sh

aggregate:
	$(RSCRIPT) R/02_aggregate.R

metrics: aggregate
	$(RSCRIPT) R/03_classification_metrics.R

figures: metrics
	$(RSCRIPT) R/04_figures.R

sessioninfo:
	$(RSCRIPT) -e 'writeLines(capture.output(sessionInfo()), "results/sessionInfo.txt")'

clean-figures:
	rm -f results/figures/beyondcell_benchmark.pdf results/figures/beyondcell_benchmark.png
