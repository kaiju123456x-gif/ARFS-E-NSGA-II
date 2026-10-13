# Ablation and Parameter Analysis

This document explains the two supplementary experiment blocks reported in the
manuscript. The manuscript PDF is the authority for the scientific scope; the
repository provides an executable implementation, a machine-readable contract,
and newly generated fold-level evidence.

## Shared controlled protocol

Both analyses use only **Colon** and **SRBCT**, as stated in the manuscript. All
settings reuse the same 25 paired outer folds, training-only preprocessing,
30-feature candidate pools, three-fold inner evaluation, distance-weighted
3-nearest-neighbor classifier, population size 12, and budget of 240 objective
evaluations. Only the component activation pattern or the named parameter value
changes.

| Fixed quantity | Value |
|---|---:|
| Outer evaluation | 5 folds x 5 repeats = 25 paired folds |
| Inner evaluation | 3 folds |
| Candidate pool | 30 training-fold-selected features |
| Population | 12 candidate masks |
| Evaluation budget | 240 objective calls per outer fold |
| Objectives | Balanced error, feature ratio, and redundancy |
| Datasets | Colon and SRBCT |

The restriction to two datasets is an analysis-scope limitation. Conclusions
from these experiments must not be generalized automatically to all ten
datasets.

## Component ablation

The letters denote component **packages**, not single operators.

| Package | Manuscript definition | Main implementation |
|---|---|---|
| A | Multi-source guided initialization combining ReliefF, mutual information, chi-square, F-score, and random-forest importance with validation-performance softmax weights | `src/arfs_ensga2/initialization.py` |
| B | DPIM search control: archive-relative population partitioning, differential parent selection, IGD-adaptive mutation, bidirectional or sparsity-biased mutation, and cumulative IGD reference handling | `src/arfs_ensga2/optimizer.py` |
| C | DRFM redundancy refinement: periodic population-level feature-usage monitoring and individual-level redundancy-relevance exchange | `src/arfs_ensga2/optimizer.py` |

The eight variants form the complete 2 x 2 x 2 activation design:

| Paper label | Repository ID | A | B | C | Interpretation |
|---|---|:---:|:---:|:---:|---|
| BASE | `none` | 0 | 0 | 0 | Common NSGA-II backbone with random sparse initialization |
| A | `a` | 1 | 0 | 0 | Guided initialization only |
| B | `b` | 0 | 1 | 0 | DPIM only |
| C | `c` | 0 | 0 | 1 | DRFM only |
| AB | `ab` | 1 | 1 | 0 | Guided initialization and DPIM |
| AC | `ac` | 1 | 0 | 1 | Guided initialization and DRFM |
| BC | `bc` | 0 | 1 | 1 | DPIM and DRFM |
| ABC | `abc` | 1 | 1 | 1 | Complete ARFS-E-NSGA-II |

Run the paper-scoped ablation with:

```bash
python -m arfs_ensga2.cli paper-ablation
```

The command creates `results/paper-ablation/Colon/` and
`results/paper-ablation/SRBCT/`. In each directory, `fold_results.csv` contains
the 25 paired observations for every variant, `summary.csv` contains their
means and sample standard deviations, and `diagnostics/` records the activated
components and generation-level behavior. `audit_report.json` is the integrity
verdict.

The manuscript interpretation is deliberately qualified: variants containing
DPIM generally produced stronger hypervolume, the DRFM effect was more
dataset-dependent, and ABC offered a balanced result rather than winning every
endpoint. These statements are limited to Colon and SRBCT.

## Parameter analysis

The complete ABC configuration is used throughout. One-factor-at-a-time (OFAT)
settings vary one parameter while holding all others at the defaults below.

| Symbol | Configuration field | Operational meaning | Default | Tested values |
|---|---|---|---:|---|
| Pm | `mutation_probability` | Base per-bit mutation probability before DPIM adaptation | 0.12 | 0.05, 0.12, 0.20 |
| W | `window_size` | Recent IGD-change window used by the stagnation test | 10 | 3, 5, 10 |
| rho | `rho` | Relative multiplier for the IGD stagnation threshold | 0.20 | 0.05, 0.20, 0.50 |
| alpha | `alpha` | Redundancy penalty in the DRFM feature-merit function | 0.35 | 0, 0.35, 0.75 |
| L | `monitor_interval` | Generations between population-level feature-usage checks | 5 | 3, 5, 10 |
| tau_MI | `mi_threshold` | Mutual-information safeguard threshold for dormant-feature monitoring | 0 | 0, 0.5 |
| Archive | `archive_capacity` | Maximum number of nondominated archive members | 50 | 20, 50 |

Two full 3 x 3 interaction grids are also evaluated:

- `window_size x rho`, which probes the relative stagnation rule;
- `alpha x monitor_interval`, which probes redundancy strength and monitoring frequency.

This gives 19 OFAT settings plus 18 interaction settings, or **37 settings per
dataset**. Across two datasets, the command creates 74 setting directories and
1,850 outer-fold result rows (74 x 25). Default configurations are intentionally
repeated within different OFAT families because each family is a separately
labelled paired comparison.

Run the analysis with:

```bash
python -m arfs_ensga2.cli paper-parameters
```

The top-level `results/paper-parameters/parameter_analysis_summary.csv` records
the mean and sample standard deviation of balanced accuracy, Macro-F1, selected
feature count, redundancy, hypervolume, and runtime for every setting. Each
nested setting directory preserves its 25 fold-level observations, archives,
splits, configuration, diagnostics, and audit report.

The manuscript reports that `mutation_probability` and `alpha` show the most
visible changes, whereas `window_size`, `rho`, `monitor_interval`,
`mi_threshold`, and archive capacity are comparatively stable under the tested
budget. It also reports little observable `window_size x rho` interaction and
no clear `alpha x monitor_interval` interaction. These are descriptive findings
for the tested grids and datasets.

## Evidence boundary

The repository does not present manuscript numbers as if they were newly run
evidence. Paper-reported values are reference targets; executable commands write
fresh results under `results/`. Figures 3--5 and the manuscript's exploratory
Wilcoxon/Holm table are publication-level presentations derived from paired
fold observations. The current command preserves all observations needed to
reconstruct them, but it does not automatically typeset those figures or the
significance table.

