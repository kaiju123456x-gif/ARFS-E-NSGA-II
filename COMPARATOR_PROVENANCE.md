# Comparator implementation provenance

Eight comparator cores were adapted from the author's pre-unification experiment
snapshot named `E-NSGA-2-plus`; HMO-NSGA-II was reconstructed separately from
its open-access paper. The snapshot is not a Git repository and does not contain
a source-code license or commit identifier. Consequently, this release records
the exact input-file hashes and describes every adaptation. It does not claim
that these files are official implementations published by the original
algorithm authors.

| Method | Legacy source within the snapshot | SHA-256 | Release status |
|---|---|---|---|
| MOPSO | `experiments/baselines/mopso_baseline.py` | `3f60f0d97660100c8352d5b2220f72c44b740dec8cb1a5b327aa10922eeaeda3` | Adapted |
| NSGA-III | `experiments/baselines/nsga3_baseline.py` | `4ea2526622837a7732b0093df5ec28823340aa218c74af359a108cfff613d08b` | Adapted |
| FTGGA | `experiments/ftgga/ftgga.py` | `1eecf2d4816559b01adae76fa62daa3fad2692c2ff767f80df02cb7e9ffc1459` | Adapted |
| MMODE | `experiments/mmode/mmode.py` | `37e6ba9ccfc1112b555649020d62ceef5928ef8ccd0b6351e7644459a4e607b4` | Adapted |
| MOMOGS-PCE | `experiments/revision_suite_20260918/fork_momogs_pce.py` | `94c0dc5933dfd59d0232e19aee5787b45a11b7d256174aec13cd384645c76ef4` | Adapted |
| XGBoost-MOGA | `experiments/xgb_moga/xgb_moga.py` | `fe43a3c7cad76448cdf5124911de2383edf8e7df5ce75e2fab55ba059598992f` | Adapted |
| HSIC Lasso | `experiments/revision_suite_20260918/hsic.py` | `f9c4b8a789f52833baa632c362f5f8d05a35d244d226981788f7a518e4ed778b` | Adapted |
| Block HSIC Lasso | Same file as HSIC Lasso | Same as above | Adapted |
| HMO-NSGA-II | [Vijai and P. (2025)](https://doi.org/10.1016/j.dajour.2025.100550) | Not applicable | Paper-guided reimplementation |

## Fair-protocol boundary

Only search/model cores were retained. The legacy data loaders, folds,
preprocessing, objective wrappers, result files, and method labels were not
imported. The current repository supplies all methods with the same:

1. outer fold and test holdout;
2. train-fitted median imputation and min-max scaling;
3. train-only ANOVA candidate pool;
4. inner folds, including group exclusion where required;
5. three-objective evaluator and endpoint rule;
6. random seed and objective-evaluation budget for evolutionary methods.

HSIC Lasso and Block HSIC Lasso return a single tuned subset, so hypervolume is
not defined for them and the evolutionary evaluation budget does not apply.

MMODE's two views are rebuilt from ANOVA F and mutual-information rankings using
only the current outer-training fold. XGBoost-MOGA requires XGBoost and fails
explicitly if that dependency is unavailable; it never substitutes another
model. MOMOGS-PCE uses four exhaustive, non-overlapping subspaces even when the
candidate dimension is not divisible by four.

## HMO-NSGA-II paper-guided reimplementation

The legacy snapshot contains an RMODE-E implementation and a plotting label
that renames `rmode-e` to `HMO-NSGA-II`. Its own status note warns that this
rename is not evidence of algorithmic equivalence. RMODE-E is therefore not
used. The `hmo-nsga-ii` implementation instead follows the CC BY 4.0 open-access 2025
paper: Information Gain, random-forest and ReliefF feature evidence; guided
initialization by feature tournaments; three-parent crossover; balanced
score-guided add/remove mutation; and NSGA-II environmental selection.

The paper names SVM-Rank for rank aggregation but does not provide the training
preferences, threshold, source code, or enough detail to reproduce that stage.
The implementation consequently uses equal-weight normalized score averaging
and retains the continuous fused score instead of an unspecified thresholded
vector. For the text-described three-parent operator, majority-selected features
are preserved and singly selected features are sampled by fused score. The
current shared protocol supplies crossover probability, mutation probability,
three-objective evaluation, and the exact evaluation budget. These choices are
stored in every run's diagnostics and must not be described as author-supplied
parameters.
