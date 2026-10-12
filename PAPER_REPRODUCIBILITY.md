# Paper-to-repository map

This file maps every experimental claim in the manuscript to a public, inspectable repository artifact. The submitted manuscript PDF is the authority for dataset names, source identities, experimental settings, and reported results.

| Manuscript content | Repository evidence | Reproduction entry point |
|---|---|---|
| Ten datasets and linked-sample structure | `paper/dataset_table.csv`, `data/dataset_manifest.csv`, `paper/experiment_contract.yaml` | `python -m arfs_ensga2.cli prepare-data` |
| Leakage-controlled nested evaluation | `src/arfs_ensga2/experiment.py`, `src/arfs_ensga2/objectives.py` | `python -m arfs_ensga2.cli suite` |
| Three optimization objectives | `src/arfs_ensga2/objectives.py` | Stored in `archives.csv` |
| Multi-source initialization | `src/arfs_ensga2/initialization.py` | Variant `a` or `abc` |
| DPIM | `src/arfs_ensga2/optimizer.py` | Variant `b` or `abc` |
| DRFM | `src/arfs_ensga2/optimizer.py` | Variant `c` or `abc` |
| Complete proposed method | `configs/default.yaml`, `configs/grouped.yaml` | `python -m arfs_ensga2.cli reproduce` |
| Eight-pattern component ablation | `src/arfs_ensga2/experiment.py` | `python -m arfs_ensga2.cli paper-ablation` |
| One-factor and interaction parameter analysis | `paper/experiment_contract.yaml`, `src/arfs_ensga2/paper_experiments.py` | `python -m arfs_ensga2.cli paper-parameters` |
| Reported aggregate targets and interpretation limits | `paper/reported_results.yaml` | Compare against newly generated `summary.csv` files |
| Output integrity | `src/arfs_ensga2/audit.py` | `python -m arfs_ensga2.cli audit --output <directory>` |
| Available comparator implementations | `src/arfs_ensga2/comparators.py`, `COMPARATOR_PROVENANCE.md` | `python -m arfs_ensga2.cli paper-comparison` |

## Evidence produced by each run

| Artifact | Paper relationship |
|---|---|
| `fold_results.csv` | The 25 paired outer-fold observations underlying mean and standard deviation tables |
| `archives.csv` | Training-side Pareto sets underlying hypervolume |
| `summary.csv` | Per-method or per-variant aggregate metrics |
| `splits.json` | Fixed outer/inner indices and candidate-feature identities |
| `diagnostics/*.json` | DPIM state, IGD history, DRFM monitoring, and swaps |
| `manifest.json` | Dataset hash, exact configuration, software versions, and platform |
| `audit_report.json` | Automated completeness and integrity verdict |

## Comparator provenance

The manuscript compares ARFS-E-NSGA-II with MOPSO, NSGA-II, NSGA-III,
HMO-NSGA-II, FTGGA, MMODE, MOMOGS-PCE, XGBoost-MOGA, HSIC Lasso, and Block
HSIC Lasso. Eight audited legacy comparator cores and a paper-guided HMO-NSGA-II
reimplementation are integrated behind the same evaluation interface.
`COMPARATOR_PROVENANCE.md` records the exact legacy input-file hashes, the
adaptation boundary, the HMO paper source, and every fair-protocol completion.
The unrelated RMODE-E implementation is never relabelled as HMO-NSGA-II. These
repository-native adaptations do not redistribute or claim to be the original
authors' third-party source packages; they were added after the cited manuscript
PDF was exported so that reviewers can execute the complete comparison locally.
