# Reproducibility protocol

This protocol connects public data sources, processed matrices, experiment configuration, random seeds, folds, Pareto archives, and audit reports.

## 1. Clean environment

Use Python 3.10 or 3.11 in a fresh virtual environment.

```bash
git clone https://github.com/kaiju123456x-gif/ARFS-E-NSGA-II.git
cd ARFS-E-NSGA-II
python -m venv .venv
```

Activate `.venv`, then run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pip install "openpyxl==3.1.5"
```

## 2. Installation check

```bash
python -m pytest
python -m arfs_ensga2.cli demo --output results/demo
python -m arfs_ensga2.cli audit --output results/demo
```

All tests must pass, both commands must exit with code 0, and the audit must report `"status": "PASS"`.

## 3. Data retrieval and verification

```bash
python -m arfs_ensga2.cli prepare-geo --accession all
python -m arfs_ensga2.cli prepare-legacy --dataset all
```

Compare every generated processed checksum with `data/dataset_manifest.csv`. Required linked-sample group counts are 52 for Breast_cancer_2, 36 for Pancreatic_Cancer, and 74 for Lung_Adenocarcinoma.

## 4. Paper configuration

```bash
python -m arfs_ensga2.cli suite --data-dir data/processed --output results/paper
```

The configuration uses five outer folds repeated five times, three inner folds, base seed 2026, a 30-feature training-fold candidate pool, population size 12, evaluation budget 240, and variant `abc`. Group-aware inner and outer folds are used for linked samples. No outer-test information is used to impute, scale, screen, initialize, or optimize.

Run all eight component combinations with:

```bash
python -m arfs_ensga2.cli suite --data-dir data/processed --output results/paper-ablation --ablation
```

## 5. Evidence audit

The suite audits every dataset. For a single result directory:

```bash
python -m arfs_ensga2.cli audit --output results/paper/Colon
```

| Evidence | Purpose |
|---|---|
| `manifest.json` | Dataset checksum, configuration, dependency versions, and platform |
| `splits.json` | Outer/inner folds and training-only candidate features |
| `fold_results.csv` | Predictive, subset, redundancy, and runtime summaries |
| `archives.csv` | Pareto archives and hypervolume inputs |
| `diagnostics/*.json` | DPIM and DRFM behavior |
| `audit_report.json` | Machine-readable integrity verdict |

## 6. Determinism

Stochastic components derive from the stored base seed and fold/variant identifiers. Split indices and metadata should match exactly. Small floating-point differences can occur across BLAS implementations or platforms; retain full precision when comparing outputs.

## 7. Third-party comparators

This repository does not relicense third-party implementations. It reproduces ARFS-E-NSGA-II, its common NSGA-II backbone, all eight proposed-component combinations, and the shared data/evaluation pipeline. Literature comparators should be obtained from their original authors and run with the matched folds, candidate pool, objective evaluator, seeds, and budget described in the manuscript.

## 8. Reproduction issues

Open a GitHub issue with the operating system, Python version, failing command, complete error, relevant checksums, and generated `manifest.json` and `audit_report.json` after removing private local paths.
