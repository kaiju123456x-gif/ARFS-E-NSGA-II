# Generated result directories

Experiment outputs are generated locally because full repeated nested cross-validation is computationally expensive. Do not edit result files manually.

This repository does not bundle the complete paper-scale results or their audit
reports. This directory documents the expected layout and is the default output
location for new reproduction runs.

```bash
python -m arfs_ensga2.cli suite --data-dir data/processed --output results/paper
```

Each dataset directory must contain `fold_results.csv`, `archives.csv`, `summary.csv`, `splits.json`, `manifest.json`, `diagnostics/`, and `audit_report.json`. The suite exits unsuccessfully if an integrity audit fails.
