# Contributing

Thank you for helping improve the reproducibility of ARFS-E-NSGA-II.

## Public-language policy

All material committed to the public repository must be understandable without translation:

- Write documentation, code comments, docstrings, command-line messages, configuration notes, issue templates, and pull-request descriptions in English.
- Use descriptive English names for files, directories, variables, functions, classes, table columns, and generated outputs.
- Do not commit machine-specific absolute paths. Use command-line arguments, configuration values, or paths relative to the repository root.
- Original third-party dataset content does not need to be translated. Raw and generated datasets are excluded from the repository and must remain under the ignored `data/raw/` and `data/processed/` directories.

## Reproducibility requirements

Changes that affect experiments must preserve or update:

- the exact configuration and random-seed derivation;
- train-only preprocessing and feature screening;
- group-aware splitting for linked biological samples;
- machine-readable manifests, folds, archives, diagnostics, and audit reports;
- public-data identifiers, checksums, and preparation instructions.

Do not commit third-party data unless its license explicitly permits redistribution. Never commit credentials, access tokens, private data, or identifying local paths.

## Local verification

Use Python 3.10 or 3.11 and run:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pytest
python -m arfs_ensga2.cli demo --output results/demo
python -m arfs_ensga2.cli audit --output results/demo
```

The tests must pass and the final audit must report `"status": "PASS"` before a change is submitted.
