# Manuscript-aligned experiment record

This directory provides a machine-readable bridge between the submitted manuscript and the public implementation. It does not replace the manuscript and does not present transcribed aggregate values as newly generated evidence.

| File | Purpose |
|---|---|
| `dataset_table.csv` | Exact dataset identities, dimensions, class counts, linked-group counts, and source identifiers reported in Table I |
| `experiment_contract.yaml` | Experimental protocol, optimizer settings, comparator scope, ablations, and parameter grids |
| `reported_results.yaml` | Headline aggregate values and interpretation limits reported in Tables IV-VII and the conclusion |

## Dataset identity rule

The manuscript accession is the scientific study identifier. A retrieval identifier may differ only when the preparation command uses a public mirror. In particular:

| Manuscript accession | Retrieval identifier | Repository dataset |
|---|---|---|
| `GSE15471` | `E-GEOD-15471` | `Pancreatic_Cancer` |
| `GSE10072` | `E-GEOD-10072` | `Lung_Adenocarcinoma` |

These are identifier aliases for the same studies, not substitutions with different datasets.

## Generated evidence

Run `python -m arfs_ensga2.cli reproduce` to generate auditable fold-level evidence for the proposed method. Run `paper-ablation` and `paper-parameters` for the two manuscript analyses restricted to Colon and SRBCT. Generated outputs belong under `results/` and are intentionally not confused with the transcribed aggregate targets in this directory.
