# Paper data access and provenance

This directory is the single entry point for obtaining and verifying all ten datasets used in the manuscript. The repository does not relicense third-party data. Its preparation code downloads each fixed public source, verifies its identity, validates the expected schema, and creates the analysis-ready representation locally.

## Automated preparation

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pip install "openpyxl==3.1.5"
python -m arfs_ensga2.cli prepare-geo --accession all
python -m arfs_ensga2.cli prepare-legacy --dataset all
```

Generated files are written to `data/processed/`. Every matrix has a neighboring `*.manifest.json` containing source information, checksums, shape, class counts, grouping information, and preparation metadata.

## Dataset inventory

| Paper name | Samples | Features | Classes | Public source | Group-aware CV |
|---|---:|---:|---:|---|---|
| Lung_cancer | 111 | 54,675 | 2 | GEO GSE3141 | No linked structure reported |
| Brain_Tumor | 180 | 54,613 | 4 | GEO GSE4290 | No linked structure reported |
| Breast_cancer_1 | 97 | 24,481 | 2 | Mendeley Data 10.17632/fhx5zgx2zj.1 | No linked structure reported |
| Breast_cancer_2 | 104 | 22,283 | 2 | GEO GSE3726 | 52 matched groups |
| Pancreatic_Cancer | 78 | 54,675 | 2 | ArrayExpress E-GEOD-15471 | 36 patient/replicate groups |
| Leukemia | 72 | 7,129 | 2 | Golub benchmark, commit-pinned mirror | No linked structure reported |
| Lung_Adenocarcinoma | 107 | 22,283 | 2 | ArrayExpress E-GEOD-10072 | 74 subject groups |
| Colon | 62 | 2,000 | 2 | Alon benchmark, commit-pinned mirror | No linked structure reported |
| SRBCT | 83 | 2,308 | 4 | Mendeley Data 10.17632/fhx5zgx2zj.1 | No linked structure reported |
| 11_Tumor | 174 | 12,533 | 11 | Mendeley Data 10.17632/fhx5zgx2zj.1 | No linked structure reported |

Machine-readable identifiers, landing pages, expected processed checksums, and redistribution notes are in [`dataset_manifest.csv`](dataset_manifest.csv).

## Analysis-ready schema

```text
<numeric feature columns>,label[,group]
```

- `label` is required.
- `group` is present only for linked samples.
- Missing values are imputed from each outer-training fold only.
- Scaling and ANOVA screening are learned inside each outer-training fold.
- Group identifiers are parsed from study metadata, never inferred from row order.

## Verification and rights

After preparation, compare each generated `output_sha256` with `processed_sha256` in `dataset_manifest.csv`. Experiment `manifest.json` files record the same checksum, linking source, matrix, configuration, and result.

Mendeley Data DOI `10.17632/fhx5zgx2zj.1` states CC BY-NC 3.0. GEO and ArrayExpress records come from their public repositories. The pinned mirror used for Colon and Leukemia has no declared redistribution license, so those files are downloaded from their fixed upstream URL rather than copied here. Users must follow the original providers' terms.
