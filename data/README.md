# Data layout

Raw and processed research data are intentionally ignored by Git. Place analysis-ready CSV files in `data/processed/`.

```bash
pip install -e ".[data]"
arfs-ensga2 prepare-geo --accession all
arfs-ensga2 prepare-legacy --dataset all
```

The commands download fixed public versions, verify SHA-256 checksums where the provider exposes a stable file, validate dimensions and class counts, and create `*.manifest.json` provenance records. GEO matrices come from NCBI GEO or its official EMBL-EBI mirror. Breast_cancer_1, SRBCT, and 11_Tumor come from Mendeley Data DOI `10.17632/fhx5zgx2zj.1` (CC BY-NC 3.0). Colon and Leukemia are pinned to commit `5681201bb4e2ba7024fe15d4c71046017219b658` of `kivancguckiran/microarray-data`; that repository declares no data license, so its files are downloaded locally only.

| Paper name | Samples | Features | Classes | Source identifier | Group-aware CV |
|---|---:|---:|---:|---|---|
| Lung_cancer | 111 | 54,675 | 2 | GSE3141 | No linked structure reported |
| Brain_Tumor | 180 | 54,613 | 4 | GSE4290 | No linked structure reported |
| Breast_cancer_1 | 97 | 24,481 | 2 | van 't Veer et al. (2002) processed benchmark | No linked structure reported |
| Breast_cancer_2 | 104 | 22,283 | 2 | GSE3726 | Required: matched pair ID |
| Pancreatic_Cancer | 78 | 54,675 | 2 | GSE15471 | Required: patient/replicate ID |
| Leukemia | 72 | 7,129 | 2 | Golub et al. (1999) processed benchmark | No linked structure reported |
| Lung_Adenocarcinoma | 107 | 22,283 | 2 | GSE10072 | Required: subject ID |
| Colon | 62 | 2,000 | 2 | Alon et al. (1999) processed benchmark | No linked structure reported |
| SRBCT | 83 | 2,308 | 4 | Khan et al. (2001) processed benchmark | No linked structure reported |
| 11_Tumor | 174 | 12,533 | 11 | Su et al. (2001) processed benchmark | No linked structure reported |

Expected schema:

```text
<numeric feature columns>,label[,group]
```

Do not infer group identifiers from row order. Use verified study metadata and record the preparation script, source URL, retrieval date, checksums, exclusions, duplicate handling, transformations, and class mapping.

For `Breast_cancer_2`, `Pancreatic_Cancer`, and `Lung_Adenocarcinoma`, copy `configs/default.yaml` and set `experiment.group_column: group` before running an experiment.

