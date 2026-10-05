import csv
import gzip
from pathlib import Path

import pandas as pd

from arfs_ensga2.datasets import LegacySpec, _read_legacy, labels_and_groups, read_geo_metadata


def _write_metadata(path: Path, rows: list[list[str]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerows(rows)
        handle.write("!series_matrix_table_begin\n")


def test_gse3726_pair_ids_are_recovered(tmp_path: Path):
    path = tmp_path / "matrix.gz"
    _write_metadata(
        path,
        [
            ["!Sample_title", "B10R", "B10T", "C13R", "C13T"],
            ["!Sample_source_name_ch1", "breast", "breast", "colon", "colon"],
            ["!Sample_characteristics_ch1", "x", "x", "x", "x"],
        ],
    )
    labels, groups = labels_and_groups("GSE3726", read_geo_metadata(path))
    assert labels == ["breast", "breast", "colon", "colon"]
    assert groups == ["B10", "B10", "C13", "C13"]


def test_gse15471_patient_and_replicate_grouping():
    metadata = {
        "!Sample_title": [["N30162", "N30162_rep", "T30162", "T30162_rep"]],
        "!Sample_source_name_ch1": [["x"] * 4],
        "!Sample_characteristics_ch1": [["x"] * 4],
    }
    labels, groups = labels_and_groups("GSE15471", metadata)
    assert labels == ["normal", "normal", "tumor", "tumor"]
    assert groups == ["30162", "30162", "30162", "30162"]


def test_gse10072_subject_grouping():
    metadata = {
        "!Sample_title": [["Lung Tumor_GT00006", "Normal Lung_GT00006"]],
        "!Sample_source_name_ch1": [["x", "x"]],
        "!Sample_characteristics_ch1": [["x", "x"]],
    }
    labels, groups = labels_and_groups("GSE10072", metadata)
    assert labels == ["tumor", "non_tumor"]
    assert groups == ["GT00006", "GT00006"]


def test_legacy_xlsx_has_class_in_first_column(tmp_path: Path, monkeypatch):
    source = pd.DataFrame([["a", 1.0, 2.0], ["b", 3.0, 4.0]])
    monkeypatch.setattr(pd, "read_excel", lambda path, header=None: source)
    spec = LegacySpec("example", "example.xlsx", 2, 2, 2, "url", "hash", "license")
    features, labels = _read_legacy(tmp_path / "example.xlsx", spec)
    assert features.shape == (2, 2)
    assert labels.tolist() == ["a", "b"]

