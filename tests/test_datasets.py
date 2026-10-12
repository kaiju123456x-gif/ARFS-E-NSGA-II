import csv
import gzip
from pathlib import Path

import pandas as pd

from arfs_ensga2.datasets import (
    LegacySpec,
    _read_legacy,
    labels_and_groups,
    load_processed_checksums,
    read_geo_metadata,
    sha256,
    verify_processed_record,
    write_reproducible_csv_gz,
)


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


def test_processed_gzip_is_byte_reproducible(tmp_path: Path):
    frame = pd.DataFrame({"feature_00000": [1.0, 2.0], "label": ["a", "b"]})
    first = write_reproducible_csv_gz(frame, tmp_path / "first.csv.gz")
    second = write_reproducible_csv_gz(frame, tmp_path / "second.csv.gz")
    assert first.read_bytes() == second.read_bytes()


def test_processed_manifest_is_enforced(tmp_path: Path):
    output = write_reproducible_csv_gz(
        pd.DataFrame({"feature_00000": [1.0], "label": ["a"]}),
        tmp_path / "Toy.csv.gz",
    )
    manifest = tmp_path / "dataset_manifest.csv"
    manifest.write_text(
        f"dataset,processed_sha256\nToy,{sha256(output)}\n",
        encoding="utf-8",
    )
    checksums = load_processed_checksums(manifest)
    record = {
        "spec": {"public_name": "Toy"},
        "output_sha256": sha256(output),
    }
    verify_processed_record(record, checksums)

    record["output_sha256"] = "0" * 64
    try:
        verify_processed_record(record, checksums)
    except ValueError as error:
        assert "processed checksum mismatch" in str(error)
    else:
        raise AssertionError("checksum mismatch was not rejected")

