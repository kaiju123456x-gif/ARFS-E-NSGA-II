import csv
from pathlib import Path

import yaml

from arfs_ensga2.cli import GROUPED_DATASETS, PAPER_DATASETS
from arfs_ensga2.config import load_config
from arfs_ensga2.paper_experiments import INTERACTIONS, ONE_FACTOR_VALUES, parameter_settings


ROOT = Path(__file__).resolve().parents[1]


def test_paper_contract_matches_code_and_data_manifest() -> None:
    contract = yaml.safe_load((ROOT / "paper" / "experiment_contract.yaml").read_text(encoding="utf-8"))
    contract_datasets = [item["name"] for item in contract["datasets"]]
    contract_grouped = {item["name"] for item in contract["datasets"] if item["grouped"]}
    with (ROOT / "data" / "dataset_manifest.csv").open(encoding="utf-8", newline="") as handle:
        manifest_rows = {row["dataset"]: row for row in csv.DictReader(handle)}

    assert tuple(contract_datasets) == PAPER_DATASETS
    assert set(contract_datasets) == set(manifest_rows)
    assert contract_grouped == GROUPED_DATASETS
    for item in contract["datasets"]:
        row = manifest_rows[item["name"]]
        assert int(row["samples"]) == item["samples"]
        assert int(row["features"]) == item["features"]
        assert int(row["classes"]) == item["classes"]
        assert (int(row["groups"]) if row["groups"] else None) == item.get("groups")


def test_table_i_csv_matches_paper_contract() -> None:
    contract = yaml.safe_load((ROOT / "paper" / "experiment_contract.yaml").read_text(encoding="utf-8"))
    with (ROOT / "paper" / "dataset_table.csv").open(encoding="utf-8", newline="") as handle:
        table_rows = {row["repository_name"]: row for row in csv.DictReader(handle)}

    assert set(table_rows) == {item["name"] for item in contract["datasets"]}
    for item in contract["datasets"]:
        row = table_rows[item["name"]]
        assert row["paper_label"] == item["paper_label"]
        assert int(row["samples"]) == item["samples"]
        assert int(row["features"]) == item["features"]
        assert int(row["classes"]) == item["classes"]
        assert row["paper_accession"] == item["accession"]


def test_paper_contract_matches_default_configuration() -> None:
    contract = yaml.safe_load((ROOT / "paper" / "experiment_contract.yaml").read_text(encoding="utf-8"))
    experiment, optimizer = load_config(ROOT / "configs" / "default.yaml")

    for field in ("outer_splits", "outer_repeats", "inner_splits", "base_seed", "candidate_features"):
        assert getattr(experiment, field) == contract["protocol"][field]
    for field, value in contract["optimizer"].items():
        assert getattr(optimizer, field) == value


def test_parameter_grid_matches_paper_contract() -> None:
    contract = yaml.safe_load((ROOT / "paper" / "experiment_contract.yaml").read_text(encoding="utf-8"))
    assert contract["parameter_analysis"]["one_factor"] == {
        key: list(values) for key, values in ONE_FACTOR_VALUES.items()
    }
    assert contract["parameter_analysis"]["interactions"] == [
        list(pair) for pair in INTERACTIONS
    ]
    assert len(parameter_settings()) == 37


def test_ablation_labels_match_repository_variants() -> None:
    contract = yaml.safe_load((ROOT / "paper" / "experiment_contract.yaml").read_text(encoding="utf-8"))
    assert [item["repository_id"] for item in contract["ablation"]["variants"]] == [
        "none", "a", "b", "c", "ab", "ac", "bc", "abc"
    ]
    assert [item["paper_label"] for item in contract["ablation"]["variants"]] == [
        "BASE", "A", "B", "C", "AB", "AC", "BC", "ABC"
    ]
