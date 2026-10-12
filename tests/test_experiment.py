from pathlib import Path

import pandas as pd

from arfs_ensga2.audit import audit_results
from arfs_ensga2.config import ExperimentConfig, OptimizerConfig
from arfs_ensga2.experiment import create_demo_dataset, run_experiment


def test_end_to_end_experiment(tmp_path: Path):
    data = create_demo_dataset(tmp_path / "synthetic.csv", seed=3)
    experiment = ExperimentConfig(
        outer_splits=2,
        outer_repeats=1,
        inner_splits=2,
        candidate_features=8,
        base_seed=5,
    )
    optimizer = OptimizerConfig(
        population_size=4,
        evaluation_budget=8,
        archive_capacity=8,
        initialization_top_k=3,
    )
    output = tmp_path / "results"
    frame = run_experiment(data, output, experiment, optimizer, variants=("abc",))
    assert len(frame) == 2
    assert set(frame["method"]) == {"ARFS-E-NSGA-II"}
    assert audit_results(output)["status"] == "PASS"
    assert (output / "audit_report.json").exists()


def test_audit_detects_unpaired_method_seed(tmp_path: Path):
    data = create_demo_dataset(tmp_path / "synthetic.csv", seed=3)
    experiment = ExperimentConfig(
        outer_splits=2,
        outer_repeats=1,
        inner_splits=2,
        candidate_features=8,
        base_seed=5,
    )
    optimizer = OptimizerConfig(
        population_size=4,
        evaluation_budget=8,
        archive_capacity=8,
        initialization_top_k=3,
    )
    output = tmp_path / "results"
    run_experiment(data, output, experiment, optimizer, variants=("abc", "none"))
    frame = pd.read_csv(output / "fold_results.csv")
    frame.loc[0, "seed"] += 1
    frame.to_csv(output / "fold_results.csv", index=False)
    report = audit_results(output)
    assert report["status"] == "FAIL"
    assert "paired methods do not share exactly one seed per outer fold" in report["errors"]

