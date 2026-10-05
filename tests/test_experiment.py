from pathlib import Path

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
    assert audit_results(output)["status"] == "PASS"

