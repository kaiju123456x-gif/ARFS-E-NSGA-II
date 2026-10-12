from arfs_ensga2.cli import build_parser


def test_reproduce_command_defaults() -> None:
    args = build_parser().parse_args(["reproduce"])

    assert args.data_dir == "data/processed"
    assert args.output == "results/paper"
    assert args.skip_data is False
    assert args.ablation is False


def test_reproduce_command_flags() -> None:
    args = build_parser().parse_args(["reproduce", "--skip-data", "--ablation"])

    assert args.skip_data is True
    assert args.ablation is True


def test_paper_ablation_scope() -> None:
    args = build_parser().parse_args(["paper-ablation"])

    assert args.data_dir == "data/processed"
    assert args.output == "results/paper-ablation"


def test_paper_parameter_defaults() -> None:
    args = build_parser().parse_args(["paper-parameters"])

    assert args.data_dir == "data/processed"
    assert args.output == "results/paper-parameters"
