from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit_results
from .config import load_config
from .datasets import (
    GEO_SPECS,
    LEGACY_SPECS,
    load_processed_checksums,
    prepare_geo_dataset,
    prepare_legacy_dataset,
    verify_processed_record,
)
from .experiment import ABLATIONS, create_demo_dataset, run_experiment
from .comparators import AVAILABLE_COMPARATORS
from .paper_experiments import run_parameter_analysis


PAPER_DATASETS = (
    "Lung_cancer", "Brain_Tumor", "Breast_cancer_1", "Breast_cancer_2",
    "Pancreatic_Cancer", "Leukemia", "Lung_Adenocarcinoma", "Colon", "SRBCT",
    "11_Tumor",
)
GROUPED_DATASETS = {"Breast_cancer_2", "Pancreatic_Cancer", "Lung_Adenocarcinoma"}
PAPER_ABLATION_DATASETS = ("Colon", "SRBCT")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arfs-ensga2")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run one configuration on a CSV dataset")
    run.add_argument("--data", required=True)
    run.add_argument("--config", default="configs/default.yaml")
    run.add_argument("--output", required=True)
    run.add_argument("--variant", choices=sorted(ABLATIONS), default="abc")

    ablation = subparsers.add_parser("ablation", help="run all eight A/B/C variants")
    ablation.add_argument("--data", required=True)
    ablation.add_argument("--config", default="configs/default.yaml")
    ablation.add_argument("--output", required=True)

    compare = subparsers.add_parser(
        "compare", help="run ARFS-E-NSGA-II and the audited available comparators"
    )
    compare.add_argument("--data", required=True)
    compare.add_argument("--config", default="configs/default.yaml")
    compare.add_argument("--output", required=True)
    compare.add_argument(
        "--method",
        action="append",
        choices=["abc", "none", *AVAILABLE_COMPARATORS],
        help="repeat to select methods; default runs every available method",
    )

    demo = subparsers.add_parser("demo", help="generate synthetic data and run a smoke experiment")
    demo.add_argument("--output", default="results/demo")
    demo.add_argument("--config", default="configs/demo.yaml")

    audit = subparsers.add_parser("audit", help="validate a completed result directory")
    audit.add_argument("--output", required=True)

    prepare = subparsers.add_parser(
        "prepare-geo", help="download and prepare supported public GEO datasets"
    )
    prepare.add_argument(
        "--accession", choices=["all", *sorted(GEO_SPECS)], default="all"
    )
    prepare.add_argument("--raw-dir", default="data/raw/geo")
    prepare.add_argument("--output", default="data/processed")
    prepare.add_argument("--manifest", default="data/dataset_manifest.csv")
    legacy = subparsers.add_parser(
        "prepare-legacy", help="download and prepare five public benchmark matrices"
    )
    legacy.add_argument("--dataset", choices=["all", *LEGACY_SPECS], default="all")
    legacy.add_argument("--raw-dir", default="data/raw/legacy")
    legacy.add_argument("--output", default="data/processed")
    legacy.add_argument("--manifest", default="data/dataset_manifest.csv")
    prepare_all = subparsers.add_parser(
        "prepare-data", help="download, prepare, and verify all ten paper datasets"
    )
    prepare_all.add_argument("--raw-root", default="data/raw")
    prepare_all.add_argument("--output", default="data/processed")
    prepare_all.add_argument("--manifest", default="data/dataset_manifest.csv")
    suite = subparsers.add_parser("suite", help="run all ten paper datasets")
    suite.add_argument("--data-dir", default="data/processed")
    suite.add_argument("--config", default="configs/default.yaml")
    suite.add_argument("--grouped-config", default="configs/grouped.yaml")
    suite.add_argument("--output", default="results/paper")
    suite.add_argument("--ablation", action="store_true")
    paper_ablation = subparsers.add_parser(
        "paper-ablation", help="run all eight variants on Colon and SRBCT as reported"
    )
    paper_ablation.add_argument("--data-dir", default="data/processed")
    paper_ablation.add_argument("--config", default="configs/default.yaml")
    paper_ablation.add_argument("--output", default="results/paper-ablation")
    paper_comparison = subparsers.add_parser(
        "paper-comparison", help="run every manuscript method on all ten datasets"
    )
    paper_comparison.add_argument("--data-dir", default="data/processed")
    paper_comparison.add_argument("--config", default="configs/default.yaml")
    paper_comparison.add_argument("--grouped-config", default="configs/grouped.yaml")
    paper_comparison.add_argument("--output", default="results/paper-comparison")
    parameters = subparsers.add_parser(
        "paper-parameters", help="run the manuscript parameter analysis on Colon and SRBCT"
    )
    parameters.add_argument("--data-dir", default="data/processed")
    parameters.add_argument("--config", default="configs/default.yaml")
    parameters.add_argument("--output", default="results/paper-parameters")
    reproduce = subparsers.add_parser(
        "reproduce", help="prepare all data and run the complete proposed-method suite"
    )
    reproduce.add_argument("--raw-root", default="data/raw")
    reproduce.add_argument("--data-dir", default="data/processed")
    reproduce.add_argument("--manifest", default="data/dataset_manifest.csv")
    reproduce.add_argument("--config", default="configs/default.yaml")
    reproduce.add_argument("--grouped-config", default="configs/grouped.yaml")
    reproduce.add_argument("--output", default="results/paper")
    reproduce.add_argument("--ablation", action="store_true")
    reproduce.add_argument(
        "--skip-data", action="store_true", help="reuse already verified processed matrices"
    )
    return parser


def prepare_all_data(raw_root: str | Path, output: str | Path, manifest: str | Path) -> list[dict[str, object]]:
    expected_checksums = load_processed_checksums(manifest)
    raw_root = Path(raw_root)
    records = [
        prepare_geo_dataset(accession, raw_root / "geo", output)
        for accession in sorted(GEO_SPECS)
    ]
    records.extend(
        prepare_legacy_dataset(name, raw_root / "legacy", output)
        for name in LEGACY_SPECS
    )
    for record in records:
        verify_processed_record(record, expected_checksums)
    return records


def run_paper_suite(
    data_dir: str | Path,
    output_dir: str | Path,
    config: str | Path,
    grouped_config: str | Path,
    include_ablation: bool,
    dataset_names: tuple[str, ...] = PAPER_DATASETS,
    variants_override: tuple[str, ...] | None = None,
) -> dict[str, dict[str, object]]:
    reports: dict[str, dict[str, object]] = {}
    variants = variants_override or (tuple(ABLATIONS) if include_ablation else ("abc",))
    for name in dataset_names:
        config_path = grouped_config if name in GROUPED_DATASETS else config
        experiment, optimizer = load_config(config_path)
        output = Path(output_dir) / name
        run_experiment(
            Path(data_dir) / f"{name}.csv.gz",
            output,
            experiment,
            optimizer,
            variants=variants,
        )
        reports[name] = audit_results(output)
    return reports


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "audit":
        report = audit_results(args.output)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["status"] == "PASS" else 1)
    if args.command == "prepare-geo":
        expected_checksums = load_processed_checksums(args.manifest)
        accessions = sorted(GEO_SPECS) if args.accession == "all" else [args.accession]
        records = [
            prepare_geo_dataset(accession, args.raw_dir, args.output)
            for accession in accessions
        ]
        for record in records:
            verify_processed_record(record, expected_checksums)
        print(json.dumps(records, indent=2))
        return
    if args.command == "prepare-legacy":
        expected_checksums = load_processed_checksums(args.manifest)
        names = list(LEGACY_SPECS) if args.dataset == "all" else [args.dataset]
        records = [prepare_legacy_dataset(name, args.raw_dir, args.output) for name in names]
        for record in records:
            verify_processed_record(record, expected_checksums)
        print(json.dumps(records, indent=2))
        return
    if args.command == "prepare-data":
        records = prepare_all_data(args.raw_root, args.output, args.manifest)
        print(json.dumps(records, indent=2))
        return
    if args.command == "suite":
        reports = run_paper_suite(
            args.data_dir, args.output, args.config, args.grouped_config, args.ablation
        )
        print(json.dumps(reports, indent=2))
        raise SystemExit(0 if all(r["status"] == "PASS" for r in reports.values()) else 1)
    if args.command == "paper-ablation":
        reports = run_paper_suite(
            args.data_dir,
            args.output,
            args.config,
            args.config,
            True,
            PAPER_ABLATION_DATASETS,
        )
        print(json.dumps(reports, indent=2))
        raise SystemExit(0 if all(r["status"] == "PASS" for r in reports.values()) else 1)
    if args.command == "paper-comparison":
        methods = ("abc", "none", *AVAILABLE_COMPARATORS)
        reports = run_paper_suite(
            args.data_dir,
            args.output,
            args.config,
            args.grouped_config,
            False,
            PAPER_DATASETS,
            methods,
        )
        print(json.dumps(reports, indent=2))
        raise SystemExit(0 if all(r["status"] == "PASS" for r in reports.values()) else 1)
    if args.command == "paper-parameters":
        summary = run_parameter_analysis(args.data_dir, args.output, args.config)
        print(summary.to_json(orient="records", indent=2))
        return
    if args.command == "reproduce":
        if not args.skip_data:
            prepare_all_data(args.raw_root, args.data_dir, args.manifest)
        reports = run_paper_suite(
            args.data_dir, args.output, args.config, args.grouped_config, args.ablation
        )
        print(json.dumps(reports, indent=2))
        raise SystemExit(0 if all(r["status"] == "PASS" for r in reports.values()) else 1)
    experiment, optimizer = load_config(args.config)
    if args.command == "demo":
        output = Path(args.output)
        data = create_demo_dataset(output / "synthetic.csv", experiment.base_seed)
        run_experiment(data, output, experiment, optimizer, variants=("abc", "none"))
        report = audit_results(output)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["status"] == "PASS" else 1)
    if args.command == "compare":
        variants = tuple(args.method) if args.method else ("abc", "none", *AVAILABLE_COMPARATORS)
    else:
        variants = tuple(ABLATIONS) if args.command == "ablation" else (args.variant,)
    run_experiment(args.data, args.output, experiment, optimizer, variants=variants)
    report = audit_results(args.output)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

