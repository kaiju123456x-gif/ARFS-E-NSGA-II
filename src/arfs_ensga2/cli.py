from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit_results
from .config import load_config
from .datasets import GEO_SPECS, LEGACY_SPECS, prepare_geo_dataset, prepare_legacy_dataset
from .experiment import ABLATIONS, create_demo_dataset, run_experiment


PAPER_DATASETS = (
    "Lung_cancer", "Brain_Tumor", "Breast_cancer_1", "Breast_cancer_2",
    "Pancreatic_Cancer", "Leukemia", "Lung_Adenocarcinoma", "Colon", "SRBCT",
    "11_Tumor",
)
GROUPED_DATASETS = {"Breast_cancer_2", "Pancreatic_Cancer", "Lung_Adenocarcinoma"}


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
    legacy = subparsers.add_parser(
        "prepare-legacy", help="download and prepare five public benchmark matrices"
    )
    legacy.add_argument("--dataset", choices=["all", *LEGACY_SPECS], default="all")
    legacy.add_argument("--raw-dir", default="data/raw/legacy")
    legacy.add_argument("--output", default="data/processed")
    suite = subparsers.add_parser("suite", help="run all ten paper datasets")
    suite.add_argument("--data-dir", default="data/processed")
    suite.add_argument("--config", default="configs/default.yaml")
    suite.add_argument("--grouped-config", default="configs/grouped.yaml")
    suite.add_argument("--output", default="results/paper")
    suite.add_argument("--ablation", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "audit":
        report = audit_results(args.output)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["status"] == "PASS" else 1)
    if args.command == "prepare-geo":
        accessions = sorted(GEO_SPECS) if args.accession == "all" else [args.accession]
        records = [
            prepare_geo_dataset(accession, args.raw_dir, args.output)
            for accession in accessions
        ]
        print(json.dumps(records, indent=2))
        return
    if args.command == "prepare-legacy":
        names = list(LEGACY_SPECS) if args.dataset == "all" else [args.dataset]
        records = [prepare_legacy_dataset(name, args.raw_dir, args.output) for name in names]
        print(json.dumps(records, indent=2))
        return
    if args.command == "suite":
        reports = {}
        variants = tuple(ABLATIONS) if args.ablation else ("abc",)
        for name in PAPER_DATASETS:
            config_path = args.grouped_config if name in GROUPED_DATASETS else args.config
            experiment, optimizer = load_config(config_path)
            output = Path(args.output) / name
            run_experiment(
                Path(args.data_dir) / f"{name}.csv.gz",
                output,
                experiment,
                optimizer,
                variants=variants,
            )
            reports[name] = audit_results(output)
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
    variants = tuple(ABLATIONS) if args.command == "ablation" else (args.variant,)
    run_experiment(args.data, args.output, experiment, optimizer, variants=variants)
    report = audit_results(args.output)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

