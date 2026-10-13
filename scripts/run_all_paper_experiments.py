"""Run the complete manuscript experiment matrix with resumable task-level audits.

The runner follows ``paper/experiment_contract.yaml`` and the settings encoded in
``configs/default.yaml`` / ``configs/grouped.yaml``.  Every task writes to an
independent directory and is skipped on resume only after a successful audit.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from arfs_ensga2.audit import audit_results
from arfs_ensga2.cli import GROUPED_DATASETS, PAPER_ABLATION_DATASETS, PAPER_DATASETS
from arfs_ensga2.comparators import AVAILABLE_COMPARATORS
from arfs_ensga2.config import load_config
from arfs_ensga2.experiment import ABLATIONS, run_experiment
from arfs_ensga2.paper_experiments import SUMMARY_METRICS, parameter_settings


COMPARISON_METHODS = ("abc", "none", *AVAILABLE_COMPARATORS)
STATUS_FILE = "task_status.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _completed(output: Path, expected_variants: tuple[str, ...]) -> bool:
    status_path = output / STATUS_FILE
    if not status_path.exists():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return status.get("status") == "PASS" and status.get("expected_variants") == list(
        expected_variants
    )


def _run_task(task: dict[str, object]) -> dict[str, object]:
    output = Path(str(task["output"]))
    expected_variants = tuple(str(value) for value in task["variants"])
    if _completed(output, expected_variants):
        return {**task, "status": "SKIPPED", "reason": "previous audited PASS"}

    started = time.perf_counter()
    status: dict[str, object] = {
        **task,
        "status": "RUNNING",
        "expected_variants": list(expected_variants),
        "started_utc": _utc_now(),
    }
    _write_json(output / STATUS_FILE, status)
    try:
        experiment, optimizer = load_config(str(task["config"]))
        updates = dict(task.get("optimizer_updates", {}))
        if updates:
            optimizer = replace(optimizer, **updates)
            optimizer.validate()
        frame = run_experiment(
            str(task["data"]),
            output,
            experiment,
            optimizer,
            variants=expected_variants,
        )
        audit = audit_results(output)
        if audit["status"] != "PASS":
            raise RuntimeError(f"result audit failed: {audit}")
        observed = tuple(sorted(frame["variant"].unique().tolist()))
        expected = tuple(sorted(expected_variants))
        if observed != expected:
            raise RuntimeError(f"variant mismatch: expected {expected}, observed {observed}")
        status.update(
            {
                "status": "PASS",
                "finished_utc": _utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
                "rows": len(frame),
                "audit": audit,
                "optimizer": asdict(optimizer),
            }
        )
    except Exception as error:  # preserve complete failure evidence for resume
        status.update(
            {
                "status": "FAIL",
                "finished_utc": _utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
    _write_json(output / STATUS_FILE, status)
    return status


def _tasks(root: Path, phases: set[str]) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    data_dir = root / "data" / "processed"
    default = root / "configs" / "default.yaml"
    grouped = root / "configs" / "grouped.yaml"
    result_root = root / "results" / "paper_complete"

    if "comparison" in phases:
        for dataset in PAPER_DATASETS:
            tasks.append(
                {
                    "phase": "comparison",
                    "task_id": f"comparison/{dataset}",
                    "data": str(data_dir / f"{dataset}.csv.gz"),
                    "config": str(grouped if dataset in GROUPED_DATASETS else default),
                    "output": str(result_root / "comparison" / dataset),
                    "variants": COMPARISON_METHODS,
                }
            )

    if "ablation" in phases:
        for dataset in PAPER_ABLATION_DATASETS:
            tasks.append(
                {
                    "phase": "ablation",
                    "task_id": f"ablation/{dataset}",
                    "data": str(data_dir / f"{dataset}.csv.gz"),
                    "config": str(default),
                    "output": str(result_root / "ablation" / dataset),
                    "variants": tuple(ABLATIONS),
                }
            )

    if "parameters" in phases:
        for dataset in PAPER_ABLATION_DATASETS:
            for setting_id, updates in parameter_settings():
                tasks.append(
                    {
                        "phase": "parameters",
                        "task_id": f"parameters/{dataset}/{setting_id}",
                        "data": str(data_dir / f"{dataset}.csv.gz"),
                        "config": str(default),
                        "output": str(result_root / "parameters" / dataset / setting_id),
                        "variants": ("abc",),
                        "optimizer_updates": updates,
                    }
                )
    return tasks


def _summarize(root: Path, statuses: list[dict[str, object]]) -> None:
    result_root = root / "results" / "paper_complete"
    summary_root = result_root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(statuses).drop(columns=["traceback", "audit"], errors="ignore").to_csv(
        summary_root / "task_status.csv", index=False
    )

    parameter_rows: list[dict[str, object]] = []
    for status_path in (result_root / "parameters").glob("**/task_status.json"):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "PASS":
            continue
        output = status_path.parent
        frame = pd.read_csv(output / "fold_results.csv")
        relative = output.relative_to(result_root / "parameters")
        parts = relative.parts
        row: dict[str, object] = {
            "dataset": parts[0],
            "setting": "/".join(parts[1:]),
            **dict(status.get("optimizer_updates", {})),
        }
        for metric in SUMMARY_METRICS:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_std"] = float(frame[metric].std(ddof=1))
        parameter_rows.append(row)
    if parameter_rows:
        pd.DataFrame(parameter_rows).sort_values(["dataset", "setting"]).to_csv(
            summary_root / "parameter_analysis_summary.csv", index=False
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--phase",
        action="append",
        choices=("comparison", "ablation", "parameters"),
        help="repeat to select phases; default runs all phases",
    )
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    root = args.root.resolve()
    phases = set(args.phase or ("comparison", "ablation", "parameters"))
    tasks = _tasks(root, phases)
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

    statuses: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_run_task, task): task for task in tasks}
        for future in as_completed(futures):
            status = future.result()
            statuses.append(status)
            print(
                json.dumps(
                    {
                        "task_id": status["task_id"],
                        "status": status["status"],
                        "elapsed_seconds": status.get("elapsed_seconds"),
                        "error": status.get("error"),
                    }
                ),
                flush=True,
            )
            _summarize(root, statuses)

    failed = [status for status in statuses if status["status"] == "FAIL"]
    _summarize(root, statuses)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
