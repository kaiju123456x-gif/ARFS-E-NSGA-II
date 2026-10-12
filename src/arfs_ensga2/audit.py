from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def audit_results(output_dir: str | Path) -> dict[str, object]:
    output = Path(output_dir)
    required = ["fold_results.csv", "archives.csv", "splits.json", "manifest.json", "summary.csv"]
    missing = [name for name in required if not (output / name).exists()]
    errors: list[str] = []
    if missing:
        errors.append(f"missing files: {missing}")
        return {"status": "FAIL", "errors": errors}
    results = pd.read_csv(output / "fold_results.csv")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    splits = json.loads((output / "splits.json").read_text(encoding="utf-8"))
    expected_folds = (
        manifest["config"]["experiment"]["outer_splits"]
        * manifest["config"]["experiment"]["outer_repeats"]
    )
    expected_rows = expected_folds * len(manifest["variants"])
    if len(results) != expected_rows:
        errors.append(f"expected {expected_rows} result rows, found {len(results)}")
    if len(splits) != expected_folds:
        errors.append(f"expected {expected_folds} split records, found {len(splits)}")
    budget = manifest["config"]["optimizer"]["evaluation_budget"]
    budget_types = manifest.get("method_budget_types", {})
    evolutionary = results["variant"].map(budget_types).ne("single_subset")
    if not (results.loc[evolutionary, "evaluations"] == budget).all():
        errors.append("one or more evolutionary runs did not use the configured evaluation budget")
    if not (results.loc[~evolutionary, "evaluations"] == 1).all():
        errors.append("one or more single-subset runs used an unexpected evaluation count")
    allowed_missing = results["variant"].map(budget_types).eq("single_subset") & results[
        "hypervolume"
    ].isna()
    unexpected_missing = results.isna()
    unexpected_missing.loc[allowed_missing, "hypervolume"] = False
    if unexpected_missing.any().any():
        errors.append("fold_results.csv contains unexpected missing values")
    duplicate_keys = results.duplicated(["variant", "repeat", "fold"]).sum()
    if duplicate_keys:
        errors.append(f"found {duplicate_keys} duplicate run keys")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "result_rows": len(results),
        "split_records": len(splits),
        "evaluation_budget": budget,
    }

