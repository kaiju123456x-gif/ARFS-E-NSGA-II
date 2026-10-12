from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _json_indices(value: object, label: str, errors: list[str]) -> list[int]:
    try:
        parsed = json.loads(str(value))
        if not isinstance(parsed, list) or any(
            not isinstance(item, int) for item in parsed
        ):
            raise ValueError
        return parsed
    except (TypeError, ValueError, json.JSONDecodeError):
        errors.append(f"invalid JSON index list in {label}")
        return []


def _write_report(output: Path, report: dict[str, object]) -> dict[str, object]:
    if output.exists():
        (output / "audit_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    return report


def audit_results(output_dir: str | Path) -> dict[str, object]:
    """Validate the fold-level evidence required by the manuscript protocol."""
    output = Path(output_dir)
    required = [
        "fold_results.csv",
        "archives.csv",
        "splits.json",
        "manifest.json",
        "summary.csv",
    ]
    missing = [name for name in required if not (output / name).exists()]
    errors: list[str] = []
    if missing:
        return _write_report(
            output, {"status": "FAIL", "errors": [f"missing files: {missing}"]}
        )

    results = pd.read_csv(output / "fold_results.csv")
    archives = pd.read_csv(output / "archives.csv")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    splits = json.loads((output / "splits.json").read_text(encoding="utf-8"))
    variants = manifest["variants"]
    experiment = manifest["config"]["experiment"]
    expected_folds = experiment["outer_splits"] * experiment["outer_repeats"]
    expected_rows = expected_folds * len(variants)

    required_columns = {
        "variant",
        "method_id",
        "method",
        "repeat",
        "fold",
        "seed",
        "n_features",
        "evaluations",
        "hypervolume",
        "selected_candidate_indices",
        "selected_original_indices",
        "selected_feature_names",
    }
    absent_columns = sorted(required_columns - set(results.columns))
    if absent_columns:
        errors.append(f"fold_results.csv is missing columns: {absent_columns}")
        return _write_report(output, {"status": "FAIL", "errors": errors})

    if len(results) != expected_rows:
        errors.append(f"expected {expected_rows} result rows, found {len(results)}")
    if len(splits) != expected_folds:
        errors.append(f"expected {expected_folds} split records, found {len(splits)}")
    if set(results["variant"]) != set(variants):
        errors.append("result methods do not match manifest variants")
    if not (results["variant"] == results["method_id"]).all():
        errors.append("variant and method_id columns disagree")

    counts = results.groupby("variant").size().to_dict()
    for variant in variants:
        if counts.get(variant, 0) != expected_folds:
            errors.append(
                f"method {variant} has {counts.get(variant, 0)} folds; expected {expected_folds}"
            )

    duplicate_keys = int(results.duplicated(["variant", "repeat", "fold"]).sum())
    if duplicate_keys:
        errors.append(f"found {duplicate_keys} duplicate run keys")
    seed_counts = results.groupby(["repeat", "fold"])["seed"].nunique()
    if len(seed_counts) != expected_folds or not (seed_counts == 1).all():
        errors.append("paired methods do not share exactly one seed per outer fold")

    budget = manifest["config"]["optimizer"]["evaluation_budget"]
    budget_types = manifest.get("method_budget_types", {})
    evolutionary = results["variant"].map(budget_types).ne("single_subset")
    if not (results.loc[evolutionary, "evaluations"] == budget).all():
        errors.append("one or more evolutionary runs did not use the configured evaluation budget")
    if not (results.loc[~evolutionary, "evaluations"] == 1).all():
        errors.append("one or more single-subset runs used an unexpected evaluation count")
    if results.loc[evolutionary, "hypervolume"].isna().any():
        errors.append("one or more evolutionary runs are missing hypervolume")
    if not results.loc[~evolutionary, "hypervolume"].isna().all():
        errors.append("single-subset methods must not report Pareto hypervolume")

    allowed_missing = (~evolutionary) & results["hypervolume"].isna()
    unexpected_missing = results.isna()
    unexpected_missing.loc[allowed_missing, "hypervolume"] = False
    if unexpected_missing.any().any():
        errors.append("fold_results.csv contains unexpected missing values")

    split_by_key: dict[tuple[int, int], dict[str, object]] = {}
    original_features = int(manifest["shape"]["features"])
    expected_candidates = min(int(experiment["candidate_features"]), original_features)
    for record in splits:
        key = (int(record["repeat"]), int(record["fold"]))
        if key in split_by_key:
            errors.append(f"duplicate split record for repeat/fold {key}")
        split_by_key[key] = record
        train = list(map(int, record["train_indices"]))
        test = list(map(int, record["test_indices"]))
        if len(train) != len(set(train)) or len(test) != len(set(test)):
            errors.append(f"duplicate sample index in outer split {key}")
        if set(train) & set(test):
            errors.append(f"outer train/test overlap in split {key}")
        candidates = list(map(int, record["candidate_indices"]))
        if len(candidates) != expected_candidates or len(candidates) != len(
            set(candidates)
        ):
            errors.append(f"invalid candidate pool in split {key}")
        if any(index < 0 or index >= original_features for index in candidates):
            errors.append(f"candidate index outside original feature space in split {key}")
        if len(record.get("candidate_names", [])) != len(candidates):
            errors.append(f"candidate name count mismatch in split {key}")
        if int(record.get("outer_group_overlap_count", 0)) != 0:
            errors.append(f"group leakage in outer split {key}")
        inner_records = record["inner_splits_relative_to_outer_train"]
        if len(inner_records) != int(experiment["inner_splits"]):
            errors.append(f"wrong inner-fold count in split {key}")
        for inner_number, inner in enumerate(inner_records):
            inner_train = list(map(int, inner["train"]))
            inner_valid = list(map(int, inner["validation"]))
            if set(inner_train) & set(inner_valid):
                errors.append(
                    f"inner train/validation overlap in split {key}, fold {inner_number}"
                )
            if any(index < 0 or index >= len(train) for index in inner_train + inner_valid):
                errors.append(f"inner index outside outer-training set in split {key}")
            if int(inner.get("group_overlap_count", 0)) != 0:
                errors.append(f"group leakage in inner split {key}, fold {inner_number}")

    for row_number, row in results.iterrows():
        key = (int(row["repeat"]), int(row["fold"]))
        split = split_by_key.get(key)
        if split is None:
            errors.append(f"result row has no matching split record: {key}")
            continue
        candidate_count = len(split["candidate_indices"])
        selected_candidate = _json_indices(
            row["selected_candidate_indices"], f"row {row_number} candidate indices", errors
        )
        selected_original = _json_indices(
            row["selected_original_indices"], f"row {row_number} original indices", errors
        )
        try:
            selected_names = json.loads(str(row["selected_feature_names"]))
        except json.JSONDecodeError:
            selected_names = []
            errors.append(f"invalid selected_feature_names JSON in row {row_number}")
        expected_size = int(row["n_features"])
        if expected_size < 1 or expected_size > candidate_count:
            errors.append(f"invalid selected feature count in row {row_number}")
        if not (
            len(selected_candidate) == len(set(selected_candidate)) == expected_size
            and len(selected_original) == len(set(selected_original)) == expected_size
            and isinstance(selected_names, list)
            and len(selected_names) == expected_size
        ):
            errors.append(f"selected subset evidence mismatch in row {row_number}")
        if any(index < 0 or index >= candidate_count for index in selected_candidate):
            errors.append(f"selected candidate index outside pool in row {row_number}")
        elif selected_original != [
            int(split["candidate_indices"][index]) for index in selected_candidate
        ]:
            errors.append(f"candidate-to-original index mapping mismatch in row {row_number}")
        elif selected_names != [
            split["candidate_names"][index] for index in selected_candidate
        ]:
            errors.append(f"candidate-to-feature-name mapping mismatch in row {row_number}")

        diagnostic = output / "diagnostics" / (
            f"{row['variant']}_r{int(row['repeat'])}_f{int(row['fold'])}.json"
        )
        if not diagnostic.exists():
            errors.append(f"missing diagnostics file: {diagnostic.name}")

    archive_keys = set(zip(archives["variant"], archives["repeat"], archives["fold"]))
    for row in results.itertuples(index=False):
        if (row.variant, row.repeat, row.fold) not in archive_keys:
            errors.append(
                f"missing archive for {row.variant}, repeat {row.repeat}, fold {row.fold}"
            )

    checks = {
        "paired_outer_folds": not errors,
        "shared_method_seeds": not errors,
        "disjoint_outer_and_inner_partitions": not errors,
        "zero_group_overlap": not errors,
        "candidate_pool_integrity": not errors,
        "exact_evolutionary_budget": not errors,
        "subset_evidence_integrity": not errors,
        "archive_and_diagnostics_present": not errors,
    }
    return _write_report(
        output,
        {
            "status": "PASS" if not errors else "FAIL",
            "errors": errors,
            "result_rows": len(results),
            "split_records": len(splits),
            "methods": counts,
            "evaluation_budget": budget,
            "checks": checks,
        },
    )
