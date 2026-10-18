# Complete Manuscript Experiment Plan

This directory is an isolated execution copy for the experiments described in
the manuscript. It does not reuse numerical results from the historical code.

## Fixed protocol

| Item | Value |
|---|---|
| Datasets | 10 manuscript datasets |
| Outer evaluation | 5 repeats x 5 folds |
| Inner evaluation | 3 folds |
| Linked samples | Stratified group splits for GSE3726, GSE15471, and GSE10072 |
| Candidate pool | Training-only ANOVA, 30 features |
| Objectives | Balanced error, selected features / 30, mean pairwise MI |
| Population | 12 |
| Objective-call budget | 240 |
| Endpoint | Lexicographic training error, feature ratio, redundancy |
| Hypervolume reference | (1.1, 1.1, 1.1), with RED transformed by z/(1+z) |

## Experiment matrix

| Phase | Scope |
|---|---:|
| Comparison | 10 datasets x 11 methods x 25 outer folds |
| Ablation | 2 datasets x 8 variants x 25 outer folds |
| Parameters | 2 datasets x 37 settings x 25 outer folds |

The runner is task-level resumable. A task is reused only when
`task_status.json` records an audited `PASS` with the expected method set.

## Commands

```powershell
D:\chatgpt\envs\research\Scripts\python.exe -m arfs_ensga2.cli prepare-data
D:\chatgpt\envs\research\Scripts\python.exe scripts\run_all_paper_experiments.py --workers 3
```

Results are written under `results/paper_complete/`. Failures are retained with
their traceback and are never silently retried within the same invocation.

## Completed archive

The complete run contains 86 audited tasks, 5,000 method-fold result rows, and
2,150 split records. Generate or refresh the final aggregate tables and archive
validation report with:

```powershell
D:\chatgpt\envs\research\Scripts\python.exe scripts\summarize_completed_experiments.py
```

The principal entry point for reviewing the completed run is
`results/paper_complete/summary/experiment_validation_report.md`. An audit
`PASS` establishes archive completeness and protocol integrity; it does not by
itself assert agreement with numerical values produced by historical code.
