"""Read-only checks for the frozen, five-fold filtered development experiment."""
import json
from pathlib import Path

import numpy as np

from filtered_cohort_protocol import verify_filtered_cohort
from research_provenance import build_run_provenance, file_record


# These values are outputs of selection, not experimental settings.
SUMMARY_OUTCOMES = frozenset({
    "run_provenance", "ensemble", "single_fold_summary", "val_mcc_weights",
    "val_aupr_weights", "ensemble_threshold", "rank_threshold", "partner_transport",
})
RUN_IDENTITY = frozenset({
    "provenance_id", "created_utc", "status", "completed_fold_indices", "partner_transport",
})


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def require_equal(actual, expected, label):
    # JSON comparison also distinguishes booleans from integer indices/settings.
    if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError(f"Frozen protocol mismatch: {label}")


def summary_configuration(summary):
    return {key: value for key, value in summary.items() if key not in SUMMARY_OUTCOMES}


def run_configuration(run):
    return {key: value for key, value in run.items() if key not in RUN_IDENTITY}


def cohort_selection(selection):
    # Training records its explicit CLI selection mode in addition to the
    # shared cohort record returned by the prerequisite/manifest validator.
    if selection.get("selection_mode") != "explicit":
        raise ValueError("Frozen training must use explicit CLI dataset selection")
    return {key: value for key, value in selection.items() if key != "selection_mode"}


def read_freeze(path):
    freeze = read_json(path)
    if (freeze.get("schema_version") != 1 or freeze.get("seed") != 2101
            or freeze.get("max_folds") != 5 or freeze.get("skip_test_eval") is not True
            or freeze.get("analysis_label") != "filtered-cohort analysis"):
        raise ValueError("Unsupported frozen development experiment")
    run = freeze["run_configuration"]
    if (run.get("scheduled_fold_indices") != list(range(5))
            or run.get("loaded_test_sets") != [] or run.get("skip_test_eval") is not True
            or set(run.get("datasets", {})) != {"Train335"}
            or run.get("cv_group_key") != "complex_code" or run.get("grouped_cv") is not True):
        raise ValueError("Freeze must specify all five grouped development folds and no external evaluation")
    if freeze["comparison"] != {
        "bootstrap_replicates": 5000, "bootstrap_seed": 2101,
        "resampling_unit": "complex_code", "difference": "OT minus no-OT",
        "primary_metric": "auc_pr_trapezoidal", "secondary_metric": "mcc",
        "threshold_policy": "hold each arm's saved pooled OOF threshold fixed",
        "confidence_interval": "paired percentile 95%",
    }:
        raise ValueError("Comparison plan differs from the frozen analysis")
    return freeze


def verify_environment(freeze, *, check_runtime=True):
    for record in freeze["source_files"]:
        require_equal(file_record(record["path"]), record, f"source {record['path']}")
    selection = freeze["run_configuration"]["dataset_selection"]
    actual, _, folds = verify_filtered_cohort(
        selection["resolved_data_dir"], selection["cohort_manifest"]["path"],
        selection["grouped_fold_manifest"]["path"], freeze["seed"],
    )
    require_equal(actual, cohort_selection(selection), "resolved cohort and hashes")
    require_equal(folds["folds"], freeze["run_configuration"]["cv_splits"], "fixed folds")
    if check_runtime:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        runtime = build_run_provenance({}, [], [], device, 0)["runtime"]
        require_equal(runtime, freeze["run_configuration"]["runtime"], "current runtime")
    return folds


def validate_run_metadata(run, summary, freeze, transport, *, require_complete=True):
    require_equal(run_configuration(run), freeze["run_configuration"], "run configuration")
    require_equal(summary_configuration(summary), freeze["summary_configuration"], "model recipe")
    require_equal(summary.get("run_provenance"), run, "embedded run provenance")
    if not isinstance(run.get("provenance_id"), str) or not run["provenance_id"].strip():
        raise ValueError("Missing run identity")
    if run.get("partner_transport") is not transport or summary.get("partner_transport") is not transport:
        raise ValueError("Incorrect transport arm")
    if require_complete:
        if run.get("status") != "complete":
            raise ValueError("Comparison requires all five folds to have completed")
        require_equal(run.get("completed_fold_indices"), list(range(5)), "all five completed fold indices")
        for name in ("val_mcc_weights", "val_aupr_weights"):
            weights = np.asarray(summary.get(name), dtype=float)
            if (weights.shape != (5,) or not np.isfinite(weights).all()
                    or np.any(weights < 0) or not np.isclose(weights.sum(), 1)):
                raise ValueError(f"Invalid five-fold {name}")
        if summary.get("single_fold_summary") != {}:
            raise ValueError("External test metrics present in development summary")
        ensembles = summary.get("ensemble")
        if not isinstance(ensembles, dict) or not ensembles or any(value != {} for value in ensembles.values()):
            raise ValueError("External ensemble metrics present in development summary")


def validate_launch_records(directory, freeze, transport):
    directory = Path(directory).resolve(strict=True)
    selection = freeze["run_configuration"]["dataset_selection"]
    launcher = read_json(directory / "launcher_provenance.json")
    expected = {
        "schema_version": 1, "component": "filtered_v1_launcher",
        "analysis_label": freeze["analysis_label"],
        "resolved_data_dir": selection["resolved_data_dir"],
        "declared_files": selection["declared_files"],
        "cohort_manifest": selection["cohort_manifest"],
        "datasets": selection["datasets"], "fold_manifest": selection["grouped_fold_manifest"],
        "seed": 2101, "max_folds": 5, "grouped_cv": True, "cv_group_key": "complex_code",
        "partner_transport": transport, "skip_test_eval": True,
        "external_test_sets": ["Test287.pkl"], "output_dir": str(directory),
    }
    # PowerShell adds creation time; all fields describing the launch must match.
    require_equal({k: v for k, v in launcher.items() if k != "created_utc"}, expected, "launcher")
    gate = read_json(directory / "prerequisite_provenance.json")
    require_equal(gate.get("dataset_selection"), cohort_selection(selection), "prerequisite cohort")
    require_equal(gate.get("loaded_datasets"), selection["datasets"], "prerequisite inputs")
    for key in ("require_plm", "require_partner", "require_pair", "require_partner_encoder", "fixed_folds_validated"):
        if gate.get(key) is not True:
            raise ValueError(f"Strict prerequisite missing: {key}")
    if gate.get("status") != "passed" or gate.get("errors") != [] or gate.get("min_partner_encoder_coverage") != 1.0:
        raise ValueError("Strict prerequisite gate did not pass")
    require_equal(gate.get("code_files"), freeze["prerequisite_code_files"], "prerequisite source hashes")


def validate_completed_run(directory, freeze, transport):
    directory = Path(directory)
    run = read_json(directory / "run_provenance.json")
    summary = read_json(directory / "ensemble_summary.json")
    validate_run_metadata(run, summary, freeze, transport)
    validate_launch_records(directory, freeze, transport)
    if list(directory.glob("ensemble_*_predictions.npz")):
        raise ValueError("Unexpected external predictions in a development run")
    for number in range(1, 6):
        if not (directory / f"fold{number}_best.pt").is_file():
            raise ValueError(f"Missing selected checkpoint for fold {number}")
    return run, summary
