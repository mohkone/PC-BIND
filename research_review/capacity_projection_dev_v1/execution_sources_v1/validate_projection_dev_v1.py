"""Read-only identity, checkpoint and initialization gates for the projection ablation."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from compare_oof_development import expected_oof, load_oof, score_metrics
from filtered_cohort_protocol import read_json_with_record, validate_fixed_folds
from frozen_development_protocol import (
    require_equal, run_configuration, summary_configuration, validate_run_metadata,
)
from multiseed_protocol_v1 import validate_prerequisite_record
from research_provenance import file_record, load_dataset_with_provenance
from validate_multiseed_dev_v3 import _finite_tree, _state_schema, reference_state_schema


BINDING_FIELDS = ("training_seed", "fold_seed", "fold_manifest", "execution_binding",
                  "projection_arm", "transport_sinkhorn_iters")
OUTPUT_FILES = frozenset({"run_provenance.json", "ensemble_summary.json", "oof_predictions.npz"}
    | {f"fold{i}_best.pt" for i in range(1, 6)}
    | {f"initialization_fold{i}.{extension}" for i in range(1, 6) for extension in ("json", "pt")})


def _read_record(record):
    value, actual = read_json_with_record(record["path"])
    require_equal(actual, record, "recorded JSON bytes")
    return value


def expected_configuration(manifest_path, manifest, protocol, job):
    """Derive the recipe from the unchanged accepted OT dry runs, with routing only."""
    from projection_protocol_v1 import execution_binding

    ancestry = protocol["accepted_ancestry"]
    acceptance = _read_record(ancestry["accepted_v2_acceptance"])
    v2 = _read_record(ancestry["accepted_v2"])
    require_equal(acceptance.get("execution_manifest"), ancestry["accepted_v2"], "accepted v2 identity")
    if (acceptance.get("status") != "validated_before_training"
            or acceptance.get("jobs_validated") != 6 or acceptance.get("training_started") is not False):
        raise ValueError("All six accepted-v2 dry runs are required")
    records = acceptance.get("dry_run_records")
    if not isinstance(records, list) or len(records) != 6:
        raise ValueError("Incomplete accepted-v2 dry-run inventory")
    index = (2102, 2103, 2104).index(job["training_seed"]) * 2
    dry_record = records[index]
    dry = _read_record(dry_record)
    old_job = v2["jobs"][index]
    require_equal({key: dry.get(key) for key in old_job}, old_job, "accepted OT dry-run job")
    for key, value in (("fold_seed", 2101), ("dry_run", True), ("training_started", False),
                       ("skip_test_eval", True), ("partner_transport", True)):
        require_equal(dry.get(key), value, "accepted OT dry-run " + key)
    require_equal(dry["cv_splits"], manifest["cv_splits"], "accepted fixed folds")
    reference_recipe = dict(protocol["reference_model_recipe"], seed=job["training_seed"])
    require_equal(dry["model_recipe"], reference_recipe, "unchanged accepted model recipe")
    attached = {"training_seed": job["training_seed"], "fold_seed": 2101,
        "fold_manifest": protocol["dataset_selection"]["grouped_fold_manifest"],
        "execution_binding": execution_binding(manifest_path, manifest, protocol),
        "projection_arm": job["arm"], "transport_sinkhorn_iters": job["transport_sinkhorn_iters"]}
    expected_run = copy.deepcopy(dry["run_provenance"])
    expected_run.update(attached)
    expected_run["code_files"] = manifest["source_files"] + manifest["reference_source_files"]
    recipe = dict(reference_recipe, transport_sinkhorn_iters=job["transport_sinkhorn_iters"])
    return {"run_configuration": run_configuration(expected_run),
            "summary_configuration": dict(recipe, **attached)}, dry_record


def validate_launch(manifest_path, manifest, protocol, job):
    from projection_protocol_v1 import execution_environment, trainer_arguments

    process_dir = Path(manifest["process_dir"]).resolve(strict=True)
    stem = f"{job['arm']}_{job['training_seed']}"
    launch, launch_record = read_json_with_record(process_dir / f"{stem}_launch.json")
    gate_path = process_dir / f"{stem}_prerequisite_provenance.json"
    expected = {"schema_version": 1, "component": "capacity_projection_dev_v1_launcher",
        "execution_manifest": file_record(manifest_path), "protocol": manifest["protocol"],
        "job": job, "environment": execution_environment(protocol, job),
        "trainer_command": trainer_arguments(manifest_path, protocol, job, dry_run=False),
        "prerequisite_provenance": file_record(gate_path),
        "training_started": True, "skip_test_eval": True}
    require_equal({key: value for key, value in launch.items() if key != "created_utc"},
                  expected, "projection launch evidence")
    gate = _read_record(launch["prerequisite_provenance"])
    validate_prerequisite_record(gate, protocol, manifest)
    return {"launch": launch_record, "prerequisite_provenance": expected["prerequisite_provenance"]}


def validate_output_inventory(directory):
    entries = list(Path(directory).iterdir())
    if {entry.name for entry in entries} != OUTPUT_FILES or any(not entry.is_file() for entry in entries):
        raise ValueError("Incomplete or unexpected projection output inventory; external artifacts are forbidden")


def validate_metadata(run, summary, expected):
    """All recipe fields remain exact; initialization outcomes require their own gate."""
    if (not isinstance(run.get("initialization_records"), list)
            or len(run["initialization_records"]) != 5):
        raise ValueError("Five initialization records are required")
    require_equal(summary.get("initialization_records"), run["initialization_records"],
                  "summary initialization inventory")
    clean_run = {key: value for key, value in run.items() if key != "initialization_records"}
    clean_summary = {key: value for key, value in summary.items() if key != "initialization_records"}
    clean_summary["run_provenance"] = clean_run
    require_equal(summary.get("run_provenance"), run, "full embedded run provenance")
    validate_run_metadata(clean_run, clean_summary, expected, True)
    _finite_tree(summary)


def validate_checkpoint(checkpoint, run, summary, number, labels, probs, schema):
    """Validate selected outcomes and the exact initialization prefix at saving."""
    fold = number - 1
    _finite_tree(checkpoint)
    require_equal(_state_schema(checkpoint.get("model_state")), schema, "approved transport model-state schema")
    require_equal(checkpoint.get("fold_index"), fold, "checkpoint fold index")
    require_equal(checkpoint.get("cv_split"), run["cv_splits"][fold], "checkpoint split membership")
    saved_run = checkpoint.get("run_provenance", {})
    require_equal(run_configuration({key: value for key, value in saved_run.items() if key != "initialization_records"}),
        run_configuration({key: value for key, value in run.items() if key != "initialization_records"}),
        "checkpoint run configuration")
    require_equal(saved_run.get("provenance_id"), run["provenance_id"], "checkpoint run identity")
    require_equal(saved_run.get("completed_fold_indices"), list(range(number)), "checkpoint completed folds")
    require_equal(saved_run.get("initialization_records"), run["initialization_records"][:number],
                  "checkpoint initialization prefix")
    entry = run["initialization_records"][fold]
    require_equal(checkpoint.get("initialization_record"), entry["record"], "checkpoint initialization JSON")
    require_equal(checkpoint.get("initialization_snapshot"), entry["snapshot"], "checkpoint initialization snapshot")
    if saved_run.get("partner_transport") is not True:
        raise ValueError("Both checkpoint arms require transport")
    for key in ("partner_transport", "skip_test_eval", "use_plm_features", "use_aux_plm_features"):
        if checkpoint.get(key) is not True:
            raise ValueError("Invalid checkpoint " + key)
    for key in BINDING_FIELDS:
        require_equal(checkpoint.get(key), run[key], "checkpoint explicit binding " + key)
    for key in summary:
        if key in checkpoint and key not in {"run_provenance", "partner_transport", "initialization_records"}:
            require_equal(checkpoint[key], summary[key], "checkpoint recipe " + key)
    threshold = checkpoint.get("threshold")
    if type(threshold) not in (float, int) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid checkpoint threshold")
    for key, metric in (("val_mcc", "mcc"), ("val_auc_pr", "auc_pr_trapezoidal")):
        actual = score_metrics(labels, probs, threshold)[metric]
        if not np.isclose(checkpoint.get(key, np.nan), actual, rtol=0, atol=1e-6):
            raise ValueError("Checkpoint does not reproduce OOF fold metric " + key)


def accepted_initializations(manifest_path, manifest, job):
    """Bind structural and first-fold evidence to the accepted real-data dry run."""
    from projection_initialization_v1 import validate_initialization

    path = Path(manifest_path)
    acceptance, _ = read_json_with_record(path.with_name(path.stem + ".acceptance.json"))
    require_equal(acceptance.get("execution_manifest"), file_record(path), "initialization acceptance freeze")
    dry_validation = _read_record(acceptance["all_six_dry_runs"])
    require_equal(dry_validation.get("execution_manifest"), file_record(path), "initialization dry-run freeze")
    records = dry_validation.get("dry_run_records")
    if (dry_validation.get("status") != "passed" or dry_validation.get("jobs_validated") != 6
            or dry_validation.get("training_started") is not False
            or not isinstance(records, list) or len(records) != 6):
        raise ValueError("Six accepted initialization dry runs are required")
    record = records[manifest["jobs"].index(job)]
    dry = _read_record(record)
    for key in ("training_seed", "arm", "output_dir", "partner_transport"):
        require_equal(dry.get(key), job[key], "accepted initialization dry-run " + key)
    for key, value in (("fold_seed", 2101), ("dry_run", True), ("training_started", False),
                       ("skip_test_eval", True)):
        require_equal(dry.get(key), value, "accepted initialization " + key)
    require_equal(dry["cv_splits"], manifest["cv_splits"], "accepted initialization folds")
    initializations = dry.get("initialization_records")
    if not isinstance(initializations, list) or len(initializations) != 5:
        raise ValueError("Five accepted initialization observations are required")
    for index, initialization in enumerate(initializations):
        validate_initialization(initialization)
        for key, value in (("fold_index", index), ("training_seed", job["training_seed"]),
            ("fold_seed", 2101), ("arm", job["arm"]),
            ("projection_iterations", job["transport_sinkhorn_iters"])):
            require_equal(initialization.get(key), value, "accepted initialization identity " + key)
    return initializations, record


def validate_initializations(directory, run, job, schema, approved):
    import torch
    from projection_initialization_v1 import validate_initialization

    validated, artifacts = [], []
    for number, entry in enumerate(run["initialization_records"], 1):
        if set(entry) != {"fold_index", "record", "snapshot"} or entry["fold_index"] != number - 1:
            raise ValueError("Incomplete or reordered initialization artifact identity")
        record_path = Path(directory) / f"initialization_fold{number}.json"
        snapshot_path = Path(directory) / f"initialization_fold{number}.pt"
        require_equal(entry["record"], file_record(record_path), "initialization JSON artifact")
        require_equal(entry["snapshot"], file_record(snapshot_path), "initialization snapshot artifact")
        record = _read_record(entry["record"])
        snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
        _finite_tree(snapshot)
        validate_initialization(record, snapshot)
        for key, expected in (("training_seed", job["training_seed"]), ("fold_seed", 2101),
            ("fold_index", number - 1), ("arm", job["arm"]),
            ("projection_iterations", job["transport_sinkhorn_iters"])):
            require_equal(record.get(key), expected, "initialization " + key)
        require_equal(_state_schema(snapshot.get("model_state")), schema, "initial transport model-state schema")
        for key in ("parameter_schema", "buffer_schema", "state_schema", "parameter_count",
                    "trainable_parameter_count", "optimizer"):
            require_equal(record[key], approved[number - 1][key], "accepted initial model/optimizer " + key)
        if number == 1:
            for key in ("parameter_sha256", "state_sha256", "rng_before", "rng_after"):
                require_equal(record[key], approved[0][key], "accepted first-fold initialization " + key)
        if validated:
            # Every fold retains the same parameter inventory and optimizer setup.
            for key in ("parameter_schema", "buffer_schema", "state_schema", "parameter_count",
                        "trainable_parameter_count", "optimizer"):
                require_equal(record[key], validated[0][key], "within-run initialization " + key)
        require_equal(file_record(record_path), entry["record"], "initialization JSON changed while reading")
        require_equal(file_record(snapshot_path), entry["snapshot"], "initialization snapshot changed while reading")
        validated.append(record)
        artifacts.append(entry)
        del snapshot
    return validated, artifacts


def validate_arm(manifest_path, job):
    import torch
    from projection_protocol_v1 import verify_manifest

    manifest_record = file_record(manifest_path)
    manifest, protocol = verify_manifest(manifest_path, require_acceptance=True)
    if sum(declared == job for declared in manifest["jobs"]) != 1:
        raise ValueError("Arm is not a unique declared projection job")
    if (type(job.get("training_seed")) is not int or job["training_seed"] not in (2102, 2103, 2104)
            or job.get("fold_seed") != 2101 or job.get("max_folds") != 5
            or job.get("skip_test_eval") is not True or job.get("partner_transport") is not True
            or job.get("arm") not in ("projected", "rowsoftmax")
            or job.get("transport_sinkhorn_iters") != (5 if job["arm"] == "projected" else 0)):
        raise ValueError("Wrong projection arm, seed, fixed folds or external-evaluation setting")
    directory = Path(job["output_dir"]).resolve(strict=True)
    validate_output_inventory(directory)
    expected, dry_record = expected_configuration(manifest_path, manifest, protocol, job)
    run, run_record = read_json_with_record(directory / "run_provenance.json")
    summary, summary_record = read_json_with_record(directory / "ensemble_summary.json")
    validate_metadata(run, summary, expected)
    artifacts = validate_launch(manifest_path, manifest, protocol, job)
    artifacts.update(run_provenance=run_record, ensemble_summary=summary_record,
        accepted_recipe_dry_run=dry_record, execution_manifest=manifest_record)
    training = run["datasets"]["Train335"]
    proteins, actual = load_dataset_with_provenance(training["path"])
    require_equal(actual, training, "loaded training dataset identity")
    require_equal(len(proteins), protocol["sample_count"], "full sample coverage")
    folds_json = _read_record(run["fold_manifest"])
    folds = validate_fixed_folds(proteins, folds_json, 2101, training["sha256"], "complex_code")
    require_equal(folds, run["cv_splits"], "actual fixed fold identities")
    labels, identity = expected_oof(proteins, folds)
    del proteins
    require_equal(len(labels), protocol["residue_count"], "full residue coverage")
    require_equal(len(np.unique(identity["sample_index"])), protocol["sample_count"], "full sample identities")
    require_equal(len(np.unique(np.char.upper(np.char.strip(identity["complex_id"])))),
                  protocol["complex_count"], "full complex identities")
    labels, probs, threshold, oof_record = load_oof(directory / "oof_predictions.npz", run, labels, identity)
    if not np.isclose(threshold, summary["ensemble_threshold"], rtol=0, atol=1e-7):
        raise ValueError("OOF threshold differs from saved summary")
    schema, schema_record = reference_state_schema(
        {"reference_comparison": protocol["observed_references"]["historical_seed2101_comparison"]}, "ot")
    approved_initializations, approved_record = accepted_initializations(manifest_path, manifest, job)
    initializations, initialization_artifacts = validate_initializations(directory, run, job, schema, approved_initializations)
    artifacts.update(oof_predictions=oof_record, reference_state_schema=schema_record,
                     initializations=initialization_artifacts,
                     accepted_initialization_dry_run=approved_record, checkpoints=[])
    for number in range(1, 6):
        path = directory / f"fold{number}_best.pt"
        before = file_record(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        mask = identity["fold_index"] == number - 1
        validate_checkpoint(checkpoint, run, summary, number, labels[mask], probs[mask], schema)
        require_equal(file_record(path), before, "checkpoint changed while reading")
        artifacts["checkpoints"].append(before)
        del checkpoint
    for record in (run_record, summary_record, manifest_record, artifacts["launch"], artifacts["prerequisite_provenance"]):
        require_equal(file_record(record["path"]), record, "evidence changed during validation")
    return {"labels": labels, "probs": probs, "threshold": threshold, "identity": identity,
        "run": run, "summary": summary, "job": job, "artifacts": artifacts, "initializations": initializations}


def validate_pair_records(projected, rowsoftmax):
    from projection_initialization_v1 import validate_initialization_pair

    left, right = projected["run"], rowsoftmax["run"]
    if (left.get("projection_arm") != "projected" or right.get("projection_arm") != "rowsoftmax"
            or left.get("training_seed") != right.get("training_seed")
            or left.get("training_seed") not in (2102, 2103, 2104)
            or left.get("partner_transport") is not True or right.get("partner_transport") is not True
            or left.get("transport_sinkhorn_iters") != 5 or right.get("transport_sinkhorn_iters") != 0):
        raise ValueError("Wrong matched projection seed or arm")
    outcomes = {"projection_arm", "transport_sinkhorn_iters", "initialization_records"}
    require_equal({key: value for key, value in run_configuration(left).items() if key not in outcomes},
        {key: value for key, value in run_configuration(right).items() if key not in outcomes},
        "paired run recipe and provenance")
    require_equal({key: value for key, value in summary_configuration(projected["summary"]).items() if key not in outcomes},
        {key: value for key, value in summary_configuration(rowsoftmax["summary"]).items() if key not in outcomes},
        "paired model recipe")
    if left["provenance_id"] == right["provenance_id"]:
        raise ValueError("Matched arms must have distinct run identities")
    if Path(projected["job"]["output_dir"]).resolve() == Path(rowsoftmax["job"]["output_dir"]).resolve():
        raise ValueError("Matched arms share an output directory")
    for key in ("sample_index", "residue_index", "complex_id", "fold_index"):
        if not np.array_equal(projected["identity"][key], rowsoftmax["identity"][key]):
            raise ValueError("Matched OOF identity differs: " + key)
    if not np.array_equal(projected["labels"], rowsoftmax["labels"]):
        raise ValueError("Matched OOF labels differ")
    if len(projected["initializations"]) != 5 or len(rowsoftmax["initializations"]) != 5:
        raise ValueError("Matched arms require five initialization records")
    evidence = [validate_initialization_pair(a, b, require_initial_values=index == 0)
        for index, (a, b) in enumerate(zip(projected["initializations"], rowsoftmax["initializations"]))]
    return {"training_seed": left["training_seed"], "fold_seed": 2101,
        "first_fold_initial_parameter_equality_required": True,
        "later_fold_initial_equality_required": False, "folds": evidence,
        "no_fold_reseeding": True, "external_evaluation": "skipped in both arms"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validate-only", action="store_true", required=True)
    args = parser.parse_args()
    from projection_protocol_v1 import verify_manifest
    manifest, _ = verify_manifest(args.manifest, require_acceptance=True)
    arms = [validate_arm(args.manifest, job) for job in manifest["jobs"]]
    pairs = [validate_pair_records(arms[index], arms[index + 1]) for index in (0, 2, 4)]
    print(json.dumps({"status": "validated_complete_six_runs", "outputs_written": False,
        "bootstrap_run": False, "artifacts": [arm["artifacts"] for arm in arms],
        "matching_evidence": pairs}, indent=2, allow_nan=False))
