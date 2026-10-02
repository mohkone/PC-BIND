"""Read-only validation of completed accepted-v2 training under v3 launch control."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from compare_oof_development import expected_oof, load_oof, validate_checkpoint
from filtered_cohort_protocol import read_json_with_record, validate_fixed_folds
from frozen_development_protocol import require_equal, run_configuration, validate_run_metadata
from multiseed_protocol_v1 import execution_environment, validate_prerequisite_record
from research_provenance import file_record, load_dataset_with_provenance
from run_multiseed_dev_v1 import trainer_arguments


BINDING_FIELDS = ("training_seed", "fold_seed", "fold_manifest", "execution_binding")
OUTPUT_FILES = frozenset({"run_provenance.json", "ensemble_summary.json", "oof_predictions.npz"}
                         | {f"fold{i}_best.pt" for i in range(1, 6)})


def _read_record(record):
    value, actual = read_json_with_record(record["path"])
    require_equal(actual, record, "recorded JSON bytes")
    return value


def accepted_binding(manifest, protocol, v2):
    """Reconstruct the immutable trainer binding without fresh-output selection."""
    require_equal(v2["protocol"], manifest["protocol"], "v2/v3 protocol binding")
    return {"version": "multiseed_dev_v1", "protocol": manifest["protocol"],
            "execution_manifest": manifest["accepted_v2"], "source_files": v2["source_files"],
            "reference_source_files": v2["reference_source_files"],
            "dataset_selection": protocol["dataset_selection"],
            "runtime": protocol["required_common_runtime"]}


def expected_configuration(manifest, protocol, v2, job):
    acceptance = _read_record(manifest["accepted_v2_acceptance"])
    require_equal(acceptance.get("execution_manifest"), manifest["accepted_v2"], "accepted v2 identity")
    if (acceptance.get("status") != "validated_before_training"
            or acceptance.get("jobs_validated") != 6 or acceptance.get("training_started") is not False):
        raise ValueError("The six accepted-v2 dry runs are required")
    records = acceptance.get("dry_run_records")
    if not isinstance(records, list) or len(records) != 6:
        raise ValueError("The accepted-v2 dry-run inventory is incomplete")
    index = manifest["jobs"].index(job)
    dry_record = records[index]
    dry = _read_record(dry_record)
    for key in ("training_seed", "arm", "output_dir", "partner_transport"):
        require_equal(dry.get(key), job[key], "accepted dry-run job " + key)
    for key, expected in (("fold_seed", 2101), ("dry_run", True),
                          ("training_started", False), ("skip_test_eval", True)):
        require_equal(dry.get(key), expected, "accepted dry-run " + key)
    require_equal(dry["cv_splits"], manifest["cv_splits"], "accepted fixed folds")
    binding = accepted_binding(manifest, protocol, v2)
    attached = {"training_seed": job["training_seed"], "fold_seed": 2101,
                "fold_manifest": protocol["dataset_selection"]["grouped_fold_manifest"],
                "execution_binding": binding}
    expected_run = dry["run_provenance"]
    for key, value in attached.items():
        require_equal(expected_run.get(key), value, "accepted trainer binding " + key)
    require_equal(expected_run["code_files"], v2["source_files"] + v2["reference_source_files"],
                  "accepted trainer source inventory")
    recipe = dict(protocol["reference_model_recipe"], seed=job["training_seed"])
    require_equal(dry["model_recipe"], recipe, "accepted model recipe")
    return {"run_configuration": run_configuration(expected_run),
            "summary_configuration": dict(recipe, **attached)}, dry_record


def validate_launch(manifest_path, manifest, protocol, v2, job):
    process_dir = Path(manifest["process_dir"]).resolve(strict=True)
    stem = f"{job['arm']}_{job['training_seed']}"
    launch_path = process_dir / f"{stem}_launch.json"
    gate_path = process_dir / f"{stem}_prerequisite_provenance.json"
    launch, launch_record = read_json_with_record(launch_path)
    expected = {
        "schema_version": 1, "component": "multiseed_dev_v3_launcher",
        "execution_manifest_v3": file_record(manifest_path),
        "accepted_v2": manifest["accepted_v2"], "protocol": manifest["protocol"],
        "job": job, "environment": execution_environment(protocol, job),
        "trainer_command": trainer_arguments(manifest["accepted_v2"]["path"], protocol, job)[:-1],
        "prerequisite_provenance": file_record(gate_path),
        "training_started": True, "skip_test_eval": True,
    }
    require_equal({key: value for key, value in launch.items() if key != "created_utc"},
                  expected, "v3 launch evidence")
    gate = _read_record(launch["prerequisite_provenance"])
    validate_prerequisite_record(gate, protocol, v2)
    return {"launch": launch_record, "prerequisite_provenance": expected["prerequisite_provenance"]}


def validate_output_inventory(directory):
    entries = list(Path(directory).iterdir())
    if {entry.name for entry in entries} != OUTPUT_FILES or any(not entry.is_file() for entry in entries):
        raise ValueError("Completed development output inventory differs; partial or external artifacts are forbidden")


def _finite_tree(value):
    import torch

    if isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_tree(item)
    elif torch.is_tensor(value):
        if (torch.is_floating_point(value) or torch.is_complex(value)) and not torch.isfinite(value).all():
            raise ValueError("Nonfinite tensor in checkpoint")
    elif isinstance(value, np.ndarray) and value.dtype.kind in "fc":
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite array in checkpoint")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(value):
        raise ValueError("Nonfinite numeric checkpoint field")
    elif isinstance(value, (complex, np.complexfloating)) and not np.isfinite(value):
        raise ValueError("Nonfinite complex checkpoint field")


def _state_schema(state):
    import torch

    if not isinstance(state, dict) or not state or any(not torch.is_tensor(value) for value in state.values()):
        raise ValueError("Checkpoint model_state must contain a nonempty tensor dictionary")
    return {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in state.items()}


def reference_state_schema(protocol, arm):
    import torch

    reference = _read_record(protocol["reference_comparison"])
    role = "ot" if arm == "ot" else "control"
    record = reference["artifacts"][role]["checkpoints"][0]
    require_equal(file_record(record["path"]), record, "frozen reference checkpoint")
    checkpoint = torch.load(record["path"], map_location="cpu", weights_only=False)
    schema = _state_schema(checkpoint.get("model_state"))
    require_equal(file_record(record["path"]), record, "reference checkpoint changed during reading")
    return schema, record


def validate_arm(manifest_path, job):
    """Return only a complete, bound and numerically valid five-fold OOF arm."""
    import torch
    from multiseed_infrastructure_v3 import verify_manifest

    manifest_record = file_record(manifest_path)
    manifest, protocol, v2 = verify_manifest(manifest_path, require_acceptance=True)
    matches = [declared for declared in manifest["jobs"] if declared == job]
    if len(matches) != 1:
        raise ValueError("Arm is not a unique declared v3 job")
    require_equal(job, matches[0], "complete declared job identity")
    directory = Path(job["output_dir"]).resolve(strict=True)
    validate_output_inventory(directory)
    expected, dry_record = expected_configuration(manifest, protocol, v2, job)
    run, run_record = read_json_with_record(directory / "run_provenance.json")
    summary, summary_record = read_json_with_record(directory / "ensemble_summary.json")
    validate_run_metadata(run, summary, expected, job["partner_transport"])
    artifacts = validate_launch(manifest_path, manifest, protocol, v2, job)
    artifacts.update(run_provenance=run_record, ensemble_summary=summary_record,
                     accepted_dry_run=dry_record, execution_manifest_v3=manifest_record)
    training = run["datasets"]["Train335"]
    proteins, actual = load_dataset_with_provenance(training["path"])
    require_equal(actual, training, "loaded training dataset identity")
    require_equal(len(proteins), protocol["sample_count"], "full training sample coverage")
    fold_json = _read_record(run["fold_manifest"])
    folds = validate_fixed_folds(proteins, fold_json, 2101, training["sha256"], "complex_code")
    require_equal(folds, run["cv_splits"], "actual saved fold identities")
    expected_labels, identity = expected_oof(proteins, folds)
    del proteins
    require_equal(len(expected_labels), protocol["residue_count"], "full residue coverage")
    require_equal(len(np.unique(identity["sample_index"])), protocol["sample_count"], "full sample identities")
    require_equal(len(np.unique(np.char.upper(np.char.strip(identity["complex_id"])))),
                  protocol["complex_count"], "full complex identities")
    labels, probs, threshold, oof_record = load_oof(directory / "oof_predictions.npz", run,
                                                   expected_labels, identity)
    if not np.isclose(threshold, summary["ensemble_threshold"], rtol=0, atol=1e-7):
        raise ValueError("OOF threshold differs from saved summary")
    schema, schema_record = reference_state_schema(protocol, job["arm"])
    artifacts.update(oof_predictions=oof_record, reference_state_schema=schema_record, checkpoints=[])
    for number in range(1, 6):
        path = directory / f"fold{number}_best.pt"
        before = file_record(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        _finite_tree(checkpoint)
        require_equal(_state_schema(checkpoint.get("model_state")), schema, "approved arm model-state schema")
        for key in BINDING_FIELDS:
            require_equal(checkpoint.get(key), run[key], "checkpoint explicit binding " + key)
        mask = identity["fold_index"] == number - 1
        validate_checkpoint(checkpoint, run, summary, number, labels[mask], probs[mask])
        require_equal(file_record(path), before, "checkpoint changed during reading")
        artifacts["checkpoints"].append(before)
        del checkpoint
    for record in (run_record, summary_record, manifest_record, artifacts["launch"], artifacts["prerequisite_provenance"]):
        require_equal(file_record(record["path"]), record, "evidence changed during validation")
    return {"labels": labels, "probs": probs, "threshold": threshold, "identity": identity,
            "run": run, "artifacts": artifacts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validate-only", action="store_true", required=True)
    args = parser.parse_args()
    from multiseed_infrastructure_v3 import verify_manifest
    manifest, _, _ = verify_manifest(args.manifest, require_acceptance=True)
    validated = [{"job": job, "artifacts": validate_arm(args.manifest, job)["artifacts"]}
                 for job in manifest["jobs"]]
    print(json.dumps({"status": "validated_complete_six_runs", "outputs_written": False,
        "bootstrap_run": False, "artifacts": validated}, indent=2, allow_nan=False))
