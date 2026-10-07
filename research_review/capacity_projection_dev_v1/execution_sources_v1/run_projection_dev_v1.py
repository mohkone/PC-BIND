"""Locked serial worker and non-training acceptance simulations for projection v1."""
import argparse
import copy
import json
import os
import sys
import traceback
from pathlib import Path

from filtered_cohort_protocol import read_json_with_record
from frozen_development_protocol import require_equal
from projection_protocol_v1 import (
    DEFAULT_MANIFEST, ROOT, execution_environment, prerequisite_arguments,
    require_fresh_destinations, require_launch_authorization, select_job,
    trainer_arguments, validate_dry_pair, validate_prerequisite_record, verify_manifest,
)
from research_provenance import file_record
from run_multiseed_dev_v1 import write_new
from run_multiseed_dev_v3 import execute_serial, now, run_logged, write_state


def validate_dry_record(value, manifest_path, manifest, protocol, job):
    from projection_initialization_v1 import validate_initialization
    from projection_protocol_v1 import execution_binding

    for key in ("training_seed", "fold_seed", "arm", "output_dir", "partner_transport", "skip_test_eval"):
        require_equal(value[key], job[key], "dry-run scheduled " + key)
    if value.get("dry_run") is not True or value.get("training_started") is not False:
        raise ValueError("Acceptance dry run must not fit a model")
    require_equal(value["cv_splits"], manifest["cv_splits"], "dry-run fixed folds")
    require_equal(value["model_recipe"]["seed"], job["training_seed"], "training RNG seed")
    require_equal(value["model_recipe"]["transport_sinkhorn_iters"], job["transport_sinkhorn_iters"], "projection routing")
    run = value["run_provenance"]
    require_equal(run["execution_binding"], execution_binding(manifest_path, manifest, protocol), "dry-run execution binding")
    require_equal(run["runtime"], manifest["runtime"], "dry-run CPU runtime")
    require_equal(run["code_files"], manifest["source_files"] + manifest["reference_source_files"], "dry-run source freeze")
    if len(value.get("initialization_records", [])) != 5:
        raise ValueError("All five initialization observations required")
    for index, initialization in enumerate(value["initialization_records"]):
        validate_initialization(initialization)
        for key, expected in (("fold_index", index), ("training_seed", job["training_seed"]),
                ("fold_seed", 2101), ("arm", job["arm"]), ("projection_iterations", job["transport_sinkhorn_iters"])):
            require_equal(initialization[key], expected, "dry-run initialization " + key)


def normalize_across_seeds(record):
    value = copy.deepcopy(record)
    for key in ("training_seed", "output_dir", "initialization_records"):
        value.pop(key, None)
    value["model_recipe"].pop("seed", None)
    for key in ("seed", "training_seed", "provenance_id", "created_utc"):
        value["run_provenance"].pop(key, None)
    return value


def dry_run(manifest_path, evidence_dir):
    manifest, protocol = verify_manifest(manifest_path)
    initial_manifest = file_record(manifest_path)
    require_fresh_destinations(manifest)
    evidence = Path(evidence_dir).resolve()
    evidence.mkdir(parents=True, exist_ok=False)
    records, paths = [], []
    for job in manifest["jobs"]:
        require_equal(file_record(manifest_path), initial_manifest, "dry-run freeze bytes")
        verify_manifest(manifest_path)
        select_job(protocol, job["training_seed"], job["fold_seed"], job["arm"], job["output_dir"])
        directory = evidence / f"{job['arm']}_{job['training_seed']}"
        directory.mkdir(exist_ok=False)
        environment = execution_environment(protocol, job)
        gate_path = directory / "prerequisite_provenance.json"
        if run_logged(prerequisite_arguments(protocol, gate_path), environment, directory, "prerequisites"):
            raise RuntimeError("Strict real-data prerequisite dry-run gate failed: " + str(directory))
        gate, _ = read_json_with_record(gate_path)
        validate_prerequisite_record(gate, protocol, manifest)
        command = trainer_arguments(manifest_path, protocol, job, dry_run=True)
        if run_logged(command, environment, directory, "trainer"):
            raise RuntimeError("Real-data trainer dry run failed: " + str(directory))
        stdout = (directory / "trainer_stdout.log").read_text(encoding="utf-8-sig")
        if stdout.count("PROJECTION_DRY_RUN_JSON") != 1:
            raise ValueError("Missing unique projection dry-run evidence")
        value = json.loads(stdout.split("PROJECTION_DRY_RUN_JSON", 1)[1].strip())
        validate_dry_record(value, manifest_path, manifest, protocol, job)
        path = directory / "dry_run.json"
        write_new(path, value)
        write_new(directory / "commands.json", {"prerequisite_command": prerequisite_arguments(protocol, gate_path),
            "trainer_command": command, "environment": environment, "training_started": False})
        records.append(value)
        paths.append(file_record(path))
        require_fresh_destinations(manifest)
    matching = [validate_dry_pair(records[i], records[i + 1]) for i in (0, 2, 4)]
    for offset in (0, 1):
        for index in (offset + 2, offset + 4):
            require_equal(normalize_across_seeds(records[offset]), normalize_across_seeds(records[index]),
                "only permitted cross-seed dry-run differences")
    state = execute_serial(manifest["jobs"], lambda *_: None, lambda *_: 0,
        lambda index, job: {"status": "passed", "dry_run": paths[index]},
        lambda: {"status": "simulation_only", "bootstrap_run": False})
    if state["status"] != "complete":
        raise ValueError(state["error"])
    state.update(simulation=True, training_started=False, execution_manifest=initial_manifest)
    write_new(evidence / "queue_simulation.json", state)
    verify_manifest(manifest_path)
    require_fresh_destinations(manifest)
    result = {"status": "passed", "jobs_validated": 6, "training_started": False,
        "external_evaluation": "skipped", "execution_manifest": initial_manifest,
        "actual_gate_and_trainer_dry_runs": 6, "dry_run_records": paths,
        "initialization_matching": matching, "declared_training_outputs_absent": True,
        "cross_seed_differences": "training RNG seed, initialization outcomes and run identities/paths only",
        "bootstrap_run": False, "serial_worker_routing_simulation": "passed",
        "evidence_files": [file_record(p) for p in sorted(evidence.rglob("*")) if p.is_file()]}
    write_new(evidence / "validation.json", result)
    return result


def simulate_failure(manifest_path, evidence_dir, validation_failure=False):
    manifest, protocol = verify_manifest(manifest_path)
    require_fresh_destinations(manifest)
    evidence = Path(evidence_dir).resolve()
    evidence.mkdir(parents=True, exist_ok=False)
    def preflight(index, job):
        verify_manifest(manifest_path)
        select_job(protocol, job["training_seed"], 2101, job["arm"], job["output_dir"])
    def launch(index, job, live, state):
        command = [sys.executable, "-c", "import sys; sys.exit(" + ("0" if validation_failure else "7") + ")"]
        live["simulation_command"] = command
        return run_logged(command, execution_environment(protocol, job), evidence, "forced_failure")
    def validate(index, job):
        raise ValueError("Deliberately injected artifact-validation failure")
    def aggregate():
        raise AssertionError("No aggregate may follow a failed arm")
    state = execute_serial(manifest["jobs"], preflight, launch, validate, aggregate)
    state.update(simulation=True, training_started=False, execution_manifest=file_record(manifest_path))
    write_new(evidence / "queue_state.json", state)
    expected = "failed_validation" if validation_failure else "failed"
    if (state["status"] != "failed" or state["jobs"][0]["status"] != expected
            or any(j["status"] != "queued" for j in state["jobs"][1:])
            or state["comparison"]["status"] != "waiting_for_all_six_complete"):
        raise ValueError("Failure simulation did not stop the queue")
    require_fresh_destinations(manifest)
    return state


def claim_queue(process_dir, manifest_record, authorization):
    # Exclusive directory creation and a retained lock forbid concurrent workers
    # and automatic retries even if the first worker exits or fails.
    directory = Path(process_dir)
    directory.mkdir(parents=True, exist_ok=False)
    write_new(directory / "queue.lock", {"worker_pid": os.getpid(), "execution_manifest": manifest_record,
        "launch_authorization": authorization, "created_utc": now()})
    return directory


def run_queue(manifest_path):
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest, protocol = verify_manifest(manifest_path, require_acceptance=True)
    # Authorization is the final gate, before any production directory or child.
    authorization = require_launch_authorization(manifest_path, manifest, protocol)
    initial_manifest = file_record(manifest_path)
    require_fresh_destinations(manifest)
    process_dir = claim_queue(manifest["process_dir"], initial_manifest, authorization)
    validated = {}
    def record(state):
        state.update(worker_pid=os.getpid(), execution_manifest=initial_manifest,
            launch_authorization=authorization, process_dir=str(process_dir), external_evaluation="skipped in every arm")
        write_state(process_dir / "queue_state.json", state)
    def preflight(index, job):
        require_equal(file_record(manifest_path), initial_manifest, "launch manifest bytes")
        verify_manifest(manifest_path, require_acceptance=True)
        require_equal(require_launch_authorization(manifest_path, manifest, protocol), authorization, "launch authorization bytes")
        select_job(protocol, job["training_seed"], 2101, job["arm"], job["output_dir"])
        if Path(manifest["aggregate_output_dir"]).exists():
            raise ValueError("Aggregate destination collision")
    def launch(index, job, live, state):
        stem = f"{job['arm']}_{job['training_seed']}"
        environment = execution_environment(protocol, job)
        gate_path = process_dir / f"{stem}_prerequisite_provenance.json"
        code = run_logged(prerequisite_arguments(protocol, gate_path), environment, process_dir, stem + "_prerequisites")
        if code:
            return code
        gate, _ = read_json_with_record(gate_path)
        validate_prerequisite_record(gate, protocol, manifest)
        command = trainer_arguments(manifest_path, protocol, job, dry_run=False)
        write_new(process_dir / f"{stem}_launch.json", {"schema_version": 1,
            "component": "capacity_projection_dev_v1_launcher", "created_utc": now(),
            "execution_manifest": initial_manifest, "protocol": manifest["protocol"], "job": job,
            "environment": environment, "trainer_command": command,
            "prerequisite_provenance": file_record(gate_path), "training_started": True, "skip_test_eval": True})
        def notify(pid, stdout, stderr):
            live.update(training_pid=pid, stdout_log=stdout, stderr_log=stderr, command=command)
            record(state)
        return run_logged(command, dict(environment, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8"), process_dir, stem, notify)
    def validate(index, job):
        from validate_projection_dev_v1 import validate_arm, validate_pair_records
        result = validate_arm(manifest_path, job)
        validated[index] = result
        pairing = None
        if index % 2:
            # Re-read the first arm rather than relying on stale cached evidence.
            first = validate_arm(manifest_path, manifest["jobs"][index - 1])
            pairing = validate_pair_records(first, result)
            validated.pop(index - 1, None)
            validated.pop(index, None)
        path = process_dir / f"{job['arm']}_{job['training_seed']}_validated.json"
        write_new(path, {"status": "complete", "job": job, "external_evaluation": "skipped",
            "execution_manifest": initial_manifest, "artifacts": result["artifacts"],
            "initializations": result["initializations"], "matched_pair_validation": pairing})
        return file_record(path)
    def aggregate():
        verify_manifest(manifest_path, require_acceptance=True)
        command = [sys.executable, str(ROOT / "compare_projection_dev_v1.py"), "--manifest", str(manifest_path)]
        code = run_logged(command, {}, process_dir, "comparison")
        if code:
            raise RuntimeError(f"Registered projection aggregate failed with exit {code}; preserved logs")
        directory = Path(manifest["aggregate_output_dir"])
        value, value_record = read_json_with_record(directory / "comparison.json")
        require_equal(value["execution_manifest"], initial_manifest, "final comparison source freeze")
        if value.get("status") != "complete" or value["analysis"]["primary_training_seeds"] != [2102, 2103, 2104]:
            raise ValueError("Incomplete registered projection comparison")
        return {"comparison": value_record, "result_markdown": file_record(directory / "RESULT.md")}
    state = execute_serial(manifest["jobs"], preflight, launch, validate, aggregate, record)
    if state["status"] != "complete":
        print(state["error"], file=sys.stderr, flush=True)
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--simulate-process-failure", action="store_true")
    modes.add_argument("--simulate-validation-failure", action="store_true")
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args()
    if args.run:
        if args.evidence_dir is not None:
            parser.error("Production destinations are frozen; evidence paths are simulation-only")
        return run_queue(args.manifest)
    if args.evidence_dir is None:
        parser.error("Simulation requires a fresh --evidence-dir")
    result = dry_run(args.manifest, args.evidence_dir) if args.dry_run else simulate_failure(
        args.manifest, args.evidence_dir, args.simulate_validation_failure)
    print(json.dumps({key: result[key] for key in ("status", "training_started")}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
