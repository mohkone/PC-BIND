"""One locked, fail-stop serial queue for the immutable six-run amendment."""
import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from filtered_cohort_protocol import read_json_with_record
from multiseed_infrastructure_v3 import (DEFAULT_MANIFEST, ROOT, require_fresh_destinations, verify_manifest)
from multiseed_protocol_v1 import (clean_environment, execution_environment, require_equal,
    select_job, validate_prerequisite_record)
from research_provenance import file_record
from run_multiseed_dev_v1 import dry_run_jobs, prerequisite_arguments, trainer_arguments, write_new


def now():
    return datetime.now(timezone.utc).isoformat()


def write_state(path, state):
    state["updated_utc"] = now()
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def execute_serial(jobs, preflight, launch, validate, aggregate, record=lambda state: None):
    """The tested state machine; zero child exit still requires artifact validation."""
    state = {"status": "preflight", "jobs": [{**job, "status": "queued"} for job in jobs],
        "comparison": {"status": "waiting_for_all_six_complete"}, "created_utc": now()}
    record(state)
    try:
        for index, job in enumerate(state["jobs"]):
            preflight(index, jobs[index])
            job.update(status="running", started_utc=now())
            state.update(status="training", current_job_index=index)
            record(state)
            code = launch(index, jobs[index], job, state)
            job.update(exit_code=code, ended_utc=now())
            if code != 0:
                job["status"] = "failed"
                raise RuntimeError(f"{job['arm']} training seed {job['training_seed']} exited {code}; no later arm launched")
            job["status"] = "validating"
            record(state)
            job["validation"] = validate(index, jobs[index])
            job["status"] = "complete"
            record(state)
        state.update(status="comparing", current_job_index=None)
        state["comparison"]["status"] = "running"
        record(state)
        state["comparison"]["artifacts"] = aggregate()
        state["comparison"]["status"] = "complete"
        state.update(status="complete", ended_utc=now())
        record(state)
    except Exception as error:
        for job in state["jobs"]:
            if job["status"] in ("running", "validating"):
                job["status"] = "failed_validation" if job["status"] == "validating" else "failed"
        if state["comparison"]["status"] == "running":
            state["comparison"]["status"] = "failed"
        state.update(status="failed", error=str(error), ended_utc=now())
        record(state)
    return state


def run_logged(command, environment, directory, stem, notify=None):
    stdout_path = Path(directory) / (stem + "_stdout.log")
    stderr_path = Path(directory) / (stem + "_stderr.log")
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        child = subprocess.Popen(command, cwd=ROOT, env=clean_environment(environment), stdout=stdout, stderr=stderr)
        if notify is not None:
            notify(child.pid, str(stdout_path), str(stderr_path))
        return child.wait()


def dry_run(manifest_path, evidence_dir):
    manifest, protocol, _ = verify_manifest(manifest_path)
    require_fresh_destinations(manifest)
    evidence_dir = Path(evidence_dir).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=False)
    dry_run_jobs(manifest["accepted_v2"]["path"], evidence_dir / "all_six")
    dry_records = []
    for job in manifest["jobs"]:
        path = evidence_dir / "all_six" / f"{job['arm']}_{job['training_seed']}" / "dry_run.json"
        value, _ = read_json_with_record(path)
        dry_records.append(value)
    def preflight(index, job):
        verify_manifest(manifest_path)
        select_job(protocol, job["training_seed"], 2101, job["arm"], job["output_dir"])
    def validate(index, job):
        dry = dry_records[index]
        require_equal(dry["cv_splits"], manifest["cv_splits"], "worker preserved folds")
        require_equal(dry["training_seed"], job["training_seed"], "worker seed routing")
        require_equal(dry["fold_seed"], 2101, "worker fixed fold seed")
        if dry["skip_test_eval"] is not True or dry["training_started"] is not False:
            raise ValueError("Dry-run training/external exclusion failed")
        return {"status": "passed", "training_started": False}
    state = execute_serial(manifest["jobs"], preflight, lambda *args: 0, validate,
        lambda: {"status": "simulation_only", "bootstrap_run": False})
    if state["status"] != "complete":
        raise ValueError(state["error"])
    state.update(simulation=True, training_started=False, execution_manifest_v3=file_record(manifest_path))
    write_new(evidence_dir / "queue_simulation.json", state)
    require_fresh_destinations(manifest)
    files = [file_record(path) for path in sorted(evidence_dir.rglob("*")) if path.is_file()]
    result = {"status": "passed", "jobs_validated": 6, "training_started": False,
        "external_evaluation": "skipped", "execution_manifest_v3": file_record(manifest_path),
        "evidence_files": files, "actual_gate_and_trainer_dry_runs": 6,
        "serial_worker_routing_simulation": "passed", "bootstrap_run": False}
    write_new(evidence_dir / "validation.json", result)
    return result


def simulate_failure(manifest_path, evidence_dir, validation_failure=False):
    manifest, protocol, _ = verify_manifest(manifest_path)
    require_fresh_destinations(manifest)
    evidence_dir = Path(evidence_dir).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=False)
    def preflight(index, job):
        verify_manifest(manifest_path)
        select_job(protocol, job["training_seed"], 2101, job["arm"], job["output_dir"])
    def launch(index, job, live_job, state):
        # Harmless clean-process fault injection, never a training command.
        command = [sys.executable, "-c", "import sys; sys.exit(" + ("0" if validation_failure else "7") + ")"]
        live_job["simulation_command"] = command
        return run_logged(command, execution_environment(protocol, job), evidence_dir, "forced_failure")
    def validate(index, job):
        raise ValueError("Deliberately injected post-run artifact-validation failure")
    def aggregate():
        raise AssertionError("Aggregate must never be reached after a forced failure")
    state = execute_serial(manifest["jobs"], preflight, launch, validate, aggregate)
    state.update(simulation=True, training_started=False, execution_manifest_v3=file_record(manifest_path))
    write_new(evidence_dir / "queue_state.json", state)
    expected = "failed_validation" if validation_failure else "failed"
    if (state["status"] != "failed" or state["jobs"][0]["status"] != expected
            or any(job["status"] != "queued" for job in state["jobs"][1:])
            or state["comparison"]["status"] != "waiting_for_all_six_complete"):
        raise ValueError("Forced failure failed to demonstrate queue stop")
    require_fresh_destinations(manifest)
    return state


def run_queue(manifest_path):
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest, protocol, v2 = verify_manifest(manifest_path, require_acceptance=True)
    initial_manifest = file_record(manifest_path)
    require_fresh_destinations(manifest)
    process_dir = Path(manifest["process_dir"])
    process_dir.mkdir(parents=True, exist_ok=True)
    # Retain the exclusive claim on failure/completion: never automatically retry.
    with (process_dir / "queue.lock").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"worker_pid": os.getpid(), "manifest": initial_manifest, "created_utc": now()}))
    def record(state):
        state.update(worker_pid=os.getpid(), execution_manifest_v3=initial_manifest,
            accepted_v2=manifest["accepted_v2"], external_evaluation="skipped in every arm",
            process_dir=str(process_dir))
        write_state(process_dir / "queue_state.json", state)
    def preflight(index, job):
        require_equal(file_record(manifest_path), initial_manifest, "launch manifest bytes")
        verify_manifest(manifest_path, require_acceptance=True)
        select_job(protocol, job["training_seed"], 2101, job["arm"], job["output_dir"])
        if Path(manifest["aggregate_output_dir"]).exists():
            raise ValueError("Aggregate destination collision")
    def launch(index, job, live_job, state):
        stem = f"{job['arm']}_{job['training_seed']}"
        environment = execution_environment(protocol, job)
        gate_path = process_dir / f"{stem}_prerequisite_provenance.json"
        code = run_logged(prerequisite_arguments(protocol, gate_path), environment, process_dir, stem + "_prerequisites")
        if code != 0:
            return code
        gate, _ = read_json_with_record(gate_path)
        validate_prerequisite_record(gate, protocol, v2)
        # The frozen trainer itself creates its directory after its v2 gate.
        command = trainer_arguments(manifest["accepted_v2"]["path"], protocol, job)[:-1]
        write_new(process_dir / f"{stem}_launch.json", {
            "schema_version": 1, "component": "multiseed_dev_v3_launcher", "created_utc": now(),
            "execution_manifest_v3": initial_manifest, "accepted_v2": manifest["accepted_v2"],
            "protocol": manifest["protocol"], "job": job, "environment": environment,
            "trainer_command": command, "prerequisite_provenance": file_record(gate_path),
            "training_started": True, "skip_test_eval": True})
        def notify(pid, stdout, stderr):
            live_job.update(training_pid=pid, stdout_log=stdout, stderr_log=stderr, command=command)
            record(state)
        return run_logged(command, dict(environment, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8"), process_dir, stem, notify)
    def validate(index, job):
        from validate_multiseed_dev_v3 import validate_arm
        result = validate_arm(manifest_path, job)
        path = process_dir / f"{job['arm']}_{job['training_seed']}_validated.json"
        write_new(path, {"status": "complete", "job": job, "external_evaluation": "skipped",
            "execution_manifest_v3": initial_manifest, "artifacts": result["artifacts"]})
        return file_record(path)
    def aggregate():
        verify_manifest(manifest_path, require_acceptance=True)
        command = [sys.executable, str(ROOT / "compare_multiseed_dev_v3.py"), "--manifest", str(manifest_path)]
        code = run_logged(command, {}, process_dir, "comparison")
        if code != 0:
            raise RuntimeError(f"Three-seed aggregate failed with exit {code}; preserved logs")
        directory = Path(manifest["aggregate_output_dir"])
        result, result_record = read_json_with_record(directory / "comparison.json")
        require_equal(result["execution_manifest"], initial_manifest, "final comparison source freeze")
        if result.get("status") != "complete" or result["analysis"]["primary_training_seeds"] != [2102, 2103, 2104]:
            raise ValueError("Aggregate returned without a complete registered result")
        return {"comparison": result_record, "result_markdown": file_record(directory / "RESULT.md")}
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
            parser.error("Production destinations are frozen; --evidence-dir is only for simulations")
        return run_queue(args.manifest)
    if args.evidence_dir is None:
        parser.error("Simulation requires a fresh --evidence-dir")
    if args.dry_run:
        result = dry_run(args.manifest, args.evidence_dir)
    else:
        result = simulate_failure(args.manifest, args.evidence_dir, args.simulate_validation_failure)
    print(json.dumps({k: result[k] for k in ("status", "training_started")}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
