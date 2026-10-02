"""One-off serial queue for the two authorized frozen development runs."""
import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from frozen_development_protocol import (
    read_freeze, require_equal, validate_completed_run, verify_environment,
)
from research_provenance import file_record


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_state(path, state):
    state["updated_utc"] = utc_now()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def training_command(powershell, root, job):
    return [str(powershell), "-NoProfile", "-NonInteractive", "-File",
            str(root / job["wrapper"]), "-Seed", "2101", "-MaxFolds", "5",
            "-SkipTestEval", "1", "-OutputDir", job["output_dir"]]


def run_queue(freeze_path, process_dir, powershell):
    freeze_path = Path(freeze_path).resolve(strict=True)
    freeze_record = file_record(freeze_path)
    freeze = read_freeze(freeze_path)
    root = Path(freeze["project_root"]).resolve(strict=True)
    process_dir = Path(process_dir).resolve()
    process_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive claim: an interrupted/failed queue must be inspected, not retried
    # over existing outputs. The lock is retained as part of the local record.
    with (process_dir / "queue.lock").open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()) + "\n")
    state_path = process_dir / "queue_state.json"
    state = {
        "schema_version": 1, "status": "preflight", "created_utc": utc_now(),
        "worker_pid": os.getpid(), "freeze_manifest": freeze_record,
        "mode": "serial, fresh five-fold development runs; no checkpoint reuse",
        "external_evaluation": "skipped in both arms",
        "jobs": [{**job, "status": "queued"} for job in freeze["jobs"]],
        "comparison": {"status": "waiting_for_both_complete", "output_dir": freeze["comparison_output_dir"]},
    }
    write_state(state_path, state)
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("PPI_")}
    env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    try:
        if [job["role"] for job in state["jobs"]] != ["ot", "control"]:
            raise ValueError("Queue must run OT then the matched no-OT control")
        for job in state["jobs"]:
            output = Path(job["output_dir"])
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                raise ValueError(f"Refusing an existing nonempty run: {output}")
        if Path(freeze["comparison_output_dir"]).exists():
            raise ValueError("Comparison output already exists")
        for job in state["jobs"]:
            require_equal(file_record(freeze_path), freeze_record, "immutable freeze manifest")
            verify_environment(freeze)
            command = training_command(powershell, root, job)
            job.update({"status": "running", "started_utc": utc_now(), "command": command,
                        "stdout_log": str(process_dir / (job["role"] + "_stdout.log")),
                        "stderr_log": str(process_dir / (job["role"] + "_stderr.log"))})
            state["status"] = "training"
            state["current_arm"] = job["role"]
            with Path(job["stdout_log"]).open("xb") as stdout, Path(job["stderr_log"]).open("xb") as stderr:
                child = subprocess.Popen(command, cwd=root, env=env, stdout=stdout, stderr=stderr)
                job["launcher_pid"] = child.pid
                write_state(state_path, state)
                code = child.wait()
            job["exit_code"] = code
            job["ended_utc"] = utc_now()
            if code != 0:
                job["status"] = "failed"
                raise RuntimeError(f"{job['role']} training exited with code {code}; queue stopped")
            # A zero process exit alone is insufficient for a complete run.
            verify_environment(freeze)
            validate_completed_run(job["output_dir"], freeze, job["partner_transport"])
            job["status"] = "complete"
            write_state(state_path, state)
        require_equal(file_record(freeze_path), freeze_record, "immutable freeze manifest")
        state["status"] = "comparing"
        state["current_arm"] = None
        state["comparison"]["status"] = "running"
        write_state(state_path, state)
        command = [sys.executable, str(root / "compare_oof_development.py"),
                   "--model-dir", state["jobs"][0]["output_dir"],
                   "--control-dir", state["jobs"][1]["output_dir"],
                   "--freeze-manifest", str(freeze_path),
                   "--output-dir", freeze["comparison_output_dir"]]
        with (process_dir / "comparison_stdout.log").open("xb") as stdout, \
                (process_dir / "comparison_stderr.log").open("xb") as stderr:
            code = subprocess.call(command, cwd=root, env=env, stdout=stdout, stderr=stderr)
        state["comparison"]["exit_code"] = code
        if code != 0:
            state["comparison"]["status"] = "failed"
            raise RuntimeError("Complete OOF comparison failed validation or resampling; see comparison_stderr.log")
        state["comparison"]["status"] = "complete"
        state["status"] = "complete"
        state["ended_utc"] = utc_now()
        write_state(state_path, state)
        return 0
    except Exception as error:
        state.update(status="failed", error=str(error), ended_utc=utc_now())
        for job in state["jobs"]:
            if job["status"] == "running":
                job["status"] = "failed_validation"
        write_state(state_path, state)
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", required=True)
    parser.add_argument("--process-dir", required=True)
    parser.add_argument("--powershell", required=True)
    args = parser.parse_args()
    return run_queue(args.freeze_manifest, args.process_dir, args.powershell)


if __name__ == "__main__":
    raise SystemExit(main())
