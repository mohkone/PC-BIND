"""Freeze and dry-run the six preregistered jobs. This CLI never starts training."""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from filtered_cohort_protocol import read_json_with_record
from multiseed_protocol_v1 import (
    NEW_SOURCES, FOLD_SEED, assert_preserved_recipe, clean_environment,
    execution_environment, read_protocol, require_equal, select_job, validate_pair_records,
    validate_prerequisite_record, verify_record,
)
from research_provenance import file_record


ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = ROOT / "research_review/multiseed_dev_v1/protocol.json"
DEFAULT_EXECUTION = ROOT / "research_review/multiseed_dev_v1/execution_manifest_v2.json"


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def freeze_execution(protocol_path, output_path):
    protocol, record = read_protocol(protocol_path)
    reference, _ = read_json_with_record(protocol["reference_freeze_manifest"]["path"])
    for source in reference["source_files"]:
        verify_record(source)
    protected = assert_preserved_recipe(ROOT / "CROSS5FOLD_multi_test.py", ROOT / "CROSS5FOLD_multiseed_dev_v1.py")
    manifest = {"schema_version": 1, "execution_version": "multiseed_dev_v1",
        "status": "frozen_for_validation", "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": record, "fold_seed": FOLD_SEED, "jobs": protocol["seed_and_arm_order"],
        "source_files": [file_record(ROOT / name) for name in NEW_SOURCES],
        "reference_source_files": reference["source_files"],
        "prerequisite_code_files": reference["prerequisite_code_files"],
        "reference_environment": file_record(ROOT / "research_review/frozen_5fold_dev_v1/ot_dry_run.txt"),
        "runtime": protocol["required_common_runtime"], "dataset_selection": protocol["dataset_selection"],
        "model_recipe": protocol["reference_model_recipe"], "training_policy": protocol["training_policy"],
        "cv_splits": reference["run_configuration"]["cv_splits"],
        "protected_recipe_definitions_verified": protected,
        "external_evaluation": "skipped", "training_started": False}
    for job in manifest["jobs"]:
        select_job(protocol, job["training_seed"], FOLD_SEED, job["arm"], job["output_dir"])
    write_new(output_path, manifest)
    return file_record(output_path)


def trainer_arguments(manifest_path, protocol, job):
    selection = protocol["dataset_selection"]
    return [sys.executable, str(ROOT / "CROSS5FOLD_multiseed_dev_v1.py"),
        "--training-seed", str(job["training_seed"]), "--fold-seed", str(FOLD_SEED),
        "--arm", job["arm"], "--execution-manifest", str(Path(manifest_path).resolve()),
        "--data-dir", selection["resolved_data_dir"],
        "--cohort-manifest", selection["cohort_manifest"]["path"],
        "--fold-manifest", selection["grouped_fold_manifest"]["path"], "--dry-run"]


def prerequisite_arguments(protocol, output):
    selection = protocol["dataset_selection"]
    return [sys.executable, str(ROOT / "check_pcbind_prereqs.py"),
        "--data-dir", selection["resolved_data_dir"],
        "--cohort-manifest", selection["cohort_manifest"]["path"],
        "--fold-manifest", selection["grouped_fold_manifest"]["path"],
        "--seed", str(FOLD_SEED), "--require-plm", "--require-partner", "--require-pair",
        "--require-partner-encoder", "--min-partner-encoder-coverage", "1.0",
        "--provenance-output", str(output), "Train335.pkl", "Test287.pkl"]


def run_logged(command, environment, prefix):
    prefix = Path(prefix)
    with Path(str(prefix) + "_stdout.txt").open("x", encoding="utf-8", newline="\n") as stdout:
        with Path(str(prefix) + "_stderr.txt").open("x", encoding="utf-8", newline="\n") as stderr:
            process = subprocess.run(command, cwd=ROOT, env=clean_environment(environment),
                                     stdout=stdout, stderr=stderr, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"Read-only validation failed ({process.returncode}): {prefix}")


def dry_run_job(manifest_path, job, evidence_dir):
    manifest, manifest_record = read_json_with_record(manifest_path)
    protocol, protocol_record = read_protocol(manifest["protocol"]["path"])
    require_equal(protocol_record, manifest["protocol"], "dry-run protocol bytes")
    select_job(protocol, job["training_seed"], FOLD_SEED, job["arm"], job["output_dir"])
    evidence = Path(evidence_dir)
    evidence.mkdir(parents=True, exist_ok=False)
    environment = execution_environment(protocol, job)
    gate_path = evidence / "prerequisite_provenance.json"
    gate_command = prerequisite_arguments(protocol, gate_path)
    run_logged(gate_command, environment, evidence / "prerequisites")
    gate, _ = read_json_with_record(gate_path)
    validate_prerequisite_record(gate, protocol, manifest)
    command = trainer_arguments(manifest_path, protocol, job)
    run_logged(command, environment, evidence / "trainer")
    text = (evidence / "trainer_stdout.txt").read_text(encoding="utf-8-sig")
    offset = text.index('MULTISEED_DRY_RUN_JSON\n') + len('MULTISEED_DRY_RUN_JSON\n')
    result = json.loads(text[offset:])
    if (result.get("dry_run") is not True or result.get("training_started") is not False
            or result.get("training_seed") != job["training_seed"]
            or result.get("fold_seed") != FOLD_SEED or result.get("output_dir") != job["output_dir"]
            or result.get("partner_transport") is not job["partner_transport"]):
        raise ValueError("Dry-run did not validate the declared job")
    require_equal(result["cv_splits"], manifest["cv_splits"], "all five dry-run fold identities")
    require_equal(file_record(manifest_path), manifest_record, "manifest changed during validation")
    if Path(job["output_dir"]).exists():
        raise ValueError("Dry-run created a training output directory")
    write_new(evidence / "dry_run.json", result)
    write_new(evidence / "launch_validation.json", {"status": "passed", "job": job,
        "fold_seed": FOLD_SEED, "protocol": protocol_record, "execution_manifest": manifest_record,
        "environment": environment, "prerequisite_command": gate_command, "trainer_command": command,
        "prerequisite_provenance": file_record(gate_path), "dry_run": file_record(evidence / "dry_run.json"),
        "training_started": False, "training_output_created": False})
    return result


def dry_run_jobs(manifest_path, evidence_dir, selected_seeds=None):
    manifest, record = read_json_with_record(manifest_path)
    evidence = Path(evidence_dir)
    evidence.mkdir(parents=True, exist_ok=False)
    results = []
    for job in manifest["jobs"]:
        if selected_seeds is not None and job["training_seed"] not in selected_seeds:
            continue
        print(f"Validating {job['arm']} training seed {job['training_seed']}; fixed fold seed 2101", flush=True)
        results.append(dry_run_job(manifest_path, job, evidence / f"{job['arm']}_{job['training_seed']}"))
        if len(results) % 2 == 0:
            validate_pair_records(results[-2], results[-1])
    if not results:
        raise ValueError("No declared seed pair selected")
    # Across training seeds only seed and run identity fields may differ.
    def across_seed(record):
        value = json.loads(json.dumps(record))
        for key in ("training_seed", "output_dir"):
            value.pop(key)
        value["model_recipe"].pop("seed")
        for key in ("seed", "training_seed", "provenance_id", "created_utc"):
            value["run_provenance"].pop(key, None)
        return value
    for arm in (True, False):
        group = [r for r in results if r["partner_transport"] is arm]
        for result in group[1:]:
            require_equal(across_seed(result), across_seed(group[0]), "only allowed cross-seed differences")
    verification = {"status": "passed", "execution_manifest": record,
        "jobs_validated": len(results), "training_started": False, "external_evaluation": "skipped",
        "paired_recipe_and_fold_identity": "passed", "cross_seed_differences": "training seed and run identities/paths only",
        "dry_run_records": [file_record(evidence / f"{r['arm']}_{r['training_seed']}" / "dry_run.json") for r in results]}
    write_new(evidence / "validation.json", verification)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execution-manifest", type=Path, default=DEFAULT_EXECUTION)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--training-seed", type=int, choices=(2102, 2103, 2104))
    args = parser.parse_args()
    if args.freeze:
        result = freeze_execution(args.protocol, args.execution_manifest)
    else:
        if args.evidence_dir is None:
            parser.error("--dry-run requires a fresh --evidence-dir")
        result = dry_run_jobs(args.execution_manifest, args.evidence_dir,
                              None if args.training_seed is None else [args.training_seed])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
