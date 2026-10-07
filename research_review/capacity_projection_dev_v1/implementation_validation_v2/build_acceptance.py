"""Independently bind completed non-training checks to infrastructure freeze 2."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from filtered_cohort_protocol import read_json_with_record
from frozen_development_protocol import require_equal, run_configuration
from projection_protocol_v1 import (DEFAULT_MANIFEST, REVIEW, assert_preserved_recipe,
    require_fresh_destinations, require_infrastructure_acceptance, require_launch_authorization,
    validate_dry_pair, validate_prerequisite_record, verify_manifest, verify_record)
from research_provenance import file_record
from run_multiseed_dev_v1 import write_new
from run_projection_dev_v1 import validate_dry_record
from validate_multiseed_dev_v3 import reference_state_schema
from validate_projection_dev_v1 import BINDING_FIELDS, expected_configuration


def main():
    directory = Path(__file__).resolve().parent
    manifest, protocol = verify_manifest(DEFAULT_MANIFEST)
    freeze = file_record(DEFAULT_MANIFEST)
    require_fresh_destinations(manifest)
    if Path(manifest["launch_authorization_path"]).exists():
        raise ValueError("Acceptance must precede explicit launch authorization")
    tests, test_record = read_json_with_record(directory / "full_tests.json")
    if tests["status"] != "passed" or tests["failures"] or tests["errors"] or tests["tests_run"] != 213:
        raise ValueError("The complete 213-test suite must pass with only the declared existing skip")
    require_equal(tests["source_files"], manifest["source_files"], "full suite source identity")
    require_equal(tests["execution_manifest"], freeze, "full suite freeze identity")
    old_tests, _ = read_json_with_record(REVIEW / "implementation_validation_v1/full_tests.json")
    require_equal(tests["skipped"], old_tests["skipped"], "only the pre-existing host symlink skip")
    for record in tests["test_source_files"] + [tests["test_log"], tests["runner"]]:
        verify_record(record)
    dry, dry_record = read_json_with_record(directory / "all_six_dry_runs/validation.json")
    require_equal(dry["execution_manifest"], freeze, "all-six dry-run freeze")
    if dry["status"] != "passed" or dry["jobs_validated"] != 6 or dry["training_started"] is not False:
        raise ValueError("All six complete non-training real-data dry runs are required")
    for record in dry["evidence_files"]:
        verify_record(record)
    schema, schema_record = reference_state_schema(
        {"reference_comparison": protocol["observed_references"]["historical_seed2101_comparison"]}, "ot")
    values, descriptions = [], []
    for job, record in zip(manifest["jobs"], dry["dry_run_records"]):
        verify_record(record)
        value, _ = read_json_with_record(record["path"])
        validate_dry_record(value, DEFAULT_MANIFEST, manifest, protocol, job)
        gate_path = Path(record["path"]).parent / "prerequisite_provenance.json"
        gate, _ = read_json_with_record(gate_path)
        validate_prerequisite_record(gate, protocol, manifest)
        expected, accepted_reference = expected_configuration(DEFAULT_MANIFEST, manifest, protocol, job)
        actual_run = {k: v for k, v in value["run_provenance"].items() if k != "initialization_records"}
        require_equal(run_configuration(actual_run), expected["run_configuration"], "actual trainer metadata matches validator")
        candidate_summary = dict(value["model_recipe"], **{key: actual_run[key] for key in BINDING_FIELDS})
        require_equal(candidate_summary, expected["summary_configuration"], "actual prospective summary matches validator")
        for initial in value["initialization_records"]:
            shape_map = {item["name"]: {"shape": item["shape"], "dtype": item["dtype"]}
                for item in initial["state_schema"]}
            require_equal(shape_map, schema, "unchanged approved transport architecture")
            for key in ("parameter_schema", "buffer_schema", "state_schema", "parameter_count",
                    "trainable_parameter_count", "optimizer"):
                require_equal(initial[key], value["initialization_records"][0][key], "all-fold structure " + key)
        first = value["initialization_records"][0]
        descriptions.append({"job": job, "dry_run": record, "accepted_recipe_reference": accepted_reference,
            "parameter_count": first["parameter_count"], "trainable_parameter_count": first["trainable_parameter_count"],
            "optimizer_groups": len(first["optimizer"]["groups"]), "first_fold_parameter_sha256": first["parameter_sha256"],
            "first_fold_state_sha256": first["state_sha256"], "first_fold_rng_before": first["rng_before"],
            "first_fold_rng_after": first["rng_after"], "schema_and_recipe_validated": True})
        values.append(value)
    if len(values) != 6:
        raise ValueError("Exact six-arm dry-run identity required")
    matching = [validate_dry_pair(values[i], values[i+1]) for i in (0, 2, 4)]
    if len({values[i]["initialization_records"][0]["parameter_sha256"] for i in (0, 2, 4)}) != 3:
        raise ValueError("The three training seeds must produce distinct model initializations")
    failure_records = {}
    for name, expected_status in (("forced_process_failure", "failed"), ("forced_validation_failure", "failed_validation")):
        state, record = read_json_with_record(directory / name / "queue_state.json")
        require_equal(state["execution_manifest"], freeze, "failure simulation source identity")
        if (state["status"] != "failed" or state["training_started"] is not False or state["simulation"] is not True
                or state["jobs"][0]["status"] != expected_status
                or any(job["status"] != "queued" for job in state["jobs"][1:])
                or state["comparison"]["status"] != "waiting_for_all_six_complete"):
            raise ValueError("The forced failure advanced later arms or analysis")
        failure_records[name] = record
    require_fresh_destinations(manifest)
    independent = {"status": "passed", "execution_manifest": freeze, "jobs_validated": 6,
        "training_started": False, "launch_authorized": False, "external_evaluation": "skipped",
        "preserved_model_training_definitions": assert_preserved_recipe(
            ROOT / "CROSS5FOLD_multiseed_dev_v1.py", ROOT / "CROSS5FOLD_projection_dev_v1.py"),
        "reference_state_schema": schema_record, "arms": descriptions, "initialization_matching": matching,
        "distinct_first_fold_parameter_fingerprints_across_seeds": 3,
        "dry_run_later_fold_fingerprints_are_initialization_only": True,
        "future_later_fold_equality_is_not_required_and_no_reseeding_is_added": True,
        "declared_destinations_absent": [job["output_dir"] for job in manifest["jobs"]]
            + [manifest["aggregate_output_dir"], manifest["process_dir"]],
        "launch_authorization_file_absent": True, "bootstrap_run": False}
    independent_path = directory / "independent_metadata_validation.json"
    write_new(independent_path, independent)
    evidence_files = [file_record(path) for path in sorted(directory.rglob("*")) if path.is_file()]
    acceptance = {"schema_version": 1, "execution_version": "capacity_projection_dev_v1", "execution_revision": 2,
        "status": "infrastructure_accepted_training_unstarted", "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_started": False, "launch_authorized": False, "jobs_validated": 6,
        "execution_manifest": freeze, "protocol": manifest["protocol"], "source_files": manifest["source_files"],
        "source_archive": manifest["source_archive"], "reference_source_files": manifest["reference_source_files"],
        "superseded_preacceptance": manifest["superseded_preacceptance"],
        "infrastructure_amendment": manifest["infrastructure_amendment"],
        "dataset_selection": manifest["dataset_selection"], "runtime": manifest["runtime"],
        "cv_splits": manifest["cv_splits"], "jobs": manifest["jobs"], "external_evaluation": "skipped",
        "full_tests": test_record, "all_six_dry_runs": dry_record, **failure_records,
        "independent_metadata_validation": file_record(independent_path), "evidence_files": evidence_files,
        "parameter_initialization_matching": descriptions,
        "no_training_outputs_or_process_directory_created": True, "bootstrap_run": False,
        "launch_requires_later_explicit_user_instruction": True}
    acceptance_path = DEFAULT_MANIFEST.with_name(DEFAULT_MANIFEST.stem + ".acceptance.json")
    write_new(acceptance_path, acceptance)
    require_infrastructure_acceptance(DEFAULT_MANIFEST, manifest)
    verify_manifest(DEFAULT_MANIFEST, require_acceptance=True)
    try:
        require_launch_authorization(DEFAULT_MANIFEST, manifest, protocol)
    except ValueError as error:
        if "explicit user launch authorization has not been recorded" not in str(error):
            raise
        refusal = str(error)
    else:
        raise ValueError("Missing explicit authorization failed to block launch")
    require_fresh_destinations(manifest)
    verification = {"status": "passed", "execution_manifest": freeze, "acceptance": file_record(acceptance_path),
        "training_started": False, "launch_authorized": False, "launch_gate_refusal": refusal,
        "worker_or_trainer_launch_attempted": False, "all_declared_destinations_absent": True,
        "source_ancestry_cohort_fold_runtime_and_evidence_hashes_verified": True,
        "bootstrap_run": False, "external_evaluation": "skipped"}
    write_new(REVIEW / "ACCEPTANCE_VERIFICATION_v2.json", verification)
    print(json.dumps({"status": acceptance["status"], "acceptance": file_record(acceptance_path),
        "tests_run": tests["tests_run"], "tests_passed": tests["tests_run"] - len(tests["skipped"]),
        "parameter_count": descriptions[0]["parameter_count"], "jobs_validated": 6}, indent=2))


if __name__ == "__main__":
    main()
