"""Immutable outer infrastructure freeze; the accepted v2 trainer stays unchanged."""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from filtered_cohort_protocol import read_json_with_record
from multiseed_protocol_v1 import (read_protocol, require_equal, require_launch_validation,
    validate_jobs, verify_execution_context, verify_record)
from research_provenance import file_record
from run_multiseed_dev_v1 import write_new

ROOT = Path(__file__).resolve().parent
REVIEW = ROOT / "research_review/multiseed_dev_v1"
V2 = REVIEW / "execution_manifest_v2.json"
V2_SHA = "ca69112d633de00ee80d23a7c7913a939d780cf2561f403393588f95a5eb0519"
DEFAULT_MANIFEST = REVIEW / "execution_manifest_v3.json"
SOURCES = ("multiseed_infrastructure_v3.py", "run_multiseed_dev_v3.py",
    "validate_multiseed_dev_v3.py", "compare_multiseed_dev_v3.py",
    "tests/test_multiseed_worker_v3.py", "tests/test_multiseed_validation_v3.py",
    "tests/test_multiseed_analysis_v3.py")
ANALYSIS = {"attempted_draws": 5000, "rng": "PCG64", "bootstrap_seed": 2101,
    "historical_seed_2101_in_primary": False, "thresholds": "all six saved thresholds fixed",
    "complex_order": "sorted uppercase stripped identifiers", "quantile_method": "linear"}


def v2_context(protocol, v2, record, protocol_record):
    binding = {"version": "multiseed_dev_v1", "protocol": protocol_record,
        "execution_manifest": record, "source_files": v2["source_files"],
        "reference_source_files": v2["reference_source_files"],
        "dataset_selection": protocol["dataset_selection"], "runtime": protocol["required_common_runtime"]}
    return {"manifest": v2, "manifest_record": record, "protocol": protocol,
        "protocol_record": protocol_record, "execution_binding": binding}


def verify_manifest(path, require_acceptance=False):
    manifest, record = read_json_with_record(path)
    if (manifest.get("schema_version") != 1 or manifest.get("infrastructure_version") != "multiseed_dev_v3"
            or manifest.get("amendment_scope") != "queue, artifact validation and preregistered analysis only"):
        raise ValueError("Unsupported infrastructure amendment")
    v2, v2_record = read_json_with_record(V2)
    if v2_record["sha256"] != V2_SHA:
        raise ValueError("Accepted v2 execution freeze changed")
    require_equal(manifest["accepted_v2"], v2_record, "accepted v2 freeze")
    protocol, protocol_record = read_protocol(manifest["protocol"]["path"])
    require_equal(manifest["protocol"], protocol_record, "registered protocol")
    require_equal(v2["protocol"], protocol_record, "v2 protocol")
    validate_jobs(protocol, manifest["jobs"])
    require_equal(manifest["accepted_v2_source_files"], v2["source_files"], "preserved v2 sources")
    require_equal(manifest["runtime"], protocol["required_common_runtime"], "accepted runtime")
    require_equal(manifest["dataset_selection"], protocol["dataset_selection"], "fixed data registry")
    require_equal(manifest["cv_splits"], v2["cv_splits"], "five immutable fold identities")
    require_equal(manifest["fold_seed"], 2101, "fixed fold seed")
    require_equal(manifest["external_evaluation"], "skipped", "external exclusion")
    require_equal(manifest["analysis"], ANALYSIS, "registered aggregate computation")
    require_equal(manifest["aggregate_output_dir"], str(REVIEW / "complete_oof_v3"), "aggregate destination")
    require_equal(manifest["process_dir"], str(ROOT / "outputs_multiseed_dev_v3_process"), "queue destination")
    require_equal([r["path"] for r in manifest["source_files"]],
                  [str(ROOT / name) for name in SOURCES], "new infrastructure inventory")
    for source in manifest["source_files"]:
        verify_record(source)
    for preserved in manifest["preserved_records"]:
        verify_record(preserved)
    require_equal([r["path"] for r in manifest["preserved_records"]],
        [str(REVIEW / name) for name in ("PREREGISTRATION.md", "protocol.json",
         "execution_manifest_v2.acceptance.json", "execution_source_archive_v2.json")], "preserved record inventory")
    acceptance_record = file_record(REVIEW / "execution_manifest_v2.acceptance.json")
    require_equal(manifest["accepted_v2_acceptance"], acceptance_record, "accepted v2 gate bytes")
    archive, _ = read_json_with_record(REVIEW / "execution_source_archive_v2.json")
    require_equal(archive["execution_manifest"], v2_record, "v2 archive binding")
    require_equal([item["original"] for item in archive["sources"]], v2["source_files"], "v2 archive inventory")
    for item in archive["sources"]:
        verify_record(item["archive"])
        require_equal(item["archive"]["sha256"], item["original"]["sha256"], "preserved archive bytes")
    verify_record(manifest["source_archive"])
    current_archive, _ = read_json_with_record(manifest["source_archive"]["path"])
    require_equal([item["original"] for item in current_archive["sources"]], manifest["source_files"], "v3 archive inventory")
    for item, name in zip(current_archive["sources"], SOURCES):
        require_equal(item["archive"]["path"], str(REVIEW / "execution_sources_v3" / name), "v3 archive path")
        verify_record(item["archive"])
        require_equal(item["archive"]["sha256"], item["original"]["sha256"], "archived source bytes")
    context = v2_context(protocol, v2, v2_record, protocol_record)
    verify_execution_context(context)
    for source in v2["prerequisite_code_files"]:
        verify_record(source)
    require_launch_validation(context)
    if require_acceptance:
        verify_acceptance(path, manifest, record)
    require_equal(file_record(path), record, "unchanged infrastructure manifest")
    return manifest, protocol, v2


def verify_acceptance(path, manifest, record):
    acceptance, _ = read_json_with_record(Path(path).with_name(Path(path).stem + ".acceptance.json"))
    if (acceptance.get("status") != "accepted_before_training" or acceptance.get("training_started") is not False
            or acceptance.get("jobs_validated") != 6 or acceptance.get("external_evaluation") != "skipped"):
        raise ValueError("Missing complete v3 infrastructure acceptance")
    require_equal(acceptance["execution_manifest"], record, "accepted v3 manifest")
    require_equal(acceptance["source_files"], manifest["source_files"], "tested infrastructure source bytes")
    require_equal(acceptance["accepted_v2"], manifest["accepted_v2"], "preserved v2 acceptance binding")
    for evidence in acceptance["evidence_files"]:
        verify_record(evidence)
    for key in ("full_tests", "all_six_dry_runs", "forced_process_failure", "forced_validation_failure"):
        evidence = acceptance[key]
        verify_record(evidence)
        data, _ = read_json_with_record(evidence["path"])
        if key == "full_tests":
            if data.get("status") != "passed" or data.get("failures") != 0 or data.get("errors") != 0:
                raise ValueError("Full suite did not pass")
            require_equal(data["source_files"], manifest["source_files"], "full-suite source freeze")
            verify_record(data["log"])
        elif key == "all_six_dry_runs":
            if data.get("status") != "passed" or data.get("jobs_validated") != 6 or data.get("training_started") is not False:
                raise ValueError("All six worker dry runs are required")
            require_equal(data["execution_manifest_v3"], record, "worker dry-run source freeze")
            for item in data["evidence_files"]:
                verify_record(item)
        else:
            expected = "failed" if key == "forced_process_failure" else "failed_validation"
            if (data.get("status") != "failed" or data.get("simulation") is not True
                    or data.get("training_started") is not False
                    or data["jobs"][0]["status"] != expected
                    or any(j["status"] != "queued" for j in data["jobs"][1:])
                    or data["comparison"]["status"] != "waiting_for_all_six_complete"):
                raise ValueError("Forced failure did not halt before later arms/aggregate")
            require_equal(data["execution_manifest_v3"], record, "tested fail-stop source freeze")


def require_fresh_destinations(manifest):
    # v2 intentionally refuses even empty existing run directories. Preserve it.
    for job in manifest["jobs"]:
        if Path(job["output_dir"]).exists():
            raise ValueError("Refusing existing training destination: " + job["output_dir"])
    if Path(manifest["aggregate_output_dir"]).exists():
        raise ValueError("Refusing existing aggregate destination")


def freeze():
    if DEFAULT_MANIFEST.exists() or (REVIEW / "execution_source_archive_v3.json").exists():
        raise ValueError("Refusing replacement infrastructure freeze")
    v2, record = read_json_with_record(V2)
    if record["sha256"] != V2_SHA:
        raise ValueError("Accepted v2 freeze changed")
    protocol, protocol_record = read_protocol(REVIEW / "protocol.json")
    archive_dir = REVIEW / "execution_sources_v3"
    archive_dir.mkdir(exist_ok=False)
    sources = [file_record(ROOT / name) for name in SOURCES]
    archived = []
    for name, original in zip(SOURCES, sources):
        target = archive_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original["path"], target)
        saved = file_record(target)
        require_equal(saved["sha256"], original["sha256"], "archive copy")
        archived.append({"original": original, "archive": saved})
    archive_path = REVIEW / "execution_source_archive_v3.json"
    write_new(archive_path, {"schema_version": 1, "accepted_v2": record, "sources": archived})
    manifest = {"schema_version": 1, "infrastructure_version": "multiseed_dev_v3",
        "amendment_scope": "queue, artifact validation and preregistered analysis only",
        "amendment_reason": "Accepted v2 implements seed routing and dry runs; v3 adds the missing serial worker, per-run artifact validator and registered three-seed analyzer without changing v2 or the protocol.",
        "created_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol_record,
        "accepted_v2": record, "accepted_v2_acceptance": file_record(REVIEW / "execution_manifest_v2.acceptance.json"),
        "accepted_v2_source_files": v2["source_files"], "source_files": sources,
        "source_archive": file_record(archive_path),
        "preserved_records": [file_record(REVIEW / name) for name in ("PREREGISTRATION.md", "protocol.json",
            "execution_manifest_v2.acceptance.json", "execution_source_archive_v2.json")],
        "jobs": protocol["seed_and_arm_order"], "fold_seed": 2101, "cv_splits": v2["cv_splits"],
        "runtime": protocol["required_common_runtime"], "dataset_selection": protocol["dataset_selection"],
        "aggregate_output_dir": str(REVIEW / "complete_oof_v3"),
        "process_dir": str(ROOT / "outputs_multiseed_dev_v3_process"), "external_evaluation": "skipped",
        "analysis": ANALYSIS}
    require_fresh_destinations(manifest)
    write_new(DEFAULT_MANIFEST, manifest)
    verify_manifest(DEFAULT_MANIFEST)
    return file_record(DEFAULT_MANIFEST)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--require-acceptance", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        print(json.dumps(freeze(), indent=2))
    else:
        verify_manifest(args.manifest, args.require_acceptance)
        print("Immutable infrastructure, accepted v2, protocol, source, data, folds and runtime verified.")
