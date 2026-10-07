"""Fail-closed execution binding for the registered projection-policy ablation."""
import argparse
import ast
import copy
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from filtered_cohort_protocol import read_json_with_record, verify_filtered_cohort
from frozen_development_protocol import require_equal
from multiseed_infrastructure_v3 import verify_manifest as verify_accepted_ancestry
from multiseed_protocol_v1 import (clean_environment, reference_environment, verify_record,
    validate_training_namespace as reference_validate_training_namespace,
    validate_prerequisite_record as reference_validate_prerequisite_record)
from research_provenance import build_run_provenance, file_record
from run_multiseed_dev_v1 import prerequisite_arguments as reference_prerequisite_arguments, write_new

ROOT = Path(__file__).resolve().parent
REVIEW = ROOT / "research_review/capacity_projection_dev_v1"
DEFAULT_PROTOCOL = REVIEW / "protocol.json"
DEFAULT_MANIFEST = REVIEW / "execution_manifest_v1.json"
PROTOCOL_SHA = "3cc6a2e58f331928d36849eddef80af40606505186a6e5d28eb6ada9f4079a3c"
TRAINING_SEEDS = (2102, 2103, 2104)
FOLD_SEED = 2101
SOURCES = ("projection_protocol_v1.py", "projection_initialization_v1.py",
    "CROSS5FOLD_projection_dev_v1.py", "run_projection_dev_v1.py",
    "validate_projection_dev_v1.py", "compare_projection_dev_v1.py",
    "tests/test_projection_protocol_v1.py", "tests/test_projection_training_v1.py",
    "tests/test_projection_validation_v1.py", "tests/test_projection_analysis_v1.py")


def read_protocol(path=DEFAULT_PROTOCOL):
    protocol, record = read_json_with_record(path)
    registration, _ = read_json_with_record(Path(path).parent / "registration_validation.json")
    require_equal(record, registration["protocol"], "registered projection protocol bytes")
    if record["sha256"] != PROTOCOL_SHA or protocol.get("protocol_version") != "capacity_projection_dev_v1":
        raise ValueError("Registered projection policy changed")
    require_equal(protocol["training_seeds"], list(TRAINING_SEEDS), "declared training seeds")
    require_equal(protocol["fixed_fold_seed"], FOLD_SEED, "fixed fold seed")
    require_equal(protocol["folds_per_arm"], 5, "five-fold scope")
    validate_jobs(protocol, protocol["seed_and_arm_order"])
    return protocol, record


def validate_jobs(protocol, jobs):
    if len(jobs) != 6:
        raise ValueError("Exactly six projection-ablation arms required")
    seen = set()
    for index, (seed, arm, iterations) in enumerate((s, a, i) for s in TRAINING_SEEDS
                                                   for a, i in (("projected", 5), ("rowsoftmax", 0))):
        job = jobs[index]
        expected = {"training_seed": seed, "fold_seed": FOLD_SEED, "arm": arm,
            "partner_transport": True, "transport_sinkhorn_iters": iterations,
            "max_folds": 5, "skip_test_eval": True,
            "output_dir": str(ROOT / f"outputs_capacity_projection_dev_v1_{arm}_trainseed{seed}_cvseed2101")}
        require_equal(job, expected, "exact seed/arm/projection job")
        require_equal(job, protocol["seed_and_arm_order"][index], "registered job order")
        path = os.path.normcase(str(Path(job["output_dir"]).resolve()))
        if path in seen:
            raise ValueError("Aliased ablation destinations")
        seen.add(path)


def select_job(protocol, training_seed, fold_seed, arm, output_dir):
    if type(training_seed) is not int or training_seed not in TRAINING_SEEDS:
        raise ValueError("Undeclared training seed")
    if type(fold_seed) is not int or fold_seed != FOLD_SEED:
        raise ValueError("Fold seed must stay 2101; no split regeneration")
    validate_jobs(protocol, protocol["seed_and_arm_order"])
    matches = [j for j in protocol["seed_and_arm_order"] if j["training_seed"] == training_seed and j["arm"] == arm]
    if len(matches) != 1 or Path(output_dir).resolve() != Path(matches[0]["output_dir"]).resolve():
        raise ValueError("Undeclared or mixed arm destination")
    if Path(output_dir).exists():
        raise ValueError("Refusing pre-existing training output directory")
    return copy.deepcopy(matches[0])


def execution_environment(protocol, job):
    prior, _ = read_json_with_record(protocol["accepted_ancestry"]["prior_protocol"]["path"])
    environment = reference_environment(prior)
    environment.update(PPI_SEED=str(job["training_seed"]), PPI_OUTPUT_DIR=job["output_dir"],
        PPI_PARTNER_TRANSPORT="1", PPI_TRANSPORT_SINKHORN_ITERS=str(job["transport_sinkhorn_iters"]))
    return environment


def trainer_arguments(manifest_path, protocol, job, dry_run=False):
    selection = protocol["dataset_selection"]
    result = [sys.executable, str(ROOT / "CROSS5FOLD_projection_dev_v1.py"),
        "--training-seed", str(job["training_seed"]), "--fold-seed", str(FOLD_SEED),
        "--arm", job["arm"], "--execution-manifest", str(Path(manifest_path).resolve()),
        "--data-dir", selection["resolved_data_dir"], "--cohort-manifest", selection["cohort_manifest"]["path"],
        "--fold-manifest", selection["grouped_fold_manifest"]["path"]]
    return result + (["--dry-run"] if dry_run else [])


def prerequisite_arguments(protocol, output):
    return reference_prerequisite_arguments({"dataset_selection": protocol["dataset_selection"]}, output)


def execution_binding(manifest_path, manifest, protocol):
    return {"version": "capacity_projection_dev_v1", "protocol": manifest["protocol"],
        "execution_manifest": file_record(manifest_path), "source_files": manifest["source_files"],
        "reference_source_files": manifest["reference_source_files"],
        "dataset_selection": protocol["dataset_selection"], "runtime": protocol["required_common_runtime"]}


def assert_preserved_recipe(original_path, new_path):
    original = ast.parse(Path(original_path).read_text(encoding="utf-8-sig"))
    new = ast.parse(Path(new_path).read_text(encoding="utf-8-sig"))
    def definitions(tree):
        return {n.name: ast.dump(n, include_attributes=False) for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    require_equal(definitions(new), definitions(original), "all model/data/loss/training definitions")
    def assignments(nodes):
        return {ast.dump(ast.Tuple(elts=n.targets, ctx=ast.Load()), include_attributes=False):
                ast.dump(n, include_attributes=False) for n in nodes if isinstance(n, ast.Assign)}
    for before, after in ((original.body, new.body), (original.body[-1].body, new.body[-1].body)):
        current = assignments(after)
        for target, value in assignments(before).items():
            require_equal(current.get(target), value, "protected recipe assignment")
    class RemoveEvidence(ast.NodeTransformer):
        def visit_Assign(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id in {
                    "capture_rng_states", "record_initialization"}:
                return None
            return self.generic_visit(node)
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "record_initialization":
                return None
            return self.generic_visit(node)
        def visit_Dict(self, node):
            allowed = {"projection_arm", "initialization_records", "initialization_record", "initialization_snapshot"}
            pairs = [(k, v) for k, v in zip(node.keys, node.values)
                     if not isinstance(k, ast.Constant) or k.value not in allowed]
            node.keys, node.values = [k for k, _ in pairs], [v for _, v in pairs]
            return self.generic_visit(node)
    def loop(tree):
        candidates = [n for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
            and n.target.id == "fold" and isinstance(n.iter, ast.Call) and isinstance(n.iter.func, ast.Name)
            and n.iter.func.id == "range" and len(n.iter.args) == 1 and isinstance(n.iter.args[0], ast.Name)
            and n.iter.args[0].id == "folds_to_run"]
        if len(candidates) != 1:
            raise ValueError("Missing unique original five-fold training loop")
        return ast.dump(RemoveEvidence().visit(candidates[0]), include_attributes=False)
    require_equal(loop(new), loop(original), "unchanged optimization/loss/checkpoint-selection loop")
    return len(definitions(original))


def verify_execution_context(context, check_runtime=True):
    manifest, protocol = context["manifest"], context["protocol"]
    verify_record(context["manifest_record"])
    verify_record(context["protocol_record"])
    for record in manifest["source_files"] + manifest["reference_source_files"] + manifest["preserved_records"]:
        verify_record(record)
    assert_preserved_recipe(ROOT / "CROSS5FOLD_multiseed_dev_v1.py", ROOT / "CROSS5FOLD_projection_dev_v1.py")
    selection = protocol["dataset_selection"]
    actual, _, folds = verify_filtered_cohort(selection["resolved_data_dir"], selection["cohort_manifest"]["path"],
        selection["grouped_fold_manifest"]["path"], FOLD_SEED)
    require_equal(actual, {k: v for k, v in selection.items() if k != "selection_mode"}, "fixed cohort bytes")
    require_equal(folds["folds"], manifest["cv_splits"], "fixed fold indices/groups/order")
    if check_runtime:
        runtime = build_run_provenance({}, [], [], "cpu", 0)["runtime"]
        require_equal(runtime, protocol["required_common_runtime"], "accepted CPU runtime")
    return folds


def verify_manifest(path=DEFAULT_MANIFEST, require_acceptance=False):
    manifest, record = read_json_with_record(path)
    protocol, protocol_record = read_protocol(manifest["protocol"]["path"])
    if manifest.get("execution_version") != "capacity_projection_dev_v1" or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported projection execution freeze")
    require_equal(manifest["protocol"], protocol_record, "execution protocol binding")
    validate_jobs(protocol, manifest["jobs"])
    require_equal([r["path"] for r in manifest["source_files"]], [str(ROOT / name) for name in SOURCES], "new source inventory")
    require_equal(manifest["reference_source_files"], protocol["accepted_ancestry"]["unchanged_source_files"], "accepted ancestry sources")
    require_equal(manifest["dataset_selection"], protocol["dataset_selection"], "declared data selection")
    require_equal(manifest["runtime"], protocol["required_common_runtime"], "declared CPU runtime")
    require_equal(manifest["analysis"], protocol["analysis"], "registered analysis")
    require_equal(manifest["fold_seed"], FOLD_SEED, "execution fixed-fold seed")
    require_equal(manifest["external_evaluation"], "skipped", "execution external exclusion")
    if (manifest.get("status") != "frozen_for_infrastructure_acceptance"
            or manifest.get("training_started") is not False or manifest.get("launch_authorized") is not False):
        raise ValueError("Immutable infrastructure-only source freeze required")
    require_equal(manifest["process_dir"], protocol["process_dir"], "process destination")
    require_equal(manifest["aggregate_output_dir"], protocol["aggregate_output_dir"], "aggregate destination")
    require_equal(manifest["launch_authorization_path"], str(REVIEW / "launch_authorization.json"), "separate launch authorization")
    accepted, prior, _ = verify_accepted_ancestry(protocol["accepted_ancestry"]["accepted_v3"]["path"], require_acceptance=True)
    require_equal(manifest["cv_splits"], accepted["cv_splits"], "unchanged original five folds")
    require_equal(manifest["prerequisite_code_files"], json.loads(Path(protocol["accepted_ancestry"]["accepted_v2"]["path"]).read_text(encoding="utf-8-sig"))["prerequisite_code_files"], "unchanged prerequisite code")
    verify_record(manifest["source_archive"])
    archive, _ = read_json_with_record(manifest["source_archive"]["path"])
    require_equal([item["original"] for item in archive["sources"]], manifest["source_files"], "source archive inventory")
    for item, name in zip(archive["sources"], SOURCES):
        verify_record(item["archive"])
        require_equal(item["archive"]["path"], str(REVIEW / "execution_sources_v1" / name), "archived source path")
        require_equal(item["archive"]["sha256"], item["original"]["sha256"], "archived source bytes")
    verify_execution_context({"manifest": manifest, "manifest_record": record,
        "protocol": protocol, "protocol_record": protocol_record})
    if require_acceptance:
        require_infrastructure_acceptance(path, manifest)
    require_equal(file_record(path), record, "execution freeze unchanged")
    return manifest, protocol


def prepare_execution(manifest_path, training_seed, fold_seed, arm, output_dir):
    manifest, protocol = verify_manifest(manifest_path)
    job = select_job(protocol, training_seed, fold_seed, arm, output_dir)
    environment = execution_environment(protocol, job)
    require_equal({k: v for k, v in os.environ.items() if k.upper().startswith("PPI_")}, environment, "isolated arm environment")
    return {"manifest": manifest, "manifest_record": file_record(manifest_path), "protocol": protocol,
        "protocol_record": manifest["protocol"], "job": job, "environment": environment,
        "execution_binding": execution_binding(manifest_path, manifest, protocol)}


def validate_training_namespace(namespace, context):
    protocol = copy.deepcopy(context["protocol"])
    protocol["reference_model_recipe"]["transport_sinkhorn_iters"] = context["job"]["transport_sinkhorn_iters"]
    protocol["training_policy"] = protocol["unchanged_training_policy"]
    return reference_validate_training_namespace(namespace, dict(context, protocol=protocol))


def attach_execution(run, context):
    run.update(training_seed=context["job"]["training_seed"], fold_seed=FOLD_SEED,
        fold_manifest=context["protocol"]["dataset_selection"]["grouped_fold_manifest"],
        execution_binding=context["execution_binding"], projection_arm=context["job"]["arm"],
        transport_sinkhorn_iters=context["job"]["transport_sinkhorn_iters"], initialization_records=[])
    require_equal(run["runtime"], context["protocol"]["required_common_runtime"], "actual run runtime")
    require_equal(run["cv_splits"], context["manifest"]["cv_splits"], "actual run fixed folds")
    if run["partner_transport"] is not True or run["skip_test_eval"] is not True:
        raise ValueError("Both projection arms must retain transport and skip external evaluation")
    return run


def validate_prerequisite_record(gate, protocol, manifest):
    return reference_validate_prerequisite_record(gate, protocol, manifest)


def validate_dry_pair(projected, rowsoftmax):
    from projection_initialization_v1 import validate_initialization_pair
    for record, arm, iterations in ((projected, "projected", 5), (rowsoftmax, "rowsoftmax", 0)):
        if (record.get("arm") != arm or record.get("partner_transport") is not True or record.get("dry_run") is not True
                or record.get("training_started") is not False or record.get("skip_test_eval") is not True
                or record.get("fold_seed") != FOLD_SEED or record["model_recipe"]["transport_sinkhorn_iters"] != iterations):
            raise ValueError("Incorrect dry-run projection/transport/external state")
        if len(record.get("initialization_records", [])) != 5:
            raise ValueError("All five dry-run initialization records required")
    if projected["training_seed"] != rowsoftmax["training_seed"] or projected["training_seed"] not in TRAINING_SEEDS:
        raise ValueError("Unequal paired training seeds")
    def normalize(record):
        value = copy.deepcopy(record)
        for key in ("arm", "output_dir", "initialization_records"):
            value.pop(key, None)
        value["model_recipe"].pop("transport_sinkhorn_iters")
        for key in ("provenance_id", "created_utc", "projection_arm", "transport_sinkhorn_iters"):
            value["run_provenance"].pop(key, None)
        return value
    require_equal(normalize(projected), normalize(rowsoftmax), "only permitted paired dry-run differences")
    return [validate_initialization_pair(a, b, require_initial_values=(i == 0)) for i, (a, b) in
            enumerate(zip(projected["initialization_records"], rowsoftmax["initialization_records"]))]


def require_infrastructure_acceptance(path, manifest=None):
    path = Path(path)
    if manifest is None:
        manifest, _ = read_json_with_record(path)
    acceptance, _ = read_json_with_record(path.with_name(path.stem + ".acceptance.json"))
    if (acceptance.get("status") != "infrastructure_accepted_training_unstarted" or acceptance.get("training_started") is not False
            or acceptance.get("launch_authorized") is not False or acceptance.get("jobs_validated") != 6):
        raise ValueError("Complete infrastructure acceptance required before training")
    require_equal(acceptance["execution_manifest"], file_record(path), "accepted source freeze")
    require_equal(acceptance["source_files"], manifest["source_files"], "accepted/tested source bytes")
    for key in ("full_tests", "all_six_dry_runs", "forced_process_failure", "forced_validation_failure"):
        verify_record(acceptance[key])
        if acceptance[key] not in acceptance["evidence_files"]:
            raise ValueError("Acceptance evidence omitted required artifact: " + key)
    for record in acceptance["evidence_files"]:
        verify_record(record)
    tests, _ = read_json_with_record(acceptance["full_tests"]["path"])
    if tests.get("status") != "passed" or tests.get("failures") != 0 or tests.get("errors") != 0:
        raise ValueError("Full regression suite must pass")
    require_equal(tests["source_files"], manifest["source_files"], "full suite tested source freeze")
    require_equal(tests["execution_manifest"], file_record(path), "full suite execution freeze")
    verify_record(tests["test_log"])
    dry, _ = read_json_with_record(acceptance["all_six_dry_runs"]["path"])
    if dry.get("status") != "passed" or dry.get("jobs_validated") != 6 or dry.get("training_started") is not False:
        raise ValueError("All six real-data dry runs required")
    require_equal(dry["execution_manifest"], file_record(path), "dry-run source freeze")
    if not isinstance(dry.get("dry_run_records"), list) or len(dry["dry_run_records"]) != 6:
        raise ValueError("Exact six accepted dry-run records required")
    for record in dry["evidence_files"]:
        verify_record(record)
    loaded = []
    for job, record in zip(manifest["jobs"], dry["dry_run_records"]):
        verify_record(record)
        item, _ = read_json_with_record(record["path"])
        if (item["training_seed"], item["arm"], item["output_dir"]) != (job["training_seed"], job["arm"], job["output_dir"]):
            raise ValueError("Accepted dry-run identity differs from scheduled arm")
        require_equal(item["cv_splits"], manifest["cv_splits"], "accepted dry-run exact folds")
        loaded.append(item)
    if len(loaded) != 6:
        raise ValueError("Six dry-run identities required")
    for index in (0, 2, 4):
        validate_dry_pair(loaded[index], loaded[index+1])
    for key, status in (("forced_process_failure", "failed"), ("forced_validation_failure", "failed_validation")):
        verify_record(acceptance[key])
        state, _ = read_json_with_record(acceptance[key]["path"])
        if (state.get("simulation") is not True or state.get("training_started") is not False
                or state.get("status") != "failed" or state["jobs"][0]["status"] != status
                or any(j["status"] != "queued" for j in state["jobs"][1:])
                or state["comparison"]["status"] != "waiting_for_all_six_complete"):
            raise ValueError("Failure injection did not stop before later arms/aggregate")
    return acceptance


def require_launch_authorization(manifest_path, manifest=None, protocol=None):
    if manifest is None:
        manifest, protocol = verify_manifest(manifest_path, require_acceptance=True)
    require_infrastructure_acceptance(manifest_path, manifest)
    path = Path(manifest["launch_authorization_path"])
    if not path.is_file():
        raise ValueError("Training is unstarted: explicit user launch authorization has not been recorded")
    authorization, record = read_json_with_record(path)
    if (authorization.get("status") != "explicit_user_launch_authorized"
            or authorization.get("source") != "explicit_user_instruction"
            or not isinstance(authorization.get("user_instruction"), str) or not authorization["user_instruction"].strip()):
        raise ValueError("Explicit user launch authorization required")
    require_equal(authorization["execution_manifest"], file_record(manifest_path), "authorized execution freeze")
    require_equal(authorization["acceptance"], file_record(Path(manifest_path).with_name(Path(manifest_path).stem + ".acceptance.json")), "authorized concrete acceptance")
    require_equal(authorization["jobs"], manifest["jobs"], "authorized six-arm serial scope")
    return record


def require_launch_validation(context):
    return require_launch_authorization(context["manifest_record"]["path"], context["manifest"], context["protocol"])


def require_fresh_destinations(manifest):
    for job in manifest["jobs"]:
        if Path(job["output_dir"]).exists():
            raise ValueError("Training output collision: " + job["output_dir"])
    for key in ("aggregate_output_dir", "process_dir"):
        if Path(manifest[key]).exists():
            raise ValueError("Existing declared destination: " + manifest[key])


def freeze():
    protocol, protocol_record = read_protocol()
    if DEFAULT_MANIFEST.exists():
        raise ValueError("Execution freeze already exists; refusing replacement")
    accepted, _, _ = verify_accepted_ancestry(protocol["accepted_ancestry"]["accepted_v3"]["path"], require_acceptance=True)
    protected = assert_preserved_recipe(ROOT / "CROSS5FOLD_multiseed_dev_v1.py", ROOT / "CROSS5FOLD_projection_dev_v1.py")
    archive_dir = REVIEW / "execution_sources_v1"
    archive_dir.mkdir(exist_ok=False)
    sources = [file_record(ROOT / name) for name in SOURCES]
    archives = []
    for source, name in zip(sources, SOURCES):
        target = archive_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source["path"], target)
        saved = file_record(target)
        require_equal(saved["sha256"], source["sha256"], "exact source archive copy")
        archives.append({"original": source, "archive": saved})
    archive_path = REVIEW / "execution_source_archive_v1.json"
    write_new(archive_path, {"schema_version": 1, "protocol": protocol_record, "sources": archives})
    v2, _ = read_json_with_record(protocol["accepted_ancestry"]["accepted_v2"]["path"])
    preserved = [file_record(REVIEW / "PREREGISTRATION.md"), file_record(REVIEW / "registration_validation.json")]
    preserved += [record for record in protocol["accepted_ancestry"].values() if isinstance(record, dict) and "sha256" in record]
    preserved += [record for record in protocol["observed_references"].values() if isinstance(record, dict) and "sha256" in record]
    preserved += [file_record(ROOT / "research_review/multiseed_dev_v1/COMPLETION_RECORD.json"),
        file_record(ROOT / "research_review/multiseed_dev_v1/completion_archive_v1/archive_manifest.json")]
    manifest = {"schema_version": 1, "execution_version": "capacity_projection_dev_v1",
        "status": "frozen_for_infrastructure_acceptance", "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol_record, "source_files": sources, "source_archive": file_record(archive_path),
        "reference_source_files": protocol["accepted_ancestry"]["unchanged_source_files"],
        "preserved_records": preserved, "prerequisite_code_files": v2["prerequisite_code_files"],
        "jobs": protocol["seed_and_arm_order"], "cv_splits": accepted["cv_splits"], "fold_seed": FOLD_SEED,
        "dataset_selection": protocol["dataset_selection"], "runtime": protocol["required_common_runtime"],
        "analysis": protocol["analysis"], "process_dir": protocol["process_dir"],
        "aggregate_output_dir": protocol["aggregate_output_dir"], "external_evaluation": "skipped",
        "launch_authorization_path": str(REVIEW / "launch_authorization.json"),
        "training_started": False, "launch_authorized": False,
        "protected_recipe_definitions_verified": protected}
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
        verify_manifest(args.manifest, require_acceptance=args.require_acceptance)
        print("Registered projection execution sources, ancestry, data, folds and runtime verified.")
