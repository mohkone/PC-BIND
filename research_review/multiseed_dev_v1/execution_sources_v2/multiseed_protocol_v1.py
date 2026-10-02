"""Fail-closed routing for the preregistered fixed-fold training-seed repeat."""
import ast
import copy
import json
import os
from pathlib import Path

from filtered_cohort_protocol import read_json_with_record, verify_filtered_cohort
from frozen_development_protocol import require_equal
from research_provenance import build_run_provenance, file_record


TRAINING_SEEDS = (2102, 2103, 2104)
FOLD_SEED = 2101
NEW_SOURCES = ("multiseed_protocol_v1.py", "run_multiseed_dev_v1.py",
               "CROSS5FOLD_multiseed_dev_v1.py", "tests/test_multiseed_protocol_v1.py")
ROUTING_FUNCTIONS = {"configure_training_data", "prepare_training_splits"}
FREEZE_SHA = "7f8f5f642cff676c8ef3438554dbf8b5ee8b6f5529b0e8deff58248277909265"


def verify_record(record):
    require_equal(file_record(record["path"]), record, "file bytes: " + record["path"])


def read_protocol(path):
    protocol, record = read_json_with_record(path)
    registration, _ = read_json_with_record(Path(path).parent / "registration_validation.json")
    require_equal(record, registration["protocol"], "registered protocol bytes")
    if (protocol.get("schema_version") != 1 or protocol.get("protocol_version") != "multiseed_dev_v1"
            or protocol.get("additional_training_seeds") != list(TRAINING_SEEDS)
            or type(protocol.get("fixed_fold_seed")) is not int or protocol["fixed_fold_seed"] != FOLD_SEED
            or protocol.get("reference_in_primary_new_seed_estimate") is not False
            or protocol.get("external_evaluation") != "skipped in every arm"
            or protocol.get("folds_per_arm") != 5):
        raise ValueError("Unsupported preregistered multiseed design")
    if protocol["reference_freeze_manifest"]["sha256"] != FREEZE_SHA:
        raise ValueError("Unexpected reference freeze identity")
    for key in ("reference_freeze_manifest", "reference_comparison", "exact_recipe_sources"):
        verify_record(protocol[key])
    validate_jobs(protocol, protocol["seed_and_arm_order"])
    return protocol, record


def validate_jobs(protocol, jobs):
    expected = protocol["seed_and_arm_order"]
    if len(jobs) != 6 or len(expected) != 6:
        raise ValueError("Exactly six preregistered jobs are required")
    seen = set()
    root = Path(protocol["reference_freeze_manifest"]["path"]).resolve().parents[2]
    for index, (seed, arm) in enumerate((s, a) for s in TRAINING_SEEDS for a in ("ot", "noot")):
        job = jobs[index]
        if (type(job.get("training_seed")) is not int or job["training_seed"] != seed
                or job.get("arm") != arm or job.get("partner_transport") is not (arm == "ot")):
            raise ValueError("Unequal seed pair, incorrect arm or changed serial job order")
        desired = root / f"outputs_multiseed_dev_v1_{arm}_trainseed{seed}_cvseed2101"
        path = Path(job["output_dir"]).resolve()
        if path != desired or os.path.normcase(str(path)) in seen:
            raise ValueError("Changed, mixed or aliased output directory")
        if job.get("fold_seed", FOLD_SEED) != FOLD_SEED or type(job.get("fold_seed", FOLD_SEED)) is not int:
            raise ValueError("Fold seed must stay 2101")
        seen.add(os.path.normcase(str(path)))
        require_equal(job, expected[index], "declared seed/arm job")


def select_job(protocol, training_seed, fold_seed, arm, output_dir):
    if type(training_seed) is not int or training_seed not in TRAINING_SEEDS:
        raise ValueError("Training seed must be one of 2102, 2103, 2104")
    if type(fold_seed) is not int or fold_seed != FOLD_SEED:
        raise ValueError("Fold seed must remain 2101; fold regeneration is forbidden")
    validate_jobs(protocol, protocol["seed_and_arm_order"])
    matches = [job for job in protocol["seed_and_arm_order"]
               if job["training_seed"] == training_seed and job["arm"] == arm]
    if len(matches) != 1:
        raise ValueError("Undeclared arm")
    job = copy.deepcopy(matches[0])
    if Path(output_dir).resolve() != Path(job["output_dir"]).resolve():
        raise ValueError("Output directory is bound to another seed or arm")
    if Path(output_dir).exists():
        raise ValueError("Refusing pre-existing output directory; fresh paths are required")
    return job


def reference_environment(protocol):
    root = Path(protocol["reference_freeze_manifest"]["path"]).parent
    text = (root / "ot_dry_run.txt").read_text(encoding="utf-8-sig")
    settings = json.loads(text[text.index("{"):])
    return {key: value for key, value in settings.items() if key.startswith("PPI_")}


def execution_environment(protocol, job):
    environment = reference_environment(protocol)
    environment.update(PPI_SEED=str(job["training_seed"]), PPI_OUTPUT_DIR=job["output_dir"],
                       PPI_PARTNER_TRANSPORT="1" if job["partner_transport"] else "0")
    return environment


def clean_environment(environment, inherited=None):
    result = {key: value for key, value in (os.environ if inherited is None else inherited).items()
              if not key.upper().startswith("PPI_")}
    result.update(environment)
    return result


def assert_preserved_recipe(original_path, new_path):
    """All reference definitions except two routing helpers must be identical."""
    original = ast.parse(Path(original_path).read_text(encoding="utf-8-sig"))
    new = ast.parse(Path(new_path).read_text(encoding="utf-8-sig"))
    definitions = lambda tree: {node.name: ast.dump(node, include_attributes=False)
                               for node in tree.body
                               if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    before, after = definitions(original), definitions(new)
    if set(before) != set(after):
        raise ValueError("Reference model/training definitions were added or removed")
    for name in before.keys() - ROUTING_FUNCTIONS:
        if before[name] != after[name]:
            raise ValueError("Protected recipe definition changed: " + name)
    def assignments(nodes):
        return {ast.dump(ast.Tuple(elts=node.targets, ctx=ast.Load()), include_attributes=False):
                ast.dump(node, include_attributes=False)
                for node in nodes if isinstance(node, ast.Assign)}
    for scope, old_nodes, new_nodes in (("module", original.body, new.body),
            ("main", original.body[-1].body, new.body[-1].body)):
        old_assignments, new_assignments = assignments(old_nodes), assignments(new_nodes)
        for target, value in old_assignments.items():
            if new_assignments.get(target) != value:
                raise ValueError("Protected recipe assignment changed in " + scope + ": " + target)
    class StripProvenance(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                if node.value.func.id == "verify_execution_context":
                    return None
            return self.generic_visit(node)

        def visit_Dict(self, node):
            pairs = [(k, v) for k, v in zip(node.keys, node.values)
                     if not isinstance(k, ast.Constant) or k.value not in
                     {"training_seed", "fold_seed", "fold_manifest", "execution_binding"}]
            node.keys, node.values = [p[0] for p in pairs], [p[1] for p in pairs]
            return self.generic_visit(node)

    def training_loop(tree):
        candidates = [n for n in ast.walk(tree) if isinstance(n, ast.For)
                      and isinstance(n.target, ast.Name) and n.target.id == "fold"
                      and isinstance(n.iter, ast.Call) and isinstance(n.iter.func, ast.Name)
                      and n.iter.func.id == "range" and len(n.iter.args) == 1
                      and isinstance(n.iter.args[0], ast.Name) and n.iter.args[0].id == "folds_to_run"]
        if len(candidates) != 1:
            raise ValueError("Missing unique original training/selection loop")
        return ast.dump(StripProvenance().visit(candidates[0]), include_attributes=False)
    if training_loop(original) != training_loop(new):
        raise ValueError("Protected model/optimizer/loss/selection training loop changed")
    return len(before) - len(ROUTING_FUNCTIONS)


def verify_execution_context(context, *, check_runtime=True):
    manifest = context["manifest"]
    verify_record(context["manifest_record"])
    verify_record(context["protocol_record"])
    for record in manifest["source_files"] + manifest["reference_source_files"] + [manifest["reference_environment"]]:
        verify_record(record)
    protocol = context["protocol"]
    selection = protocol["dataset_selection"]
    actual, _, folds = verify_filtered_cohort(selection["resolved_data_dir"],
        selection["cohort_manifest"]["path"], selection["grouped_fold_manifest"]["path"], FOLD_SEED)
    require_equal(actual, {k: v for k, v in selection.items() if k != "selection_mode"}, "fixed cohort")
    freeze, _ = read_json_with_record(protocol["reference_freeze_manifest"]["path"])
    require_equal(folds["folds"], freeze["run_configuration"]["cv_splits"], "exact original fold identity/order")
    if check_runtime:
        runtime = build_run_provenance({}, [], [], "cpu", 0)["runtime"]
        require_equal(runtime, protocol["required_common_runtime"], "preregistered CPU runtime")
    return folds


def prepare_execution(manifest_path, training_seed, fold_seed, arm, output_dir):
    manifest, manifest_record = read_json_with_record(manifest_path)
    if (manifest.get("schema_version") != 1 or manifest.get("execution_version") != "multiseed_dev_v1"
            or manifest.get("fold_seed") != FOLD_SEED or manifest.get("status") != "frozen_for_validation"):
        raise ValueError("Unsupported execution manifest")
    protocol, protocol_record = read_protocol(manifest["protocol"]["path"])
    require_equal(protocol_record, manifest["protocol"], "execution protocol binding")
    require_equal(manifest["jobs"], protocol["seed_and_arm_order"], "execution job schedule")
    job = select_job(protocol, training_seed, fold_seed, arm, output_dir)
    context = {"protocol": protocol, "protocol_record": protocol_record,
               "manifest": manifest, "manifest_record": manifest_record, "job": job,
               "environment": execution_environment(protocol, job)}
    require_equal(manifest["runtime"], protocol["required_common_runtime"], "bound runtime")
    require_equal(manifest["dataset_selection"], protocol["dataset_selection"], "bound data selection")
    require_equal(manifest["model_recipe"], protocol["reference_model_recipe"], "bound model recipe")
    require_equal({key: value for key, value in os.environ.items() if key.upper().startswith("PPI_")},
                  context["environment"], "explicit isolated training environment")
    root = Path(protocol["reference_freeze_manifest"]["path"]).resolve().parents[2]
    require_equal([r["path"] for r in manifest["source_files"]],
                  [str((root / rel).resolve()) for rel in NEW_SOURCES], "execution source inventory")
    freeze, _ = read_json_with_record(protocol["reference_freeze_manifest"]["path"])
    require_equal(manifest["reference_source_files"], freeze["source_files"], "reference source inventory")
    require_equal(manifest["prerequisite_code_files"], freeze["prerequisite_code_files"], "prerequisite source inventory")
    require_equal(manifest["reference_environment"]["path"],
                  str(root / "research_review/frozen_5fold_dev_v1/ot_dry_run.txt"), "reference environment inventory")
    assert_preserved_recipe(root / "CROSS5FOLD_multi_test.py", root / "CROSS5FOLD_multiseed_dev_v1.py")
    verify_execution_context(context)
    context["execution_binding"] = {"version": "multiseed_dev_v1", "protocol": protocol_record,
        "execution_manifest": manifest_record, "source_files": manifest["source_files"],
        "reference_source_files": manifest["reference_source_files"],
        "dataset_selection": protocol["dataset_selection"], "runtime": protocol["required_common_runtime"]}
    return context


def validate_training_namespace(namespace, context):
    protocol, job = context["protocol"], context["job"]
    aliases = {"weight_decay": "OPT_WEIGHT_DECAY", "test_sets": "TEST_PKL_FILES"}
    actual = {}
    for key in protocol["reference_model_recipe"]:
        if key == "pair_contact_contrast_mismatch_policy":
            actual[key] = "in_batch_nearest_length"
        elif key == "pair_contact_contrast_coverage":
            actual[key] = namespace["pair_contact_contrast_coverage"]
        else:
            actual[key] = namespace[aliases.get(key, key.upper())]
    expected = dict(protocol["reference_model_recipe"], seed=job["training_seed"])
    require_equal(actual, expected, "effective loaded-data model recipe")
    policy = protocol["training_policy"]
    require_equal({"max_epochs": namespace["epochs"], "early_stopping_patience": namespace["patience"],
                   "top_checkpoint_average_candidates": namespace["TOP_K_CHECKPOINTS"]},
                  {key: policy[key] for key in ("max_epochs", "early_stopping_patience", "top_checkpoint_average_candidates")},
                  "effective epoch, patience and checkpoint-average policy")
    if namespace["PARTNER_TRANSPORT"] is not job["partner_transport"]:
        raise ValueError("Wrong effective transport arm")
    if namespace["FOLD_SEED"] != FOLD_SEED:
        raise ValueError("Wrong effective fold seed")
    require_equal(namespace["training_splits"], context["manifest"]["cv_splits"], "loaded saved fold identity/order")
    require_equal(namespace["dataset_selection"], protocol["dataset_selection"], "effective data selection")
    return actual


def attach_execution(run, context):
    run.update(training_seed=context["job"]["training_seed"], fold_seed=FOLD_SEED,
               fold_manifest=context["protocol"]["dataset_selection"]["grouped_fold_manifest"],
               execution_binding=context["execution_binding"])
    require_equal(run["runtime"], context["protocol"]["required_common_runtime"], "actual training runtime")
    require_equal(run["cv_splits"], context["manifest"]["cv_splits"], "actual run splits")
    return run


def validate_prerequisite_record(gate, protocol, manifest):
    for key in ("require_plm", "require_partner", "require_pair", "require_partner_encoder", "fixed_folds_validated"):
        if gate.get(key) is not True:
            raise ValueError("Strict prerequisite requirement omitted: " + key)
    if (gate.get("status") != "passed" or gate.get("errors") != []
            or gate.get("min_partner_encoder_coverage") != 1.0):
        raise ValueError("Strict prerequisite gate did not pass")
    selection = protocol["dataset_selection"]
    require_equal(gate.get("dataset_selection"), {k: v for k, v in selection.items() if k != "selection_mode"}, "gated cohort selection")
    require_equal(gate.get("loaded_datasets"), selection["datasets"], "gated dataset hashes")
    require_equal(gate.get("code_files"), manifest["prerequisite_code_files"], "gated source hashes")


def require_launch_validation(context):
    """A training invocation requires the completed tests + six dry-run gate."""
    path = Path(context["manifest_record"]["path"])
    acceptance, _ = read_json_with_record(path.with_name(path.stem + ".acceptance.json"))
    if (acceptance.get("status") != "validated_before_training"
            or acceptance.get("jobs_validated") != 6
            or acceptance.get("training_started") is not False):
        raise ValueError("All six dry runs and the full test suite must pass before training")
    require_equal(acceptance["execution_manifest"], context["manifest_record"], "accepted execution source freeze")
    verify_record(acceptance["full_test_record"])
    tests, _ = read_json_with_record(acceptance["full_test_record"]["path"])
    if tests.get("status") != "passed" or tests.get("errors") != 0 or tests.get("failures") != 0:
        raise ValueError("Full-suite acceptance is not passing")
    require_equal(tests["source_files"], context["manifest"]["source_files"], "tested execution sources")
    verify_record(tests["log"])
    records = acceptance.get("dry_run_records")
    gate_records = acceptance.get("prerequisite_records")
    if not isinstance(records, list) or len(records) != 6 or not isinstance(gate_records, list) or len(gate_records) != 6:
        raise ValueError("Six ordered dry-run records required")
    loaded = []
    for job, record in zip(context["manifest"]["jobs"], records):
        verify_record(record)
        dry, _ = read_json_with_record(record["path"])
        if (dry.get("training_seed") != job["training_seed"] or dry.get("arm") != job["arm"]
                or dry.get("output_dir") != job["output_dir"]):
            raise ValueError("Accepted dry-run job identity mismatch")
        require_equal(dry["cv_splits"], context["manifest"]["cv_splits"], "accepted fold identity/order")
        require_equal(dry["run_provenance"]["execution_binding"], context["execution_binding"], "accepted provenance binding")
        loaded.append(dry)
    for index in (0, 2, 4):
        validate_pair_records(loaded[index], loaded[index + 1])
    for record in gate_records:
        verify_record(record)
        gate, _ = read_json_with_record(record["path"])
        validate_prerequisite_record(gate, context["protocol"], context["manifest"])


def validate_pair_records(ot, noot):
    if (type(ot.get("training_seed")) is not int or ot["training_seed"] not in TRAINING_SEEDS
            or type(noot.get("training_seed")) is not int
            or ot["training_seed"] != noot.get("training_seed")
            or ot.get("partner_transport") is not True or noot.get("partner_transport") is not False):
        raise ValueError("Unequal training seeds or wrong matched arms")
    for record in (ot, noot):
        if (type(record.get("fold_seed")) is not int or record.get("fold_seed") != FOLD_SEED
                or record.get("dry_run") is not True
                or record.get("training_started") is not False or record.get("skip_test_eval") is not True):
            raise ValueError("Dry-run record changed fixed-fold/training/external status")
    allowed = {"arm", "partner_transport", "output_dir", "run_provenance"}
    require_equal({k: v for k, v in ot.items() if k not in allowed},
                  {k: v for k, v in noot.items() if k not in allowed}, "paired dry-run recipe and identity")
    allowed_run = {"provenance_id", "created_utc", "partner_transport"}
    require_equal({k: v for k, v in ot["run_provenance"].items() if k not in allowed_run},
                  {k: v for k, v in noot["run_provenance"].items() if k not in allowed_run}, "paired run provenance")
    if Path(ot["output_dir"]).resolve() == Path(noot["output_dir"]).resolve():
        raise ValueError("Matched arms share an output directory")
