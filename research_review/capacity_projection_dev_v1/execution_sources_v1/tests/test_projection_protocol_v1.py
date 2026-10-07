"""Projection routing, immutable recipe, no-launch and serial failure acceptance tests."""
import copy
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import projection_protocol_v1 as protocol_tools
import run_projection_dev_v1 as worker
from projection_initialization_v1 import capture_initialization, capture_rng_states
from research_provenance import file_record


def dry_record(arm, seed=2102):
    iterations = 5 if arm == "projected" else 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    initializations = []
    for fold in range(5):
        before = capture_rng_states()
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
        initializations.append(capture_initialization(model, optimizer, fold, seed, 2101,
            arm, iterations, before, capture_rng_states()))
    return {"arm": arm, "training_seed": seed, "fold_seed": 2101,
        "output_dir": arm + str(seed), "partner_transport": True, "dry_run": True,
        "training_started": False, "skip_test_eval": True,
        "cv_splits": [{"fold_index": i} for i in range(5)],
        "model_recipe": {"seed": seed, "transport_sinkhorn_iters": iterations, "hidden_dim": 80},
        "run_provenance": {"seed": seed, "training_seed": seed, "fold_seed": 2101,
            "provenance_id": arm + str(seed), "created_utc": arm, "runtime": {"device": "cpu"},
            "projection_arm": arm, "transport_sinkhorn_iters": iterations, "initialization_records": []},
        "initialization_records": initializations}


class ProjectionProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.protocol_record = protocol_tools.read_protocol()

    def test_registered_six_arm_order_projection_and_fixed_folds(self):
        protocol_tools.validate_jobs(self.protocol, self.protocol["seed_and_arm_order"])
        self.assertEqual([(j["training_seed"], j["arm"]) for j in self.protocol["seed_and_arm_order"]],
            [(s, a) for s in (2102, 2103, 2104) for a in ("projected", "rowsoftmax")])

    def test_wrong_seed_fold_order_transport_scope_and_collision_rejected(self):
        for key, bad in (("training_seed", 2101), ("fold_seed", 2102), ("max_folds", 1),
                ("skip_test_eval", False), ("partner_transport", False), ("transport_sinkhorn_iters", 0),
                ("output_dir", self.protocol["seed_and_arm_order"][1]["output_dir"])):
            jobs = copy.deepcopy(self.protocol["seed_and_arm_order"])
            jobs[0][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol_tools.validate_jobs(self.protocol, jobs)
        with self.assertRaises(ValueError):
            protocol_tools.validate_jobs(self.protocol, list(reversed(self.protocol["seed_and_arm_order"])))

    def test_select_job_rejects_regeneration_unequal_seed_and_output_mixing(self):
        job = self.protocol["seed_and_arm_order"][0]
        for seed, fold, arm, output in ((2101, 2101, "projected", job["output_dir"]),
                (2102, 2102, "projected", job["output_dir"]),
                (2103, 2101, "projected", job["output_dir"]),
                (2102, 2101, "rowsoftmax", job["output_dir"])):
            with self.assertRaises(ValueError):
                protocol_tools.select_job(self.protocol, seed, fold, arm, output)

    def test_clean_environment_removes_inherited_ppi_values(self):
        with mock.patch.dict(os.environ, {"PPI_SEED": "999", "ppi_extra": "contamination"}):
            actual = protocol_tools.clean_environment({"PPI_SEED": "2102"})
        self.assertEqual(actual["PPI_SEED"], "2102")
        self.assertNotIn("ppi_extra", actual)

    def test_arm_environment_differs_only_in_projection_and_output(self):
        projected, rowsoftmax = self.protocol["seed_and_arm_order"][:2]
        left = protocol_tools.execution_environment(self.protocol, projected)
        right = protocol_tools.execution_environment(self.protocol, rowsoftmax)
        self.assertEqual({k for k in left if left[k] != right[k]},
            {"PPI_TRANSPORT_SINKHORN_ITERS", "PPI_OUTPUT_DIR"})
        self.assertEqual(left["PPI_PARTNER_TRANSPORT"], right["PPI_PARTNER_TRANSPORT"])
        self.assertEqual(left["PPI_PARTNER_TRANSPORT"], "1")

    def test_actual_commands_keep_strict_gate_fixed_cohort_and_no_external_eval(self):
        job = self.protocol["seed_and_arm_order"][0]
        command = protocol_tools.trainer_arguments("manifest.json", self.protocol, job, dry_run=True)
        self.assertIn("--dry-run", command)
        self.assertEqual(command[command.index("--fold-seed") + 1], "2101")
        self.assertEqual(command[command.index("--training-seed") + 1], "2102")
        self.assertEqual(command[command.index("--data-dir") + 1], self.protocol["dataset_selection"]["resolved_data_dir"])
        gate = protocol_tools.prerequisite_arguments(self.protocol, "gate.json")
        for flag in ("--require-plm", "--require-partner", "--require-pair", "--require-partner-encoder"):
            self.assertIn(flag, gate)
        self.assertEqual(gate[-2:], ["Train335.pkl", "Test287.pkl"])

    def test_ast_guard_proves_original_math_and_rejects_recipe_or_loop_changes(self):
        original = protocol_tools.ROOT / "CROSS5FOLD_multiseed_dev_v1.py"
        current = protocol_tools.ROOT / "CROSS5FOLD_projection_dev_v1.py"
        self.assertGreater(protocol_tools.assert_preserved_recipe(original, current), 70)
        text = current.read_text(encoding="utf-8-sig")
        for before, after in (("else 3e-4", "else 9e-4"), ("patience = 8", "patience = 7"),
                ("for fold in range(folds_to_run):", "for fold in range(folds_to_run - 1):")):
            self.assertTrue(before in text, before)
            with tempfile.TemporaryDirectory() as directory:
                changed = Path(directory) / "changed.py"
                changed.write_text(text.replace(before, after), encoding="utf-8")
                with self.assertRaises(ValueError):
                    protocol_tools.assert_preserved_recipe(original, changed)

    def test_dry_pair_exact_parameter_optimizer_and_first_fold_values(self):
        left, right = dry_record("projected"), dry_record("rowsoftmax")
        evidence = protocol_tools.validate_dry_pair(left, right)
        self.assertEqual(len(evidence), 5)
        self.assertTrue(evidence[0]["initial_values_equal"])
        self.assertTrue(evidence[0]["rng_fingerprints_equal"])
        for changed in ("seed", "fold", "runtime", "recipe", "nonfinite"):
            bad = copy.deepcopy(right)
            if changed == "seed":
                bad["training_seed"] = 2103
            elif changed == "fold":
                bad["cv_splits"][0]["fold_index"] = 4
            elif changed == "runtime":
                bad["run_provenance"]["runtime"]["device"] = "cuda"
            elif changed == "recipe":
                bad["model_recipe"]["hidden_dim"] = 81
            else:
                bad["initialization_records"][0]["parameter_count"] = float("nan")
            with self.subTest(change=changed), self.assertRaises(ValueError):
                protocol_tools.validate_dry_pair(left, bad)

    def test_trainer_saves_explicit_projection_and_initialization_bindings(self):
        tree = ast.parse((protocol_tools.ROOT / "CROSS5FOLD_projection_dev_v1.py").read_text(encoding="utf-8-sig"))
        saved = [node.args[0] for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch" and node.func.attr == "save" and node.args
            and isinstance(node.args[0], ast.Dict)]
        self.assertEqual(len(saved), 1)
        keys = {key.value for key in saved[0].keys if isinstance(key, ast.Constant)}
        self.assertTrue({"projection_arm", "initialization_record", "initialization_snapshot",
            "training_seed", "fold_seed", "execution_binding"} <= keys)
        summaries = [node.args[1] for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == "save_json" and len(node.args) == 2
            and isinstance(node.args[1], ast.Dict)]
        self.assertEqual(len(summaries), 1)
        keys = {key.value for key in summaries[0].keys if isinstance(key, ast.Constant)}
        self.assertTrue({"projection_arm", "initialization_records", "training_seed", "fold_seed"} <= keys)

    def test_cross_seed_normalization_preserves_recipe_and_folds(self):
        left, right = dry_record("projected", 2102), dry_record("projected", 2103)
        self.assertEqual(worker.normalize_across_seeds(left), worker.normalize_across_seeds(right))
        right["cv_splits"][0]["fold_index"] = 4
        self.assertNotEqual(worker.normalize_across_seeds(left), worker.normalize_across_seeds(right))

    def test_missing_launch_authorization_fails_before_production_directory_or_child(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            manifest = {"launch_authorization_path": str(Path(directory) / "launch_authorization.json"),
                "process_dir": str(Path(directory) / "process")}
            with mock.patch.object(protocol_tools, "require_infrastructure_acceptance", return_value={}), \
                    mock.patch.object(worker, "verify_manifest", return_value=(manifest, {})), \
                    mock.patch.object(worker, "claim_queue") as claim, mock.patch.object(worker, "run_logged") as child:
                with self.assertRaisesRegex(ValueError, "explicit user launch"):
                    worker.run_queue(manifest_path)
                claim.assert_not_called()
                child.assert_not_called()
            self.assertFalse(Path(manifest["process_dir"]).exists())

    def test_exclusive_retained_lock_refuses_second_worker_and_preserves_first(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "process"
            worker.claim_queue(target, {"sha256": "frozen"}, {"sha256": "authorized"})
            before = file_record(target / "queue.lock")
            with self.assertRaises(FileExistsError):
                worker.claim_queue(target, {"sha256": "changed"}, {})
            self.assertEqual(file_record(target / "queue.lock"), before)

    def test_training_aggregate_and_process_collisions_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = {"jobs": [{"output_dir": str(Path(directory) / "training")}],
                "aggregate_output_dir": str(Path(directory) / "aggregate"), "process_dir": str(Path(directory) / "process")}
            protocol_tools.require_fresh_destinations(manifest)
            for path in (manifest["jobs"][0]["output_dir"], manifest["aggregate_output_dir"], manifest["process_dir"]):
                Path(path).mkdir()
                with self.assertRaises(ValueError):
                    protocol_tools.require_fresh_destinations(manifest)
                Path(path).rmdir()

    def test_every_process_failure_halts_later_arms_and_aggregate(self):
        jobs = self.protocol["seed_and_arm_order"]
        for failed in range(6):
            launches, aggregates = [], []
            def launch(index, *args):
                launches.append(index)
                return 7 if index == failed else 0
            state = worker.execute_serial(jobs, lambda *_: None, launch, lambda *_: {"passed": True},
                lambda: aggregates.append(True))
            self.assertEqual(launches, list(range(failed + 1)))
            self.assertEqual(state["jobs"][failed]["status"], "failed")
            self.assertTrue(all(j["status"] == "queued" for j in state["jobs"][failed + 1:]))
            self.assertEqual(aggregates, [])

    def test_every_validation_failure_halts_even_after_zero_exit(self):
        jobs = self.protocol["seed_and_arm_order"]
        for failed in range(6):
            launches, aggregates = [], []
            def validate(index, job):
                if index == failed:
                    raise ValueError("corrupt artifact")
                return {"passed": True}
            state = worker.execute_serial(jobs, lambda *_: None,
                lambda index, *args: launches.append(index) or 0, validate, lambda: aggregates.append(True))
            self.assertEqual(launches, list(range(failed + 1)))
            self.assertEqual(state["jobs"][failed]["status"], "failed_validation")
            self.assertEqual(state["comparison"]["status"], "waiting_for_all_six_complete")
            self.assertEqual(aggregates, [])

    def test_aggregate_only_runs_after_all_six_validations(self):
        validations, aggregates = [], []
        state = worker.execute_serial(self.protocol["seed_and_arm_order"], lambda *_: None, lambda *_: 0,
            lambda index, *args: validations.append(index) or {"passed": True},
            lambda: aggregates.append(tuple(validations)) or {"complete": True})
        self.assertEqual(state["status"], "complete")
        self.assertEqual(aggregates, [tuple(range(6))])

    def test_forced_failures_are_clean_processes_and_never_trainer_commands(self):
        for validation_failure in (False, True):
            with tempfile.TemporaryDirectory() as directory, \
                    mock.patch.object(worker, "verify_manifest", return_value=({"jobs": self.protocol["seed_and_arm_order"]}, self.protocol)), \
                    mock.patch.object(worker, "require_fresh_destinations"), \
                    mock.patch.object(worker, "select_job"), \
                    mock.patch.object(worker, "file_record", return_value={"sha256": "frozen"}), \
                    mock.patch.object(worker, "run_logged", return_value=0 if validation_failure else 7) as child:
                state = worker.simulate_failure("manifest.json", Path(directory) / "evidence", validation_failure)
                self.assertEqual(child.call_count, 1)
                self.assertEqual(child.call_args.args[0][1], "-c")
                self.assertEqual(state["training_started"], False)
                self.assertEqual(state["jobs"][0]["status"], "failed_validation" if validation_failure else "failed")


if __name__ == "__main__":
    unittest.main()
