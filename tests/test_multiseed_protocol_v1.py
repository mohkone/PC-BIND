"""Seed/fold isolation regressions. No experiment training or dataset mutations."""
import ast
import copy
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

import CROSS5FOLD_multiseed_dev_v1 as trainer
from multiseed_protocol_v1 import (
    FOLD_SEED, assert_preserved_recipe, attach_execution, clean_environment,
    execution_environment, read_protocol, require_launch_validation,
    select_job, validate_jobs, validate_pair_records, validate_prerequisite_record,
)
from run_multiseed_dev_v1 import trainer_arguments
from test_filtered_protocol import TRAIN_HASH, small_split


ROOT = Path(__file__).resolve().parents[1]


def protocol_fixture(root):
    return {"reference_freeze_manifest": {"path": str(root / "research_review/frozen_5fold_dev_v1/freeze_manifest.json")},
            "seed_and_arm_order": [{"training_seed": seed, "arm": arm,
                "partner_transport": arm == "ot",
                "output_dir": str(root / f"outputs_multiseed_dev_v1_{arm}_trainseed{seed}_cvseed2101")}
                for seed in (2102, 2103, 2104) for arm in ("ot", "noot")]}


def pair_fixture():
    result = []
    for arm, flag in (("ot", True), ("noot", False)):
        result.append({"training_seed": 2102, "fold_seed": 2101, "arm": arm,
            "partner_transport": flag, "dry_run": True, "training_started": False,
            "skip_test_eval": True, "output_dir": f"temporary_{arm}",
            "cv_splits": [{"fold_index": i, "val_indices": [i]} for i in range(5)],
            "model_recipe": {"seed": 2102, "max_folds": 5},
            "run_provenance": {"seed": 2102, "training_seed": 2102, "fold_seed": 2101,
                "partner_transport": flag, "provenance_id": arm, "created_utc": arm,
                "runtime": {"device": "cpu"}, "source_files": [{"sha256": "a" * 64}]}})
    return result


class DeclaredJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.protocol = protocol_fixture(self.root)

    def test_only_declared_seed_arm_order_and_paths_are_accepted(self):
        validate_jobs(self.protocol, self.protocol["seed_and_arm_order"])
        for job in self.protocol["seed_and_arm_order"]:
            self.assertEqual(select_job(self.protocol, job["training_seed"], 2101, job["arm"], job["output_dir"]), job)

    def test_fold_seed_cannot_be_changed_or_conflated(self):
        job = self.protocol["seed_and_arm_order"][0]
        for seed in (2102, 2103, 2104, True, 2101.0, "2101"):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "Fold seed"):
                select_job(self.protocol, 2102, seed, "ot", job["output_dir"])

    def test_training_seed_requires_declared_integer(self):
        job = self.protocol["seed_and_arm_order"][0]
        for seed in (2101, 2105, True, 2102.0, "2102"):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                select_job(self.protocol, seed, 2101, "ot", job["output_dir"])

    def test_mixed_seed_arm_output_and_nonempty_paths_fail(self):
        jobs = self.protocol["seed_and_arm_order"]
        for other in jobs[1:]:
            with self.subTest(other=other), self.assertRaisesRegex(ValueError, "bound to another"):
                select_job(self.protocol, 2102, 2101, "ot", other["output_dir"])
        output = Path(jobs[0]["output_dir"])
        output.mkdir()
        (output / "evidence.txt").write_text("preserve")
        with self.assertRaisesRegex(ValueError, "pre-existing"):
            select_job(self.protocol, 2102, 2101, "ot", str(output))

    def test_preexisting_empty_directory_is_also_rejected(self):
        job = self.protocol["seed_and_arm_order"][0]
        Path(job["output_dir"]).mkdir()
        with self.assertRaisesRegex(ValueError, "pre-existing"):
            select_job(self.protocol, 2102, 2101, "ot", job["output_dir"])

    def test_swapped_unequal_seed_or_duplicate_jobs_fail(self):
        jobs = self.protocol["seed_and_arm_order"]
        for field, value in (("training_seed", 2103), ("partner_transport", False),
                             ("output_dir", jobs[1]["output_dir"]), ("fold_seed", 2102)):
            modified = copy.deepcopy(jobs)
            modified[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_jobs(self.protocol, modified)
        with self.assertRaises(ValueError):
            validate_jobs(self.protocol, list(reversed(jobs)))

    def test_inherited_ppi_overrides_are_removed_and_parent_is_unchanged(self):
        inherited = {"PATH": "retain", "PPI_SEED": "999", "ppi_skip_test_eval": "0", "PPI_EXTRA": "1"}
        saved = copy.deepcopy(inherited)
        actual = clean_environment({"PPI_SEED": "2102", "PPI_SKIP_TEST_EVAL": "1"}, inherited)
        self.assertEqual(actual, {"PATH": "retain", "PPI_SEED": "2102", "PPI_SKIP_TEST_EVAL": "1"})
        self.assertEqual(inherited, saved)


class MatchedDryRunTests(unittest.TestCase):
    def test_only_allowed_pair_differences_pass(self):
        validate_pair_records(*pair_fixture())

    def test_unequal_seeds_fold_external_or_recipe_changes_fail(self):
        for field, value in (("training_seed", 2103), ("training_seed", 2102.0),
                             ("fold_seed", 2102), ("fold_seed", 2101.0),
                             ("skip_test_eval", False), ("training_started", True),
                             ("model_recipe", {"seed": 2102, "max_folds": 1}),
                             ("cv_splits", [])):
            ot, noot = pair_fixture()
            noot[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_pair_records(ot, noot)

    def test_source_runtime_or_shared_output_changes_fail(self):
        for field in ("runtime", "source_files"):
            ot, noot = pair_fixture()
            noot["run_provenance"][field] = None
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_pair_records(ot, noot)
        ot, noot = pair_fixture()
        noot["output_dir"] = ot["output_dir"]
        with self.assertRaisesRegex(ValueError, "share an output"):
            validate_pair_records(ot, noot)

    def test_training_is_blocked_without_completed_launch_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "execution_manifest_v2.json"
            context = {"manifest_record": {"path": str(path)}}
            with self.assertRaises(FileNotFoundError):
                require_launch_validation(context)
            path.with_name(path.stem + ".acceptance.json").write_text(json.dumps({"status": "validated_before_training", "jobs_validated": 2, "training_started": False}))
            with self.assertRaisesRegex(ValueError, "All six"):
                require_launch_validation(context)


class SeedRoutingTests(unittest.TestCase):
    def test_fixed_split_order_is_identical_for_every_training_seed(self):
        proteins, document = small_split()
        for seed in (2102, 2103, 2104):
            with patch.object(trainer, "SEED", seed), patch.object(trainer, "FOLD_SEED", 2101), patch.object(trainer, "make_cv_folds", side_effect=AssertionError("must never generate")):
                self.assertEqual(trainer.prepare_training_splits(proteins, {"sha256": TRAIN_HASH}, document), document["folds"])

    def test_missing_manifest_or_changed_fixed_identity_is_rejected(self):
        proteins, document = small_split()
        with patch.object(trainer, "make_cv_folds", side_effect=AssertionError("must never generate")):
            with self.assertRaisesRegex(ValueError, "regeneration"):
                trainer.prepare_training_splits(proteins, {"sha256": TRAIN_HASH}, None)
        changed = copy.deepcopy(document)
        changed["folds"][0]["train_indices"].reverse()
        with self.assertRaisesRegex(ValueError, "identity/order"):
            trainer.prepare_training_splits(proteins, {"sha256": TRAIN_HASH}, changed)
        with patch.object(trainer, "FOLD_SEED", 2102), self.assertRaises(ValueError):
            trainer.prepare_training_splits(proteins, {"sha256": TRAIN_HASH}, document)

    def test_training_seed_changes_rng_initialization_shuffle_and_dropout(self):
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        self.addCleanup(random.setstate, python_state)
        self.addCleanup(np.random.set_state, numpy_state)
        self.addCleanup(torch.set_rng_state, torch_state)
        def snapshot(seed):
            trainer.seed_everything(seed)
            weights = torch.nn.Linear(8, 4).weight.detach().clone()
            order = torch.cat(list(DataLoader(torch.arange(64), batch_size=8, shuffle=True)))
            dropped = torch.nn.Dropout(0.25)(torch.ones(128))
            return weights, order, dropped, random.random(), np.random.random(4)
        for seed in (2102, 2103, 2104):
            for first, second in zip(snapshot(seed), snapshot(seed)):
                np.testing.assert_array_equal(first, second)
        first, second = snapshot(2102), snapshot(2103)
        for index in (0, 1, 2):
            self.assertFalse(torch.equal(first[index], second[index]))

    def test_all_protected_definitions_and_training_loop_are_identical(self):
        self.assertEqual(assert_preserved_recipe(ROOT / "CROSS5FOLD_multi_test.py", ROOT / "CROSS5FOLD_multiseed_dev_v1.py"), 77)
        text = (ROOT / "CROSS5FOLD_multiseed_dev_v1.py").read_text()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "altered.py"
            path.write_text(text.replace("lr=lr, weight_decay=OPT_WEIGHT_DECAY", "lr=1.0, weight_decay=OPT_WEIGHT_DECAY"))
            with self.assertRaisesRegex(ValueError, "training loop changed"):
                assert_preserved_recipe(ROOT / "CROSS5FOLD_multi_test.py", path)

    def test_epoch_patience_average_count_and_optimizer_settings_are_protected(self):
        text = (ROOT / "CROSS5FOLD_multiseed_dev_v1.py").read_text()
        replacements = (("epochs = 30", "epochs = 31"), ("patience = 8", "patience = 9"),
                        ("TOP_K_CHECKPOINTS = 3", "TOP_K_CHECKPOINTS = 2"))
        with tempfile.TemporaryDirectory() as temp:
            for old, new in replacements:
                self.assertIn(old, text)
                path = Path(temp) / "altered.py"
                path.write_text(text.replace(old, new))
                with self.subTest(setting=old), self.assertRaisesRegex(ValueError, "assignment changed"):
                    assert_preserved_recipe(ROOT / "CROSS5FOLD_multi_test.py", path)

    def test_dry_run_terminates_before_output_creation_or_training_loop(self):
        tree = ast.parse((ROOT / "CROSS5FOLD_multiseed_dev_v1.py").read_text())
        main = tree.body[-1]
        dry = next(node for node in main.body if isinstance(node, ast.If) and
                   isinstance(node.test, ast.BoolOp) and any(isinstance(x, ast.Attribute) and x.attr == "dry_run" for x in node.test.values))
        self.assertTrue(any(isinstance(node, ast.Raise) for node in dry.body))
        mkdir = next(node for node in ast.walk(main) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "makedirs")
        self.assertLess(dry.end_lineno, mkdir.lineno)


class RegisteredRecipeTests(unittest.TestCase):
    def test_gate_must_bind_exact_sources_data_folds_and_all_strict_features(self):
        selection = {"selection_mode": "explicit", "datasets": {"Train335.pkl": {"sha256": "a" * 64}}, "seed": 2101}
        protocol = {"dataset_selection": selection}
        manifest = {"prerequisite_code_files": [{"sha256": "b" * 64}]}
        gate = {key: True for key in ("require_plm", "require_partner", "require_pair", "require_partner_encoder", "fixed_folds_validated")}
        gate.update(status="passed", errors=[], min_partner_encoder_coverage=1.0,
                    dataset_selection={k: v for k, v in selection.items() if k != "selection_mode"},
                    loaded_datasets=selection["datasets"], code_files=manifest["prerequisite_code_files"])
        validate_prerequisite_record(gate, protocol, manifest)
        for field, value in (("require_plm", False), ("require_partner_encoder", False),
                             ("min_partner_encoder_coverage", 0.95), ("fixed_folds_validated", False),
                             ("loaded_datasets", {}), ("code_files", [])):
            changed = dict(gate, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_prerequisite_record(changed, protocol, manifest)

    def test_reference_environment_changes_only_training_seed_arm_output(self):
        protocol, _ = read_protocol(ROOT / "research_review/multiseed_dev_v1/protocol.json")
        environments = [execution_environment(protocol, job) for job in protocol["seed_and_arm_order"]]
        for env in environments:
            self.assertEqual(env["PPI_SKIP_TEST_EVAL"], "1")
            self.assertEqual(env["PPI_TEST_SETS"], "Test287.pkl")
            self.assertEqual(env["PPI_MAX_FOLDS"], "5")
            self.assertEqual({k: v for k, v in env.items() if k not in {"PPI_SEED", "PPI_OUTPUT_DIR", "PPI_PARTNER_TRANSPORT"}},
                             {k: v for k, v in environments[0].items() if k not in {"PPI_SEED", "PPI_OUTPUT_DIR", "PPI_PARTNER_TRANSPORT"}})
        args = trainer_arguments(ROOT / "execution.json", protocol, protocol["seed_and_arm_order"][0])
        self.assertIn("--dry-run", args)
        self.assertEqual(args[args.index("--fold-seed") + 1], "2101")
        self.assertEqual(args[args.index("--training-seed") + 1], "2102")

    def test_run_and_checkpoint_embedded_binding_carries_both_seeds(self):
        context = {"job": {"training_seed": 2103}, "protocol": {"required_common_runtime": {"device": "cpu"},
            "dataset_selection": {"grouped_fold_manifest": {"sha256": "b" * 64}}},
            "manifest": {"cv_splits": [1]}, "execution_binding": {"source_files": [{"sha256": "c" * 64}]}}
        run = {"runtime": {"device": "cpu"}, "cv_splits": [1]}
        attach_execution(run, context)
        self.assertEqual(run["training_seed"], 2103)
        self.assertEqual(run["fold_seed"], FOLD_SEED)
        self.assertEqual(run["fold_manifest"]["sha256"], "b" * 64)
        self.assertEqual(run["execution_binding"], context["execution_binding"])


if __name__ == "__main__":
    unittest.main()
