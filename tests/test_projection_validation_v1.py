"""Small on-disk fixtures reject corrupted ablation identity and saved artifacts."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from compare_oof_development import score_metrics
from projection_initialization_v1 import capture_initialization, capture_rng_states, initialization_snapshot
from research_provenance import file_record
from validate_projection_dev_v1 import (
    BINDING_FIELDS, _state_schema, validate_arm, validate_pair_records,
)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


class ProjectionArtifactTests(unittest.TestCase):
    def setUp(self):
        from projection_protocol_v1 import execution_binding, trainer_arguments

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output, self.process = self.root / "output", self.root / "process"
        self.output.mkdir()
        self.process.mkdir()
        self.manifest_path = self.root / "manifest.json"
        write(self.manifest_path, {})
        dataset = self.root / "Train335.pkl"
        dataset.write_bytes(b"synthetic fixture; loading is injected")
        self.training = dict(file_record(dataset), sample_count=5)
        self.folds = [{"fold_index": index, "train_indices": [j for j in range(5) if j != index],
            "val_indices": [index]} for index in range(5)]
        fold_path = self.root / "folds.json"
        write(fold_path, {"folds": self.folds})
        self.selection = {"selection_mode": "explicit", "resolved_data_dir": str(self.root),
            "cohort_manifest": {"path": str(self.root / "cohort.json")},
            "grouped_fold_manifest": file_record(fold_path), "datasets": {"Train335": self.training}}
        self.protocol = {"dataset_selection": self.selection,
            "required_common_runtime": {"device": "cpu", "python": "accepted"},
            "reference_model_recipe": {"seed": 2101, "max_folds": 5,
                "skip_test_eval": True, "transport_sinkhorn_iters": 5},
            "sample_count": 5, "complex_count": 5, "residue_count": 10,
            "observed_references": {"historical_seed2101_comparison": {}}}
        self.jobs = [{"training_seed": seed, "fold_seed": 2101, "arm": arm,
            "partner_transport": True, "transport_sinkhorn_iters": 5 if arm == "projected" else 0,
            "max_folds": 5, "skip_test_eval": True, "output_dir": str(self.output)}
            for seed in (2102, 2103, 2104) for arm in ("projected", "rowsoftmax")]
        self.job = self.jobs[0]
        v2_jobs = [{"training_seed": seed, "arm": arm, "partner_transport": arm == "ot",
            "output_dir": str(self.root / f"previous_{seed}_{arm}")}
            for seed in (2102, 2103, 2104) for arm in ("ot", "noot")]
        v2_path = self.root / "v2.json"
        write(v2_path, {"jobs": v2_jobs})
        self.manifest = {"jobs": self.jobs, "protocol": {"path": "registered", "sha256": "fixed"},
            "source_files": [], "reference_source_files": [], "prerequisite_code_files": [],
            "process_dir": str(self.process), "cv_splits": self.folds}
        self.run = {"seed": 2102, "training_seed": 2102, "fold_seed": 2101,
            "fold_manifest": file_record(fold_path),
            "execution_binding": execution_binding(self.manifest_path, self.manifest, self.protocol),
            "projection_arm": "projected", "transport_sinkhorn_iters": 5,
            "cv_splits": self.folds, "code_files": [], "runtime": self.protocol["required_common_runtime"],
            "datasets": {"Train335": self.training}, "provenance_id": "unique-fixture",
            "created_utc": "prospective", "status": "complete", "completed_fold_indices": list(range(5)),
            "partner_transport": True, "skip_test_eval": True}
        dry_records = []
        for index, job in enumerate(v2_jobs):
            prior_run = {key: value for key, value in self.run.items()
                if key not in {"projection_arm", "transport_sinkhorn_iters"}}
            prior_run.update(seed=job["training_seed"], training_seed=job["training_seed"],
                partner_transport=job["partner_transport"], status="validated_before_training", completed_fold_indices=[])
            path = self.root / f"prior_dry_{index}.json"
            write(path, {**job, "fold_seed": 2101, "dry_run": True, "training_started": False,
                "skip_test_eval": True, "cv_splits": self.folds, "run_provenance": prior_run,
                "model_recipe": dict(self.protocol["reference_model_recipe"], seed=job["training_seed"])})
            dry_records.append(file_record(path))
        acceptance_path = self.root / "v2_acceptance.json"
        write(acceptance_path, {"execution_manifest": file_record(v2_path),
            "status": "validated_before_training", "jobs_validated": 6, "training_started": False,
            "dry_run_records": dry_records})
        self.protocol["accepted_ancestry"] = {"accepted_v2": file_record(v2_path),
            "accepted_v2_acceptance": file_record(acceptance_path)}
        self.summary = {"seed": 2102, "max_folds": 5, "skip_test_eval": True,
            **{key: self.run[key] for key in BINDING_FIELDS}, "run_provenance": self.run,
            "partner_transport": True, "val_mcc_weights": [.2] * 5, "val_aupr_weights": [.2] * 5,
            "single_fold_summary": {}, "ensemble": {"mean": {}, "rank": {}},
            "ensemble_threshold": .5, "rank_threshold": .5}
        self.labels = np.tile([0, 1], 5)
        self.probs = np.tile([.2, .8], 5)
        self.identity = {"sample_index": np.repeat(np.arange(5), 2), "residue_index": np.tile([0, 1], 5),
            "complex_id": np.repeat([f"C{i}" for i in range(5)], 2), "fold_index": np.repeat(np.arange(5), 2)}
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.0003)
        before = after = capture_rng_states()
        self.state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        self.initializations, entries = [], []
        for index in range(5):
            arguments = (index, 2102, 2101, "projected", 5, before, after)
            record = capture_initialization(model, optimizer, *arguments)
            snapshot = initialization_snapshot(model, optimizer, *arguments)
            json_path, snapshot_path = self.output / f"initialization_fold{index + 1}.json", self.output / f"initialization_fold{index + 1}.pt"
            write(json_path, record)
            torch.save(snapshot, snapshot_path)
            entries.append({"fold_index": index, "record": file_record(json_path), "snapshot": file_record(snapshot_path)})
            self.initializations.append(record)
        self.run["initialization_records"] = entries
        self.summary["initialization_records"] = entries
        approved_records = []
        for index, job in enumerate(self.jobs):
            initializations = copy.deepcopy(self.initializations)
            for record in initializations:
                record.update(training_seed=job["training_seed"], arm=job["arm"],
                    projection_iterations=job["transport_sinkhorn_iters"])
            path = self.root / f"projection_dry_{index}.json"
            write(path, {**job, "dry_run": True, "training_started": False,
                "cv_splits": self.folds, "initialization_records": initializations})
            approved_records.append(file_record(path))
        dry_validation_path = self.root / "dry_validation.json"
        write(dry_validation_path, {"execution_manifest": file_record(self.manifest_path),
            "status": "passed", "jobs_validated": 6, "training_started": False,
            "dry_run_records": approved_records})
        write(self.root / "manifest.acceptance.json", {"execution_manifest": file_record(self.manifest_path),
            "all_six_dry_runs": file_record(dry_validation_path)})
        self.save_run()
        self.save_oof()
        for number in range(1, 6):
            saved_run = dict(self.run, status="running", completed_fold_indices=list(range(number)),
                initialization_records=entries[:number])
            metrics = score_metrics(self.labels[:2], self.probs[:2], .5)
            checkpoint = {"fold_index": number - 1, "cv_split": self.folds[number - 1],
                "run_provenance": saved_run, "partner_transport": True, "skip_test_eval": True,
                "use_plm_features": True, "use_aux_plm_features": True,
                **{key: self.run[key] for key in BINDING_FIELDS},
                "initialization_record": entries[number - 1]["record"],
                "initialization_snapshot": entries[number - 1]["snapshot"],
                "threshold": .5, "val_mcc": metrics["mcc"], "val_auc_pr": metrics["auc_pr_trapezoidal"],
                "model_state": self.state}
            torch.save(checkpoint, self.output / f"fold{number}_best.pt")
        gate_path = self.process / "projected_2102_prerequisite_provenance.json"
        write(gate_path, {"require_plm": True, "require_partner": True, "require_pair": True,
            "require_partner_encoder": True, "fixed_folds_validated": True, "status": "passed", "errors": [],
            "min_partner_encoder_coverage": 1.0,
            "dataset_selection": {key: value for key, value in self.selection.items() if key != "selection_mode"},
            "loaded_datasets": self.selection["datasets"], "code_files": []})
        self.environment = {"PPI_SEED": "2102", "PPI_PARTNER_TRANSPORT": "1", "PPI_TRANSPORT_SINKHORN_ITERS": "5"}
        write(self.process / "projected_2102_launch.json", {"schema_version": 1,
            "component": "capacity_projection_dev_v1_launcher", "execution_manifest": file_record(self.manifest_path),
            "protocol": self.manifest["protocol"], "job": self.job, "environment": self.environment,
            "trainer_command": trainer_arguments(self.manifest_path, self.protocol, self.job, dry_run=False),
            "prerequisite_provenance": file_record(gate_path), "training_started": True,
            "skip_test_eval": True, "created_utc": "prospective"})

    def save_run(self):
        write(self.output / "run_provenance.json", self.run)
        self.summary["run_provenance"] = self.run
        write(self.output / "ensemble_summary.json", self.summary)

    def save_oof(self, **changes):
        data = {"labels": self.labels, "probs": self.probs, "threshold": np.array([.5]), **self.identity,
            "dataset_name": "Train335", "dataset_sha256": self.training["sha256"],
            "provenance_id": self.run["provenance_id"], "identity_schema_version": 1}
        data.update(changes)
        np.savez_compressed(self.output / "oof_predictions.npz", **data)

    def validate(self):
        with patch("projection_protocol_v1.verify_manifest", return_value=(self.manifest, self.protocol)), \
            patch("projection_protocol_v1.execution_environment", return_value=self.environment), \
            patch("validate_projection_dev_v1.load_dataset_with_provenance", return_value=([{}] * 5, self.training)), \
            patch("validate_projection_dev_v1.validate_fixed_folds", return_value=self.folds), \
            patch("validate_projection_dev_v1.expected_oof", return_value=(self.labels, self.identity)), \
            patch("validate_projection_dev_v1.reference_state_schema", return_value=(_state_schema(self.state), file_record(self.manifest_path))):
            return validate_arm(self.manifest_path, self.job)

    def pair(self):
        projected = self.validate()
        rowsoftmax = copy.deepcopy(projected)
        rowsoftmax["run"].update(projection_arm="rowsoftmax", transport_sinkhorn_iters=0, provenance_id="row-fixture")
        rowsoftmax["summary"].update(projection_arm="rowsoftmax", transport_sinkhorn_iters=0)
        rowsoftmax["job"]["output_dir"] = str(self.root / "rowsoftmax-output")
        for record in rowsoftmax["initializations"]:
            record.update(arm="rowsoftmax", projection_iterations=0)
        return projected, rowsoftmax

    def test_complete_artifacts_and_initialization_pair_pass(self):
        projected, rowsoftmax = self.pair()
        self.assertEqual(len(projected["artifacts"]["checkpoints"]), 5)
        self.assertEqual(len(projected["initializations"]), 5)
        self.assertEqual(len(validate_pair_records(projected, rowsoftmax)["folds"]), 5)

    def test_wrong_seed_folds_runtime_source_iterations_and_recipe_rejected(self):
        for key, bad in (("training_seed", 2103), ("fold_seed", 2102),
            ("fold_manifest", dict(self.run["fold_manifest"], sha256="regenerated")),
            ("runtime", {"device": "cuda"}), ("code_files", [{"sha256": "changed"}]),
            ("transport_sinkhorn_iters", 0), ("unregistered_recipe_field", .1)):
            original = copy.deepcopy(self.run)
            self.run[key] = bad
            self.save_run()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate()
            self.run.clear()
            self.run.update(original)
        self.save_run()

    def test_partial_folds_wrong_transport_and_wrong_arm_rejected(self):
        for key, bad in (("completed_fold_indices", [0, 1, 2, 3]), ("partner_transport", False),
            ("projection_arm", "rowsoftmax")):
            original = copy.deepcopy(self.run)
            self.run[key] = bad
            self.save_run()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate()
            self.run.clear()
            self.run.update(original)

    def test_external_metric_prediction_and_missing_initialization_rejected(self):
        self.summary["ensemble"]["mean"] = {"Test287": {"mcc": .8}}
        self.save_run()
        with self.assertRaises(ValueError):
            self.validate()
        self.summary["ensemble"]["mean"] = {}
        self.save_run()
        (self.output / "Test287_predictions.npz").write_bytes(b"forbidden")
        with self.assertRaises(ValueError):
            self.validate()
        (self.output / "Test287_predictions.npz").unlink()
        (self.output / "initialization_fold3.pt").unlink()
        with self.assertRaises(ValueError):
            self.validate()

    def test_partial_nonfinite_or_corrupt_oof_identity_rejected(self):
        for changes in ({"labels": self.labels[:-1], "probs": self.probs[:-1]},
            {"probs": np.full(10, np.nan)}, {"fold_index": self.identity["fold_index"][::-1]},
            {"complex_id": np.full(10, "missing")}, {"residue_index": np.full(10, 9)},
            {"sample_index": self.identity["sample_index"][::-1]}, {"threshold": np.array([np.inf])}):
            self.save_oof(**changes)
            with self.subTest(keys=list(changes)), self.assertRaises(ValueError):
                self.validate()

    def test_nonfinite_checkpoint_changed_schema_or_metric_rejected(self):
        path = self.output / "fold3_best.pt"
        original = torch.load(path, weights_only=False)
        for key, value in (("model_state", {"weight": torch.tensor([float("nan")])}),
            ("model_state", {"unexpected": torch.ones(3)}), ("val_mcc", .123), ("training_seed", 2104)):
            checkpoint = dict(original)
            checkpoint[key] = value
            torch.save(checkpoint, path)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate()

    def test_initialization_count_hash_and_checkpoint_prefix_rejected(self):
        original = copy.deepcopy(self.run["initialization_records"])
        self.run["initialization_records"] = original[:-1]
        self.summary["initialization_records"] = self.run["initialization_records"]
        self.save_run()
        with self.assertRaises(ValueError):
            self.validate()
        self.run["initialization_records"] = original
        self.summary["initialization_records"] = original
        self.save_run()
        path = self.output / "fold2_best.pt"
        checkpoint = torch.load(path, weights_only=False)
        checkpoint["run_provenance"]["initialization_records"] = original
        torch.save(checkpoint, path)
        with self.assertRaises(ValueError):
            self.validate()

    def test_initial_snapshot_tensor_mutation_rejected_even_with_new_file_hash(self):
        path = self.output / "initialization_fold2.pt"
        snapshot = torch.load(path, weights_only=False)
        snapshot["model_state"]["weight"].add_(1)
        torch.save(snapshot, path)
        self.run["initialization_records"][1]["snapshot"] = file_record(path)
        self.save_run()
        with self.assertRaises(ValueError):
            self.validate()

    def test_pair_first_fold_mismatch_and_schema_trainability_rejected(self):
        projected, rowsoftmax = self.pair()
        rowsoftmax["initializations"][0]["parameter_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_pair_records(projected, rowsoftmax)
        projected, rowsoftmax = self.pair()
        rowsoftmax["initializations"][2]["parameter_schema"][0]["requires_grad"] = False
        with self.assertRaises(ValueError):
            validate_pair_records(projected, rowsoftmax)

    def test_pair_later_divergence_recorded_without_reseeding(self):
        projected, rowsoftmax = self.pair()
        rowsoftmax["initializations"][2]["parameter_sha256"] = "0" * 64
        evidence = validate_pair_records(projected, rowsoftmax)
        self.assertFalse(evidence["folds"][2]["initial_values_equal"])
        self.assertTrue(evidence["no_fold_reseeding"])

    def test_accepted_dry_initialization_missing_or_corrupt_hash_rejected(self):
        from validate_projection_dev_v1 import accepted_initializations
        accepted, _ = accepted_initializations(self.manifest_path, self.manifest, self.job)
        self.assertEqual(accepted, self.initializations)
        path = self.root / "projection_dry_0.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["initialization_records"][0]["parameter_sha256"] = "0" * 64
        write(path, value)
        with self.assertRaises(ValueError):
            self.validate()

    def test_initialization_must_match_accepted_schema_optimizer_and_first_fold(self):
        from validate_projection_dev_v1 import validate_initializations
        approved = copy.deepcopy(self.initializations)
        approved[0]["parameter_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_initializations(self.output, self.run, self.job, _state_schema(self.state), approved)
        approved = copy.deepcopy(self.initializations)
        approved[2]["optimizer"]["groups"][0]["settings"]["lr"] = 1.0
        with self.assertRaises(ValueError):
            validate_initializations(self.output, self.run, self.job, _state_schema(self.state), approved)


if __name__ == "__main__":
    unittest.main()
