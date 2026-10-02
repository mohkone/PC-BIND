"""Small on-disk artifacts exercise fail-closed metadata/OOF/checkpoint validation."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from compare_oof_development import score_metrics
from research_provenance import file_record
from run_multiseed_dev_v1 import trainer_arguments
from validate_multiseed_dev_v3 import accepted_binding, validate_arm, _state_schema


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


class ArtifactValidatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "output"
        self.process = self.root / "process"
        self.output.mkdir()
        self.process.mkdir()
        self.manifest_path = self.root / "manifest.json"
        write(self.manifest_path, {})
        dataset = self.root / "Train335.pkl"
        dataset.write_bytes(b"synthetic fixture; loading is injected")
        self.training = dict(file_record(dataset), sample_count=5)
        folds_path = self.root / "folds.json"
        self.folds = [{"fold_index": i, "train_indices": [j for j in range(5) if j != i],
            "val_indices": [i]} for i in range(5)]
        write(folds_path, {"folds": self.folds})
        self.selection = {"selection_mode": "explicit", "resolved_data_dir": str(self.root),
            "cohort_manifest": {"path": str(self.root / "cohort.json")},
            "grouped_fold_manifest": file_record(folds_path), "datasets": {"Train335": self.training}}
        self.protocol = {"dataset_selection": self.selection, "required_common_runtime": {"device": "cpu", "python": "accepted"},
            "reference_model_recipe": {"seed": 2101, "max_folds": 5, "skip_test_eval": True},
            "sample_count": 5, "complex_count": 5, "residue_count": 10}
        self.jobs = [{"training_seed": seed, "arm": arm, "partner_transport": arm == "ot", "output_dir": str(self.output)}
            for seed in (2102, 2103, 2104) for arm in ("ot", "noot")]
        self.job = self.jobs[0]
        self.v2 = {"source_files": [], "reference_source_files": [], "prerequisite_code_files": [], "protocol": {"path": "registered", "sha256": "fixed"}}
        v2_path = self.root / "v2.json"
        write(v2_path, self.v2)
        self.manifest = {"jobs": self.jobs, "protocol": self.v2["protocol"], "accepted_v2": file_record(v2_path),
            "process_dir": str(self.process), "cv_splits": self.folds}
        binding = accepted_binding(self.manifest, self.protocol, self.v2)
        self.run = {"seed": 2102, "training_seed": 2102, "fold_seed": 2101,
            "fold_manifest": file_record(folds_path), "execution_binding": binding,
            "cv_splits": self.folds, "code_files": [], "runtime": self.protocol["required_common_runtime"],
            "datasets": {"Train335": self.training}, "provenance_id": "unique-fixture",
            "created_utc": "prospective", "status": "complete", "completed_fold_indices": list(range(5)),
            "partner_transport": True, "skip_test_eval": True}
        records = []
        for i, job in enumerate(self.jobs):
            run = dict(self.run, seed=job["training_seed"], training_seed=job["training_seed"],
                partner_transport=job["partner_transport"], status="validated_before_training", completed_fold_indices=[])
            path = self.root / f"dry_{i}.json"
            write(path, {**job, "fold_seed": 2101, "dry_run": True, "training_started": False,
                "skip_test_eval": True, "cv_splits": self.folds, "run_provenance": run,
                "model_recipe": dict(self.protocol["reference_model_recipe"], seed=job["training_seed"])})
            records.append(file_record(path))
        acceptance_path = self.root / "accepted.json"
        write(acceptance_path, {"execution_manifest": self.manifest["accepted_v2"], "status": "validated_before_training",
            "jobs_validated": 6, "training_started": False, "dry_run_records": records})
        self.manifest["accepted_v2_acceptance"] = file_record(acceptance_path)
        self.summary = {"seed": 2102, "max_folds": 5, "skip_test_eval": True,
            **{k: self.run[k] for k in ("training_seed", "fold_seed", "fold_manifest", "execution_binding")},
            "run_provenance": self.run, "partner_transport": True, "val_mcc_weights": [.2]*5,
            "val_aupr_weights": [.2]*5, "single_fold_summary": {}, "ensemble": {"mean": {}, "rank": {}},
            "ensemble_threshold": .5, "rank_threshold": .5}
        self.labels = np.tile([0, 1], 5)
        self.probs = np.tile([.2, .8], 5)
        self.identity = {"sample_index": np.repeat(np.arange(5), 2), "residue_index": np.tile([0, 1], 5),
            "complex_id": np.repeat([f"C{i}" for i in range(5)], 2), "fold_index": np.repeat(np.arange(5), 2)}
        self.save_run()
        self.save_oof()
        self.state = {"weight": torch.tensor([.25, .75])}
        for number in range(1, 6):
            saved_run = dict(self.run, status="running", completed_fold_indices=list(range(number)))
            score = score_metrics(self.labels[:2], self.probs[:2], .5)
            checkpoint = {"fold_index": number-1, "cv_split": self.folds[number-1], "run_provenance": saved_run,
                "partner_transport": True, "skip_test_eval": True, "use_plm_features": True, "use_aux_plm_features": True,
                **{k: self.run[k] for k in ("training_seed", "fold_seed", "fold_manifest", "execution_binding")},
                "threshold": .5, "val_mcc": score["mcc"], "val_auc_pr": score["auc_pr_trapezoidal"], "model_state": self.state}
            torch.save(checkpoint, self.output / f"fold{number}_best.pt")
        gate_path = self.process / "ot_2102_prerequisite_provenance.json"
        write(gate_path, {"require_plm": True, "require_partner": True, "require_pair": True,
            "require_partner_encoder": True, "fixed_folds_validated": True, "status": "passed", "errors": [],
            "min_partner_encoder_coverage": 1.0,
            "dataset_selection": {k: v for k, v in self.selection.items() if k != "selection_mode"},
            "loaded_datasets": self.selection["datasets"], "code_files": []})
        self.environment = {"PPI_SEED": "2102", "PPI_PARTNER_TRANSPORT": "1"}
        write(self.process / "ot_2102_launch.json", {"schema_version": 1, "component": "multiseed_dev_v3_launcher",
            "execution_manifest_v3": file_record(self.manifest_path), "accepted_v2": self.manifest["accepted_v2"],
            "protocol": self.manifest["protocol"], "job": self.job, "environment": self.environment,
            "trainer_command": trainer_arguments(self.manifest["accepted_v2"]["path"], self.protocol, self.job)[:-1],
            "prerequisite_provenance": file_record(gate_path), "training_started": True,
            "skip_test_eval": True, "created_utc": "prospective"})

    def save_run(self):
        write(self.output / "run_provenance.json", self.run)
        self.summary["run_provenance"] = self.run
        write(self.output / "ensemble_summary.json", self.summary)

    def save_oof(self, **changes):
        data = {"labels": self.labels, "probs": self.probs, "threshold": np.array([.5]), **self.identity,
            "dataset_name": "Train335", "dataset_sha256": self.training["sha256"], "provenance_id": self.run["provenance_id"],
            "identity_schema_version": 1}
        data.update(changes)
        np.savez_compressed(self.output / "oof_predictions.npz", **data)

    def validate(self):
        with patch("multiseed_infrastructure_v3.verify_manifest", return_value=(self.manifest, self.protocol, self.v2)), \
                patch("validate_multiseed_dev_v3.execution_environment", return_value=self.environment), \
                patch("validate_multiseed_dev_v3.load_dataset_with_provenance", return_value=([{}]*5, self.training)), \
                patch("validate_multiseed_dev_v3.validate_fixed_folds", return_value=self.folds), \
                patch("validate_multiseed_dev_v3.expected_oof", return_value=(self.labels, self.identity)), \
                patch("validate_multiseed_dev_v3.reference_state_schema", return_value=(_state_schema(self.state), file_record(self.manifest_path))):
            return validate_arm(self.manifest_path, self.job)

    def test_complete_identity_and_checkpoints_pass(self):
        result = self.validate()
        np.testing.assert_array_equal(result["labels"], self.labels)
        self.assertEqual(len(result["artifacts"]["checkpoints"]), 5)

    def test_wrong_training_seed_rejected(self):
        self.run["training_seed"] = 2103
        self.save_run()
        with self.assertRaises(ValueError):
            self.validate()

    def test_wrong_fold_seed_or_manifest_rejected(self):
        self.run["fold_seed"] = 2102
        self.save_run()
        with self.assertRaises(ValueError):
            self.validate()
        self.run["fold_seed"] = 2101
        self.run["fold_manifest"] = dict(self.run["fold_manifest"], sha256="regenerated")
        self.save_run()
        with self.assertRaises(ValueError):
            self.validate()

    def test_runtime_and_source_mismatch_rejected(self):
        for key, bad in (("runtime", {"device": "cuda"}), ("code_files", [{"sha256": "different"}])):
            original = copy.deepcopy(self.run[key])
            self.run[key] = bad
            self.save_run()
            with self.assertRaises(ValueError):
                self.validate()
            self.run[key] = original

    def test_partial_folds_and_wrong_transport_rejected(self):
        self.run["completed_fold_indices"] = [0, 1, 2, 3]
        self.save_run()
        with self.assertRaises(ValueError):
            self.validate()
        self.run["completed_fold_indices"] = list(range(5))
        self.run["partner_transport"] = False
        self.save_run()
        with self.assertRaises(ValueError):
            self.validate()

    def test_external_metrics_and_prediction_files_rejected(self):
        self.summary["ensemble"]["mean"] = {"Test287": {"mcc": .8}}
        self.save_run()
        with self.assertRaises(ValueError):
            self.validate()
        self.summary["ensemble"]["mean"] = {}
        self.save_run()
        (self.output / "Test287_predictions.npz").write_bytes(b"unexpected")
        with self.assertRaises(ValueError):
            self.validate()

    def test_partial_nonfinite_and_reordered_oof_rejected(self):
        for changes in ({"labels": self.labels[:-1], "probs": self.probs[:-1]},
            {"probs": np.full(10, np.nan)}, {"fold_index": self.identity["fold_index"][::-1]},
            {"sample_index": self.identity["sample_index"][::-1]}, {"threshold": np.array([np.inf])}):
            self.save_oof(**changes)
            with self.assertRaises(ValueError):
                self.validate()

    def test_nonfinite_checkpoint_and_changed_model_schema_rejected(self):
        path = self.output / "fold3_best.pt"
        checkpoint = torch.load(path, weights_only=False)
        checkpoint["model_state"] = {"weight": torch.tensor([float("nan"), .75])}
        torch.save(checkpoint, path)
        with self.assertRaises(ValueError):
            self.validate()
        checkpoint["model_state"] = {"unexpected_branch": torch.ones(3)}
        torch.save(checkpoint, path)
        with self.assertRaises(ValueError):
            self.validate()

    def test_checkpoint_seed_and_metrics_rejected(self):
        path = self.output / "fold2_best.pt"
        checkpoint = torch.load(path, weights_only=False)
        checkpoint["training_seed"] = 2104
        torch.save(checkpoint, path)
        with self.assertRaises(ValueError):
            self.validate()
        checkpoint["training_seed"] = 2102
        checkpoint["val_mcc"] = .123
        torch.save(checkpoint, path)
        with self.assertRaises(ValueError):
            self.validate()


if __name__ == "__main__":
    unittest.main()
