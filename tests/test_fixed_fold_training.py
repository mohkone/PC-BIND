"""Fixed-cohort training must consume recorded splits without regeneration."""
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

import CROSS5FOLD_multi_test as training


class FixedFoldTrainingTests(unittest.TestCase):
    def setUp(self):
        self.saved = [
            {"fold_index": 0, "train_indices": [3, 1], "val_indices": [2, 0]},
            {"fold_index": 1, "train_indices": [0, 2], "val_indices": [1, 3]},
        ]
        self.source = {"path": str(Path("selected/Train335.pkl").resolve()),
                       "sha256": "a" * 64, "size_bytes": 1200, "sample_count": 4}
        self.proteins = [{"complex_code": str(i)} for i in range(4)]

    def test_fixed_folds_never_call_the_random_split_generator(self):
        validator = mock.Mock(return_value=self.saved)
        module = types.SimpleNamespace(validate_fixed_folds=validator)
        manifest = {"seed": 2101, "folds": self.saved}
        with mock.patch.dict(sys.modules, {"filtered_cohort_protocol": module}), \
                mock.patch.object(training, "make_cv_folds", side_effect=AssertionError("RNG split called")), \
                mock.patch.object(training, "SEED", 2101), \
                mock.patch.object(training, "CV_GROUP_KEY", "complex_code"):
            result = training.prepare_training_splits(self.proteins, self.source, manifest)
        self.assertIs(result, self.saved)
        validator.assert_called_once_with(self.proteins, manifest, 2101, "a" * 64, "complex_code")

    def test_training_loop_indices_preserve_saved_training_and_validation_order(self):
        train_indices, val_indices = training.training_split_indices(self.saved[0])
        np.testing.assert_array_equal(train_indices, [3, 1])
        np.testing.assert_array_equal(val_indices, [2, 0])
        self.assertNotEqual(train_indices.tolist(), self.saved[1]["val_indices"])

    def test_validator_failure_propagates_without_regenerating_folds(self):
        module = types.SimpleNamespace(validate_fixed_folds=mock.Mock(side_effect=ValueError("tampered manifest")))
        with mock.patch.dict(sys.modules, {"filtered_cohort_protocol": module}), \
                mock.patch.object(training, "make_cv_folds", side_effect=AssertionError("RNG split called")):
            with self.assertRaisesRegex(ValueError, "tampered manifest"):
                training.prepare_training_splits(self.proteins, self.source, {"folds": self.saved})

    def test_loaded_training_and_test_hashes_and_counts_must_match_registry(self):
        selection = {"hashes_verified": True, "datasets": {
            "Train335.pkl": self.source,
            "Test287.pkl": {"path": str(Path("selected/Test287.pkl").resolve()),
                            "sha256": "b" * 64, "size_bytes": 600, "sample_count": 2},
        }}
        valid_sources = {"Train335": dict(self.source), "Test287": dict(selection["datasets"]["Test287.pkl"])}
        training.verify_loaded_cohort_sources(valid_sources, selection)
        for dataset, field, value in [("Train335", "sha256", "c" * 64),
                                       ("Test287", "sha256", "d" * 64),
                                       ("Train335", "sample_count", 5),
                                       ("Train335", "size_bytes", 1201),
                                       ("Train335", "path", str(Path("other/Train335.pkl").resolve()))]:
            altered = {name: dict(source) for name, source in valid_sources.items()}
            altered[dataset][field] = value
            with self.subTest(dataset=dataset, field=field), self.assertRaisesRegex(ValueError, "verified cohort"):
                training.verify_loaded_cohort_sources(altered, selection)

    def test_undeclared_external_dataset_cannot_join_the_filtered_comparison(self):
        selection = {"hashes_verified": True, "datasets": {"Train335.pkl": self.source}}
        with self.assertRaisesRegex(ValueError, "not declared"):
            training.verify_loaded_cohort_sources({"Test70": self.source}, selection)

    def test_fixed_cohort_provenance_includes_validator_and_dedicated_launchers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in ("run_filtered_ot.ps1", "filtered_common.ps1", "unrelated.ps1"):
                (root / filename).touch()
            included = {Path(path).name for path in training.training_provenance_code_paths(root, fixed_cohort=True)}
            self.assertTrue({"filtered_cohort_protocol.py", "run_filtered_ot.ps1", "filtered_common.ps1"}.issubset(included))
            self.assertNotIn("unrelated.ps1", included)


class TrainingCLIConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(training, "EXPLICIT_DATA_DIR", "stale-environment-directory").start()
        mock.patch.object(training, "GEO_DATA_DIR", "stale-environment-directory").start()
        mock.patch.object(training, "GROUPED_CV", False).start()

    def test_cli_directory_overrides_environment_compatibility_state(self):
        with mock.patch.dict(os.environ, {"PPI_DATA_DIR": "unrelated-data"}):
            selection, folds = training.configure_training_data(self.directory)
        self.assertIsNone(folds)
        self.assertEqual(training.EXPLICIT_DATA_DIR, str(self.directory))
        self.assertEqual(training.GEO_DATA_DIR, str(self.directory))
        self.assertEqual(selection["resolved_data_dir"], str(self.directory))

    def test_detected_cohort_manifest_requires_explicit_fixed_fold_selection(self):
        (self.directory / "cohort_manifest.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "--fold-manifest"):
            training.configure_training_data(self.directory)

    def test_selected_manifests_are_verified_and_effective_grouping_is_recorded(self):
        cohort_path = self.directory / "cohort_manifest.json"
        fold_path = self.directory / "grouped_folds_seed2101.json"
        cohort_path.write_text(json.dumps({"analysis_label": "filtered-cohort analysis"}), encoding="utf-8")
        verified = {"resolved_data_dir": str(self.directory), "analysis_label": "filtered-cohort analysis",
                    "hashes_verified": True, "datasets": {"Train335.pkl": {"sha256": "a" * 64},
                                                              "Test287.pkl": {"sha256": "b" * 64}}}
        folds = {"folds": []}
        verifier = mock.Mock(return_value=(verified, {}, folds))
        with mock.patch.dict(sys.modules, {"filtered_cohort_protocol": types.SimpleNamespace(verify_filtered_cohort=verifier)}):
            selection, returned_folds = training.configure_training_data(self.directory, cohort_path, fold_path)
        verifier.assert_called_once_with(self.directory, cohort_path, fold_path, training.SEED)
        self.assertIs(returned_folds, folds)
        self.assertTrue(training.GROUPED_CV)
        self.assertEqual(selection["selection_mode"], "explicit")
        self.assertEqual(selection["datasets"], verified["datasets"])


if __name__ == "__main__":
    unittest.main()
