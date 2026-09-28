"""Explicit cohort selection must not borrow files from another dataset."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import CROSS5FOLD_multi_test as training


ROOT = Path(__file__).resolve().parents[1]


class DatasetSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.selected = self.root / "cohort v1"
        self.selected.mkdir()
        self.legacy = self.root / "data" / "geo"
        self.legacy.mkdir(parents=True)
        # Empty placeholders exercise path selection only; no dataset is serialized.
        for filename in ("Train335.pkl", "Test287.pkl", "Test70.pkl"):
            (self.legacy / filename).touch()
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(training, "EXPLICIT_DATA_DIR", str(self.selected)).start()
        mock.patch.object(training, "GEO_DATA_DIR", str(self.legacy)).start()

    def test_explicit_training_file_cannot_fall_back_to_legacy(self):
        with self.assertRaisesRegex(FileNotFoundError, "fallback.*disabled"):
            training.dataset_path("Train335.pkl")
        (self.selected / "Train335.pkl").touch()
        self.assertEqual(training.dataset_path("Train335.pkl"), str((self.selected / "Train335.pkl").resolve()))

    def test_missing_requested_test_is_not_silently_skipped_or_borrowed(self):
        (self.selected / "Train335.pkl").touch()
        with mock.patch.object(training, "TEST_PKL_FILES", ["Test287.pkl"]):
            for require_geo in (True, False):
                with self.subTest(require_geo=require_geo), self.assertRaises(FileNotFoundError):
                    training.available_test_datasets(require_geo=require_geo)

    def test_only_explicitly_requested_tests_are_required(self):
        (self.selected / "Test287.pkl").touch()
        with mock.patch.object(training, "TEST_PKL_FILES", ["Test287.pkl"]):
            self.assertEqual(training.available_test_datasets(),
                             [("Test287", str((self.selected / "Test287.pkl").resolve()))])

    def test_filenames_cannot_escape_the_selected_directory(self):
        for filename in ("../Train335.pkl", "..\\Train335.pkl", str(self.legacy / "Train335.pkl")):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                training.dataset_path(filename)

    def test_unconfigured_lookup_preserves_geo_and_raw_data_behavior(self):
        with mock.patch.object(training, "EXPLICIT_DATA_DIR", None):
            self.assertEqual(training.dataset_path("Train335.pkl"), str(self.legacy / "Train335.pkl"))
            with mock.patch.object(training, "find_existing_file", side_effect=lambda directory, filename:
                                   "data/Train335.pkl" if directory == "data" else None):
                self.assertEqual(training.dataset_path("Train335.pkl"), "data/Train335.pkl")

    def test_manifest_label_and_content_identity_are_recorded(self):
        manifest = self.selected / "cohort_manifest.json"
        manifest.write_text(json.dumps({"analysis_label": "filtered-cohort analysis"}), encoding="utf-8")
        result = training.dataset_selection_provenance(self.selected / "Train335.pkl")
        self.assertEqual(result["selection_mode"], "explicit")
        self.assertEqual(result["resolved_data_dir"], str(self.selected.resolve()))
        self.assertEqual(result["analysis_label"], "filtered-cohort analysis")
        self.assertEqual(result["cohort_manifest"]["path"], str(manifest.resolve()))
        self.assertEqual(len(result["cohort_manifest"]["sha256"]), 64)


@unittest.skipUnless(os.name == "nt", "PowerShell runner integration is Windows-specific")
class RunnerDatasetSelectionTests(unittest.TestCase):
    def run_script(self, body, cwd):
        shell = shutil.which("pwsh") or shutil.which("powershell")
        if shell is None:
            self.skipTest("PowerShell is unavailable")
        return subprocess.run([shell, "-NoProfile", "-NonInteractive", "-Command", body],
                              cwd=cwd, text=True, capture_output=True)

    def test_runner_resolves_caller_relative_cohort_and_restores_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            caller = Path(directory)
            cohort = caller / "cohort v1"
            cohort.mkdir()
            (cohort / "Train335.pkl").touch()
            script = str(ROOT / "run_seed.ps1").replace("'", "''")
            result = self.run_script(
                "$ErrorActionPreference='Stop'; $env:PPI_DATA_DIR='old-selection'; "
                f"$config = & '{script}' -DataDir 'cohort v1' -SkipTestEval 1 -DryRun 6>$null | ConvertFrom-Json; "
                "[PSCustomObject]@{selected=$config.PPI_DATA_DIR; restored=$env:PPI_DATA_DIR; location=(Get-Location).Path} | ConvertTo-Json",
                caller)
            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads(result.stdout)
            self.assertEqual(Path(config["selected"]).resolve(), cohort.resolve())
            self.assertEqual(config["restored"], "old-selection")
            self.assertEqual(Path(config["location"]).resolve(), caller.resolve())

    def test_runner_preflight_rejects_missing_requested_test(self):
        with tempfile.TemporaryDirectory() as directory:
            caller = Path(directory)
            (caller / "Train335.pkl").touch()
            script = str(ROOT / "run_seed.ps1").replace("'", "''")
            result = self.run_script(
                f"& '{script}' -DataDir '.' -TestSets 'Test287.pkl' -DryRun 6>$null", caller)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Test287.pkl", result.stderr)
            self.assertIn("fallback", result.stderr)


if __name__ == "__main__":
    unittest.main()
