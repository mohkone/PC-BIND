"""Regression tests for prediction alignment and statistical estimands."""

import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import compare_pcbind_primary as comparison
import metrics
import statistical_ablation_tests as ablations
from research_provenance import file_record, prediction_identity


class MetricDefinitionTests(unittest.TestCase):
    def test_trapezoidal_auc_is_preserved_for_tied_scores(self):
        # Balanced labels with a constant score have AP=0.5 but trapezoidal PR AUC=0.75.
        self.assertAlmostEqual(metrics.compute_auc_pr([0, 1], [0.5, 0.5]), 0.75)
        self.assertAlmostEqual(metrics.compute_auc_roc([0, 1], [0.5, 0.5]), 0.5)

    def test_no_positive_auc_is_undefined(self):
        self.assertTrue(np.isnan(metrics.compute_auc_pr([0, 0], [0.2, 0.3])))
        self.assertTrue(np.isnan(metrics.compute_auc_roc([1, 1], [0.2, 0.3])))

    def test_macro_metrics_explicitly_include_both_binary_classes(self):
        _, recall, precision, f1, sensitivity, specificity = metrics.compute_performance(
            [0, 0, 1, 1], [0, 0, 0, 1]
        )
        self.assertAlmostEqual(recall, 0.75)
        self.assertAlmostEqual(precision, (2 / 3 + 1) / 2)
        self.assertAlmostEqual(f1, (0.8 + 2 / 3) / 2)
        self.assertEqual((sensitivity, specificity), (0.5, 1.0))

    def test_invalid_metric_inputs_are_rejected(self):
        for labels, scores in [([0, 0.5], [0.1, 0.9]), ([0, 1], [np.nan, 0.9]),
                               ([0, 1], [0.1]), ([], [])]:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                metrics.compute_auc_pr(labels, scores)


class AlignmentAndBootstrapTests(unittest.TestCase):
    def write_dataset(self, directory, samples):
        path = Path(directory) / "TestTiny.pkl"
        with path.open("wb") as handle:
            pickle.dump(samples, handle)
        return path

    @staticmethod
    def sample(code, labels):
        return {"complex_code": code, "label": np.asarray(labels),
                "residue_graph_node": np.zeros((len(labels), 2))}

    def test_shared_complex_chains_are_one_resampling_unit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_dataset(directory, [self.sample(" abcd ", [0, 1]),
                                                  self.sample("other", [1]),
                                                  self.sample("ABCD", [0])])
            groups = comparison.complex_indices(path, 4, np.array([0, 1, 1, 0]))
            self.assertEqual([group.tolist() for group in groups], [[0, 1, 3], [2]])

    def test_same_length_wrong_dataset_order_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_dataset(directory, [self.sample("ABCD", [0, 1, 0, 1])])
            with self.assertRaisesRegex(ValueError, "dataset order"):
                comparison.complex_indices(path, 4, np.array([1, 0, 1, 0]))

    def test_unknown_complex_identifier_cannot_become_independent_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            for code in [None, "", "   "]:
                path = self.write_dataset(directory, [self.sample(code, [0, 1])])
                with self.subTest(code=code), self.assertRaisesRegex(ValueError, "complex_code"):
                    comparison.complex_indices(path, 2, np.array([0, 1]))

    def test_fractional_labels_are_not_silently_cast_to_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            np.savez(Path(directory) / "ensemble_TestTiny_predictions.npz",
                     labels=[0.5, 1], probs=[0.1, 0.9], threshold=[0.5])
            with self.assertRaisesRegex(ValueError, "only 0 and 1"):
                comparison.load_predictions(directory, "TestTiny")

    def test_bootstrap_uses_the_same_sample_for_both_models(self):
        labels = np.array([0, 1, 0, 1])
        scores = np.array([0.2, 0.8, 0.7, 0.3])
        result = comparison.cluster_bootstrap(labels, scores, scores,
                                             [np.array([0, 1]), np.array([2, 3])], 100, 7)
        self.assertEqual((result["mean"], result["ci_low"], result["ci_high"]), (0, 0, 0))
        self.assertEqual(result["valid_replicates"], 100)

    def test_overlapping_or_incomplete_groups_are_rejected(self):
        labels = np.array([0, 1, 0, 1])
        scores = np.array([0.2, 0.8, 0.7, 0.3])
        for groups in [[np.array([0, 1]), np.array([1, 2, 3])], [np.array([0, 1])]]:
            with self.assertRaisesRegex(ValueError, "partition"):
                comparison.cluster_bootstrap(labels, scores, scores, groups, 20, 7)

    def test_degenerate_resamples_are_counted(self):
        labels = np.array([0, 1])
        result = comparison.cluster_bootstrap(labels, np.array([0.8, 0.2]),
                                             np.array([0.1, 0.9]),
                                             [np.array([0]), np.array([1])], 100, 7)
        self.assertGreater(result["invalid_no_positive_replicates"], 0)
        self.assertEqual(result["valid_replicates"] + result["invalid_no_positive_replicates"], 100)


class StatisticalEstimandTests(unittest.TestCase):
    def test_fold_mean_is_not_substituted_for_ensemble_metric(self):
        rows = [{"test_set": "Test60", "group": "fold1", "auc_pr": "0.4"}]
        self.assertIsNone(ablations.auc_pr_for_test(rows, "Test60"))

    def test_duplicate_ensemble_metrics_are_rejected(self):
        row = {"test_set": "Test60", "group": "ensemble_mean", "auc_pr": "0.4"}
        with self.assertRaisesRegex(ValueError, "exactly one"):
            ablations.auc_pr_for_test([row, row], "Test60")

    def test_four_benchmark_exact_test_cannot_reach_point_zero_five(self):
        p, _, count = ablations.wilcoxon_exact_two_sided([0.1, 0.2, 0.3, 0.4])
        self.assertEqual((p, count), (0.125, 4))
        self.assertEqual(ablations.wilcoxon_exact_two_sided([1, 1, -1, -1])[0], 1)
        self.assertEqual(ablations.wilcoxon_exact_two_sided([0, 0, 0, 0])[0], 1)

    def test_holm_adjustment_is_monotone_and_preserves_original_order(self):
        np.testing.assert_allclose(ablations.holm_adjust([0.04, 0.01, 0.03]), [0.06, 0.03, 0.06])

    def test_empty_and_nonfinite_differences_are_rejected(self):
        for differences in [[], [0.1, np.nan], [np.inf]]:
            with self.subTest(differences=differences), self.assertRaises(ValueError):
                ablations.paired_bootstrap_ci(differences)


class PredictionIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.samples = [AlignmentAndBootstrapTests.sample("A", [0, 1]),
                        AlignmentAndBootstrapTests.sample("B", [0, 1])]
        self.dataset = self.root / "TestTiny.pkl"
        with self.dataset.open("wb") as handle:
            pickle.dump(self.samples, handle)
        self.identity = dict(prediction_identity(self.samples), dataset_name=np.asarray("TestTiny"),
                             dataset_sha256=np.asarray(file_record(self.dataset)["sha256"]),
                             provenance_id=np.asarray("control-run"), identity_schema_version=np.asarray(1))

    def archive(self, role="control", identity=None):
        directory = self.root / role
        directory.mkdir(exist_ok=True)
        np.savez(directory / "ensemble_TestTiny_predictions.npz",
                 labels=[0, 1, 0, 1], probs=[0.2, 0.8, 0.4, 0.6], threshold=[0.5],
                 **(self.identity if identity is None else identity))
        return directory

    def manifest(self, run_id, swapped=False, training_hash="a" * 64):
        folds = [[0, 2], [1, 3]] if swapped else [[0, 1], [2, 3]]
        splits = []
        for fold, indices in enumerate(folds):
            split = {"fold_index": fold}
            for kind, members in [("val", indices), ("train", folds[1 - fold])]:
                split[f"{kind}_indices"] = members
                split[f"{kind}_complex_ids"] = [f"complex{i}" for i in members]
                split[f"{kind}_group_ids"] = [f"COMPLEX{i}" for i in members]
            splits.append(split)
        return {"schema_version": 1, "provenance_id": run_id, "status": "complete",
                "grouped_cv": True, "datasets": {
                    "Train335": {"sha256": training_hash, "sample_count": 4},
                    "TestTiny": file_record(self.dataset)}, "cv_splits": splits,
                "scheduled_fold_indices": [0, 1], "completed_fold_indices": [0, 1]}

    def test_partial_identity_is_rejected_without_changing_prediction_api(self):
        run = self.archive(identity={"sample_index": self.identity["sample_index"]})
        with self.assertRaisesRegex(ValueError, "partial prediction identity"):
            comparison.load_predictions(run, "TestTiny")
        self.archive()
        self.assertEqual(len(comparison.load_predictions(run, "TestTiny")), 3)

    def test_identical_labels_do_not_hide_permuted_residue_identity(self):
        permuted = dict(self.identity)
        permuted["sample_index"] = np.array([1, 1, 0, 0])
        permuted["complex_id"] = np.array(["B", "B", "A", "A"])
        run = self.archive(identity=permuted)
        identity = comparison.load_prediction_identity(run, "TestTiny")
        with self.assertRaisesRegex(ValueError, "actual dataset residue order"):
            comparison.complex_indices(self.dataset, 4, np.array([0, 1, 0, 1]), {"control": identity})

    def test_same_labels_and_ids_do_not_hide_changed_dataset_features(self):
        run = self.archive()
        identity = comparison.load_prediction_identity(run, "TestTiny")
        self.samples[0]["residue_graph_node"][0, 0] = 12
        with self.dataset.open("wb") as handle:
            pickle.dump(self.samples, handle)
        with self.assertRaisesRegex(ValueError, "dataset hash"):
            comparison.complex_indices(self.dataset, 4, np.array([0, 1, 0, 1]), {"control": identity})

    def test_independent_run_ids_bind_to_matching_manifests(self):
        identities = {}
        for role in ("control", "model"):
            values = dict(self.identity, provenance_id=np.asarray(f"{role}-run"))
            run = self.archive(role, values)
            (run / "run_provenance.json").write_text(json.dumps(self.manifest(f"{role}-run")), encoding="utf-8")
            identities[role] = comparison.load_prediction_identity(run, "TestTiny")
        provenance = comparison.run_provenance(self.root / "control", self.root / "model")
        comparison.complex_indices(self.dataset, 4, np.array([0, 1, 0, 1]), identities)
        status = comparison.verify_prediction_sources(identities, provenance, "TestTiny")
        self.assertTrue(all("bound to saved run manifest" in value for value in status.values()))
        self.assertIn("matching recorded", provenance["training_pairing"])
        identities["model"]["provenance_id"] = "unrelated-run"
        with self.assertRaisesRegex(ValueError, "provenance_id"):
            comparison.verify_prediction_sources(identities, provenance, "TestTiny")

    def test_recorded_training_hash_and_fold_mismatches_are_rejected(self):
        for changed in ("hash", "folds"):
            for role in ("control", "model"):
                run = self.archive(role)
                manifest = self.manifest(role, swapped=role == "model" and changed == "folds",
                                         training_hash="b" * 64 if role == "model" and changed == "hash" else "a" * 64)
                (run / "run_provenance.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "differ"):
                comparison.run_provenance(self.root / "control", self.root / "model")


class ComparisonReportTests(unittest.TestCase):
    def test_cli_reports_actual_scope_and_protocol_differences(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for role, plm_mode in [("control", "disabled"), ("model", "esm2")]:
                run = root / role
                run.mkdir()
                np.savez(run / "ensemble_TestTiny_predictions.npz",
                         labels=[0, 1, 0, 1], probs=[0.1, 0.9, 0.7, 0.4], threshold=[0.5])
                (run / "ensemble_summary.json").write_text(json.dumps(
                    {"seed": 2101, "val_mcc_weights": [1.0], "plm_feature_mode": plm_mode}
                ), encoding="utf-8")
            with (root / "TestTiny.pkl").open("wb") as handle:
                pickle.dump([AlignmentAndBootstrapTests.sample("one", [0, 1]),
                             AlignmentAndBootstrapTests.sample("two", [0, 1])], handle)
            result = subprocess.run([sys.executable, str(ROOT / "compare_pcbind_primary.py"),
                                     "--control-dir", str(root / "control"), "--model-dir", str(root / "model"),
                                     "--data-dir", str(root), "--tests", "TestTiny", "--bootstrap", "20",
                                     "--output-json", str(root / "report.json"),
                                     "--output-md", str(root / "report.md")],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root / "report.json").read_text(encoding="utf-8"))
            self.assertIn("plm_feature_mode", report["provenance"]["protocol_differences"])
            self.assertIn("legacy:", report["tests"]["TestTiny"]["identity_verification"]["model"])
            markdown = (root / "report.md").read_text(encoding="utf-8")
            self.assertIn("Across 1 selected tests", markdown)
            self.assertIn("1 checkpoint weights", markdown)
            self.assertNotIn("identical grouped five-fold splits", markdown)


if __name__ == "__main__":
    unittest.main()
