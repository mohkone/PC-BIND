import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from compare_oof_development import expected_oof, load_oof, paired_complex_bootstrap, validate_checkpoint
from frozen_development_protocol import cohort_selection, run_configuration, summary_configuration, validate_run_metadata
from research_provenance import prediction_source
from run_frozen_development import run_queue


def metadata_fixture():
    run = {
        "schema_version": 1, "provenance_id": "synthetic-run", "created_utc": "synthetic",
        "status": "complete", "partner_transport": True, "seed": 2101,
        "scheduled_fold_indices": list(range(5)), "completed_fold_indices": list(range(5)),
        "datasets": {"Train335": {"sha256": "a" * 64}}, "skip_test_eval": True,
        "runtime": {"device": "cpu"}, "code_files": [{"sha256": "b" * 64}],
        "cv_splits": [{"fold_index": n} for n in range(5)],
    }
    summary = {
        "run_provenance": copy.deepcopy(run), "partner_transport": True,
        "max_folds": 5, "skip_test_eval": True, "plm_feature_dim": 480,
        "val_mcc_weights": [0.2] * 5, "val_aupr_weights": [0.2] * 5,
        "single_fold_summary": {}, "ensemble": {"mean": {}, "rank": {}},
    }
    freeze = {"run_configuration": run_configuration(run),
              "summary_configuration": summary_configuration(summary)}
    return run, summary, freeze


class FrozenMetadataTests(unittest.TestCase):
    def test_shared_cohort_record_requires_explicit_training_selection(self):
        self.assertEqual(cohort_selection({"selection_mode": "explicit", "seed": 2101}), {"seed": 2101})
        with self.assertRaises(ValueError):
            cohort_selection({"seed": 2101})

    def test_complete_matches_and_selection_outcomes_may_differ(self):
        run, summary, freeze = metadata_fixture()
        summary["ensemble_threshold"] = 0.7
        validate_run_metadata(run, summary, freeze, True)

    def test_partial_or_running_run_is_rejected(self):
        for change in ({"completed_fold_indices": [0]}, {"status": "running"},
                       {"scheduled_fold_indices": [0]}, {"completed_fold_indices": [False, True, 2, 3, 4]}):
            with self.subTest(change=change):
                run, summary, freeze = metadata_fixture()
                run.update(change)
                summary["run_provenance"] = copy.deepcopy(run)
                with self.assertRaises(ValueError):
                    validate_run_metadata(run, summary, freeze, True)

    def test_material_protocol_differences_are_rejected(self):
        for field in ("runtime", "code_files", "cv_splits", "datasets", "skip_test_eval"):
            with self.subTest(field=field):
                run, summary, freeze = metadata_fixture()
                run[field] = None
                summary["run_provenance"] = copy.deepcopy(run)
                with self.assertRaisesRegex(ValueError, "run configuration"):
                    validate_run_metadata(run, summary, freeze, True)
        run, summary, freeze = metadata_fixture()
        summary["plm_feature_dim"] = 320
        with self.assertRaisesRegex(ValueError, "model recipe"):
            validate_run_metadata(run, summary, freeze, True)

    def test_wrong_arm_and_external_metrics_are_rejected(self):
        for mutation in ("wrong_arm", "external", "weights"):
            run, summary, freeze = metadata_fixture()
            if mutation == "wrong_arm":
                run["partner_transport"] = False
                summary["run_provenance"] = copy.deepcopy(run)
            elif mutation == "external":
                summary["ensemble"]["mean"]["Test287"] = {"mcc": 0.2}
            else:
                summary["val_mcc_weights"] = [1]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_run_metadata(run, summary, freeze, True)


class OOFIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "oof_predictions.npz"
        self.proteins = [{"complex_code": f"C{n}", "label": np.array([0, 1])} for n in range(5)]
        # OOF order follows folds, deliberately different from source order.
        self.folds = [{"fold_index": n, "val_indices": [index]}
                      for n, index in enumerate([3, 0, 4, 1, 2])]
        self.labels, self.identity = expected_oof(self.proteins, self.folds)
        self.run = metadata_fixture()[0]
        self.fields = {"labels": self.labels, "probs": np.tile([0.2, 0.8], 5),
                       "threshold": np.array([0.5]), **self.identity,
                       **prediction_source(self.run, "Train335")}

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_identity_accepts_saved_fold_order(self):
        np.savez(self.path, **self.fields)
        labels, _, _, _ = load_oof(self.path, self.run, self.labels, self.identity)
        np.testing.assert_array_equal(labels, self.labels)

    def test_equal_labels_do_not_hide_identity_corruption(self):
        for key in ("sample_index", "residue_index", "complex_id", "fold_index"):
            with self.subTest(key=key):
                fields = copy.deepcopy(self.fields)
                fields[key][0] = fields[key][-1]
                np.savez(self.path, **fields)
                with self.assertRaisesRegex(ValueError, "identity/order/coverage"):
                    load_oof(self.path, self.run, self.labels, self.identity)

    def test_missing_fold_identity_or_foreign_run_is_rejected(self):
        for key in ("fold_index", "provenance_id", "dataset_sha256"):
            fields = copy.deepcopy(self.fields)
            if key == "fold_index":
                del fields[key]
            else:
                fields[key] = np.asarray("foreign" if key == "provenance_id" else "f" * 64)
            with self.subTest(key=key):
                np.savez(self.path, **fields)
                with self.assertRaises(ValueError):
                    load_oof(self.path, self.run, self.labels, self.identity)

    def test_partial_population_and_repeated_sample_are_rejected(self):
        fields = copy.deepcopy(self.fields)
        for key in ("labels", "probs", "sample_index", "residue_index", "complex_id", "fold_index"):
            fields[key] = fields[key][:-2]
        np.savez(self.path, **fields)
        with self.assertRaisesRegex(ValueError, "full fixed validation population"):
            load_oof(self.path, self.run, self.labels, self.identity)
        fields = copy.deepcopy(self.fields)
        fields["sample_index"][-2:] = fields["sample_index"][:2]
        np.savez(self.path, **fields)
        with self.assertRaisesRegex(ValueError, "sample_index"):
            load_oof(self.path, self.run, self.labels, self.identity)


class PairedBootstrapTests(unittest.TestCase):
    def test_identical_arms_have_zero_intervals(self):
        labels = np.tile([0, 1], 4)
        probs = np.array([0.1, 0.9, 0.2, 0.8, 0.4, 0.6, 0.3, 0.7])
        groups = np.repeat(["A", "B", "C", "D"], 2)
        result = paired_complex_bootstrap(labels, probs, probs, 0.5, 0.5, groups, 40)
        self.assertEqual(result["valid_replicates"], 40)
        for interval in result["paired_percentile_95_ci"].values():
            self.assertEqual(interval, [0, 0])

    def test_draws_keep_whole_complexes_and_same_indices_for_both_arms(self):
        labels = np.tile([0, 1], 4)
        probs = np.arange(1, 9, dtype=float) / 10
        groups = np.array(["A", " a ", "A", "A", "B", "B", "C", "C"])
        from compare_oof_development import compute_auc_pr
        calls = []

        def capture(y, p):
            calls.append(np.asarray(p).copy())
            return compute_auc_pr(y, p)

        with patch("compare_oof_development.compute_auc_pr", side_effect=capture):
            result = paired_complex_bootstrap(labels, probs, 1 - probs, 0.5, 0.5, groups, 30)
        self.assertEqual(result["complex_count"], 3)
        for model, control in zip(calls[::2], calls[1::2]):
            np.testing.assert_allclose(model, 1 - control)
            counts = [int(np.count_nonzero(np.isclose(model, value))) for value in probs]
            self.assertEqual(len(set(counts[:4])), 1)
            self.assertEqual(counts[4], counts[5])
            self.assertEqual(counts[6], counts[7])

    def test_single_class_draws_are_counted(self):
        result = paired_complex_bootstrap(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9]),
                                          np.array([.2, .3, .7, .8]), .5, .5,
                                          np.array(["A", "A", "B", "B"]), 50)
        self.assertGreater(result["single_class_draws_excluded"], 0)
        self.assertEqual(result["requested_replicates"],
                         result["valid_replicates"] + result["single_class_draws_excluded"])


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_binding_metrics_and_finite_parameters(self):
        import torch
        run, summary, _ = metadata_fixture()
        saved_run = copy.deepcopy(run)
        saved_run["completed_fold_indices"] = [0]
        saved_run["status"] = "running"
        checkpoint = {
            "fold_index": 0, "cv_split": run["cv_splits"][0], "run_provenance": saved_run,
            "partner_transport": True, "skip_test_eval": True,
            "use_plm_features": True, "use_aux_plm_features": True,
            "threshold": 0.5, "val_mcc": 1.0, "val_auc_pr": 1.0,
            "model_state": {"weight": torch.ones(2)},
        }
        labels, probs = np.array([0, 1]), np.array([0.2, 0.8])
        validate_checkpoint(checkpoint, run, summary, 1, labels, probs)
        for mutation in ({"fold_index": 1}, {"val_auc_pr": .3}, {"partner_transport": False},
                         {"model_state": {"weight": torch.tensor(float("nan"))}}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_checkpoint({**checkpoint, **mutation}, run, summary, 1, labels, probs)


class QueueFailureTests(unittest.TestCase):
    def test_failure_stops_before_control_or_comparison_and_preserves_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "freeze.json"
            manifest.write_text("{}")
            freeze = {"project_root": str(root), "comparison_output_dir": str(root / "comparison"),
                      "jobs": [{"role": role, "partner_transport": role == "ot",
                                "wrapper": role + ".ps1", "output_dir": str(root / role)}
                               for role in ("ot", "control")]}
            child = Mock(pid=123)
            child.wait.return_value = 7
            with patch("run_frozen_development.read_freeze", return_value=freeze), \
                    patch("run_frozen_development.verify_environment"), \
                    patch("run_frozen_development.subprocess.Popen", return_value=child) as launch, \
                    patch("run_frozen_development.subprocess.call") as compare, \
                    patch("run_frozen_development.traceback.print_exc"):
                code = run_queue(manifest, root / "process", "powershell.exe")
            self.assertEqual(code, 1)
            self.assertEqual(launch.call_count, 1)
            compare.assert_not_called()
            state = json.loads((root / "process" / "queue_state.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["jobs"][0]["exit_code"], 7)
            self.assertEqual(state["jobs"][1]["status"], "queued")


if __name__ == "__main__":
    unittest.main()
