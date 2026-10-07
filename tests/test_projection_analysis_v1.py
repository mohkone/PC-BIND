"""Independent synthetic checks of the registered projection-policy analysis."""
import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from sklearn.metrics import matthews_corrcoef, precision_recall_curve, roc_auc_score

import compare_projection_dev_v1 as analysis


def projection_pairs():
    labels = np.array([0, 0, 1, 1, 0, 1])
    identity = {"sample_index": np.array([0, 0, 1, 1, 2, 2]),
                "residue_index": np.tile([0, 1], 3), "fold_index": np.array([0, 0, 1, 1, 2, 2]),
                "complex_id": np.array([" b ", " B", "a", "A", " c", "C "])}
    projected = ([.1, .8, .7, .3, .2, .6], [.2, .6, .4, .5, .8, .7], [.3, .2, .4, .8, .7, .9])
    rowsoftmax = ([.2, .3, .8, .7, .6, .5], [.1, .4, .8, .7, .6, .9], [.8, .2, .3, .5, .1, .7])
    result = []
    for index, seed in enumerate(analysis.TRAINING_SEEDS):
        pair = {"training_seed": seed}
        for arm, probs, threshold, iterations in (
                ("projected", projected[index], .45 + .1 * index, 5),
                ("rowsoftmax", rowsoftmax[index], .5 + .1 * index, 0)):
            pair[arm] = {"labels": labels.copy(), "probs": np.array(probs), "threshold": threshold,
                         "identity": copy.deepcopy(identity),
                         "run": {"training_seed": seed, "fold_seed": 2101,
                                 "projection_arm": arm, "partner_transport": True,
                                 "transport_sinkhorn_iters": iterations}}
        result.append(pair)
    return result


def independent_metrics(y, p, threshold):
    precision, recall, _ = precision_recall_curve(y, p)
    return np.array([np.trapezoid(precision[::-1], recall[::-1]),
                     matthews_corrcoef(y, p >= threshold), roc_auc_score(y, p)])


def registered_protocol():
    # Read immutable policy text only; no real predictions or resampling.
    path = Path(__file__).resolve().parents[1] / "research_review/capacity_projection_dev_v1/protocol.json"
    return json.loads(path.read_text(encoding="utf-8"))


class ProjectionAnalysisTests(unittest.TestCase):
    def test_direction_per_seed_mean_sd_range_and_signs_independently(self):
        pairs = projection_pairs()
        result = analysis.analyze_projection_pairs(pairs, attempts=20)
        deltas = np.array([
            independent_metrics(pair["projected"]["labels"], pair["projected"]["probs"],
                                pair["projected"]["threshold"])
            - independent_metrics(pair["rowsoftmax"]["labels"], pair["rowsoftmax"]["probs"],
                                  pair["rowsoftmax"]["threshold"])
            for pair in pairs])
        for column, metric in enumerate(analysis.METRICS):
            summary = result["seed_difference_summary"][metric]
            np.testing.assert_allclose([entry["projected_minus_rowsoftmax"][metric]
                                        for entry in result["per_seed"]], deltas[:, column], atol=1e-14)
            self.assertAlmostEqual(summary["mean"], deltas[:, column].mean())
            self.assertAlmostEqual(summary["sample_standard_deviation"], deltas[:, column].std(ddof=1))
            np.testing.assert_allclose(summary["range"], [deltas[:, column].min(), deltas[:, column].max()])
            self.assertEqual(summary["standard_deviation_ddof"], 1)
            self.assertEqual(summary["sign_counts"], {
                "positive": int((deltas[:, column] > 0).sum()),
                "zero": int((deltas[:, column] == 0).sum()),
                "negative": int((deltas[:, column] < 0).sum())})
        self.assertEqual(result["primary_training_seeds"], [2102, 2103, 2104])
        self.assertFalse(result["reference_seed2101_in_primary"])
        self.assertFalse(result["historical_comparisons_in_primary_or_bootstrap"])

    def test_shared_sorted_normalized_pcg64_draws_and_fixed_thresholds(self):
        pairs = projection_pairs()
        attempts = 50
        result = analysis.analyze_projection_pairs(pairs, attempts=attempts)
        rows = [np.array([2, 3]), np.array([0, 1]), np.array([4, 5])]
        rng = np.random.Generator(np.random.PCG64(2101))
        values, excluded = [], 0
        for _ in range(attempts):
            indices = np.concatenate([rows[index] for index in rng.integers(3, size=3)])
            labels = pairs[0]["projected"]["labels"][indices]
            if np.unique(labels).size == 1:
                excluded += 1
                continue
            values.append(np.mean([
                independent_metrics(labels, pair["projected"]["probs"][indices], pair["projected"]["threshold"])[:2]
                - independent_metrics(labels, pair["rowsoftmax"]["probs"][indices], pair["rowsoftmax"]["threshold"])[:2]
                for pair in pairs], axis=0))
        bootstrap = result["paired_complex_bootstrap"]
        self.assertEqual(bootstrap["sorted_normalized_complex_ids"], ["A", "B", "C"])
        self.assertEqual(bootstrap["single_class_draws_excluded"], excluded)
        self.assertGreater(excluded, 0)
        self.assertEqual(bootstrap["valid_draws"] + excluded, attempts)
        self.assertFalse(bootstrap["replenish_excluded_draws"])
        self.assertTrue(bootstrap["same_draw_for_all_six_runs"])
        expected = np.quantile(values, [.025, .975], axis=0, method="linear")
        for column, key in enumerate(analysis.BOOTSTRAP_METRICS):
            np.testing.assert_allclose(bootstrap["paired_percentile_95_ci"][key], expected[:, column], atol=1e-14)
        self.assertNotIn("auroc", bootstrap["paired_percentile_95_ci"])
        with patch("compare_multiseed_dev_v3._bootstrap_scores", wraps=__import__(
                "compare_multiseed_dev_v3")._bootstrap_scores) as scores:
            short = analysis.analyze_projection_pairs(pairs, attempts=10)
        saved = [pair[arm]["threshold"] for pair in pairs for arm in analysis.ARMS]
        self.assertEqual([call.args[2] for call in scores.call_args_list],
                         saved * short["paired_complex_bootstrap"]["valid_draws"])

    def test_adapter_retains_actual_transport_enabled_metadata_and_new_labels(self):
        pairs = projection_pairs()
        before = copy.deepcopy(pairs)
        with patch.object(analysis, "_frozen_analyze_pairs", wraps=analysis._frozen_analyze_pairs) as math:
            result = analysis.analyze_projection_pairs(pairs, attempts=2)
        adapted = math.call_args.args[0]
        for pair in adapted:
            self.assertNotIn("run", pair["ot"])
            self.assertNotIn("run", pair["control"])
        for pair, original in zip(pairs, before):
            for arm in analysis.ARMS:
                self.assertEqual(pair[arm]["run"], original[arm]["run"])
                self.assertTrue(pair[arm]["run"]["partner_transport"])
                np.testing.assert_array_equal(pair[arm]["probs"], original[arm]["probs"])
        for pair in result["per_seed"]:
            self.assertEqual(set(pair["metrics"]), set(analysis.ARMS))
            self.assertNotIn("ot_minus_control", pair)
        report = analysis.render_markdown({"analysis": result, "limitations": []})
        self.assertIn("Projected − rowsoftmax", report)
        self.assertNotIn("no-OT", report)
        self.assertNotIn("OT −", report)

    def test_actual_seed_arm_projection_or_transport_mismatch_fails_before_math(self):
        edits = [lambda record: record["run"].update(partner_transport=False),
                 lambda record: record["run"].update(transport_sinkhorn_iters=5),
                 lambda record: record["run"].update(transport_sinkhorn_iters=False),
                 lambda record: record["run"].update(training_seed=2102),
                 lambda record: record["run"].update(fold_seed=2102),
                 lambda record: record["run"].update(projection_arm="projected")]
        for edit in edits:
            pairs = projection_pairs()
            edit(pairs[1]["rowsoftmax"])
            with patch.object(analysis, "_frozen_analyze_pairs") as math:
                with self.assertRaises(ValueError):
                    analysis.analyze_projection_pairs(pairs, attempts=1)
                math.assert_not_called()
        for seed in (2101, 2105, True):
            pairs = projection_pairs()
            pairs[0]["training_seed"] = seed
            with self.assertRaises(ValueError):
                analysis.validate_projection_pairs(pairs)
        pairs = projection_pairs()
        pairs.reverse()
        with self.assertRaises(ValueError):
            analysis.validate_projection_pairs(pairs)

    def test_identity_corruption_partial_population_and_nonfinite_data_are_failures(self):
        changes = [lambda record: record["identity"].pop("fold_index"),
                   lambda record: record["identity"]["sample_index"].__setitem__(0, 7),
                   lambda record: record["identity"]["complex_id"].__setitem__(0, ""),
                   lambda record: record["identity"]["residue_index"].__setitem__(0, 9),
                   lambda record: record["identity"]["fold_index"].__setitem__(0, 4),
                   lambda record: record.update(labels=record["labels"][:-1], probs=record["probs"][:-1]),
                   lambda record: record["labels"].__setitem__(0, 1),
                   lambda record: record["probs"].__setitem__(0, np.nan),
                   lambda record: record["probs"].__setitem__(0, np.inf),
                   lambda record: record["probs"].__setitem__(0, 1.1),
                   lambda record: record.update(threshold=np.nan)]
        for change in changes:
            pairs = projection_pairs()
            change(pairs[1]["rowsoftmax"])
            with self.assertRaises((ValueError, KeyError)):
                analysis.analyze_projection_pairs(pairs, attempts=1)
        pairs = projection_pairs()
        for pair in pairs:
            for arm in analysis.ARMS:
                pair[arm]["identity"]["residue_index"][1] = 0
        with self.assertRaises(ValueError):
            analysis.validate_projection_pairs(pairs)
        with patch("compare_multiseed_dev_v3._bootstrap_scores", return_value={
                "auc_pr_trapezoidal": np.nan, "mcc": 0.0}):
            with self.assertRaisesRegex(ValueError, "not an excluded draw"):
                analysis.analyze_projection_pairs(projection_pairs(), attempts=5)

    def test_registered_production_constants_cannot_be_changed(self):
        self.assertEqual((analysis.ATTEMPTS, analysis.BOOTSTRAP_SEED), (5000, 2101))
        protocol = registered_protocol()
        analysis.validate_registered_analysis(protocol)
        for key, value in (("replicates", 4999), ("seed", 2102), ("replenish_excluded_draws", True),
                           ("draw_operation", "rng.integers(209, size=208)"),
                           ("thresholds", "select thresholds on each bootstrap draw")):
            wrong = copy.deepcopy(protocol)
            wrong["analysis"]["bootstrap"][key] = value
            with self.assertRaises(ValueError):
                analysis.validate_registered_analysis(wrong)


class AggregateGatesTests(unittest.TestCase):
    """Mock filesystem-bound validators to exercise aggregate gates without fitting."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.path = self.directory / "execution_manifest.json"
        self.path.write_text("{}\n", encoding="utf-8")
        self.acceptance_path = self.path.with_name("execution_manifest.acceptance.json")
        self.acceptance_path.write_text("{}\n", encoding="utf-8")
        self.output = self.directory / "complete_oof"
        self.protocol = registered_protocol()
        self.manifest = {"jobs": self.protocol["seed_and_arm_order"], "aggregate_output_dir": str(self.output),
                         "protocol": {"path": "registered-protocol", "sha256": "a" * 64},
                         "source_files": [], "runtime": self.protocol["required_common_runtime"],
                         "dataset_selection": self.protocol["dataset_selection"], "cv_splits": []}
        self.records = [pair[arm] for pair in projection_pairs() for arm in analysis.ARMS]
        for record, job in zip(self.records, self.manifest["jobs"]):
            record.update(job=job, artifacts={"checkpoints": []}, initializations=[{"fold_index": i} for i in range(5)])
        self.verify = Mock(return_value=(self.manifest, self.protocol))
        self.authorize = Mock(return_value={"path": "launch-authorization", "sha256": "b" * 64})
        self.validate_arm = Mock(side_effect=self.records + copy.deepcopy(self.records))
        self.pair_check = Mock(return_value={"folds": [
            {"fold_index": i, "parameter_schema_equal": True, "optimizer_equal": True,
             "initial_tensors_equal": i == 0, "rng_equal": i == 0} for i in range(5)]})
        common = types.ModuleType("projection_protocol_v1")
        common.verify_manifest = self.verify
        common.require_launch_authorization = self.authorize
        validator = types.ModuleType("validate_projection_dev_v1")
        validator.validate_arm = self.validate_arm
        validator.validate_pair_records = self.pair_check
        modules = patch.dict(sys.modules, {"projection_protocol_v1": common, "validate_projection_dev_v1": validator})
        modules.start()
        self.addCleanup(modules.stop)
        # Production row-count gate only; this test does not invent real OOF data.
        population = (np.zeros(66208, dtype=np.int64),
                      {"sample_index": np.arange(334), "fold_index": np.arange(5)},
                      np.arange(209), [])
        validation = patch.object(analysis, "validate_projection_pairs", return_value=population)
        validation.start()
        self.addCleanup(validation.stop)

    def test_validate_only_checks_all_pairs_without_bootstrap_outputs_or_authorization(self):
        with patch.object(analysis, "analyze_projection_pairs") as bootstrap:
            result = analysis.compare(self.path, validate_only=True)
        self.assertEqual(self.validate_arm.call_count, 6)
        self.assertEqual(self.pair_check.call_count, 3)
        self.assertEqual(len(result["initialization_matching"]), 3)
        self.assertFalse(result["bootstrap_run"])
        self.assertFalse(result["outputs_written"])
        self.assertFalse(self.output.exists())
        self.authorize.assert_not_called()
        bootstrap.assert_not_called()
        self.assertTrue(all(call.kwargs["require_acceptance"] is True for call in self.verify.call_args_list))

    def test_first_pair_initialization_validation_failure_prevents_later_arms_and_analysis(self):
        self.pair_check.side_effect = ValueError("First-fold initial parameters differ")
        with patch.object(analysis, "analyze_projection_pairs") as bootstrap:
            with self.assertRaisesRegex(ValueError, "First-fold"):
                analysis.compare(self.path, validate_only=True)
        self.assertEqual(self.validate_arm.call_count, 2)
        bootstrap.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_production_requires_explicit_authorization_and_refuses_existing_destination(self):
        self.authorize.side_effect = ValueError("Explicit launch authorization is absent")
        with patch.object(analysis, "analyze_projection_pairs") as bootstrap:
            with self.assertRaisesRegex(ValueError, "authorization"):
                analysis.compare(self.path)
        self.validate_arm.assert_not_called()
        bootstrap.assert_not_called()
        self.output.mkdir()
        self.authorize.reset_mock()
        with self.assertRaisesRegex(ValueError, "rerun or overwrite"):
            analysis.compare(self.path)
        self.authorize.assert_not_called()

    def test_production_exact_draw_constants_rehashes_and_preserves_historical_context_only(self):
        synthetic = analysis.analyze_projection_pairs(projection_pairs(), attempts=2)
        with patch.object(analysis, "analyze_projection_pairs", return_value=synthetic) as bootstrap:
            result = analysis.compare(self.path)
        self.assertEqual(bootstrap.call_args.kwargs, {"attempts": 5000, "bootstrap_seed": 2101})
        self.assertEqual(self.validate_arm.call_count, 12)
        self.assertEqual(self.pair_check.call_count, 6)
        self.assertEqual(self.authorize.call_count, 2)
        self.assertEqual(result["status"], "complete")
        saved = json.loads((self.output / "comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["acceptance"], analysis.file_record(self.acceptance_path))
        historical = saved["historical_descriptive_context"]
        self.assertFalse(historical["included_in_primary_estimate_or_bootstrap"])
        self.assertNotIn("metrics", historical)
        self.assertNotIn("threshold", historical)
        self.assertNotIn("ot_minus_control", historical)
        self.assertEqual([entry["training_seed"] for entry in saved["initialization_matching"]], [2102, 2103, 2104])
        self.assertTrue(all(entry["run"]["partner_transport"] for entry in saved["artifacts"]))

    def test_changed_artifact_or_manifest_after_analysis_refuses_publication(self):
        synthetic = analysis.analyze_projection_pairs(projection_pairs(), attempts=2)

        def mutate_manifest(*args, **kwargs):
            self.path.write_text('{"changed":true}\n', encoding="utf-8")
            return synthetic

        with patch.object(analysis, "analyze_projection_pairs", side_effect=mutate_manifest):
            with self.assertRaisesRegex(ValueError, "unchanged after bootstrap"):
                analysis.compare(self.path)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
