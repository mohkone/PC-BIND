"""Independent checks of the registered three-seed statistic and shared draws."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import precision_recall_curve, matthews_corrcoef

from compare_multiseed_dev_v3 import analyze_pairs, validate_pairs, ATTEMPTS, BOOTSTRAP_SEED


def pairs_fixture():
    labels = np.array([0, 0, 1, 1, 0, 1])
    identity = {"sample_index": np.array([0, 0, 1, 1, 2, 2]),
        "residue_index": np.tile([0, 1], 3), "fold_index": np.array([0, 0, 1, 1, 2, 2]),
        "complex_id": np.array([" b ", " B", "a", "A", " c", "C "])}
    ot = ([.1, .8, .7, .3, .2, .6], [.2, .6, .4, .5, .8, .7], [.3, .2, .4, .8, .7, .9])
    control = ([.2, .3, .8, .7, .6, .5], [.1, .4, .8, .7, .6, .9], [.8, .2, .3, .5, .1, .7])
    return [{"training_seed": seed,
        "ot": {"labels": labels.copy(), "probs": np.array(ot[i]), "threshold": .45 + .1*i,
            "identity": copy.deepcopy(identity)},
        "control": {"labels": labels.copy(), "probs": np.array(control[i]), "threshold": .5 + .1*i,
            "identity": copy.deepcopy(identity)}} for i, seed in enumerate((2102, 2103, 2104))]


def independent_score(y, p, threshold):
    precision, recall, _ = precision_recall_curve(y, p)
    return np.array([np.trapezoid(precision[::-1], recall[::-1]),
                     matthews_corrcoef(y, p >= threshold)])


class AnalysisTests(unittest.TestCase):
    def test_per_seed_mean_and_sample_sd_are_not_concatenated_residue_scores(self):
        pairs = pairs_fixture()
        result = analyze_pairs(pairs, attempts=20)
        independent = np.array([independent_score(p["ot"]["labels"], p["ot"]["probs"], p["ot"]["threshold"])
            - independent_score(p["control"]["labels"], p["control"]["probs"], p["control"]["threshold"])
            for p in pairs])
        for column, key in enumerate(("auc_pr_trapezoidal", "mcc")):
            summary = result["seed_difference_summary"][key]
            self.assertAlmostEqual(summary["mean"], independent[:, column].mean())
            self.assertAlmostEqual(summary["sample_standard_deviation"], independent[:, column].std(ddof=1))
            self.assertEqual(summary["range"], [independent[:, column].min(), independent[:, column].max()])
            self.assertEqual(sum(summary["sign_counts"].values()), 3)
        self.assertFalse(result["reference_seed2101_in_primary"])

    def test_exact_shared_pcg64_draws_and_single_class_exclusion(self):
        pairs = pairs_fixture()
        attempts = 50
        result = analyze_pairs(pairs, attempts=attempts)
        # Independently construct sorted A/B/C rows and average the three
        # per-seed differences on the same whole-complex draw.
        rows = [np.array([2, 3]), np.array([0, 1]), np.array([4, 5])]
        rng = np.random.Generator(np.random.PCG64(2101))
        values, excluded = [], 0
        for _ in range(attempts):
            ix = np.concatenate([rows[i] for i in rng.integers(3, size=3)])
            y = pairs[0]["ot"]["labels"][ix]
            if np.unique(y).size == 1:
                excluded += 1
                continue
            values.append(np.mean([independent_score(y, pair["ot"]["probs"][ix], pair["ot"]["threshold"])
                - independent_score(y, pair["control"]["probs"][ix], pair["control"]["threshold"])
                for pair in pairs], axis=0))
        boot = result["paired_complex_bootstrap"]
        self.assertEqual(boot["sorted_normalized_complex_ids"], ["A", "B", "C"])
        self.assertEqual(boot["single_class_draws_excluded"], excluded)
        self.assertGreater(excluded, 0)
        self.assertEqual(boot["valid_draws"] + excluded, attempts)
        expected = np.quantile(values, [.025, .975], axis=0, method="linear")
        for column, key in enumerate(("auc_pr_trapezoidal", "mcc")):
            np.testing.assert_allclose(boot["paired_percentile_95_ci"][key], expected[:, column], atol=1e-14)

    def test_six_thresholds_are_saved_and_not_reselected(self):
        pairs = pairs_fixture()
        with patch("compare_multiseed_dev_v3._bootstrap_scores", wraps=__import__(
                "compare_multiseed_dev_v3")._bootstrap_scores) as scores:
            result = analyze_pairs(pairs, attempts=10)
        seen = [call.args[2] for call in scores.call_args_list]
        expected = [pair[arm]["threshold"] for pair in pairs for arm in ("ot", "control")]
        self.assertEqual(seen, expected * result["paired_complex_bootstrap"]["valid_draws"])

    def test_only_declared_new_seed_pairs_enter_primary_analysis(self):
        for replacement in (2101, 2105, True):
            pairs = pairs_fixture()
            pairs[0]["training_seed"] = replacement
            with self.assertRaises(ValueError):
                analyze_pairs(pairs, attempts=1)
        pairs = pairs_fixture()
        pairs.reverse()
        with self.assertRaises(ValueError):
            validate_pairs(pairs)
        self.assertEqual((ATTEMPTS, BOOTSTRAP_SEED), (5000, 2101))

    def test_bad_or_unequal_identity_is_failure_not_excluded_draw(self):
        changes = [lambda r: r["identity"].pop("fold_index"),
            lambda r: r["identity"]["sample_index"].__setitem__(0, 7),
            lambda r: r["identity"]["residue_index"].__setitem__(0, 9),
            lambda r: r["identity"]["complex_id"].__setitem__(0, ""),
            lambda r: r["identity"]["fold_index"].__setitem__(0, 4),
            lambda r: r.update(labels=r["labels"][:-1], probs=r["probs"][:-1]),
            lambda r: r["labels"].__setitem__(0, 1)]
        for change in changes:
            pairs = pairs_fixture()
            change(pairs[1]["control"])
            with self.assertRaises((ValueError, KeyError)):
                analyze_pairs(pairs, attempts=1)

    def test_nonfinite_probabilities_and_thresholds_fail(self):
        for value in (np.nan, np.inf, -.1, 1.1):
            pairs = pairs_fixture()
            pairs[2]["ot"]["probs"][0] = value
            with self.assertRaises(ValueError):
                analyze_pairs(pairs, attempts=1)
            pairs = pairs_fixture()
            pairs[2]["ot"]["threshold"] = value
            with self.assertRaises(ValueError):
                analyze_pairs(pairs, attempts=1)

    def test_duplicate_source_residue_rejected_even_if_shared_by_all_runs(self):
        pairs = pairs_fixture()
        for pair in pairs:
            for arm in ("ot", "control"):
                pair[arm]["identity"]["residue_index"][1] = 0
        with self.assertRaises(ValueError):
            validate_pairs(pairs)

    def test_unequal_metadata_seed_or_transport_rejected(self):
        pairs = pairs_fixture()
        pairs[1]["control"]["run"] = {"training_seed": 2102, "fold_seed": 2101, "partner_transport": False}
        with self.assertRaises(ValueError):
            validate_pairs(pairs)


if __name__ == "__main__":
    unittest.main()
