import tempfile
import unittest
from pathlib import Path

import numpy as np

from build_filtered_cohort import content_hash, select_samples, validate_folds, validate_retained, write_json_new


class FilteredCohortIntegrityTests(unittest.TestCase):
    def test_exclusion_uses_original_index_and_preserves_valid_same_complex(self):
        samples = [{"complex_code": "2J3R", "label": [1]},
                   {"complex_code": "2J3R", "label": [0]},
                   {"complex_code": "4M0W", "label": [1]}]
        kept, indices = select_samples(samples, {1: "2J3R"}, 3)
        self.assertEqual(indices, [0, 2])
        self.assertIs(kept[0], samples[0])
        self.assertEqual(samples[1]["label"], [0])

    def test_changed_population_or_index_fails_before_filtering(self):
        samples = [{"complex_code": "2J3R"}]
        with self.assertRaises(ValueError):
            select_samples(samples, {0: "2J3R"}, 2)
        with self.assertRaises(ValueError):
            select_samples(samples, {0: "4M0W"}, 1)

    def test_sample_hash_detects_array_order_dtype_and_label_changes(self):
        original = {"coords": np.array([[1., 2.], [3., 4.]], dtype=np.float32), "label": [0, 1]}
        reference = content_hash(original)
        self.assertEqual(reference, content_hash({"label": [0, 1], "coords": original["coords"].copy()}))
        for changed in ({**original, "label": [1, 0]},
                        {**original, "coords": original["coords"][::-1]},
                        {**original, "coords": original["coords"].astype(np.float64)}):
            self.assertNotEqual(reference, content_hash(changed))

    def test_empty_arrays_have_shape_sensitive_fingerprints(self):
        self.assertNotEqual(content_hash(np.empty((0, 3))), content_hash(np.empty((0, 4))))
        self.assertEqual(content_hash(np.empty((0, 3))), content_hash(np.empty((0, 3))))

    def test_folds_reject_complex_leakage_and_missing_or_duplicate_members(self):
        samples = [{"complex_code": code} for code in ["1AAA", "1AAA", "2BBB", "3CCC"]]
        validate_folds(samples, [[0, 1], [2, 3]])
        for folds in ([[0, 2], [1, 3]], [[0, 1], [2]], [[0, 1], [2, 2, 3]]):
            with self.assertRaises(ValueError):
                validate_folds(samples, folds)

    def test_missing_features_fail_instead_of_silent_fallback(self):
        with self.assertRaisesRegex(ValueError, "target PLM"):
            validate_retained([{"residue_graph_node": np.zeros((2, 4)),
                                "residue_sequence": "AC", "label": [0, 1]}])

    def test_versioned_provenance_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_json_new(path, {"original": True})
            with self.assertRaises(FileExistsError):
                write_json_new(path, {"original": False})
            self.assertIn('"original": true', path.read_text())


if __name__ == "__main__":
    unittest.main()
