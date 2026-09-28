import hashlib
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research_provenance import (
    build_run_provenance, cv_split_manifest, file_record, load_dataset_with_provenance,
    prediction_identity, prediction_source,
)


class ResearchProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.proteins = [
            {"complex_code": "A", "label": np.array([0, 1])},
            {"complex_code": "B", "label": np.array([1])},
            {"complex_code": "A", "label": np.array([1, 0, 1])},
        ]

    def test_dataset_hash_describes_the_loaded_bytes_and_resolved_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.pkl"
            payload = pickle.dumps(self.proteins)
            path.write_bytes(payload)
            loaded, source = load_dataset_with_provenance(path)
            self.assertEqual(source["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(source["path"], str(path.resolve()))
            self.assertEqual(source["size_bytes"], len(payload))
            self.assertEqual(source["sample_count"], len(loaded))
            path.write_bytes(payload + b"changed")
            self.assertNotEqual(source["sha256"], file_record(path)["sha256"])

    def test_fold_manifest_preserves_actual_order_and_duplicate_complex_ids(self):
        folds = [np.array([0, 2]), np.array([1])]
        manifest = cv_split_manifest(self.proteins, folds)
        self.assertEqual(manifest[0]["train_indices"], [1])
        self.assertEqual(manifest[0]["val_indices"], [0, 2])
        self.assertEqual(manifest[0]["val_complex_ids"], ["A", "A"])
        self.assertEqual(manifest[1]["train_indices"], [0, 2])

    def test_residue_identity_tracks_nonsequential_source_samples(self):
        labels = np.array([1, 0, 1, 0, 1])
        identity = prediction_identity(self.proteins, [2, 0], fold_index=1, expected_labels=labels)
        np.testing.assert_array_equal(identity["sample_index"], [2, 2, 2, 0, 0])
        np.testing.assert_array_equal(identity["residue_index"], [0, 1, 2, 0, 1])
        np.testing.assert_array_equal(identity["fold_index"], [1, 1, 1, 1, 1])
        with self.assertRaisesRegex(ValueError, "declared sample/residue order"):
            prediction_identity(self.proteins, [2, 0], expected_labels=np.roll(labels, 1))

    def test_runtime_and_identity_can_be_read_without_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            code = Path(directory) / "model.py"
            code.write_text("# synthetic provenance fixture\n", encoding="utf-8")
            source = {"Train335": {"sha256": "a" * 64, "path": "synthetic.pkl"}}
            run = build_run_provenance(source, [code], cv_split_manifest(self.proteins, [[0, 2], [1]]), "cpu", 1)
            json.dumps(run)  # manifests contain no non-JSON runtime objects
            self.assertEqual(run["scheduled_fold_indices"], [0])
            self.assertIn("python", run["runtime"])
            self.assertIn("torch", run["runtime"]["packages"])
            self.assertEqual(run["code_files"][0]["sha256"], file_record(code)["sha256"])
            archive = Path(directory) / "predictions.npz"
            identity = prediction_identity(self.proteins, [1], fold_index=0)
            np.savez_compressed(archive, **identity, **prediction_source(run, "Train335"))
            with np.load(archive, allow_pickle=False) as loaded:
                for name in loaded.files:
                    self.assertNotEqual(loaded[name].dtype, object)
                self.assertEqual(loaded["dataset_sha256"].item(), "a" * 64)
                np.testing.assert_array_equal(loaded["sample_index"], [1])


if __name__ == "__main__":
    unittest.main()
