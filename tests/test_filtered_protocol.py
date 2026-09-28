"""Offline protocol regressions; all files are disposable synthetic fixtures."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from filtered_cohort_protocol import validate_fixed_folds, verify_filtered_cohort


PROTOCOL_ERRORS = (ValueError, RuntimeError)
TRAIN_HASH = "a" * 64


def record(path):
    contents = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(contents).hexdigest(), "size_bytes": len(contents)}


def fold_item(proteins, index, train, val):
    result = {"fold_index": index}
    for side, indices in (("train", train), ("val", val)):
        result[side + "_indices"] = list(indices)
        result[side + "_complex_ids"] = [p["complex_code"] for i in indices for p in [proteins[i]]]
        result[side + "_group_ids"] = [proteins[i]["complex_code"].strip().upper() for i in indices]
    return result


def small_split():
    proteins = [{"complex_code": code} for code in (" a ", "A", "b", "B", "C", " c ")]
    # Deliberately preserve a train order that is neither sorted nor generated
    # by concatenating the other validation folds.
    folds = [fold_item(proteins, 0, [5, 3, 4, 2], [1, 0]),
             fold_item(proteins, 1, [4, 0, 5, 1], [3, 2]),
             fold_item(proteins, 2, [2, 0, 3, 1], [5, 4])]
    document = {"schema_version": 1, "analysis_label": "filtered-cohort analysis",
                "seed": 2101, "num_folds": 3, "group_key": "complex_code",
                "sample_count": 6, "complex_count": 3, "train_sha256": TRAIN_HASH,
                "folds": folds}
    return proteins, document


class FixedFoldTests(unittest.TestCase):
    def setUp(self):
        self.proteins, self.document = small_split()

    def validate(self, document=None, **kwargs):
        return validate_fixed_folds(self.proteins, document or self.document,
                                    seed=kwargs.get("seed", 2101),
                                    train_sha256=kwargs.get("train_sha256", TRAIN_HASH),
                                    group_key=kwargs.get("group_key", "complex_code"))

    def test_saved_full_folds_and_arbitrary_train_order_are_preserved(self):
        before = copy.deepcopy(self.document)
        result = self.validate()
        self.assertEqual(result, before["folds"])
        self.assertEqual(result[0]["train_indices"], [5, 3, 4, 2])
        self.assertEqual(self.document, before)

    def test_request_seed_hash_and_group_key_must_match(self):
        for argument in ({"seed": 2102}, {"train_sha256": "b" * 64}, {"group_key": "family"}):
            with self.subTest(argument=argument), self.assertRaises(PROTOCOL_ERRORS):
                self.validate(**argument)

    def test_metadata_must_match_actual_samples_and_folds(self):
        changes = {"schema_version": 2, "seed": 2102, "num_folds": 5,
                   "sample_count": 7, "complex_count": 4, "group_key": "family",
                   "train_sha256": "b" * 64}
        for key, value in changes.items():
            modified = copy.deepcopy(self.document)
            modified[key] = value
            with self.subTest(key=key), self.assertRaises(PROTOCOL_ERRORS):
                self.validate(modified)

    def test_indices_must_be_true_integers_within_range(self):
        for side in ("train", "val"):
            for invalid in (True, False, 1.0, 0.5, -1, 6, "1"):
                modified = copy.deepcopy(self.document)
                modified["folds"][0][side + "_indices"][0] = invalid
                with self.subTest(side=side, invalid=invalid), self.assertRaises(PROTOCOL_ERRORS):
                    self.validate(modified)

    def test_ids_must_agree_with_samples_in_saved_order(self):
        for key in ("train_complex_ids", "val_complex_ids", "train_group_ids", "val_group_ids"):
            modified = copy.deepcopy(self.document)
            modified["folds"][0][key][0] = "WRONG"
            with self.subTest(key=key), self.assertRaises(PROTOCOL_ERRORS):
                self.validate(modified)

    def test_train_validation_indices_must_be_disjoint(self):
        modified = copy.deepcopy(self.document)
        modified["folds"][0] = fold_item(self.proteins, 0, [0, 3, 4, 2], [1, 0])
        with self.assertRaises(PROTOCOL_ERRORS):
            self.validate(modified)

    def test_normalized_complex_groups_cannot_cross_train_validation(self):
        modified = copy.deepcopy(self.document)
        modified["folds"][0] = fold_item(self.proteins, 0, [1, 3, 4, 5], [0, 2])
        with self.assertRaises(PROTOCOL_ERRORS):
            self.validate(modified)

    def test_duplicate_train_or_validation_indices_are_rejected(self):
        for side in ("train", "val"):
            modified = copy.deepcopy(self.document)
            indices = modified["folds"][0][side + "_indices"]
            indices[0] = indices[1]
            for suffix in ("complex_ids", "group_ids"):
                modified["folds"][0][side + "_" + suffix][0] = modified["folds"][0][side + "_" + suffix][1]
            with self.subTest(side=side), self.assertRaises(PROTOCOL_ERRORS):
                self.validate(modified)

    def test_every_sample_must_validate_exactly_once(self):
        modified = copy.deepcopy(self.document)
        # Each individual fold remains valid; the complete CV repeats A and
        # omits C, so only the global validation-coverage check catches this.
        modified["folds"][2] = fold_item(self.proteins, 2, [5, 3, 4, 2], [1, 0])
        with self.assertRaises(PROTOCOL_ERRORS):
            self.validate(modified)

    def test_each_fold_must_cover_all_samples(self):
        modified = copy.deepcopy(self.document)
        modified["folds"][0] = fold_item(self.proteins, 0, [5, 3, 4], [1, 0])
        with self.assertRaises(PROTOCOL_ERRORS):
            self.validate(modified)

    def test_duplicate_fold_identifiers_are_rejected(self):
        modified = copy.deepcopy(self.document)
        modified["folds"][1]["fold_index"] = 0
        with self.assertRaises(PROTOCOL_ERRORS):
            self.validate(modified)


class CohortVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "geo_filtered_v1"
        self.directory.mkdir()
        self.train_path = self.directory / "Train335.pkl"
        self.test_path = self.directory / "Test287.pkl"
        # Verification must hash files without unpickling them.
        self.train_path.write_bytes(b"synthetic training placeholder, not pickle")
        self.test_path.write_bytes(b"synthetic test placeholder, not pickle")
        self.fold_path = self.directory / "grouped_folds_seed2101.json"
        self.cohort_path = self.directory / "cohort_manifest.json"
        self.folds = {"schema_version": 1, "analysis_label": "filtered-cohort analysis",
                      "seed": 2101, "num_folds": 5, "group_key": "complex_code",
                      "sample_count": 334, "complex_count": 209,
                      "train_sha256": record(self.train_path)["sha256"],
                      "folds": [{"fold_index": index} for index in range(5)]}
        self.cohort = {"schema_version": 1, "analysis_label": "filtered-cohort analysis",
                       "cohort_version": "geo_filtered_v1", "status": "content_verified",
                       "external_set": "Test287.pkl", "all_original_pickle_hashes_unchanged": True,
                       "datasets": {"Train335.pkl": {**record(self.train_path), "sample_count": 334},
                                    "Test287.pkl": {**record(self.test_path), "sample_count": 285}},
                       "split_manifests": {}}
        self.write_documents()

    def write_documents(self):
        self.fold_path.write_text(json.dumps(self.folds), encoding="utf-8")
        self.cohort["split_manifests"]["2101"] = record(self.fold_path)
        self.cohort_path.write_text(json.dumps(self.cohort), encoding="utf-8")

    def verify(self, **kwargs):
        return verify_filtered_cohort(kwargs.get("data_dir", self.directory),
                                      kwargs.get("cohort_manifest_path", self.cohort_path),
                                      kwargs.get("fold_manifest_path", self.fold_path),
                                      kwargs.get("seed", 2101))

    def test_exact_fixture_is_accepted_without_unpickling(self):
        provenance, cohort, folds = self.verify()
        self.assertIsInstance(provenance, dict)
        self.assertEqual(cohort, self.cohort)
        self.assertEqual(folds, self.folds)

    def test_corruption_of_either_pickle_is_rejected(self):
        for path in (self.train_path, self.test_path):
            original = path.read_bytes()
            path.write_bytes(original + b"corruption")
            with self.subTest(file=path.name), self.assertRaises(PROTOCOL_ERRORS):
                self.verify()
            path.write_bytes(original)

    def test_changed_fold_bytes_are_rejected_even_with_valid_json(self):
        self.fold_path.write_text(json.dumps(self.folds) + "\n", encoding="utf-8")
        with self.assertRaises(PROTOCOL_ERRORS):
            self.verify()

    def test_incomplete_build_marker_rejects_cohort(self):
        (self.directory / "INCOMPLETE.txt").write_text("unfinished", encoding="utf-8")
        with self.assertRaises(PROTOCOL_ERRORS):
            self.verify()

    def test_cohort_version_status_label_and_declared_counts_are_strict(self):
        original = copy.deepcopy(self.cohort)
        for key, value in (("schema_version", 2), ("cohort_version", "geo_filtered_v2"),
                           ("status", "building"), ("analysis_label", "repaired full benchmark")):
            self.cohort = copy.deepcopy(original)
            self.cohort[key] = value
            self.write_documents()
            with self.subTest(key=key), self.assertRaises(PROTOCOL_ERRORS):
                self.verify()
        for name, wrong_count in (("Train335.pkl", 335), ("Test287.pkl", 287)):
            self.cohort = copy.deepcopy(original)
            self.cohort["datasets"][name]["sample_count"] = wrong_count
            self.write_documents()
            with self.subTest(dataset=name), self.assertRaises(PROTOCOL_ERRORS):
                self.verify()

    def test_requested_seed_is_frozen(self):
        with self.assertRaises(PROTOCOL_ERRORS):
            self.verify(seed=2102)

    def test_fold_metadata_cannot_change_with_recomputed_hash(self):
        original = copy.deepcopy(self.folds)
        for key, value in (("schema_version", 2), ("seed", 2102), ("num_folds", 3),
                           ("train_sha256", "b" * 64), ("group_key", "family")):
            self.folds = copy.deepcopy(original)
            self.folds[key] = value
            self.write_documents()
            with self.subTest(key=key), self.assertRaises(PROTOCOL_ERRORS):
                self.verify()

    def test_manifest_dataset_path_cannot_escape_data_directory(self):
        outside = self.root / "Train335.pkl"
        outside.write_bytes(self.train_path.read_bytes())
        self.cohort["datasets"]["Train335.pkl"]["path"] = str(self.directory / ".." / outside.name)
        self.write_documents()
        with self.assertRaises(PROTOCOL_ERRORS):
            self.verify()

    def test_external_fold_path_cannot_escape_data_directory(self):
        outside = self.root / self.fold_path.name
        outside.write_bytes(self.fold_path.read_bytes())
        self.cohort["split_manifests"]["2101"] = record(outside)
        self.cohort_path.write_text(json.dumps(self.cohort), encoding="utf-8")
        with self.assertRaises(PROTOCOL_ERRORS):
            self.verify(fold_manifest_path=outside)

    def test_fixed_fold_filename_is_required(self):
        alternate = self.directory / "alternative_folds.json"
        alternate.write_bytes(self.fold_path.read_bytes())
        self.cohort["split_manifests"]["2101"] = record(alternate)
        self.cohort_path.write_text(json.dumps(self.cohort), encoding="utf-8")
        with self.assertRaises(PROTOCOL_ERRORS):
            self.verify(fold_manifest_path=alternate)

    def test_dataset_symlink_to_outside_is_rejected(self):
        outside = self.root / "outside_test.pkl"
        outside.write_bytes(self.test_path.read_bytes())
        self.test_path.unlink()
        try:
            self.test_path.symlink_to(outside)
        except OSError as exc:
            self.skipTest(f"Creating symlinks is unavailable on this host: {exc.__class__.__name__}")
        with self.assertRaises(PROTOCOL_ERRORS):
            self.verify()


if __name__ == "__main__":
    unittest.main()
