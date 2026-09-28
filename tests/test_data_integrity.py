"""Data preparation regressions using synthetic records; no benchmark/network access."""
import contextlib
import copy
import io
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import augment_with_pdb_coordinates as augmentation
import audit_input_integrity as input_audit
import check_pcbind_prereqs as prerequisites


def pdb_atom(serial, name, resname="ALA", chain="A", residue=1, x=0.0, alt="", occupancy=1.0, record="ATOM"):
    return (
        f"{record:<6}{serial:5d} {name:^4}{alt:1}{resname:>3} {chain:1}{residue:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{occupancy:6.2f}{20.0:6.2f}          {name[0]:>2}\n"
    )


def chain(resname="ALA"):
    key = ("A", "1", "", resname)
    return {"residues": [key], "atoms": [(key, "CA", np.zeros(3, dtype=np.float32))], "backbone": {}}


def target(sequence=None):
    sample = {"complex_code": "1ABC", "residue_graph_node": np.zeros((1, 2)), "atom_graph_node": np.zeros((1, 2)), "label": np.array([1])}
    if sequence is not None:
        sample["residue_sequence"] = sequence
    return sample


def valid_sample():
    sample = target("A")
    sample.update({
        "residue_plm_embedding": np.zeros((1, 480)),
        "residue_plm_embedding_8m": np.zeros((1, 320)),
        "partner_residue_surface_features": np.zeros((2, 14)),
        "partner_residue_sequence_features": np.zeros((2, 39)),
        "partner_residue_coords": np.zeros((2, 3)),
        "partner_residue_frames": np.repeat(np.eye(3)[None], 2, axis=0),
        "partner_residue_geo_edge": np.array([[0, 1], [1, 0]], dtype=np.int64),
        "partner_chain_slices": np.array([[0, 2]], dtype=np.int64),
        "partner_residue_sequences": ["AC"],
        "partner_residue_plm_embedding": np.zeros((2, 480)),
        "partner_residue_plm_embedding_8m": np.zeros((2, 320)),
        "partner_pair_contact_index": np.array([[0], [1]], dtype=np.int64),
    })
    return sample


class PDBPreparationTests(unittest.TestCase):
    def parse(self, text):
        with mock.patch("builtins.open", return_value=io.StringIO(text)):
            return augmentation.parse_pdb_chains("synthetic.pdb")

    def test_first_model_only(self):
        parsed = self.parse("MODEL        1\n" + pdb_atom(1, "CA", x=1) + "ENDMDL\nMODEL        2\n" + pdb_atom(2, "CA", x=90) + "ENDMDL\n")
        self.assertEqual(len(parsed["A"]["atoms"]), 1)
        self.assertEqual(float(parsed["A"]["atoms"][0][2][0]), 1.0)

    def test_one_residue_conformer_and_shared_atoms(self):
        parsed = self.parse(pdb_atom(1, "N") + pdb_atom(2, "CA", x=1, alt="A", occupancy=0.3) + pdb_atom(3, "CA", x=2, alt="B", occupancy=0.7) + pdb_atom(4, "C", x=3, alt="A", occupancy=0.3) + pdb_atom(5, "C", x=4, alt="B", occupancy=0.7))
        self.assertEqual(len(parsed["A"]["residues"]), 1)
        self.assertEqual(len(parsed["A"]["atoms"]), 3)
        self.assertEqual(float(parsed["A"]["backbone"][parsed["A"]["residues"][0]]["CA"][0]), 2.0)

    def test_modified_amino_acids_are_included_but_water_is_not(self):
        parsed = self.parse(pdb_atom(1, "CA", resname="MSE", record="HETATM") + pdb_atom(2, "O", resname="HOH", residue=2, record="HETATM"))
        self.assertEqual(augmentation.residue_sequence(parsed["A"]), "M")

    def test_ambiguous_sizes_cannot_be_resolved_by_sample_order(self):
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            augmentation.find_matching_chain(target(), {"A": chain(), "B": chain("CYS")}, {"A"})

    def test_sequence_disambiguates_equal_length_chains(self):
        self.assertEqual(augmentation.find_matching_chain(target("C"), {"A": chain(), "B": chain("CYS")}, set()), ("B", True))

    def test_declared_chain_never_silently_switches_on_identity_mismatch(self):
        sample = target("C")
        sample["pdb_chain"] = "A"
        self.assertEqual(augmentation.find_matching_chain(sample, {"A": chain(), "B": chain("CYS")}, set()), (None, False))

    def test_chain_case_is_preserved(self):
        self.assertEqual(augmentation.parse_complex_code("1abc_a"), ("1ABC", "a"))

    def test_unverified_mapping_stays_unverified_and_keeps_base_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.pkl"
            base = target()
            with path.open("wb") as handle:
                pickle.dump([base], handle)
            with mock.patch.object(augmentation, "OUT_DIR", directory), mock.patch.object(augmentation, "download_pdb", return_value="unused"), mock.patch.object(augmentation, "parse_pdb_chains", return_value={"A": chain()}), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                augmentation.augment_file(path.name)
                augmentation.augment_file(path.name)
            with path.open("rb") as handle:
                result = pickle.load(handle)[0]
            self.assertFalse(result["pdb_mapping_verified"])
            self.assertEqual(result["pdb_mapping_method"], "unique_length_unverified")
            np.testing.assert_array_equal(result["label"], base["label"])
            np.testing.assert_array_equal(result["residue_graph_node"], base["residue_graph_node"])


class PrerequisiteTests(unittest.TestCase):
    def test_valid_record(self):
        self.assertTrue(prerequisites.valid_partner_encoder_sample(valid_sample()))
        self.assertEqual(prerequisites.count_positive_pairs([valid_sample()]), (1, 1))

    def test_invalid_partner_values_fail_validation(self):
        changes = [
            ("partner_residue_coords", np.full((2, 3), np.nan)),
            ("partner_residue_plm_embedding", np.zeros((2, 1))),
            ("partner_residue_plm_model", "missing_sequence"),
            ("partner_residue_geo_edge", np.array([[0.0], [1.0]])),
            ("partner_chain_slices", np.array([[0.0, 2.0]])),
            ("partner_residue_sequences", ["A", "C"]),
            ("pdb_augmentation_status", "failed"),
        ]
        for key, value in changes:
            with self.subTest(key=key):
                sample = valid_sample()
                sample[key] = value
                self.assertFalse(prerequisites.valid_partner_encoder_sample(sample))

    def test_pair_indices_require_integer_type_and_valid_bounds(self):
        for indices in (np.array([[0], [2]]), np.array([[-1], [0]]), np.array([[0.0], [1.0]]), np.array([0, 1])):
            with self.subTest(indices=indices.tolist()):
                sample = valid_sample()
                sample["partner_pair_contact_index"] = indices
                self.assertEqual(prerequisites.count_positive_pairs([sample]), (0, 0))

    def test_target_plm_checks_width_finiteness_and_failure_marker(self):
        sample = valid_sample()
        self.assertTrue(prerequisites.valid_target_plm(sample, "residue_plm_embedding"))
        sample["residue_plm_model"] = "missing_sequence"
        self.assertFalse(prerequisites.valid_target_plm(sample, "residue_plm_embedding"))
        sample.pop("residue_plm_model")
        sample["residue_plm_embedding"][0, 0] = np.inf
        self.assertFalse(prerequisites.valid_target_plm(sample, "residue_plm_embedding"))

    def test_require_plm_rejects_partial_target_coverage(self):
        samples = [valid_sample(), copy.deepcopy(valid_sample())]
        samples[1].pop("residue_plm_embedding")
        with tempfile.TemporaryDirectory() as directory:
            with (Path(directory) / "synthetic.pkl").open("wb") as handle:
                pickle.dump(samples, handle)
            with mock.patch("sys.argv", ["check_pcbind_prereqs.py", "--data-dir", directory, "--require-plm", "synthetic.pkl"]), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(prerequisites.main(), 1)


class ExactOverlapTests(unittest.TestCase):
    def test_cross_side_overlap_counts_samples_once_and_reports_missing_inputs(self):
        train = [{"sequence_sha256": "a", "partner_sequence_sha256": ["b", "b"]}]
        test = [
            {"index": 0, "complex_code": "TEST", "sequence_sha256": "b",
             "partner_sequence_sha256": ["a", "a"]},
            {"index": 1, "complex_code": "MISSING", "sequence_sha256": None},
        ]
        target = input_audit.cross_side_overlap(train, test, "partner", "target")
        partner = input_audit.cross_side_overlap(train, test, "target", "partner")
        for result in (target, partner):
            self.assertEqual(result["available_test_samples"], 1)
            self.assertEqual(result["affected_test_samples"], 1)
            self.assertEqual(result["distinct_matched_sequences"], 1)
        self.assertEqual(partner["matching_test_chain_occurrences"], 2)

    def test_suffixes_do_not_hide_shared_pdb_and_missing_sequences_are_not_matches(self):
        train = [{"complex_code": "1ABC", "pdb_id": "1ABC", "sequence_sha256": None}]
        test = [{"complex_code": "1ABC_B", "pdb_id": "1ABC", "sequence_sha256": None}]
        result = input_audit.exact_overlap(train, test)
        self.assertEqual(result["shared_exact_complex_codes"], [])
        self.assertEqual(result["shared_normalized_pdb_ids"], ["1ABC"])
        self.assertEqual(result["test_samples_with_exact_train_sequence"], 0)

    def test_duplicate_training_sequences_do_not_double_count_test_samples(self):
        train = [{"complex_code": code, "pdb_id": code, "sequence_sha256": "same"} for code in ("1ABC", "2BCD")]
        test = [{"complex_code": "3CDE", "pdb_id": "3CDE", "sequence_sha256": "same"}]
        result = input_audit.exact_overlap(train, test)
        self.assertEqual(result["test_samples_with_exact_train_sequence"], 1)
        self.assertEqual(len(result["exact_sequence_matches"][0]["train"]), 2)


if __name__ == "__main__":
    unittest.main()
