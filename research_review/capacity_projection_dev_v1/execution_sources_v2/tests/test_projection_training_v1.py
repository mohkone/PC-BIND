"""Initialization and unchanged transport-policy checks; no model fitting."""
import ast
import copy
import json
from pathlib import Path
import random
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

import CROSS5FOLD_multiseed_dev_v1 as frozen_trainer
from projection_initialization_v1 import (
    capture_initialization, capture_rng_states, dry_run_initializations,
    initialization_snapshot, record_initialization, rng_fingerprints,
    validate_initialization, validate_initialization_pair,
)


ROOT = Path(__file__).resolve().parents[1]


class EvidenceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.register_buffer("normalization", torch.zeros(2))

    def forward(self, *args, **kwargs):
        raise AssertionError("Initialization checks may not run a model forward")


def initialized(arm="projected", seed=2102, fold=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    before = capture_rng_states()
    model = EvidenceModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-4)
    after = capture_rng_states()
    arguments = (fold, seed, 2101, arm, {"projected": 5, "rowsoftmax": 0}[arm], before, after)
    snapshot = initialization_snapshot(model, optimizer, *arguments)
    record = capture_initialization(model, optimizer, *arguments)
    return model, optimizer, before, after, record, snapshot


class InitializationEvidenceTests(unittest.TestCase):
    def test_capture_does_not_consume_rng_or_change_parameters(self):
        model, optimizer, before, after, record, snapshot = initialized()
        self.assertNotEqual(record["rng_before"]["torch_cpu_sha256"], record["rng_after"]["torch_cpu_sha256"])
        self.assertEqual(record["rng_before"]["python_sha256"], record["rng_after"]["python_sha256"])
        self.assertEqual(record["rng_before"]["numpy_sha256"], record["rng_after"]["numpy_sha256"])
        actual_before = rng_fingerprints(capture_rng_states())
        validate_initialization(record, snapshot)
        self.assertEqual(rng_fingerprints(capture_rng_states()), actual_before)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, snapshot["model_state"][name], rtol=0, atol=0)
        self.assertFalse(optimizer.state)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_same_seed_first_fold_has_exact_parameter_optimizer_and_rng_equality(self):
        projected = initialized("projected")[-2]
        rowsoftmax = initialized("rowsoftmax")[-2]
        result = validate_initialization_pair(projected, rowsoftmax)
        self.assertTrue(result["initial_values_equal"])
        self.assertTrue(result["rng_fingerprints_equal"])
        self.assertTrue(result["schema_trainability_counts_optimizer_equal"])
        self.assertEqual(projected["parameter_count"], 8)
        self.assertEqual(projected["trainable_parameter_count"], 8)

    def test_pair_rejects_trainability_optimizer_schema_seed_and_initial_tensor_changes(self):
        projected = initialized("projected")[-2]
        rowsoftmax = initialized("rowsoftmax")[-2]
        mutations = []
        changed = copy.deepcopy(rowsoftmax)
        changed["parameter_schema"][0]["requires_grad"] = False
        changed["trainable_parameter_count"] -= changed["parameter_schema"][0]["numel"]
        mutations.append(changed)
        changed = copy.deepcopy(rowsoftmax)
        changed["optimizer"]["groups"][0]["settings"]["lr"] = 0.123
        mutations.append(changed)
        changed = copy.deepcopy(rowsoftmax)
        changed["parameter_sha256"] = "a" * 64
        mutations.append(changed)
        changed = copy.deepcopy(rowsoftmax)
        changed["rng_before"]["numpy_sha256"] = "a" * 64
        mutations.append(changed)
        mutations.append(initialized("rowsoftmax", seed=2103)[-2])
        for index, changed in enumerate(mutations):
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                validate_initialization_pair(projected, changed)

    def test_later_fold_divergence_is_recorded_without_requiring_equal_random_tensors(self):
        projected = initialized("projected", fold=1)[-2]
        _, _, before, _, _, _ = initialized("rowsoftmax", fold=1)
        # Mimic a different elapsed stochastic history without changing seed metadata.
        torch.rand(11)
        actual_before = capture_rng_states()
        model = EvidenceModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-4)
        rowsoftmax = capture_initialization(model, optimizer, 1, 2102, 2101, "rowsoftmax", 0,
                                            actual_before, capture_rng_states())
        result = validate_initialization_pair(projected, rowsoftmax, require_initial_values=False)
        self.assertFalse(result["initial_values_equal"])
        self.assertFalse(result["rng_fingerprints_equal"])
        self.assertFalse(result["initial_value_equality_required"])
        with self.assertRaisesRegex(ValueError, "First-fold"):
            validate_initialization_pair(projected, rowsoftmax, require_initial_values=True)

    def test_snapshot_recomputes_hashes_and_rejects_corrupt_or_nonfinite_state(self):
        record, snapshot = initialized()[-2:]
        for invalid in ("value", "nonfinite", "schema", "rng"):
            changed = copy.deepcopy(snapshot)
            if invalid == "value":
                changed["model_state"]["linear.weight"][0, 0] += 0.25
            elif invalid == "nonfinite":
                changed["model_state"]["normalization"][0] = float("nan")
            elif invalid == "schema":
                changed["parameter_schema"][0]["requires_grad"] = False
            else:
                changed["rng_before"]["torch_cpu"][0] ^= 1
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_initialization(record, changed)

    def test_invalid_fold_seed_or_projection_metadata_is_rejected(self):
        record = initialized()[-2]
        for field, value in (("fold_seed", 2102), ("training_seed", 2101),
                             ("fold_index", 5), ("projection_iterations", 0),
                             ("training_seed", True), ("fold_seed", 2101.0)):
            changed = copy.deepcopy(record)
            changed[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_initialization(changed)

    def test_raw_rng_state_shape_and_optimizer_step_state_are_checked(self):
        model, optimizer, before, after, record, snapshot = initialized()
        changed = copy.deepcopy(before)
        changed["numpy"] = ("MT19937", np.zeros(2, dtype=np.uint32), 0, 0, 0.0)
        with self.assertRaisesRegex(ValueError, "NumPy"):
            rng_fingerprints(changed)
        optimizer.state[next(model.parameters())]["step"] = torch.tensor(1.0)
        with self.assertRaisesRegex(ValueError, "before its first step"):
            capture_initialization(model, optimizer, 0, 2102, 2101, "projected", 5, before, after)

    def test_initialization_artifacts_are_exclusive_and_bind_saved_values(self):
        model, optimizer, before, after, record, snapshot = initialized()
        namespace = {"SEED": 2102, "FOLD_SEED": 2101, "TRANSPORT_SINKHORN_ITERS": 5,
                     "cli_args": types.SimpleNamespace(arm="projected")}
        run = {"initialization_records": []}
        with tempfile.TemporaryDirectory() as directory:
            entry = record_initialization(namespace, 0, model, optimizer, run, directory, before, after)
            self.assertEqual(run["initialization_records"], [entry])
            saved = json.loads(Path(entry["record"]["path"]).read_text(encoding="utf-8"))
            tensor = torch.load(entry["snapshot"]["path"], map_location="cpu", weights_only=False)
            self.assertEqual(saved, record)
            validate_initialization(saved, tensor)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                record_initialization(namespace, 0, model, optimizer, {"initialization_records": []},
                                      directory, before, after)

    def test_all_five_dry_initializations_never_reseed_fit_forward_or_step(self):
        frozen_trainer.seed_everything(2102)
        namespace = {"SEED": 2102, "FOLD_SEED": 2101, "TRANSPORT_SINKHORN_ITERS": 5,
                     "cli_args": types.SimpleNamespace(arm="projected"), "MODEL_MODE": "innovative_triview",
                     "folds_to_run": 5, "OPT_WEIGHT_DECAY": 2e-4,
                     "build_model": lambda *args: EvidenceModel(), "in_dim": 3,
                     "atom_dim": 3, "device": torch.device("cpu")}
        with mock.patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("step")), \
                mock.patch.object(torch, "manual_seed", side_effect=AssertionError("reseed")), \
                mock.patch.object(random, "seed", side_effect=AssertionError("reseed")), \
                mock.patch.object(np.random, "seed", side_effect=AssertionError("reseed")):
            records = dry_run_initializations(namespace)
        self.assertEqual([record["fold_index"] for record in records], list(range(5)))
        self.assertEqual(len({record["parameter_sha256"] for record in records}), 5)
        for previous, current in zip(records, records[1:]):
            self.assertEqual(previous["rng_after"], current["rng_before"])

    def test_frozen_feature_dataset_and_loader_construction_do_not_consume_rng(self):
        protein = {"residue_graph_node": np.array([[1.0, 2.0], [3.0, 4.0]])}
        before = rng_fingerprints(capture_rng_states())
        mean, std = frozen_trainer.fit_feature_stats([protein])
        dataset = frozen_trainer.ProteinDataset([protein], mean, std)
        torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
        torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        self.assertEqual(before, rng_fingerprints(capture_rng_states()))


class ProjectionPolicyPreservationTests(unittest.TestCase):
    def test_all_existing_model_and_training_function_bodies_are_unchanged(self):
        old = ast.parse((ROOT / "CROSS5FOLD_multiseed_dev_v1.py").read_text(encoding="utf-8"))
        new = ast.parse((ROOT / "CROSS5FOLD_projection_dev_v1.py").read_text(encoding="utf-8"))
        def definitions(tree):
            return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        self.assertEqual(definitions(old), definitions(new))
        old_assignments = [ast.dump(node, include_attributes=False) for node in old.body
                           if isinstance(node, (ast.Assign, ast.AnnAssign))]
        new_assignments = [ast.dump(node, include_attributes=False) for node in new.body
                           if isinstance(node, (ast.Assign, ast.AnnAssign))]
        self.assertEqual(old_assignments, new_assignments)

    def test_both_existing_projection_policies_have_finite_unsupported_column_gradients(self):
        for iterations in (0, 5):
            with self.subTest(iterations=iterations):
                layer = frozen_trainer.SparseTransportLayer(hidden_dim=8, sinkhorn_iters=iterations, tau=1.0)
                raw = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.4, 0.1]], requires_grad=True)
                support = torch.tensor([[True, False, False], [False, False, False]])
                dustbin = torch.tensor([0.1, 0.3], requires_grad=True)
                plan = layer.sparse_sinkhorn(raw.masked_fill(~support, -torch.inf), dustbin)
                self.assertTrue(torch.isfinite(plan).all())
                torch.testing.assert_close(plan.sum(1), torch.ones(2))
                self.assertEqual(float(plan[1, -1].detach()), 1.0)
                (plan * torch.arange(4, dtype=plan.dtype)).sum().backward()
                self.assertTrue(torch.isfinite(raw.grad).all())
                self.assertTrue(torch.isfinite(dustbin.grad).all())
                self.assertGreater(float(raw.grad.abs().sum()), 0.0)
                self.assertGreater(float(dustbin.grad.abs().sum()), 0.0)

    def test_five_and_zero_iterations_retain_identical_learned_architecture(self):
        records = []
        for arm, iterations in (("projected", 5), ("rowsoftmax", 0)):
            frozen_trainer.seed_everything(2102)
            before = capture_rng_states()
            model = frozen_trainer.SparseTransportLayer(hidden_dim=8, sinkhorn_iters=iterations)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-4)
            records.append(capture_initialization(model, optimizer, 0, 2102, 2101, arm, iterations,
                                                 before, capture_rng_states()))
        self.assertTrue(validate_initialization_pair(*records)["initial_values_equal"])
        affinities = torch.tensor([[3.0, 0.0], [3.0, 0.0], [3.0, 0.0]])
        row = frozen_trainer.SparseTransportLayer(hidden_dim=8, sinkhorn_iters=0, tau=0.5)
        projected = frozen_trainer.SparseTransportLayer(hidden_dim=8, sinkhorn_iters=5, tau=0.5)
        row_plan = row.sparse_sinkhorn(affinities, torch.zeros(3))
        projected_plan = projected.sparse_sinkhorn(affinities, torch.zeros(3))
        expected = torch.softmax(torch.cat((affinities, torch.zeros(3, 1)), dim=1) / 0.5, dim=1)
        torch.testing.assert_close(row_plan, expected, rtol=0, atol=0)
        self.assertGreater(float(row_plan[:, 0].sum()), 1.5)
        self.assertLessEqual(float(projected_plan[:, 0].sum()), 1.5 + 1e-6)


if __name__ == "__main__":
    unittest.main()
