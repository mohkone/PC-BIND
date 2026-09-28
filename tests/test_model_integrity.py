"""Data-free regression tests for scientific validity and numerical invariants."""
import contextlib
import io
import unittest
from unittest import mock

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef

import CROSS5FOLD_multi_test as model_code


class ThresholdTests(unittest.TestCase):
    def test_selected_threshold_uses_the_deployed_boundary_rule(self):
        # The optimal strict-'greater than' cut is 0.500, whereas applying
        # >= to that value misclassifies the tied negative examples.
        labels = np.array([0, 0, 1, 1])
        for boundary in [0.5, 0.009]:
            with self.subTest(boundary=boundary):
                scores = np.array([boundary, boundary, round(boundary + 0.001, 3), 0.9])
                threshold = model_code.best_mcc_threshold(labels, scores)
                with contextlib.redirect_stdout(io.StringIO()):
                    metrics = model_code.metrics_from_scores(labels, scores, threshold=threshold)
                self.assertEqual(metrics["mcc"], 1.0)
                self.assertEqual(metrics["mcc"], matthews_corrcoef(labels, scores >= threshold))


class GroupedCVTests(unittest.TestCase):
    def test_groups_are_disjoint_and_all_samples_are_covered(self):
        proteins = [{"complex_code": code} for code in ["a", "A ", "b", "B", "c", "d"]]
        folds = model_code.make_cv_folds(proteins, seed=42, num_folds=3, grouped=True)
        self.assertEqual(sorted(np.concatenate(folds).tolist()), list(range(len(proteins))))
        group_sets = [{proteins[i]["complex_code"].strip().upper() for i in fold} for fold in folds]
        for i in range(len(folds)):
            for j in range(i):
                self.assertTrue(group_sets[i].isdisjoint(group_sets[j]))
        repeated = model_code.make_cv_folds(proteins, seed=42, num_folds=3, grouped=True)
        for original, repeat in zip(folds, repeated):
            np.testing.assert_array_equal(original, repeat)

    def test_missing_group_ids_fail_instead_of_silently_splitting_samples(self):
        for missing in [None, "", "  ", float("nan")]:
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "nonmissing"):
                model_code.make_cv_folds(
                    [{"complex_code": "a"}, {"complex_code": missing}],
                    seed=42, num_folds=2, grouped=True,
                )

    def test_insufficient_groups_fail_before_producing_an_empty_fold(self):
        with self.assertRaisesRegex(ValueError, "distinct group"):
            model_code.make_cv_folds(
                [{"complex_code": "a"}, {"complex_code": "A"}],
                seed=42, num_folds=2, grouped=True,
            )
        for count in [0, 1, 3]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                model_code.make_cv_folds([{}, {}], seed=42, num_folds=count)


class SparseTransportTests(unittest.TestCase):
    def test_final_rows_and_column_caps_both_hold_under_extreme_competition(self):
        # Every row chooses the same partner; the remaining partner columns
        # are masked. A final row softmax would undo the column projection.
        affinities = torch.full((6, 6), -torch.inf)
        affinities[:, 0] = 20.0
        for iterations in [1, 5, 10]:
            with self.subTest(iterations=iterations):
                layer = model_code.SparseTransportLayer(hidden_dim=8, sinkhorn_iters=iterations)
                plan = layer.sparse_sinkhorn(affinities, torch.zeros(6))
                self.assertTrue(torch.isfinite(plan).all())
                torch.testing.assert_close(plan.sum(1), torch.ones(6))
                self.assertTrue((plan[:, :-1].sum(0) <= 1.0 + 1e-6).all())
                self.assertTrue((plan[:, 1:-1] == 0).all())
                self.assertTrue((plan[:, -1] >= 5.0 / 6.0 - 1e-6).all())

    def test_unsupported_columns_and_rows_have_finite_gradients(self):
        raw_affinities = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.4, 0.1]], requires_grad=True)
        support = torch.tensor([[True, False, False], [False, False, False]])
        affinities = raw_affinities.masked_fill(~support, -torch.inf)
        dustbin = torch.tensor([0.1, 0.3], requires_grad=True)
        layer = model_code.SparseTransportLayer(hidden_dim=8, sinkhorn_iters=5, tau=1.0)
        plan = layer.sparse_sinkhorn(affinities, dustbin)
        self.assertEqual(plan[1, -1].item(), 1.0)
        (plan * torch.arange(4, dtype=plan.dtype)).sum().backward()
        self.assertTrue(torch.isfinite(raw_affinities.grad).all())
        self.assertTrue(torch.isfinite(dustbin.grad).all())
        self.assertGreater(raw_affinities.grad.abs().sum().item(), 0)
        self.assertGreater(dustbin.grad.abs().sum().item(), 0)

    def test_zero_iterations_preserves_uncapped_row_softmax_ablation(self):
        layer = model_code.SparseTransportLayer(hidden_dim=8, sinkhorn_iters=0, tau=0.5)
        affinities = torch.tensor([[3.0, 0.0], [3.0, 0.0], [3.0, 0.0]])
        dustbin = torch.zeros(3)
        plan = layer.sparse_sinkhorn(affinities, dustbin)
        expected = torch.softmax(torch.cat((affinities, dustbin[:, None]), dim=1) / 0.5, dim=1)
        torch.testing.assert_close(plan, expected)
        self.assertGreater(plan[:, 0].sum().item(), 1.5)

    def test_capped_projection_requires_a_dustbin(self):
        layer = model_code.SparseTransportLayer(hidden_dim=8, sinkhorn_iters=5, use_dustbin=False)
        with self.assertRaisesRegex(ValueError, "requires a dustbin"):
            layer.sparse_sinkhorn(torch.ones(2, 3))

    def test_invalid_transport_parameters_fail_early(self):
        for kwargs in [dict(tau=0.0), dict(tau=-1.0), dict(tau=float("nan")),
                       dict(tau=float("inf")), dict(sinkhorn_iters=-1), dict(top_k=0)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                model_code.SparseTransportLayer(hidden_dim=8, **kwargs)


class AuxiliaryTrainingTests(unittest.TestCase):
    def test_attention_contacts_count_each_positive_pair_once(self):
        with mock.patch.object(model_code, "PARTNER_TRANSPORT", False):
            model = model_code.TriViewAtomResidueNet(
                res_dim=4, atom_dim=3, plm_dim=0, aux_plm_dim=0,
                hidden_dim=8, partner_conditioning=True, partner_residue_encoder=False,
                partner_top_k=3, dropout=0.0, edge_dropout=0.0,
            )
        model.eval()
        fused = torch.randn(2, 8, requires_grad=True)
        _, pair_output = model.partner_condition(
            fused, torch.zeros(2, dtype=torch.long),
            partner_feats=torch.randn(3, model_code.PARTNER_FEATURE_DIM),
            partner_batch=torch.zeros(3, dtype=torch.long),
            # One positive is duplicated in input and both are in top-k.
            pair_contact_index=torch.tensor([[0, 0, 1], [1, 1, 2]]),
            pair_contact_graphs=torch.tensor([0]), return_pair=True,
        )
        logits, labels, _, explicit = pair_output
        self.assertEqual(logits.numel(), 6)
        self.assertEqual(int(labels.sum()), 2)
        self.assertEqual(int(explicit.sum()), 2)
        logits.square().mean().backward()
        self.assertTrue(torch.isfinite(fused.grad).all())

    def test_regularizer_keeps_true_partner_plan_after_a_contrastive_forward(self):
        class MinimalModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.2))
                self.plans = []

            def forward(self, *args, **kwargs):
                plan = [self.weight * (len(self.plans) + 1)]
                self.plans.append(plan)
                self._transport_plans = plan
                logits = self.weight.expand(2)
                return (logits, None) if kwargs.get("return_aux") else logits

        class Progress(list):
            def set_postfix(self, **kwargs):
                pass

        model = MinimalModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        empty_edges = torch.empty(2, 0, dtype=torch.long)
        batch = (
            torch.zeros(2, 4), empty_edges, torch.zeros(2, 3), empty_edges,
            torch.arange(2), torch.zeros(2, 3), torch.zeros(2, 3),
            torch.eye(3).repeat(2, 1, 1), torch.zeros(2, 14), torch.zeros(2, 39),
            torch.empty(2, 0), torch.empty(2, 0),
            torch.zeros(2, model_code.PARTNER_FEATURE_DIM), torch.zeros(2, dtype=torch.long),
            empty_edges, torch.zeros(2, 3), torch.eye(3).repeat(2, 1, 1),
            empty_edges, torch.empty(0, dtype=torch.long),
            torch.zeros(2), torch.zeros(2), torch.zeros(2, dtype=torch.long),
            torch.tensor([0.0, 1.0]),
        )
        captured = []

        def regularizer(_model, plans):
            captured.append(plans)
            return plans[0].square()

        settings = dict(TWO_HEAD_BINDING=False, PARTNER_CONDITIONING=True,
                        PARTNER_CONTRAST=True, PAIR_CONTACT_LOSS=False,
                        PAIR_CONTACT_CONTRAST=False, PAIR_MARGINAL_CONSISTENCY=False,
                        PAIR_MARGINAL_CONTRAST=False, PATCH_LABEL_DISTRIBUTION=False)
        with contextlib.ExitStack() as stack:
            for name, value in settings.items():
                stack.enter_context(mock.patch.object(model_code, name, value))
            stack.enter_context(mock.patch.object(model_code, "tqdm", side_effect=lambda items, **kw: Progress(items)))
            stack.enter_context(mock.patch.object(model_code, "make_mismatched_partner_graph_batch",
                                                  side_effect=lambda *args, **kw: (*args[:5], True)))
            stack.enter_context(mock.patch.object(model_code, "partner_contrastive_loss", return_value=None))
            stack.enter_context(mock.patch.object(model_code, "transport_regularization_loss", side_effect=regularizer))
            criterion = lambda logits, targets, **kwargs: logits.square().mean()
            model_code.train_one_epoch(model, [batch], optimizer, criterion, torch.device("cpu"), epoch=1)
        self.assertEqual(len(model.plans), 2)
        self.assertIs(captured[0], model.plans[0])
        self.assertIsNot(captured[0], model._transport_plans)
        self.assertIsNotNone(model.weight.grad)
        self.assertTrue(torch.isfinite(model.weight.grad))


if __name__ == "__main__":
    unittest.main()
