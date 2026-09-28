"""Data-free forward/backward check for the optional OT training path."""
import os

# Smoke checks define their own experiment, independent of the caller's run.
for name in list(os.environ):
    if name.startswith("PPI_"):
        del os.environ[name]

os.environ["PPI_PARTNER_TRANSPORT"] = "1"
os.environ["PPI_PARTNER_CONDITIONING"] = "1"
os.environ["PPI_PARTNER_RESIDUE_ENCODER"] = "0"
os.environ["PPI_TRANSPORT_DUSTBIN"] = "1"

import torch

import CROSS5FOLD_multi_test as cfg


def main():
    torch.manual_seed(2101)
    model = cfg.TriViewAtomResidueNet(
        res_dim=8, atom_dim=6, plm_dim=0, aux_plm_dim=0, hidden_dim=32,
        partner_conditioning=True, partner_residue_encoder=False,
        dropout=0.0, edge_dropout=0.0,
    )
    model.train()
    n, p, a = 7, 5, 11
    args = (
        torch.randn(n, 8), torch.randint(n, (2, 18)),
        torch.randn(a, 6), torch.randint(a, (2, 18)),
        torch.randint(n, (a,)), torch.zeros(n, dtype=torch.long),
    )
    kwargs = dict(
        res_coords=torch.randn(n, 3), atom_coords=torch.randn(a, 3),
        res_frames=torch.randn(n, 3, 3),
        surface_feats=torch.randn(n, 14),
        sequence_feats=torch.randn(n, 39),
        partner_feats=torch.randn(p, cfg.PARTNER_FEATURE_DIM),
        partner_batch=torch.zeros(p, dtype=torch.long),
        partner_edge=torch.randint(p, (2, 10)),
        partner_coords=torch.randn(p, 3),
        partner_frames=torch.randn(p, 3, 3),
        pair_contact_index=torch.tensor([[0, 1], [0, 1]]),
        pair_contact_graphs=torch.tensor([0]),
        return_aux=True, return_heads=True, return_pair=True,
        return_pair_marginal=True, return_pair_logit_matrix=True,
    )
    cls, rank, aux, pairs, marginal, matrix = model(*args, **kwargs)
    assert cls.shape == rank.shape == aux.shape == marginal.shape == (n,)
    assert matrix.shape == (n, cfg.TRANSPORT_TOP_K)
    assert torch.isfinite(cls).all() and torch.isfinite(rank).all()
    assert torch.all((marginal >= 0) & (marginal <= 1))
    assert pairs[0].numel() == pairs[1].numel()
    assert int((pairs[1] > 0.5).sum()) == 2  # no duplicated positive

    plan = model._transport_plans[0]
    assert plan.shape == (n, p + 1)
    assert torch.allclose(plan.sum(1), torch.ones(n), atol=1e-5)
    # An unbalanced plan must be able to route all target mass to no-match.
    layer = cfg.SparseTransportLayer(
        hidden_dim=32, sinkhorn_iters=5, tau=0.1, top_k=3
    )
    with torch.no_grad():
        layer.dustbin_proj[-1].weight.zero_()
        layer.dustbin_proj[-1].bias.fill_(20.0)
    _, no_match_mass, _, _ = layer(torch.randn(n, 32), torch.randn(p, 32))
    assert torch.all(no_match_mass < 1e-3)
    loss = (cls.square().mean() + rank.square().mean()
            + cfg.transport_regularization_loss(model, model._transport_plans))
    loss.backward()
    for module in (model.transport_layer.pair_scorer[0],
                   model.transport_layer.dustbin_proj[0]):
        grad = module.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0

    # Missing partner should preserve target-only outputs, with no stale plan.
    no_partner = kwargs.copy()
    no_partner.update(partner_feats=torch.empty(0, cfg.PARTNER_FEATURE_DIM),
                      partner_batch=torch.empty(0, dtype=torch.long),
                      partner_edge=torch.empty(2, 0, dtype=torch.long),
                      partner_coords=torch.empty(0, 3),
                      partner_frames=torch.empty(0, 3, 3),
                      return_pair=False, return_pair_marginal=False,
                      return_pair_logit_matrix=False)
    model(*args, **no_partner)
    assert model._transport_plans is None
    print("PC-BIND-OT data-free smoke test passed.")


if __name__ == "__main__":
    main()
