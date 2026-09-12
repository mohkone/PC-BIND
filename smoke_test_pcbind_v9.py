import os
import pickle

import torch


os.environ.setdefault("PPI_PLM_DIM", "480")
os.environ.setdefault("PPI_AUX_PLM_DIM", "320")
os.environ["PPI_PARTNER_CONDITIONING"] = "1"
os.environ["PPI_PARTNER_RESIDUE_ENCODER"] = "1"
os.environ["PPI_PARTNER_DIRECT_FUSION"] = "0"
os.environ["PPI_PARTNER_LOGIT_MODE"] = "residual"
os.environ["PPI_PARTNER_DELTA_SCALE"] = "0.25"
os.environ["PPI_PAIR_CONTACT_LOSS"] = "1"
os.environ["PPI_PAIR_CONTACT_HEAD"] = "mlp"
os.environ["PPI_PAIR_MARGINAL_CONSISTENCY"] = "0"
os.environ["PPI_PAIR_MARGINAL_CONTRAST"] = "0"
os.environ["PPI_PAIR_CONTACT_CONTRAST"] = "0"

import CROSS5FOLD_multi_test as train_cfg
from check_pcbind_prereqs import valid_partner_encoder_sample
from evaluate_mismatched_partner_control import make_mismatched_partner_list


def load_samples():
    with open(os.path.join("data", "geo", "Train335.pkl"), "rb") as handle:
        return pickle.load(handle)


def make_batch(samples):
    dataset = train_cfg.ProteinDataset(
        samples,
        plm_feature_dim=480,
        aux_plm_feature_dim=320,
    )
    return train_cfg.collate_proteins([dataset[idx] for idx in range(len(dataset))])


def forward(
    model,
    batch,
    null_partner=False,
    require_partner_grad=False,
    return_pair=False,
    return_pair_marginal=False,
    return_pair_logit_matrix=False,
):
    (
        x, edge_index, atom_x, atom_edge_index, atom2res,
        res_coords, atom_coords, res_frames, surface_feats, sequence_feats,
        plm_feats, aux_plm_feats, partner_feats, partner_batch, partner_edge,
        partner_coords, partner_frames, partner_pair_contact_index,
        pair_contact_graphs, _partner_contact, _partner_mask, batch_vec, _labels,
    ) = batch
    if null_partner:
        partner_feats = partner_feats[:0]
        partner_batch = partner_batch[:0]
        partner_edge = partner_edge[:, :0]
        partner_coords = partner_coords[:0]
        partner_frames = partner_frames[:0]
    elif require_partner_grad:
        partner_feats = partner_feats.detach().requires_grad_(True)
    output = model(
        x,
        edge_index,
        atom_x,
        atom_edge_index,
        atom2res,
        batch_vec,
        res_coords,
        atom_coords,
        res_frames,
        surface_feats,
        sequence_feats,
        plm_feats,
        aux_plm_feats,
        partner_feats,
        partner_batch,
        partner_edge=partner_edge,
        partner_coords=partner_coords,
        partner_frames=partner_frames,
        pair_contact_index=partner_pair_contact_index,
        pair_contact_graphs=pair_contact_graphs,
        return_heads=True,
        return_pair=return_pair,
        return_pair_marginal=return_pair_marginal,
        return_pair_logit_matrix=return_pair_logit_matrix,
    )
    return output, partner_feats


def main():
    torch.manual_seed(29)
    samples = [sample for sample in load_samples() if valid_partner_encoder_sample(sample)]
    samples.sort(
        key=lambda sample: (
            int(sample["residue_graph_node"].shape[0])
            + int(sample["atom_graph_node"].shape[0])
            + int(sample["partner_residue_surface_features"].shape[0])
        )
    )
    targets = samples[:2]
    mismatches, _ = make_mismatched_partner_list(targets, seed=9917, neighbor_count=1)
    true_batch = make_batch(targets)
    mismatch_batch = make_batch(mismatches)
    in_dim = int(targets[0]["residue_graph_node"].shape[1])
    atom_dim = int(targets[0]["atom_graph_node"].shape[1])
    model = train_cfg.build_model(
        in_dim,
        atom_dim,
        torch.device("cpu"),
        plm_dim=480,
        aux_plm_dim=320,
        partner_conditioning=True,
        partner_top_k=16,
        partner_direct_fusion=False,
        partner_logit_mode="residual",
        partner_delta_scale=0.25,
        partner_residue_encoder=True,
        partner_encoder_layers=1,
        partner_target_fusion=0.25,
        pair_contact_head="mlp",
    )

    model.eval()
    with torch.no_grad():
        true_heads, _ = forward(model, true_batch)
        mismatch_heads, _ = forward(model, mismatch_batch)
        null_heads, _ = forward(model, true_batch, null_partner=True)
        true_logits = 0.65 * true_heads[0] + 0.35 * true_heads[1]
        mismatch_logits = 0.65 * mismatch_heads[0] + 0.35 * mismatch_heads[1]
        null_logits = 0.65 * null_heads[0] + 0.35 * null_heads[1]
        true_null_delta = (true_logits - null_logits).abs()
        identity_delta = (true_logits - mismatch_logits).abs()
        assert float(true_null_delta.max()) <= 0.250001, (
            "bounded residual exceeded alpha"
        )
        assert float(true_null_delta.max()) > 1e-7, "partner residual did not change logits"
        assert float(identity_delta.max()) > 1e-7, "partner substitution did not change logits"

        model.partner_delta_scale = 0.0
        zero_heads, _ = forward(model, true_batch)
        zero_logits = 0.65 * zero_heads[0] + 0.35 * zero_heads[1]
        assert torch.allclose(zero_logits, null_logits, atol=1e-7, rtol=1e-6), (
            "alpha=0 control does not reproduce the intrinsic prediction"
        )
        model.partner_delta_scale = 0.25

        full_contract, _ = forward(
            model,
            true_batch,
            return_pair=True,
            return_pair_marginal=True,
            return_pair_logit_matrix=True,
        )
        assert len(full_contract) == 5, "residual pair/marginal return contract changed"
        assert full_contract[3].shape == true_logits.shape
        assert full_contract[4].shape[0] == true_logits.shape[0]

    model.train()
    model.zero_grad(set_to_none=True)
    true_pair_result, partner_feats = forward(
        model,
        true_batch,
        require_partner_grad=True,
        return_pair=True,
    )
    cls_logits, rank_logits, pair_output = true_pair_result
    loss = cls_logits.square().mean() + rank_logits.square().mean()
    pair_loss = train_cfg.sparse_pair_contact_loss(pair_output)
    assert pair_loss is not None and torch.isfinite(pair_loss), "pair loss is unavailable"
    (loss + 0.005 * pair_loss).backward()
    residual_gradient = float(sum(
        parameter.grad.abs().sum().item()
        for module in (
            model.partner_residual_gate,
            model.partner_residual_cls,
            model.partner_residual_rank_cls,
        )
        for parameter in module.parameters()
        if parameter.grad is not None
    ))
    partner_gradient = float(partner_feats.grad.abs().sum().item())
    assert residual_gradient > 0.0, "site loss did not reach residual modules"
    assert partner_gradient > 0.0, "site/pair loss did not reach partner features"

    print("PC-BIND v9 residual smoke test passed.")
    print(
        f"mean|true-null|={float(true_null_delta.mean()):.8f}, "
        f"mean|true-mismatch|={float(identity_delta.mean()):.8f}"
    )
    print(
        f"residual_gradient={residual_gradient:.6f}, "
        f"partner_feature_gradient={partner_gradient:.6f}, "
        f"pair_loss={float(pair_loss.detach()):.6f}"
    )


if __name__ == "__main__":
    main()
