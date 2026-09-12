import copy
import os
import pickle

import numpy as np
import torch


os.environ.setdefault("PPI_PLM_DIM", "480")
os.environ.setdefault("PPI_AUX_PLM_DIM", "320")
os.environ["PPI_PARTNER_CONDITIONING"] = "1"
os.environ["PPI_PARTNER_RESIDUE_ENCODER"] = "1"
os.environ["PPI_PARTNER_DIRECT_FUSION"] = "1"
os.environ["PPI_PARTNER_CONTRAST"] = "0"
os.environ["PPI_PAIR_CONTACT_LOSS"] = "1"
os.environ["PPI_PAIR_MARGINAL_CONSISTENCY"] = "1"
os.environ["PPI_PAIR_MARGINAL_CONTRAST"] = "1"
os.environ["PPI_PAIR_CONTACT_CONTRAST"] = "1"

import CROSS5FOLD_multi_test as train_cfg
from check_pcbind_prereqs import valid_partner_encoder_sample


PARTNER_FIELDS = (
    "partner_residue_surface_features",
    "partner_residue_sequence_features",
    "partner_residue_coords",
    "partner_residue_frames",
    "partner_residue_geo_edge",
    "partner_residue_sequences",
    "partner_residue_chains",
    "partner_chain_ids",
    "partner_chain_slices",
    "partner_residue_plm_embedding",
    "partner_residue_plm_model",
    "partner_residue_plm_embedding_8m",
    "partner_residue_plm_model_8m",
)


def load_samples(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def sample_size(sample):
    return (
        int(sample["residue_graph_node"].shape[0])
        + int(sample["atom_graph_node"].shape[0])
        + int(sample["partner_residue_surface_features"].shape[0])
    )


def choose_target_and_donor(samples):
    valid = [sample for sample in samples if valid_partner_encoder_sample(sample)]
    valid.sort(key=sample_size)
    target = valid[0]
    target_group = str(target.get("complex_code", "")).upper()
    donor = next(
        sample
        for sample in valid[1:]
        if str(sample.get("complex_code", "")).upper() != target_group
    )
    return target, donor


def mismatched_copy(target, donor):
    sample = copy.copy(target)
    for key in PARTNER_FIELDS:
        if key in donor:
            sample[key] = donor[key]
        else:
            sample.pop(key, None)
    sample["partner_pair_contact_index"] = np.empty((2, 0), dtype=np.int64)
    return sample


def make_batch(sample):
    dataset = train_cfg.ProteinDataset(
        [sample],
        plm_feature_dim=480,
        aux_plm_feature_dim=320,
    )
    return train_cfg.collate_proteins([dataset[0]])


def forward_batch(
    model,
    batch,
    require_partner_grad=False,
    return_pair=False,
    return_pair_marginal=False,
    return_aux=False,
    return_heads=False,
    raw_output=False,
):
    (
        x,
        edge_index,
        atom_x,
        atom_edge_index,
        atom2res,
        res_coords,
        atom_coords,
        res_frames,
        surface_feats,
        sequence_feats,
        plm_feats,
        aux_plm_feats,
        partner_feats,
        partner_batch,
        partner_edge,
        partner_coords,
        partner_frames,
        partner_pair_contact_index,
        pair_contact_graphs,
        _partner_contact,
        _partner_mask,
        batch_vec,
        _labels,
    ) = batch
    if require_partner_grad:
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
        return_pair=return_pair,
        return_pair_marginal=return_pair_marginal,
        return_aux=return_aux,
        return_heads=return_heads,
    )
    if raw_output:
        return output
    if return_pair and return_pair_marginal:
        logits, pair_output, pair_marginal = output
        return logits, partner_feats, pair_output, pair_marginal, batch_vec, pair_contact_graphs
    if return_pair:
        logits, pair_output = output
        return logits, partner_feats, pair_output
    return output, partner_feats


def main():
    torch.manual_seed(17)
    samples = load_samples(os.path.join("data", "geo", "Train335.pkl"))
    target, donor = choose_target_and_donor(samples)
    mismatch = mismatched_copy(target, donor)
    true_batch = make_batch(target)
    mismatch_batch = make_batch(mismatch)

    in_dim = int(target["residue_graph_node"].shape[1])
    atom_dim = int(target["atom_graph_node"].shape[1])
    model = train_cfg.build_model(
        in_dim,
        atom_dim,
        torch.device("cpu"),
        plm_dim=480,
        aux_plm_dim=320,
        partner_conditioning=True,
        partner_direct_fusion=True,
        partner_residue_encoder=True,
        partner_encoder_layers=1,
        partner_target_fusion=0.25,
        pair_contact_head="mlp",
    )

    paired_dataset = train_cfg.ProteinDataset(
        [target, donor],
        plm_feature_dim=480,
        aux_plm_feature_dim=320,
    )
    paired_batch = train_cfg.collate_proteins([paired_dataset[0], paired_dataset[1]])
    (
        paired_x, paired_edge, paired_atom_x, paired_atom_edge, paired_atom2res,
        paired_res_coords, paired_atom_coords, paired_res_frames,
        paired_surface, paired_sequence, paired_plm, paired_aux_plm,
        paired_partner_feats, paired_partner_batch, paired_partner_edge,
        paired_partner_coords, paired_partner_frames, paired_pair_index,
        paired_pair_graphs, _paired_contact, _paired_mask, paired_batch_vec,
        paired_labels,
    ) = paired_batch
    (
        mismatch_partner_feats,
        mismatch_partner_batch,
        mismatch_partner_edge,
        mismatch_partner_coords,
        mismatch_partner_frames,
        has_mismatch,
    ) = train_cfg.make_mismatched_partner_graph_batch(
        paired_partner_feats,
        paired_partner_batch,
        paired_partner_edge,
        paired_partner_coords,
        paired_partner_frames,
        paired_batch_vec,
    )
    assert has_mismatch, "two-sample smoke batch did not produce mismatched partners"
    assert mismatch_partner_coords.shape == paired_partner_coords.shape
    assert mismatch_partner_frames.shape == paired_partner_frames.shape
    if mismatch_partner_edge.numel() > 0:
        assert mismatch_partner_edge.min().item() >= 0
        assert mismatch_partner_edge.max().item() < mismatch_partner_feats.size(0)

    model.train()
    model.zero_grad(set_to_none=True)
    true_cls, true_rank, true_pair_output, true_marginal = model(
        paired_x, paired_edge, paired_atom_x, paired_atom_edge, paired_atom2res,
        paired_batch_vec, paired_res_coords, paired_atom_coords, paired_res_frames,
        paired_surface, paired_sequence, paired_plm, paired_aux_plm,
        paired_partner_feats, paired_partner_batch,
        partner_edge=paired_partner_edge,
        partner_coords=paired_partner_coords,
        partner_frames=paired_partner_frames,
        pair_contact_index=paired_pair_index,
        pair_contact_graphs=paired_pair_graphs,
        return_heads=True,
        return_pair=True,
        return_pair_marginal=True,
    )
    mismatch_cls, mismatch_rank, mismatch_pair_output, mismatch_marginal = model(
        paired_x, paired_edge, paired_atom_x, paired_atom_edge, paired_atom2res,
        paired_batch_vec, paired_res_coords, paired_atom_coords, paired_res_frames,
        paired_surface, paired_sequence, paired_plm, paired_aux_plm,
        mismatch_partner_feats, mismatch_partner_batch,
        partner_edge=mismatch_partner_edge,
        partner_coords=mismatch_partner_coords,
        partner_frames=mismatch_partner_frames,
        pair_contact_graphs=paired_pair_graphs,
        return_heads=True,
        return_pair=True,
        return_pair_marginal=True,
    )
    marginal_contrast = train_cfg.pair_marginal_partner_contrastive_loss(
        true_marginal,
        mismatch_marginal,
        paired_labels,
        paired_batch_vec,
        paired_pair_graphs,
    )
    assert marginal_contrast is not None and torch.isfinite(marginal_contrast), (
        "pair-marginal partner contrast was not finite"
    )
    marginal_contrast.backward(retain_graph=True)
    marginal_contrast_grad = float(sum(
        parameter.grad.abs().sum().item()
        for parameter in model.pair_scorer.parameters()
        if parameter.grad is not None
    ))
    assert marginal_contrast_grad > 0.0, (
        "pair-marginal partner contrast did not reach the pair-contact head"
    )
    model.zero_grad(set_to_none=True)
    contact_contrast, contact_differences = train_cfg.pair_contact_hard_partner_loss(
        true_pair_output,
        mismatch_pair_output,
        paired_labels,
        paired_batch_vec,
    )
    assert contact_contrast is not None and torch.isfinite(contact_contrast), (
        "pair-contact hard partner contrast was not finite"
    )
    assert contact_differences is not None and contact_differences.numel() > 0, (
        "pair-contact hard partner contrast found no eligible contact residues"
    )
    contact_contrast.backward()
    contact_contrast_grad = float(sum(
        parameter.grad.abs().sum().item()
        for parameter in model.pair_scorer.parameters()
        if parameter.grad is not None
    ))
    assert contact_contrast_grad > 0.0, (
        "pair-contact hard partner contrast did not reach the pair-contact head"
    )

    model.eval()
    with torch.no_grad():
        true_logits, _ = forward_batch(model, true_batch)
        mismatch_logits, _ = forward_batch(model, mismatch_batch)
        training_contract = forward_batch(
            model,
            true_batch,
            return_pair=True,
            return_pair_marginal=True,
            return_aux=True,
            return_heads=True,
            raw_output=True,
        )
    assert len(training_contract) == 5, (
        "two-head pair/marginal training forward must return five outputs"
    )
    mean_abs_delta = float(torch.mean(torch.abs(true_logits - mismatch_logits)).item())
    max_abs_delta = float(torch.max(torch.abs(true_logits - mismatch_logits)).item())

    model.train()
    model.zero_grad(set_to_none=True)
    grad_logits, partner_feats, pair_output, pair_marginal, batch_vec, pair_contact_graphs = forward_batch(
        model,
        true_batch,
        require_partner_grad=True,
        return_pair=True,
        return_pair_marginal=True,
    )
    pair_logits, pair_targets = pair_output[:2]
    assert pair_logits.numel() > 0, "pair-contact head returned no labelled candidates"
    assert pair_targets.min().item() == 0.0 and pair_targets.max().item() == 1.0, (
        "pair-contact smoke batch must contain positive and negative candidates"
    )
    pair_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pair_logits,
        pair_targets,
        pos_weight=pair_logits.new_tensor(8.0),
    )
    marginal_consistency = train_cfg.pair_marginal_consistency_loss(
        pair_marginal,
        grad_logits,
        None,
        batch_vec,
        pair_contact_graphs,
    )
    assert marginal_consistency is not None and torch.isfinite(marginal_consistency), (
        "pair-marginal consistency loss was not finite"
    )
    pair_marginal.retain_grad()
    (grad_logits.square().mean() + 0.005 * pair_loss + 0.02 * marginal_consistency).backward()
    partner_grad = float(partner_feats.grad.abs().sum().item())
    shared_grad = float(sum(
        parameter.grad.abs().sum().item()
        for parameter in model.shared_residue_convs.parameters()
        if parameter.grad is not None
    ))
    pair_head_grad = float(sum(
        parameter.grad.abs().sum().item()
        for parameter in model.pair_scorer.parameters()
        if parameter.grad is not None
    ))
    marginal_grad = float(pair_marginal.grad.abs().sum().item())

    assert mean_abs_delta > 1e-8, "mismatched partner did not change target logits"
    assert partner_grad > 0.0, "no gradient reached partner residue features"
    assert shared_grad > 0.0, "no gradient reached shared residue encoder"
    assert pair_head_grad > 0.0, "no gradient reached pair-contact head"
    assert marginal_grad > 0.0, "no gradient passed through the pair-contact marginal"
    print("PC-BIND v5 smoke test passed.")
    print(f"Target: {target.get('complex_code')} | donor: {donor.get('complex_code')}")
    print(
        f"Target residues={true_logits.numel()}, "
        f"true partner residues={true_batch[12].size(0)}, "
        f"mismatched partner residues={mismatch_batch[12].size(0)}"
    )
    print(f"Logit delta: mean_abs={mean_abs_delta:.8f}, max_abs={max_abs_delta:.8f}")
    print(f"Gradient sums: partner_features={partner_grad:.6f}, shared_encoder={shared_grad:.6f}")
    print(
        f"Pair-contact candidates={pair_logits.numel()}, "
        f"positives={int(pair_targets.sum().item())}, "
        f"loss={pair_loss.item():.6f}, head_gradient={pair_head_grad:.6f}"
    )
    print(
        f"Marginal consistency loss={marginal_consistency.item():.6f}, "
        f"marginal_gradient={marginal_grad:.6f}"
    )
    print(
        f"Marginal partner contrast={marginal_contrast.item():.6f}, "
        f"pair_head_gradient={marginal_contrast_grad:.6f}"
    )
    print(
        f"Contact hard-partner contrast={contact_contrast.item():.6f}, "
        f"eligible_residues={contact_differences.numel()}, "
        f"mean_D={contact_differences.mean().item():.6f}, "
        f"pair_head_gradient={contact_contrast_grad:.6f}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = train_cfg.TwoHeadBindingLoss(
        pos_weight=torch.tensor(5.0, dtype=torch.float32),
    )
    train_loss = train_cfg.train_one_epoch(
        model,
        [paired_batch],
        optimizer,
        criterion,
        torch.device("cpu"),
        epoch=1,
        fold_idx=0,
        ema=None,
    )
    assert np.isfinite(train_loss), "v8 one-step training loss was not finite"
    print(f"V8 one-step training loss={train_loss:.6f}")


if __name__ == "__main__":
    main()
