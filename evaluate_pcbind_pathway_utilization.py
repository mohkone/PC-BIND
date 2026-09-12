import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import CROSS5FOLD_multi_test as train_cfg
from check_pcbind_prereqs import valid_partner_encoder_sample
from CROSS5FOLD_multi_test import (
    best_mcc_threshold,
    collate_proteins,
    configure_plm_feature_config,
    dataset_path,
    make_cv_folds,
    metrics_from_scores,
    save_json,
    smooth_node_scores,
)
from evaluate_mismatched_partner_control import make_mismatched_partner_list
from evaluate_pcbind_marginal import infer_run_seed, load_fold_model, load_pickle, make_dataset
from evaluate_pcbind_oof_mismatch import available_folds, cluster_bootstrap_auc_pr
from evaluate_saved_ensembles import checkpoint_rank_fusion
from metrics import compute_auc_pr


CONDITIONS = ("intrinsic", "true", "mismatch", "feature_shuffled")


def safe_correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if left.size < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def variance(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.var(values)) if values.size else float("nan")


def feature_shuffle_partner(partner_feats, partner_batch, rng):
    shuffled = partner_feats.clone()
    if partner_feats.numel() == 0:
        return shuffled
    for graph_id in torch.unique(partner_batch).tolist():
        indices = torch.nonzero(partner_batch == int(graph_id), as_tuple=False).flatten()
        if indices.numel() <= 1:
            continue
        permutation = torch.as_tensor(
            rng.permutation(indices.numel()),
            dtype=torch.long,
            device=indices.device,
        )
        shuffled[indices] = partner_feats[indices[permutation]]
    return shuffled


def predict_condition(
    model,
    checkpoint,
    samples,
    checkpoint_plm_dim,
    checkpoint_aux_plm_dim,
    batch_size,
    device,
    condition,
    shuffle_seed,
):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    dataset = make_dataset(
        samples,
        checkpoint,
        checkpoint_plm_dim,
        checkpoint_aux_plm_dim,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_proteins,
        num_workers=0,
    )
    rank_fusion = checkpoint_rank_fusion(checkpoint)
    rng = np.random.RandomState(shuffle_seed)
    labels = []
    raw_logits = []
    probabilities = []
    model.eval()

    with torch.no_grad():
        for (
            x, edge_index, atom_x, atom_edge_index, atom2res,
            res_coords, atom_coords, res_frames, surface_feats, sequence_feats,
            plm_feats, aux_plm_feats, partner_feats, partner_batch, partner_edge,
            partner_coords, partner_frames, _partner_pair_contact_index,
            _pair_contact_graphs, _partner_contact, _partner_mask, batch_vec, y,
        ) in loader:
            x = x.to(device)
            edge_index = edge_index.to(device)
            atom_x = atom_x.to(device)
            atom_edge_index = atom_edge_index.to(device)
            atom2res = atom2res.to(device)
            res_coords = res_coords.to(device)
            atom_coords = atom_coords.to(device)
            res_frames = res_frames.to(device)
            surface_feats = surface_feats.to(device)
            sequence_feats = sequence_feats.to(device)
            plm_feats = plm_feats.to(device)
            aux_plm_feats = aux_plm_feats.to(device)
            partner_feats = partner_feats.to(device)
            partner_batch = partner_batch.to(device)
            partner_edge = partner_edge.to(device)
            partner_coords = partner_coords.to(device)
            partner_frames = partner_frames.to(device)
            batch_vec = batch_vec.to(device)

            if condition == "intrinsic":
                partner_feats = partner_feats[:0]
                partner_batch = partner_batch[:0]
                partner_edge = partner_edge[:, :0]
                partner_coords = partner_coords[:0]
                partner_frames = partner_frames[:0]
            elif condition == "feature_shuffled":
                partner_feats = feature_shuffle_partner(partner_feats, partner_batch, rng)

            cls_logits, rank_logits = model(
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
                return_heads=True,
            )
            if not getattr(model, "use_rank_head", False):
                rank_logits = cls_logits
            fused_logits = (1.0 - rank_fusion) * cls_logits + rank_fusion * rank_logits
            fused_probs = smooth_node_scores(
                torch.sigmoid(fused_logits),
                edge_index,
            )
            labels.append(y.numpy().astype(np.float32))
            raw_logits.append(fused_logits.cpu().numpy().astype(np.float32))
            probabilities.append(fused_probs.cpu().numpy().astype(np.float32))

    return (
        np.concatenate(labels),
        np.concatenate(raw_logits),
        np.concatenate(probabilities),
    )


def summarize_pathways(raw, probabilities):
    z_intrinsic = raw["intrinsic"]
    z_true = raw["true"]
    z_mismatch = raw["mismatch"]
    z_shuffled = raw["feature_shuffled"]
    delta_true = z_true - z_intrinsic
    delta_mismatch = z_mismatch - z_intrinsic
    delta_shuffled = z_shuffled - z_intrinsic
    identity_delta = delta_true - delta_mismatch
    final_variance = variance(z_true)

    return {
        "variance": {
            "z_intrinsic": variance(z_intrinsic),
            "z_true": final_variance,
            "z_mismatch": variance(z_mismatch),
            "z_feature_shuffled": variance(z_shuffled),
            "partner_contribution_true": variance(delta_true),
            "partner_contribution_mismatch": variance(delta_mismatch),
            "partner_contribution_feature_shuffled": variance(delta_shuffled),
            "partner_identity_delta": variance(identity_delta),
            "true_partner_relative_to_final": (
                variance(delta_true) / final_variance if final_variance > 0.0 else float("nan")
            ),
        },
        "correlation": {
            "intrinsic_vs_true_final": safe_correlation(z_intrinsic, z_true),
            "true_partner_contribution_vs_true_final": safe_correlation(delta_true, z_true),
            "true_vs_mismatch_partner_contribution": safe_correlation(
                delta_true, delta_mismatch
            ),
            "true_vs_feature_shuffled_partner_contribution": safe_correlation(
                delta_true, delta_shuffled
            ),
            "true_vs_mismatch_probability": safe_correlation(
                probabilities["true"], probabilities["mismatch"]
            ),
        },
        "magnitude": {
            "mean_abs_true_partner_contribution": float(np.mean(np.abs(delta_true))),
            "mean_abs_mismatch_partner_contribution": float(
                np.mean(np.abs(delta_mismatch))
            ),
            "mean_abs_feature_shuffled_partner_contribution": float(
                np.mean(np.abs(delta_shuffled))
            ),
            "mean_abs_true_minus_mismatch_contribution": float(
                np.mean(np.abs(identity_delta))
            ),
            "mean_abs_true_minus_feature_shuffled_contribution": float(
                np.mean(np.abs(delta_true - delta_shuffled))
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Audit PC-BIND pathway utilization using grouped Train335 OOF predictions "
            "only. No external benchmark data are loaded."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mismatch-seed", type=int, default=12017)
    parser.add_argument("--shuffle-seed", type=int, default=13017)
    parser.add_argument("--neighbor-count", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    folds = available_folds(run_dir)
    if not folds:
        raise FileNotFoundError(f"No fold checkpoints found in {run_dir}")

    first_checkpoint = torch.load(
        os.path.join(run_dir, f"fold{folds[0]}_best.pt"),
        map_location="cpu",
        weights_only=False,
    )
    seed = infer_run_seed(
        run_dir,
        args.seed
        if args.seed is not None
        else int(first_checkpoint.get("seed", train_cfg.SEED)),
    )
    grouped_cv = bool(first_checkpoint.get("grouped_cv", False))
    group_key = str(first_checkpoint.get("cv_group_key", "complex_code"))
    if not grouped_cv:
        raise ValueError("Pathway audit requires grouped_cv=True checkpoints")

    train_list = load_pickle(dataset_path("Train335.pkl"))
    cv_folds = make_cv_folds(
        train_list,
        seed=seed,
        num_folds=5,
        grouped=True,
        group_key=group_key,
    )
    in_dim = train_list[0]["residue_graph_node"].shape[1]
    atom_dim = train_list[0]["atom_graph_node"].shape[1]
    plm_mode, plm_dim = configure_plm_feature_config([train_list])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Run dir: {run_dir}")
    print(f"OOF folds: {folds}")
    print(f"Fold seed: {seed}")
    print(f"CV grouping: {group_key}")
    print(f"PLM feature mode: {plm_mode} ({plm_dim})")
    print("Data scope: grouped Train335 OOF only; external benchmarks are not loaded")
    print(
        "Feature-shuffled condition: partner feature rows are permuted within each "
        "partner graph while geometry and graph edges remain fixed"
    )

    labels_parts = []
    raw_parts = {condition: [] for condition in CONDITIONS}
    probability_parts = {condition: [] for condition in CONDITIONS}
    sample_records = []
    mismatch_mapping = []
    excluded = []
    residue_offset = 0

    for fold in folds:
        fold_samples = []
        for sample_idx in cv_folds[fold - 1]:
            sample = train_list[int(sample_idx)]
            if valid_partner_encoder_sample(sample):
                fold_samples.append(sample)
            else:
                excluded.append({
                    "fold": fold,
                    "index": int(sample_idx),
                    "complex_code": str(sample.get("complex_code", "?")),
                })
        if len(fold_samples) < 2:
            raise ValueError(f"Fold {fold} has fewer than two partner-complete samples")
        mismatch_samples, fold_mapping = make_mismatched_partner_list(
            fold_samples,
            seed=args.mismatch_seed + fold - 1,
            neighbor_count=args.neighbor_count,
        )
        for item in fold_mapping:
            mismatch_mapping.append({"fold": fold, **item})

        model, checkpoint, checkpoint_plm_dim, checkpoint_aux_plm_dim = load_fold_model(
            run_dir,
            fold,
            in_dim,
            atom_dim,
            plm_dim,
            device,
        )
        fold_outputs = {}
        for condition in CONDITIONS:
            condition_samples = mismatch_samples if condition == "mismatch" else fold_samples
            fold_outputs[condition] = predict_condition(
                model,
                checkpoint,
                condition_samples,
                checkpoint_plm_dim,
                checkpoint_aux_plm_dim,
                args.batch_size,
                device,
                condition,
                args.shuffle_seed + 100 * fold,
            )
        fold_labels = fold_outputs["true"][0]
        for condition in CONDITIONS:
            condition_labels, condition_raw, condition_probabilities = fold_outputs[condition]
            if not np.array_equal(fold_labels, condition_labels):
                raise RuntimeError(f"Fold {fold} labels differ for condition {condition}")
            raw_parts[condition].append(condition_raw)
            probability_parts[condition].append(condition_probabilities)
        labels_parts.append(fold_labels)

        for sample in fold_samples:
            length = int(len(sample["label"]))
            sample_records.append({
                "fold": fold,
                "complex_code": str(
                    sample.get(group_key, sample.get("complex_code", "?"))
                ),
                "start": residue_offset,
                "end": residue_offset + length,
            })
            residue_offset += length
        fold_true = fold_outputs["true"][2]
        fold_intrinsic = fold_outputs["intrinsic"][2]
        fold_mismatch = fold_outputs["mismatch"][2]
        print(
            f"Fold {fold}: samples={len(fold_samples)}, residues={fold_labels.size}, "
            f"true-intrinsic AUPRC={compute_auc_pr(fold_labels, fold_true) - compute_auc_pr(fold_labels, fold_intrinsic):+.4f}, "
            f"true-mismatch AUPRC={compute_auc_pr(fold_labels, fold_true) - compute_auc_pr(fold_labels, fold_mismatch):+.4f}"
        )

    labels = np.concatenate(labels_parts)
    raw = {condition: np.concatenate(raw_parts[condition]) for condition in CONDITIONS}
    probabilities = {
        condition: np.concatenate(probability_parts[condition]) for condition in CONDITIONS
    }

    grouped_segments = {}
    for record in sample_records:
        grouped_segments.setdefault(record["complex_code"], []).append(
            np.arange(record["start"], record["end"], dtype=np.int64)
        )
    cluster_codes = sorted(grouped_segments)
    cluster_indices = [np.concatenate(grouped_segments[code]) for code in cluster_codes]

    true_threshold = best_mcc_threshold(labels, probabilities["true"])
    performance = {}
    for condition in CONDITIONS:
        performance[condition] = metrics_from_scores(
            labels,
            probabilities[condition],
            split_name=f"OOF-PathwayAudit-{condition}",
            threshold=true_threshold,
        )
    performance["true_threshold"] = float(true_threshold)
    performance["aupr_differences"] = {
        "true_minus_intrinsic": float(
            performance["true"]["auc_pr"] - performance["intrinsic"]["auc_pr"]
        ),
        "true_minus_mismatch": float(
            performance["true"]["auc_pr"] - performance["mismatch"]["auc_pr"]
        ),
        "true_minus_feature_shuffled": float(
            performance["true"]["auc_pr"]
            - performance["feature_shuffled"]["auc_pr"]
        ),
    }
    performance["mcc_differences_at_true_threshold"] = {
        "true_minus_intrinsic": float(
            performance["true"]["mcc"] - performance["intrinsic"]["mcc"]
        ),
        "true_minus_mismatch": float(
            performance["true"]["mcc"] - performance["mismatch"]["mcc"]
        ),
        "true_minus_feature_shuffled": float(
            performance["true"]["mcc"]
            - performance["feature_shuffled"]["mcc"]
        ),
    }

    pathway = summarize_pathways(raw, probabilities)
    bootstrap = {
        "true_minus_intrinsic": cluster_bootstrap_auc_pr(
            labels,
            probabilities["true"],
            probabilities["intrinsic"],
            cluster_indices,
            args.bootstrap_replicates,
            args.mismatch_seed + 2000,
        ),
        "true_minus_mismatch": cluster_bootstrap_auc_pr(
            labels,
            probabilities["true"],
            probabilities["mismatch"],
            cluster_indices,
            args.bootstrap_replicates,
            args.mismatch_seed + 2001,
        ),
        "true_minus_feature_shuffled": cluster_bootstrap_auc_pr(
            labels,
            probabilities["true"],
            probabilities["feature_shuffled"],
            cluster_indices,
            args.bootstrap_replicates,
            args.mismatch_seed + 2002,
        ),
    }

    per_complex = []
    delta_true = raw["true"] - raw["intrinsic"]
    delta_mismatch = raw["mismatch"] - raw["intrinsic"]
    for code, indices in zip(cluster_codes, cluster_indices):
        per_complex.append({
            "complex_code": code,
            "residues": int(indices.size),
            "variance_intrinsic": variance(raw["intrinsic"][indices]),
            "variance_true_final": variance(raw["true"][indices]),
            "variance_true_partner_contribution": variance(delta_true[indices]),
            "mean_abs_true_partner_contribution": float(
                np.mean(np.abs(delta_true[indices]))
            ),
            "mean_abs_true_minus_mismatch_contribution": float(
                np.mean(np.abs(delta_true[indices] - delta_mismatch[indices]))
            ),
            "intrinsic_vs_true_final_correlation": safe_correlation(
                raw["intrinsic"][indices], raw["true"][indices]
            ),
            "true_vs_mismatch_contribution_correlation": safe_correlation(
                delta_true[indices], delta_mismatch[indices]
            ),
        })

    predictions_path = os.path.join(output_dir, "pcbind_pathway_predictions.npz")
    np.savez_compressed(
        predictions_path,
        labels=labels.astype(np.float32),
        **{f"raw_{key}": value.astype(np.float32) for key, value in raw.items()},
        **{
            f"probability_{key}": value.astype(np.float32)
            for key, value in probabilities.items()
        },
        delta_true=(raw["true"] - raw["intrinsic"]).astype(np.float32),
        delta_mismatch=(raw["mismatch"] - raw["intrinsic"]).astype(np.float32),
        delta_feature_shuffled=(
            raw["feature_shuffled"] - raw["intrinsic"]
        ).astype(np.float32),
    )
    output = {
        "run_dir": run_dir,
        "seed": seed,
        "folds": folds,
        "grouped_cv": True,
        "cv_group_key": group_key,
        "data_scope": "Train335 grouped out-of-fold validation only",
        "external_benchmarks_loaded": False,
        "conditions": {
            "intrinsic": "same checkpoint with an empty partner tensor",
            "true": "observed partner",
            "mismatch": "length-matched partner donor from the same validation fold",
            "feature_shuffled": (
                "partner feature rows permuted within each graph while coordinates, "
                "frames, and graph edges remain fixed"
            ),
        },
        "mismatch_seed": args.mismatch_seed,
        "shuffle_seed": args.shuffle_seed,
        "neighbor_count": args.neighbor_count,
        "evaluated_samples": len(sample_records),
        "evaluated_clusters": len(cluster_codes),
        "evaluated_residues": int(labels.size),
        "excluded": excluded,
        "performance": performance,
        "pathway_utilization": pathway,
        "cluster_bootstrap_auc_pr": bootstrap,
        "per_complex": per_complex,
        "mismatch_mapping": mismatch_mapping,
        "predictions_path": predictions_path,
    }
    summary_path = os.path.join(output_dir, "pcbind_pathway_utilization_summary.json")
    save_json(summary_path, output)

    print("\n------------ Pathway Utilization Summary ------------")
    print(
        f"AUPRC true={performance['true']['auc_pr']:.4f}, "
        f"intrinsic={performance['intrinsic']['auc_pr']:.4f}, "
        f"mismatch={performance['mismatch']['auc_pr']:.4f}, "
        f"feature_shuffled={performance['feature_shuffled']['auc_pr']:.4f}"
    )
    print(
        f"AUPRC true-intrinsic={performance['aupr_differences']['true_minus_intrinsic']:+.4f}, "
        f"true-mismatch={performance['aupr_differences']['true_minus_mismatch']:+.4f}, "
        f"true-feature_shuffled={performance['aupr_differences']['true_minus_feature_shuffled']:+.4f}"
    )
    print(
        f"Var(z_intrinsic)={pathway['variance']['z_intrinsic']:.6f}, "
        f"Var(z_true)={pathway['variance']['z_true']:.6f}, "
        f"Var(delta_true)={pathway['variance']['partner_contribution_true']:.6f}, "
        f"relative={pathway['variance']['true_partner_relative_to_final']:.6f}"
    )
    print(
        f"corr(z_intrinsic,z_true)={pathway['correlation']['intrinsic_vs_true_final']:.6f}, "
        f"corr(delta_true,delta_mismatch)={pathway['correlation']['true_vs_mismatch_partner_contribution']:.6f}"
    )
    print(
        f"mean|delta_true|={pathway['magnitude']['mean_abs_true_partner_contribution']:.6f}, "
        f"mean|delta_true-delta_mismatch|="
        f"{pathway['magnitude']['mean_abs_true_minus_mismatch_contribution']:.6f}"
    )
    for name, item in bootstrap.items():
        if "ci95_low" in item:
            print(
                f"{name}: {item['observed_drop']:+.4f} "
                f"95% CI [{item['ci95_low']:+.4f}, {item['ci95_high']:+.4f}]"
            )
    print(f"Saved pathway summary to: {summary_path}")
    print(f"Saved raw pathway predictions to: {predictions_path}")


if __name__ == "__main__":
    main()
