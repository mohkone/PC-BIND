import argparse
import json
import os
import pickle

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import CROSS5FOLD_multi_test as train_cfg
from check_pcbind_prereqs import valid_partner_encoder_sample
from CROSS5FOLD_multi_test import (
    GEOMETRY_FEATURE_MODE,
    SURFACE_FEATURE_MODE,
    SEQUENCE_FEATURE_MODE,
    ProteinDataset,
    best_mcc_threshold,
    build_model,
    collate_proteins,
    combine_fold_scores,
    configure_plm_feature_config,
    dataset_path,
    make_cv_folds,
    metrics_from_scores,
    normalized_positive_weights,
    rank_normalize,
    save_json,
    smooth_node_scores,
    write_metrics_csv,
)
from evaluate_saved_ensembles import (
    apply_checkpoint_runtime_config,
    checkpoint_rank_fusion,
    load_compatible_state,
)
from metrics import compute_auc_pr


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_run_seed(run_dir, fallback_seed):
    summary_path = os.path.join(run_dir, "ensemble_summary.json")
    if os.path.exists(summary_path):
        summary = load_json(summary_path)
        if "seed" in summary:
            return int(summary["seed"])
    ckpt_path = os.path.join(run_dir, "fold1_best.pt")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "seed" in checkpoint:
            return int(checkpoint["seed"])
    return int(fallback_seed)


def make_dataset(data_list, checkpoint, checkpoint_plm_dim, checkpoint_aux_plm_dim):
    return ProteinDataset(
        data_list,
        checkpoint["feat_mean"],
        checkpoint["feat_std"],
        checkpoint["atom_feat_mean"],
        checkpoint["atom_feat_std"],
        checkpoint.get("surface_feat_mean"),
        checkpoint.get("surface_feat_std"),
        checkpoint.get("sequence_feat_mean"),
        checkpoint.get("sequence_feat_std"),
        checkpoint.get("plm_feat_mean"),
        checkpoint.get("plm_feat_std"),
        checkpoint.get("plm_feature_dim", checkpoint_plm_dim),
        checkpoint.get("aux_plm_feat_mean"),
        checkpoint.get("aux_plm_feat_std"),
        checkpoint.get("aux_plm_feature_dim", checkpoint_aux_plm_dim),
    )


def load_fold_model(run_dir, fold, in_dim, atom_dim, plm_dim, device):
    ckpt_path = os.path.join(run_dir, f"fold{fold}_best.pt")
    print(f"Loading {ckpt_path}", flush=True)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    checkpoint_plm_dim = checkpoint.get("plm_feature_dim")
    checkpoint_aux_plm_dim = checkpoint.get("aux_plm_feature_dim")
    checkpoint_plm_dim = int(plm_dim if checkpoint_plm_dim is None else checkpoint_plm_dim)
    checkpoint_aux_plm_dim = int(0 if checkpoint_aux_plm_dim is None else checkpoint_aux_plm_dim)
    model = build_model(
        in_dim,
        atom_dim,
        device,
        plm_dim=checkpoint_plm_dim,
        aux_plm_dim=checkpoint_aux_plm_dim,
        partner_conditioning=bool(checkpoint.get("partner_conditioning", False)),
        partner_top_k=int(checkpoint.get("partner_top_k", train_cfg.PARTNER_TOP_K)),
        partner_direct_fusion=bool(checkpoint.get("partner_direct_fusion", False)),
        partner_logit_mode=checkpoint.get("partner_logit_mode", "standard"),
        partner_delta_scale=float(checkpoint.get("partner_delta_scale", 1.0)),
        partner_residue_encoder=bool(checkpoint.get("partner_residue_encoder", False)),
        partner_encoder_layers=int(
            checkpoint.get("partner_encoder_layers", train_cfg.PARTNER_ENCODER_LAYERS)
        ),
        partner_target_fusion=float(
            checkpoint.get("partner_target_fusion", train_cfg.PARTNER_TARGET_FUSION)
        ),
        pair_contact_head=checkpoint.get("pair_contact_head", "attention"),
    )
    missing, unexpected, skipped = load_compatible_state(model, checkpoint["model_state"])
    apply_checkpoint_runtime_config(model, checkpoint)
    if missing or unexpected or skipped:
        print(
            f"Checkpoint compatibility: missing={missing}, unexpected={unexpected}, skipped={skipped}",
            flush=True,
        )
    return model, checkpoint, checkpoint_plm_dim, checkpoint_aux_plm_dim


def predict_scores_with_marginal(
    model,
    loader,
    device,
    rank_fusion=None,
    smooth_marginal=True,
    return_contact_scores=False,
    contact_hard_k=10,
):
    model.eval()
    if rank_fusion is None:
        rank_fusion = train_cfg.TWO_HEAD_RANK_FUSION
    all_labels = []
    all_fused_probs = []
    all_aux_probs = []
    all_cls_probs = []
    all_rank_probs = []
    all_marginal_probs = []
    all_pair_logit_matrices = []
    all_positive_contact_scores = []
    all_hard_candidate_scores = []

    with torch.no_grad():
        for (
            x, edge_index, atom_x, atom_edge_index, atom2res,
            res_coords, atom_coords, res_frames, surface_feats, sequence_feats, plm_feats, aux_plm_feats,
            partner_feats, partner_batch, partner_edge, partner_coords, partner_frames,
            partner_pair_contact_index, pair_contact_graphs,
            partner_contact, partner_mask, batch_vec, y
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
            partner_pair_contact_index = partner_pair_contact_index.to(device)
            pair_contact_graphs = pair_contact_graphs.to(device)
            batch_vec = batch_vec.to(device)
            y_device = y.to(device)

            model_output = model(
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
                return_aux=True,
                return_heads=True,
                return_pair=return_contact_scores,
                return_pair_marginal=True,
                return_pair_logit_matrix=True,
            )
            if return_contact_scores:
                (
                    cls_logits,
                    rank_logits,
                    aux_logits,
                    pair_output,
                    pair_marginal,
                    pair_logit_matrix,
                ) = model_output
                positive_scores, hard_scores = train_cfg.pair_contact_score_vectors(
                    pair_output,
                    y_device,
                    y_device.numel(),
                    hard_k=contact_hard_k,
                )
                if positive_scores is None:
                    positive_scores = pair_marginal.new_full(pair_marginal.shape, float("nan"))
                if hard_scores is None:
                    hard_scores = pair_marginal.new_full(pair_marginal.shape, float("nan"))
                all_positive_contact_scores.append(
                    positive_scores.cpu().numpy().astype(np.float32)
                )
                all_hard_candidate_scores.append(
                    hard_scores.cpu().numpy().astype(np.float32)
                )
            else:
                cls_logits, rank_logits, aux_logits, pair_marginal, pair_logit_matrix = model_output
            if not getattr(model, "use_rank_head", False):
                rank_logits = cls_logits
            fused_logits = (1.0 - rank_fusion) * cls_logits + rank_fusion * rank_logits

            fused_probs = smooth_node_scores(torch.sigmoid(fused_logits), edge_index).cpu().numpy()
            cls_probs = smooth_node_scores(torch.sigmoid(cls_logits), edge_index).cpu().numpy()
            rank_probs = smooth_node_scores(torch.sigmoid(rank_logits), edge_index).cpu().numpy()
            if aux_logits is None:
                aux_probs = fused_probs
            else:
                aux_probs = smooth_node_scores(torch.sigmoid(aux_logits), edge_index).cpu().numpy()
            pair_marginal = pair_marginal.clamp(0.0, 1.0)
            if smooth_marginal:
                pair_marginal = smooth_node_scores(pair_marginal, edge_index).clamp(0.0, 1.0)
            marginal_probs = pair_marginal.cpu().numpy()
            pair_logit_matrix = pair_logit_matrix.cpu().numpy().astype(np.float32)

            all_fused_probs.append(fused_probs)
            all_aux_probs.append(aux_probs)
            all_cls_probs.append(cls_probs)
            all_rank_probs.append(rank_probs)
            all_marginal_probs.append(marginal_probs)
            all_pair_logit_matrices.append(pair_logit_matrix)
            all_labels.append(y.numpy())

    output = (
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_fused_probs, axis=0),
        np.concatenate(all_aux_probs, axis=0),
        np.concatenate(all_cls_probs, axis=0),
        np.concatenate(all_rank_probs, axis=0),
        np.concatenate(all_marginal_probs, axis=0),
        np.concatenate(all_pair_logit_matrices, axis=0),
    )
    if return_contact_scores:
        return output + (
            np.concatenate(all_positive_contact_scores, axis=0),
            np.concatenate(all_hard_candidate_scores, axis=0),
        )
    return output


def marginal_stack_features(site_score, aux_score, marginal_score):
    site_score = np.asarray(site_score, dtype=np.float64)
    aux_score = np.asarray(aux_score, dtype=np.float64)
    marginal_score = np.asarray(marginal_score, dtype=np.float64)
    eps = 1e-8
    return np.column_stack(
        (
            site_score,
            aux_score,
            marginal_score,
            rank_normalize(site_score),
            rank_normalize(aux_score),
            rank_normalize(marginal_score),
            np.maximum(site_score, marginal_score),
            np.minimum(site_score, marginal_score),
            np.abs(site_score - marginal_score),
            np.sqrt(np.clip(site_score * marginal_score, eps, None)),
            np.log(np.clip(site_score, eps, 1.0)),
            np.log(np.clip(marginal_score, eps, 1.0)),
        )
    )


def sigmoid_np(values):
    values = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-values))


def calibrated_marginal_from_logits(pair_logit_matrix, scale=1.0, bias=0.0):
    logits = np.asarray(pair_logit_matrix, dtype=np.float64)
    finite = np.isfinite(logits)
    probs = np.zeros_like(logits, dtype=np.float64)
    if finite.any():
        probs[finite] = sigmoid_np(float(scale) * logits[finite] + float(bias))
    no_contact = np.prod(1.0 - probs, axis=1)
    return np.clip(1.0 - no_contact, 0.0, 1.0)


def select_pair_logit_calibration(y_true, pair_logit_matrix):
    scale_grid = np.array([0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 6.00])
    bias_grid = np.linspace(-10.0, 4.0, 57)
    best = {
        "scale": 1.0,
        "bias": 0.0,
        "auc_pr": -np.inf,
        "average_precision": -np.inf,
        "score_std": 0.0,
        "unique_score_count": 0,
        "score": calibrated_marginal_from_logits(pair_logit_matrix, 1.0, 0.0),
    }
    for scale in scale_grid:
        for bias in bias_grid:
            score = calibrated_marginal_from_logits(pair_logit_matrix, scale, bias)
            score_std = float(np.std(score))
            unique_score_count = int(np.unique(np.round(score, decimals=12)).size)
            if score_std <= 1e-10 or unique_score_count < 2:
                continue
            average_precision = float(average_precision_score(y_true, score))
            auc_pr = compute_auc_pr(y_true, score)
            if (average_precision, auc_pr) > (best["average_precision"], best["auc_pr"]):
                best = {
                    "scale": float(scale),
                    "bias": float(bias),
                    "auc_pr": float(auc_pr),
                    "average_precision": average_precision,
                    "score_std": score_std,
                    "unique_score_count": unique_score_count,
                    "score": score,
                }
    if not np.isfinite(best["average_precision"]):
        raise RuntimeError("No non-degenerate pair-logit calibration candidate was found")
    return best


def select_alpha(y_true, site_score, marginal_score, rank_space=False):
    alphas = np.linspace(0.0, 1.0, 41)
    if rank_space:
        site_score = rank_normalize(site_score)
        marginal_score = rank_normalize(marginal_score)
    scores = [
        compute_auc_pr(y_true, (1.0 - alpha) * site_score + alpha * marginal_score)
        for alpha in alphas
    ]
    best_idx = int(np.argmax(scores))
    alpha = float(alphas[best_idx])
    blend = (1.0 - alpha) * site_score + alpha * marginal_score
    return alpha, float(scores[best_idx]), blend


def add_metrics(rows, summary, group, test_name, metrics):
    summary.setdefault(group, {})[test_name] = metrics
    rows.append({
        "group": group,
        "test_set": test_name,
        "acc": metrics["acc"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "mcc": metrics["mcc"],
        "auc_roc": metrics["auc_roc"],
        "auc_pr": metrics["auc_pr"],
        "threshold": metrics["t_opt"],
    })


def print_strategy_ranking(summary):
    ranked = []
    for name, by_test in summary.items():
        auprs = [metrics["auc_pr"] for metrics in by_test.values()]
        mccs = [metrics["mcc"] for metrics in by_test.values()]
        ranked.append((
            name,
            float(np.mean(auprs)),
            float(np.mean(mccs)),
            float(np.min(auprs)),
            float(np.min(mccs)),
        ))
    ranked.sort(key=lambda row: (row[1], row[2]), reverse=True)
    print("\n------------ PC-BIND Marginal Strategy Ranking ------------")
    for name, avg_aupr, avg_mcc, min_aupr, min_mcc in ranked:
        print(
            f"{name:<36} AvgAUPRC={avg_aupr:.4f} AvgMCC={avg_mcc:.4f} "
            f"MinAUPRC={min_aupr:.4f} MinMCC={min_mcc:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PC-BIND contact-marginal readout from saved pair-head checkpoints."
    )
    parser.add_argument(
        "--run-dir",
        default="outputs_pcbind_pairmlp_w005_seed2073",
        help="Directory containing fold*_best.pt checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for marginal-evaluation outputs. Defaults to --run-dir.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Fold seed. Defaults to checkpoint metadata.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--test-sets",
        default="Test60.pkl,Test287.pkl,TestB25.pkl",
        help=(
            "Comma-separated partner-capable test files. TestUB25 is excluded by "
            "default because it contains monomeric targets without benchmark partner inputs."
        ),
    )
    parser.add_argument(
        "--no-smooth-marginal",
        action="store_true",
        help="Use raw contact-marginal probabilities without residue-graph smoothing.",
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    output_dir = os.path.abspath(args.output_dir or args.run_dir)
    os.makedirs(output_dir, exist_ok=True)
    seed = infer_run_seed(run_dir, args.seed if args.seed is not None else train_cfg.SEED)
    first_checkpoint = torch.load(
        os.path.join(run_dir, "fold1_best.pt"),
        map_location="cpu",
        weights_only=False,
    )
    grouped_cv = bool(first_checkpoint.get("grouped_cv", False))
    cv_group_key = str(first_checkpoint.get("cv_group_key", "complex_code"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_list = load_pickle(dataset_path("Train335.pkl"))
    requested_test_files = [
        filename.strip() for filename in args.test_sets.split(",") if filename.strip()
    ]
    if not requested_test_files:
        raise ValueError("--test-sets must contain at least one test file")
    test_sets = {
        os.path.splitext(filename)[0]: load_pickle(dataset_path(filename))
        for filename in requested_test_files
    }
    partner_coverage = {
        "Train335": {
            "total": len(train_list),
            "evaluated": sum(valid_partner_encoder_sample(sample) for sample in train_list),
        }
    }
    for test_name, samples in list(test_sets.items()):
        valid_samples = []
        excluded = []
        for idx, sample in enumerate(samples):
            if valid_partner_encoder_sample(sample):
                valid_samples.append(sample)
            else:
                excluded.append({
                    "index": idx,
                    "complex_code": str(sample.get("complex_code", "?")),
                })
        if not valid_samples:
            raise ValueError(
                f"{test_name} has no samples with a nonempty, valid partner encoder input"
            )
        partner_coverage[test_name] = {
            "total": len(samples),
            "evaluated": len(valid_samples),
            "excluded": excluded,
        }
        test_sets[test_name] = valid_samples
    in_dim = train_list[0]["residue_graph_node"].shape[1]
    atom_dim = train_list[0]["atom_graph_node"].shape[1]
    plm_mode, plm_dim = configure_plm_feature_config([train_list, test_sets])

    print(f"Device: {device}")
    print(f"Run dir: {run_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Fold seed: {seed}")
    print(f"Grouped CV: {grouped_cv} (key={cv_group_key})")
    print(f"Test sets: {', '.join(test_sets)}")
    for data_name, coverage in partner_coverage.items():
        print(
            f"Partner-complete {data_name}: "
            f"{coverage['evaluated']}/{coverage['total']}"
        )
    print(f"PLM feature mode: {plm_mode}")
    print(f"PLM feature dim: {plm_dim}")

    validation_folds = make_cv_folds(
        train_list,
        seed,
        num_folds=5,
        grouped=grouped_cv,
        group_key=cv_group_key,
    )

    artifacts = []
    oof_labels = []
    oof_site_probs = []
    oof_aux_probs = []
    oof_cls_probs = []
    oof_rank_probs = []
    oof_marginal_probs = []
    oof_pair_logit_matrices = []

    for fold in range(1, 6):
        model, checkpoint, checkpoint_plm_dim, checkpoint_aux_plm_dim = load_fold_model(
            run_dir,
            fold,
            in_dim,
            atom_dim,
            plm_dim,
            device,
        )
        if checkpoint.get("geometry_feature_mode", "legacy_or_unknown") != GEOMETRY_FEATURE_MODE:
            print(
                f"Warning: fold {fold} geometry mode is {checkpoint.get('geometry_feature_mode')}, "
                f"current code expects {GEOMETRY_FEATURE_MODE}.",
                flush=True,
            )
        val_idx = validation_folds[fold - 1]
        val_samples = [
            train_list[i] for i in val_idx if valid_partner_encoder_sample(train_list[i])
        ]
        val_dataset = make_dataset(
            val_samples,
            checkpoint,
            checkpoint_plm_dim,
            checkpoint_aux_plm_dim,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_proteins,
            num_workers=0,
        )
        labels, site_probs, aux_probs, cls_probs, rank_probs, marginal_probs, pair_logit_matrix = predict_scores_with_marginal(
            model,
            val_loader,
            device,
            rank_fusion=checkpoint_rank_fusion(checkpoint),
            smooth_marginal=not args.no_smooth_marginal,
        )
        val_metrics = metrics_from_scores(labels, site_probs, split_name=f"PCBindMarginal-Fold{fold}-Site-Val")
        metrics_from_scores(labels, marginal_probs, split_name=f"PCBindMarginal-Fold{fold}-Marginal-Val")
        artifacts.append({
            "checkpoint": checkpoint,
            "val_mcc": val_metrics["mcc"],
            "val_auc_pr": val_metrics["auc_pr"],
            "rank_fusion": checkpoint_rank_fusion(checkpoint),
        })
        oof_labels.append(labels)
        oof_site_probs.append(site_probs)
        oof_aux_probs.append(aux_probs)
        oof_cls_probs.append(cls_probs)
        oof_rank_probs.append(rank_probs)
        oof_marginal_probs.append(marginal_probs)
        oof_pair_logit_matrices.append(pair_logit_matrix)

    oof_y = np.concatenate(oof_labels, axis=0)
    oof_site = np.concatenate(oof_site_probs, axis=0)
    oof_aux = np.concatenate(oof_aux_probs, axis=0)
    oof_marginal = np.concatenate(oof_marginal_probs, axis=0)
    oof_pair_logits = np.concatenate(oof_pair_logit_matrices, axis=0)
    calibrated = select_pair_logit_calibration(oof_y, oof_pair_logits)
    oof_calibrated_marginal = calibrated["score"]
    prob_threshold = best_mcc_threshold(oof_y, oof_site)
    rank_threshold = best_mcc_threshold(oof_y, rank_normalize(oof_site))
    marginal_threshold = best_mcc_threshold(oof_y, oof_marginal)
    marginal_rank_threshold = best_mcc_threshold(oof_y, rank_normalize(oof_marginal))
    calibrated_marginal_threshold = best_mcc_threshold(oof_y, oof_calibrated_marginal)
    calibrated_marginal_rank_threshold = best_mcc_threshold(oof_y, rank_normalize(oof_calibrated_marginal))
    marginal_alpha, marginal_blend_aupr, oof_marginal_blend = select_alpha(
        oof_y,
        oof_site,
        oof_marginal,
        rank_space=False,
    )
    marginal_rank_alpha, marginal_rank_blend_aupr, oof_marginal_rank_blend = select_alpha(
        oof_y,
        oof_site,
        oof_marginal,
        rank_space=True,
    )
    calibrated_alpha, calibrated_blend_aupr, oof_calibrated_blend = select_alpha(
        oof_y,
        oof_site,
        oof_calibrated_marginal,
        rank_space=False,
    )
    calibrated_rank_alpha, calibrated_rank_blend_aupr, oof_calibrated_rank_blend = select_alpha(
        oof_y,
        oof_site,
        oof_calibrated_marginal,
        rank_space=True,
    )
    marginal_blend_threshold = best_mcc_threshold(oof_y, oof_marginal_blend)
    marginal_rank_blend_threshold = best_mcc_threshold(oof_y, oof_marginal_rank_blend)
    calibrated_blend_threshold = best_mcc_threshold(oof_y, oof_calibrated_blend)
    calibrated_rank_blend_threshold = best_mcc_threshold(oof_y, oof_calibrated_rank_blend)

    stacker = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
    )
    stacker.fit(marginal_stack_features(oof_site, oof_aux, oof_marginal), oof_y)
    oof_stack = stacker.predict_proba(marginal_stack_features(oof_site, oof_aux, oof_marginal))[:, 1]
    stack_threshold = best_mcc_threshold(oof_y, oof_stack)
    calibrated_stacker = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
    )
    calibrated_stacker.fit(marginal_stack_features(oof_site, oof_aux, oof_calibrated_marginal), oof_y)
    oof_calibrated_stack = calibrated_stacker.predict_proba(
        marginal_stack_features(oof_site, oof_aux, oof_calibrated_marginal)
    )[:, 1]
    calibrated_stack_threshold = best_mcc_threshold(oof_y, oof_calibrated_stack)

    print("\n===== OOF PC-BIND Marginal Calibration =====")
    print(f"Site AUPRC={compute_auc_pr(oof_y, oof_site):.4f}, threshold={prob_threshold:.3f}")
    print(f"Marginal AUPRC={compute_auc_pr(oof_y, oof_marginal):.4f}, threshold={marginal_threshold:.3f}")
    print(
        f"Calibrated marginal scale={calibrated['scale']:.2f}, bias={calibrated['bias']:.2f}, "
        f"AUPRC={calibrated['auc_pr']:.4f}, AP={calibrated['average_precision']:.4f}, "
        f"threshold={calibrated_marginal_threshold:.3f}"
    )
    print(
        f"Marginal blend alpha={marginal_alpha:.2f}, "
        f"AUPRC={marginal_blend_aupr:.4f}, threshold={marginal_blend_threshold:.3f}"
    )
    print(
        f"Marginal rank blend alpha={marginal_rank_alpha:.2f}, "
        f"AUPRC={marginal_rank_blend_aupr:.4f}, threshold={marginal_rank_blend_threshold:.3f}"
    )
    print(
        f"Calibrated blend alpha={calibrated_alpha:.2f}, "
        f"AUPRC={calibrated_blend_aupr:.4f}, threshold={calibrated_blend_threshold:.3f}"
    )
    print(
        f"Calibrated rank blend alpha={calibrated_rank_alpha:.2f}, "
        f"AUPRC={calibrated_rank_blend_aupr:.4f}, threshold={calibrated_rank_blend_threshold:.3f}"
    )
    print(
        f"Marginal stack AUPRC={compute_auc_pr(oof_y, oof_stack):.4f}, "
        f"threshold={stack_threshold:.3f}"
    )
    print(
        f"Calibrated marginal stack AUPRC={compute_auc_pr(oof_y, oof_calibrated_stack):.4f}, "
        f"threshold={calibrated_stack_threshold:.3f}"
    )

    val_mcc_weights = normalized_positive_weights([a["val_mcc"] for a in artifacts])
    val_aupr_weights = normalized_positive_weights([a["val_auc_pr"] for a in artifacts])
    strategies = {
        "mean": {"mode": "mean", "weights": None, "site_threshold": prob_threshold},
        "weighted_mcc": {"mode": "weighted", "weights": val_mcc_weights, "site_threshold": prob_threshold},
        "weighted_aupr": {"mode": "weighted", "weights": val_aupr_weights, "site_threshold": prob_threshold},
        "rank": {"mode": "rank", "weights": None, "site_threshold": rank_threshold},
    }

    rows = []
    summary = {}
    for test_name, test_list in test_sets.items():
        print(f"\nEvaluating PC-BIND marginal readout on {test_name}", flush=True)
        fold_site_scores = []
        fold_aux_scores = []
        fold_marginal_scores = []
        fold_calibrated_marginal_scores = []
        y_true = None
        for fold, artifact in enumerate(artifacts, start=1):
            model, checkpoint, checkpoint_plm_dim, checkpoint_aux_plm_dim = load_fold_model(
                run_dir,
                fold,
                in_dim,
                atom_dim,
                plm_dim,
                device,
            )
            dataset = make_dataset(test_list, checkpoint, checkpoint_plm_dim, checkpoint_aux_plm_dim)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_proteins,
                num_workers=0,
            )
            labels, site_probs, aux_probs, _, _, marginal_probs, pair_logit_matrix = predict_scores_with_marginal(
                model,
                loader,
                device,
                rank_fusion=artifact["rank_fusion"],
                smooth_marginal=not args.no_smooth_marginal,
            )
            y_true = labels if y_true is None else y_true
            fold_site_scores.append(site_probs)
            fold_aux_scores.append(aux_probs)
            fold_marginal_scores.append(marginal_probs)
            fold_calibrated_marginal_scores.append(
                calibrated_marginal_from_logits(
                    pair_logit_matrix,
                    scale=calibrated["scale"],
                    bias=calibrated["bias"],
                )
            )

        for strategy_name, strategy in strategies.items():
            site_score = combine_fold_scores(fold_site_scores, strategy["mode"], strategy["weights"])
            aux_score = combine_fold_scores(fold_aux_scores, strategy["mode"], strategy["weights"])
            marginal_score = combine_fold_scores(fold_marginal_scores, strategy["mode"], strategy["weights"])
            calibrated_marginal_score = combine_fold_scores(
                fold_calibrated_marginal_scores,
                strategy["mode"],
                strategy["weights"],
            )
            if strategy_name == "rank":
                marginal_only_threshold = marginal_rank_threshold
                calibrated_marginal_only_threshold = calibrated_marginal_rank_threshold
                blend_score = (
                    (1.0 - marginal_rank_alpha) * site_score
                    + marginal_rank_alpha * marginal_score
                )
                blend_threshold = marginal_rank_blend_threshold
                calibrated_blend_score = (
                    (1.0 - calibrated_rank_alpha) * site_score
                    + calibrated_rank_alpha * calibrated_marginal_score
                )
                calibrated_blend_threshold_for_strategy = calibrated_rank_blend_threshold
            else:
                marginal_only_threshold = marginal_threshold
                calibrated_marginal_only_threshold = calibrated_marginal_threshold
                blend_score = (
                    (1.0 - marginal_alpha) * site_score
                    + marginal_alpha * marginal_score
                )
                blend_threshold = marginal_blend_threshold
                calibrated_blend_score = (
                    (1.0 - calibrated_alpha) * site_score
                    + calibrated_alpha * calibrated_marginal_score
                )
                calibrated_blend_threshold_for_strategy = calibrated_blend_threshold

            site_metrics = metrics_from_scores(
                y_true,
                site_score,
                split_name=f"PCBindMarginal-site_{strategy_name}-{test_name}",
                threshold=strategy["site_threshold"],
            )
            add_metrics(rows, summary, f"site_{strategy_name}", test_name, site_metrics)

            marginal_metrics = metrics_from_scores(
                y_true,
                marginal_score,
                split_name=f"PCBindMarginal-marginal_{strategy_name}-{test_name}",
                threshold=marginal_only_threshold,
            )
            add_metrics(rows, summary, f"marginal_{strategy_name}", test_name, marginal_metrics)

            calibrated_marginal_metrics = metrics_from_scores(
                y_true,
                calibrated_marginal_score,
                split_name=f"PCBindMarginal-calibrated_marginal_{strategy_name}-{test_name}",
                threshold=calibrated_marginal_only_threshold,
            )
            add_metrics(rows, summary, f"calibrated_marginal_{strategy_name}", test_name, calibrated_marginal_metrics)

            blend_metrics = metrics_from_scores(
                y_true,
                blend_score,
                split_name=f"PCBindMarginal-blend_{strategy_name}-{test_name}",
                threshold=blend_threshold,
            )
            add_metrics(rows, summary, f"blend_{strategy_name}", test_name, blend_metrics)

            calibrated_blend_metrics = metrics_from_scores(
                y_true,
                calibrated_blend_score,
                split_name=f"PCBindMarginal-calibrated_blend_{strategy_name}-{test_name}",
                threshold=calibrated_blend_threshold_for_strategy,
            )
            add_metrics(rows, summary, f"calibrated_blend_{strategy_name}", test_name, calibrated_blend_metrics)

            if strategy_name != "rank":
                stack_score = stacker.predict_proba(
                    marginal_stack_features(site_score, aux_score, marginal_score)
                )[:, 1]
                stack_metrics = metrics_from_scores(
                    y_true,
                    stack_score,
                    split_name=f"PCBindMarginal-stack_{strategy_name}-{test_name}",
                    threshold=stack_threshold,
                )
                add_metrics(rows, summary, f"stack_{strategy_name}", test_name, stack_metrics)

                calibrated_stack_score = calibrated_stacker.predict_proba(
                    marginal_stack_features(site_score, aux_score, calibrated_marginal_score)
                )[:, 1]
                calibrated_stack_metrics = metrics_from_scores(
                    y_true,
                    calibrated_stack_score,
                    split_name=f"PCBindMarginal-calibrated_stack_{strategy_name}-{test_name}",
                    threshold=calibrated_stack_threshold,
                )
                add_metrics(rows, summary, f"calibrated_stack_{strategy_name}", test_name, calibrated_stack_metrics)

    metrics_csv = os.path.join(output_dir, "pcbind_marginal_metrics.csv")
    summary_json = os.path.join(output_dir, "pcbind_marginal_summary.json")
    write_metrics_csv(metrics_csv, rows)
    save_json(summary_json, {
        "run_dir": run_dir,
        "seed": seed,
        "grouped_cv": grouped_cv,
        "cv_group_key": cv_group_key,
        "test_sets": list(test_sets),
        "partner_coverage": partner_coverage,
        "geometry_feature_mode": GEOMETRY_FEATURE_MODE,
        "surface_feature_mode": SURFACE_FEATURE_MODE,
        "sequence_feature_mode": SEQUENCE_FEATURE_MODE,
        "plm_feature_mode": plm_mode,
        "plm_feature_dim": plm_dim,
        "smooth_marginal": not args.no_smooth_marginal,
        "prob_threshold": prob_threshold,
        "rank_threshold": rank_threshold,
        "marginal_threshold": marginal_threshold,
        "marginal_rank_threshold": marginal_rank_threshold,
        "calibrated_marginal_scale": calibrated["scale"],
        "calibrated_marginal_bias": calibrated["bias"],
        "calibrated_marginal_oof_auc_pr": calibrated["auc_pr"],
        "calibrated_marginal_oof_average_precision": calibrated["average_precision"],
        "calibrated_marginal_oof_score_std": calibrated["score_std"],
        "calibrated_marginal_oof_unique_score_count": calibrated["unique_score_count"],
        "calibrated_marginal_threshold": calibrated_marginal_threshold,
        "calibrated_marginal_rank_threshold": calibrated_marginal_rank_threshold,
        "marginal_blend_alpha": marginal_alpha,
        "marginal_blend_oof_auc_pr": marginal_blend_aupr,
        "marginal_blend_threshold": marginal_blend_threshold,
        "marginal_rank_blend_alpha": marginal_rank_alpha,
        "marginal_rank_blend_oof_auc_pr": marginal_rank_blend_aupr,
        "marginal_rank_blend_threshold": marginal_rank_blend_threshold,
        "calibrated_blend_alpha": calibrated_alpha,
        "calibrated_blend_oof_auc_pr": calibrated_blend_aupr,
        "calibrated_blend_threshold": calibrated_blend_threshold,
        "calibrated_rank_blend_alpha": calibrated_rank_alpha,
        "calibrated_rank_blend_oof_auc_pr": calibrated_rank_blend_aupr,
        "calibrated_rank_blend_threshold": calibrated_rank_blend_threshold,
        "marginal_stack_oof_auc_pr": compute_auc_pr(oof_y, oof_stack),
        "marginal_stack_threshold": stack_threshold,
        "calibrated_marginal_stack_oof_auc_pr": compute_auc_pr(oof_y, oof_calibrated_stack),
        "calibrated_marginal_stack_threshold": calibrated_stack_threshold,
        "val_mcc_weights": val_mcc_weights,
        "val_aupr_weights": val_aupr_weights,
        "summary": summary,
    })
    print_strategy_ranking(summary)
    print(f"\nSaved marginal metrics to: {metrics_csv}")
    print(f"Saved marginal summary to: {summary_json}")


if __name__ == "__main__":
    main()
