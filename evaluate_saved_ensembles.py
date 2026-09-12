import os
import pickle

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import CROSS5FOLD_multi_test as train_cfg
from CROSS5FOLD_multi_test import (
    OUTPUT_DIR,
    SEED,
    GEOMETRY_FEATURE_MODE,
    SURFACE_FEATURE_MODE,
    SEQUENCE_FEATURE_MODE,
    PLM_FEATURE_MODE,
    AUX_PLM_FEATURE_MODE,
    AUX_PLM_FEATURE_DIM,
    configure_plm_feature_config,
    PARTNER_CONTACT_AUX,
    PARTNER_CONTACT_AUX_WEIGHT,
    ProteinDataset,
    best_mcc_threshold,
    build_model,
    collate_proteins,
    combine_fold_scores,
    metrics_from_scores,
    normalized_positive_weights,
    predict_scores,
    predict_scores_with_aux,
    predict_scores_with_heads,
    rank_normalize,
    dataset_path,
    available_test_datasets,
    write_metrics_csv,
    save_json,
)
from metrics import compute_auc_pr


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def stack_features(main_score, aux_score):
    """OOF-calibrated meta-features from binding and teacher-contact scores."""
    main_score = np.asarray(main_score, dtype=np.float64)
    aux_score = np.asarray(aux_score, dtype=np.float64)
    eps = 1e-8
    return np.column_stack(
        (
            main_score,
            aux_score,
            rank_normalize(main_score),
            rank_normalize(aux_score),
            np.sqrt(np.clip(main_score * aux_score, eps, None)),
            np.maximum(main_score, aux_score),
            np.minimum(main_score, aux_score),
            np.abs(main_score - aux_score),
            np.log(np.clip(main_score, eps, 1.0)),
            np.log(np.clip(aux_score, eps, 1.0)),
        )
    )


def load_compatible_state(model, state):
    """Load checkpoints across small architecture changes by skipping reshaped layers."""
    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }
    skipped = sorted(set(state.keys()) - set(compatible.keys()))
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if hasattr(model, "use_rank_head") and any(key.startswith("rank_cls.") for key in missing):
        model.use_rank_head = False
    elif hasattr(model, "use_rank_head") and any(key.startswith("rank_cls.") for key in state):
        model.use_rank_head = True
    if hasattr(model, "partner_cls") and model.partner_cls is not None:
        if not any(key.startswith("partner_cls.") for key in state):
            model.partner_cls = None
    return missing, unexpected, skipped


def blend_head_scores(cls_score, rank_score, alpha):
    return (1.0 - alpha) * np.asarray(cls_score) + alpha * np.asarray(rank_score)


def checkpoint_rank_fusion(checkpoint):
    return float(checkpoint.get("two_head_rank_fusion", train_cfg.TWO_HEAD_RANK_FUSION))


def checkpoint_view_logit_fusion(checkpoint):
    return float(checkpoint.get("view_logit_fusion", 0.0))


def apply_checkpoint_runtime_config(model, checkpoint):
    if hasattr(model, "view_logit_fusion"):
        model.view_logit_fusion = checkpoint_view_logit_fusion(checkpoint)
    if hasattr(model, "partner_top_k"):
        model.partner_top_k = int(checkpoint.get("partner_top_k", model.partner_top_k))
    if hasattr(model, "partner_direct_fusion"):
        model.partner_direct_fusion = bool(checkpoint.get("partner_direct_fusion", model.partner_direct_fusion))
    if hasattr(model, "partner_logit_mode"):
        model.partner_logit_mode = str(checkpoint.get("partner_logit_mode", model.partner_logit_mode)).lower()
    if hasattr(model, "partner_delta_scale"):
        model.partner_delta_scale = float(checkpoint.get("partner_delta_scale", model.partner_delta_scale))
    if hasattr(model, "partner_cls") and not bool(checkpoint.get("partner_contact_aux", True)):
        model.partner_cls = None
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_list = load_pickle(dataset_path("Train335.pkl"))
    test_sets = {
        name: load_pickle(path)
        for name, path in available_test_datasets(require_geo=True)
    }

    in_dim = train_list[0]["residue_graph_node"].shape[1]
    atom_dim = train_list[0]["atom_graph_node"].shape[1]
    plm_mode, plm_dim = configure_plm_feature_config([train_list, test_sets])
    print(f"PLM feature mode: {plm_mode}")
    print(f"PLM feature dim: {plm_dim}")
    print(f"Aux PLM feature mode: {train_cfg.AUX_PLM_FEATURE_MODE}")
    print(f"Aux PLM feature dim: {train_cfg.AUX_PLM_FEATURE_DIM}")
    batch_size = 4

    np.random.seed(SEED)
    indices = np.random.permutation(len(train_list))
    fold_ids = np.arange(len(train_list)) % 5

    artifacts = []
    oof_labels = []
    oof_probs = []
    oof_aux_probs = []
    oof_cls_probs = []
    oof_rank_probs = []

    for fold in range(5):
        ckpt_path = os.path.join(OUTPUT_DIR, f"fold{fold + 1}_best.pt")
        print(f"Loading {ckpt_path}", flush=True)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        ckpt_geometry_mode = checkpoint.get("geometry_feature_mode", "legacy_or_unknown")
        if ckpt_geometry_mode != GEOMETRY_FEATURE_MODE:
            print(
                f"Warning: checkpoint geometry mode is {ckpt_geometry_mode}, "
                f"current code expects {GEOMETRY_FEATURE_MODE}. Retrain folds for valid local-frame results.",
                flush=True,
            )
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
        pair_contact_head=checkpoint.get("pair_contact_head", "attention"),
    )
        missing, unexpected, skipped = load_compatible_state(model, checkpoint["model_state"])
        apply_checkpoint_runtime_config(model, checkpoint)
        if missing or unexpected or skipped:
            print(
                f"Checkpoint compatibility: missing={missing}, unexpected={unexpected}, "
                f"skipped={skipped}",
                flush=True,
            )

        val_idx = indices[fold_ids == fold]
        val_list = [train_list[i] for i in val_idx]
        val_dataset = ProteinDataset(
            val_list,
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
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_proteins,
            num_workers=0,
        )
        rank_fusion = checkpoint_rank_fusion(checkpoint)
        val_y, val_score, val_aux_score, val_cls_score, val_rank_score = predict_scores_with_heads(
            model, val_loader, device, rank_fusion=rank_fusion
        )
        val_metrics = metrics_from_scores(val_y, val_score, split_name=f"Saved-Fold{fold + 1}-Val")
        oof_labels.append(val_y)
        oof_probs.append(val_score)
        oof_aux_probs.append(val_aux_score)
        oof_cls_probs.append(val_cls_score)
        oof_rank_probs.append(val_rank_score)
        artifacts.append({
            "checkpoint": checkpoint,
            "val_mcc": val_metrics["mcc"],
            "val_auc_pr": val_metrics["auc_pr"],
            "rank_fusion": rank_fusion,
            "view_logit_fusion": checkpoint_view_logit_fusion(checkpoint),
        })

    oof_y = np.concatenate(oof_labels, axis=0)
    oof_score = np.concatenate(oof_probs, axis=0)
    oof_aux_score = np.concatenate(oof_aux_probs, axis=0)
    oof_cls_score = np.concatenate(oof_cls_probs, axis=0)
    oof_rank_score = np.concatenate(oof_rank_probs, axis=0)
    head_alphas = np.linspace(0.0, 1.0, 21)
    head_auc_pr = [
        compute_auc_pr(oof_y, blend_head_scores(oof_cls_score, oof_rank_score, alpha))
        for alpha in head_alphas
    ]
    head_alpha = float(head_alphas[int(np.argmax(head_auc_pr))])
    oof_head_score = blend_head_scores(oof_cls_score, oof_rank_score, head_alpha)
    blend_alphas = np.linspace(0.0, 1.0, 21)
    blend_auc_pr = []
    for alpha in blend_alphas:
        blend_auc_pr.append(compute_auc_pr(oof_y, (1.0 - alpha) * oof_score + alpha * oof_aux_score))
    blend_alpha = float(blend_alphas[int(np.argmax(blend_auc_pr))])
    oof_blend_score = (1.0 - blend_alpha) * oof_score + blend_alpha * oof_aux_score
    head_threshold = best_mcc_threshold(oof_y, oof_head_score)
    head_rank_threshold = best_mcc_threshold(oof_y, rank_normalize(oof_head_score))
    prob_threshold = best_mcc_threshold(oof_y, oof_score)
    rank_threshold = best_mcc_threshold(oof_y, rank_normalize(oof_score))
    blend_threshold = best_mcc_threshold(oof_y, oof_blend_score)
    blend_rank_threshold = best_mcc_threshold(oof_y, rank_normalize(oof_blend_score))
    stacker = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=SEED,
        ),
    )
    stacker.fit(stack_features(oof_score, oof_aux_score), oof_y)
    oof_stack_score = stacker.predict_proba(stack_features(oof_score, oof_aux_score))[:, 1]
    stack_threshold = best_mcc_threshold(oof_y, oof_stack_score)
    print(
        f"\nOOF head blend rank_alpha={head_alpha:.2f}, "
        f"AUPRC={max(head_auc_pr):.4f}, threshold={head_threshold:.3f}",
        flush=True,
    )
    print(
        f"\nOOF teacher blend alpha={blend_alpha:.2f}, "
        f"AUPRC={max(blend_auc_pr):.4f}, threshold={blend_threshold:.3f}",
        flush=True,
    )
    print(
        f"OOF stacked teacher fusion AUPRC={compute_auc_pr(oof_y, oof_stack_score):.4f}, "
        f"threshold={stack_threshold:.3f}",
        flush=True,
    )
    val_mcc_weights = normalized_positive_weights([a["val_mcc"] for a in artifacts])
    val_aupr_weights = normalized_positive_weights([a["val_auc_pr"] for a in artifacts])

    strategies = {
        "mean": {"mode": "mean", "weights": None, "threshold": prob_threshold},
        "weighted_mcc": {"mode": "weighted", "weights": val_mcc_weights, "threshold": prob_threshold},
        "weighted_aupr": {"mode": "weighted", "weights": val_aupr_weights, "threshold": prob_threshold},
        "rank": {"mode": "rank", "weights": None, "threshold": rank_threshold},
    }
    head_strategies = {
        "head_blend": {"mode": "mean", "weights": None, "threshold": head_threshold},
        "head_blend_rank": {"mode": "rank", "weights": None, "threshold": head_rank_threshold},
    }

    rows = []
    summary = {}
    for test_name, test_list in test_sets.items():
        print(f"\nEvaluating saved ensembles on {test_name}", flush=True)
        fold_scores = []
        fold_aux_scores = []
        fold_cls_scores = []
        fold_rank_scores = []
        y_true = None

        for fold_idx, artifact in enumerate(artifacts, start=1):
            print(f"  Predicting fold {fold_idx}/5", flush=True)
            checkpoint = artifact["checkpoint"]
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
                pair_contact_head=checkpoint.get("pair_contact_head", "attention"),
            )
            load_compatible_state(model, checkpoint["model_state"])
            apply_checkpoint_runtime_config(model, checkpoint)
            dataset = ProteinDataset(
                test_list,
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
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_proteins,
                num_workers=0,
            )
            labels, probs, aux_probs, cls_probs, rank_probs = predict_scores_with_heads(
                model, loader, device, rank_fusion=artifact["rank_fusion"]
            )
            y_true = labels if y_true is None else y_true
            fold_scores.append(probs)
            fold_aux_scores.append(aux_probs)
            fold_cls_scores.append(cls_probs)
            fold_rank_scores.append(rank_probs)

        for strategy_name, strategy in strategies.items():
            print(f"  Combining strategy: {strategy_name}", flush=True)
            score = combine_fold_scores(fold_scores, strategy["mode"], strategy["weights"])
            metrics = metrics_from_scores(
                y_true,
                score,
                split_name=f"SavedEnsemble-{strategy_name}-{test_name}",
                threshold=strategy["threshold"],
            )
            summary.setdefault(strategy_name, {})[test_name] = metrics
            rows.append({
                "group": f"saved_ensemble_{strategy_name}",
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

            blend_score = None
            if strategy["mode"] == "rank":
                main_score = combine_fold_scores(fold_scores, "rank", strategy["weights"])
                aux_score = combine_fold_scores(fold_aux_scores, "rank", strategy["weights"])
                blend_score = (1.0 - blend_alpha) * main_score + blend_alpha * aux_score
                blend_threshold_for_strategy = blend_rank_threshold
            else:
                main_score = combine_fold_scores(fold_scores, strategy["mode"], strategy["weights"])
                aux_score = combine_fold_scores(fold_aux_scores, strategy["mode"], strategy["weights"])
                blend_score = (1.0 - blend_alpha) * main_score + blend_alpha * aux_score
                blend_threshold_for_strategy = blend_threshold

            blend_name = f"teacher_blend_{strategy_name}"
            print(f"  Combining strategy: {blend_name}", flush=True)
            blend_metrics = metrics_from_scores(
                y_true,
                blend_score,
                split_name=f"SavedEnsemble-{blend_name}-{test_name}",
                threshold=blend_threshold_for_strategy,
            )
            summary.setdefault(blend_name, {})[test_name] = blend_metrics
            rows.append({
                "group": f"saved_ensemble_{blend_name}",
                "test_set": test_name,
                "acc": blend_metrics["acc"],
                "precision": blend_metrics["precision"],
                "recall": blend_metrics["recall"],
                "f1": blend_metrics["f1"],
                "mcc": blend_metrics["mcc"],
                "auc_roc": blend_metrics["auc_roc"],
                "auc_pr": blend_metrics["auc_pr"],
                "threshold": blend_metrics["t_opt"],
            })

            stack_name = f"teacher_stack_{strategy_name}"
            print(f"  Combining strategy: {stack_name}", flush=True)
            stack_score = stacker.predict_proba(stack_features(main_score, aux_score))[:, 1]
            stack_metrics = metrics_from_scores(
                y_true,
                stack_score,
                split_name=f"SavedEnsemble-{stack_name}-{test_name}",
                threshold=stack_threshold,
            )
            summary.setdefault(stack_name, {})[test_name] = stack_metrics
            rows.append({
                "group": f"saved_ensemble_{stack_name}",
                "test_set": test_name,
                "acc": stack_metrics["acc"],
                "precision": stack_metrics["precision"],
                "recall": stack_metrics["recall"],
                "f1": stack_metrics["f1"],
                "mcc": stack_metrics["mcc"],
                "auc_roc": stack_metrics["auc_roc"],
                "auc_pr": stack_metrics["auc_pr"],
                "threshold": stack_metrics["t_opt"],
            })

        for strategy_name, strategy in head_strategies.items():
            print(f"  Combining strategy: {strategy_name}", flush=True)
            if strategy["mode"] == "rank":
                cls_score = combine_fold_scores(fold_cls_scores, "rank", None)
                rank_score = combine_fold_scores(fold_rank_scores, "rank", None)
            else:
                cls_score = combine_fold_scores(fold_cls_scores, "mean", None)
                rank_score = combine_fold_scores(fold_rank_scores, "mean", None)
            head_score = blend_head_scores(cls_score, rank_score, head_alpha)
            metrics = metrics_from_scores(
                y_true,
                head_score,
                split_name=f"SavedEnsemble-{strategy_name}-{test_name}",
                threshold=strategy["threshold"],
            )
            summary.setdefault(strategy_name, {})[test_name] = metrics
            rows.append({
                "group": f"saved_ensemble_{strategy_name}",
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

    write_metrics_csv(os.path.join(OUTPUT_DIR, "saved_ensemble_metrics.csv"), rows)
    save_json(os.path.join(OUTPUT_DIR, "saved_ensemble_summary.json"), {
        "geometry_feature_mode": GEOMETRY_FEATURE_MODE,
        "surface_feature_mode": SURFACE_FEATURE_MODE,
        "sequence_feature_mode": SEQUENCE_FEATURE_MODE,
        "plm_feature_mode": plm_mode,
        "plm_feature_dim": plm_dim,
        "aux_plm_feature_mode": train_cfg.AUX_PLM_FEATURE_MODE,
        "aux_plm_feature_dim": train_cfg.AUX_PLM_FEATURE_DIM,
        "partner_contact_aux": PARTNER_CONTACT_AUX,
        "partner_contact_aux_weight": PARTNER_CONTACT_AUX_WEIGHT,
        "two_head_binding": train_cfg.TWO_HEAD_BINDING,
        "two_head_rank_loss_weight": train_cfg.TWO_HEAD_RANK_LOSS_WEIGHT,
        "two_head_consistency_weight": train_cfg.TWO_HEAD_CONSISTENCY_WEIGHT,
        "two_head_rank_fusion": train_cfg.TWO_HEAD_RANK_FUSION,
        "checkpoint_rank_fusions": [artifact["rank_fusion"] for artifact in artifacts],
        "checkpoint_view_logit_fusions": [artifact["view_logit_fusion"] for artifact in artifacts],
        "head_blend_rank_alpha": head_alpha,
        "head_blend_threshold": head_threshold,
        "head_blend_rank_threshold": head_rank_threshold,
        "head_blend_oof_auc_pr": float(max(head_auc_pr)),
        "prob_threshold": prob_threshold,
        "rank_threshold": rank_threshold,
        "teacher_blend_alpha": blend_alpha,
        "teacher_blend_threshold": blend_threshold,
        "teacher_blend_rank_threshold": blend_rank_threshold,
        "teacher_blend_oof_auc_pr": float(max(blend_auc_pr)),
        "teacher_stack_threshold": stack_threshold,
        "teacher_stack_oof_auc_pr": compute_auc_pr(oof_y, oof_stack_score),
        "val_mcc_weights": val_mcc_weights,
        "val_aupr_weights": val_aupr_weights,
        "summary": summary,
    })
    print(f"\nSaved ensemble-only outputs to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
