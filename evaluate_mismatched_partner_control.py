import argparse
import copy
import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

import CROSS5FOLD_multi_test as train_cfg
from CROSS5FOLD_multi_test import (
    GEOMETRY_FEATURE_MODE,
    SURFACE_FEATURE_MODE,
    SEQUENCE_FEATURE_MODE,
    ProteinDataset,
    available_test_datasets,
    best_mcc_threshold,
    collate_proteins,
    combine_fold_scores,
    configure_plm_feature_config,
    dataset_path,
    make_cv_folds,
    metrics_from_scores,
    normalized_positive_weights,
    predict_scores_with_heads,
    rank_normalize,
    save_json,
    write_metrics_csv,
)
from evaluate_pcbind_marginal import (
    infer_run_seed,
    load_fold_model,
)


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


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_saved_ensemble_calibration(run_dir, fold_count):
    summary_path = os.path.join(run_dir, "ensemble_summary.json")
    if not os.path.isfile(summary_path):
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    required = ("ensemble_threshold", "rank_threshold", "val_mcc_weights", "val_aupr_weights")
    if any(key not in summary for key in required):
        return None

    val_mcc_weights = np.asarray(summary["val_mcc_weights"], dtype=np.float64)
    val_aupr_weights = np.asarray(summary["val_aupr_weights"], dtype=np.float64)
    if val_mcc_weights.size != fold_count or val_aupr_weights.size != fold_count:
        print(
            f"Warning: ignoring {summary_path}; saved fold weights do not match {fold_count} folds.",
            flush=True,
        )
        return None

    return {
        "source": summary_path,
        "prob_threshold": float(summary["ensemble_threshold"]),
        "rank_threshold": float(summary["rank_threshold"]),
        "val_mcc_weights": val_mcc_weights,
        "val_aupr_weights": val_aupr_weights,
    }


def partner_length(sample):
    for key in ("partner_residue_surface_features", "partner_residue_sequence_features"):
        value = sample.get(key)
        if value is not None:
            arr = np.asarray(value)
            if arr.ndim >= 1:
                return int(arr.shape[0])
    return 0


def sample_name(sample, idx):
    return str(sample.get("complex_code") or sample.get("pdb_chain") or idx)


def choose_partner_donors(data_list, seed=0, neighbor_count=10):
    n = len(data_list)
    if n <= 1:
        return np.arange(n, dtype=np.int64)
    rng = np.random.RandomState(seed)
    lengths = np.asarray([partner_length(sample) for sample in data_list], dtype=np.float64)
    donors = np.zeros(n, dtype=np.int64)
    k = max(1, min(int(neighbor_count), n - 1))
    for i in range(n):
        distances = np.abs(lengths - lengths[i])
        order = np.argsort(distances + rng.uniform(0.0, 1e-6, size=n))
        candidates = [int(j) for j in order if int(j) != i and lengths[j] > 0]
        if not candidates:
            candidates = [int(j) for j in order if int(j) != i]
        pool = candidates[:k]
        donors[i] = int(rng.choice(pool))
    return donors


def make_mismatched_partner_list(data_list, seed=0, neighbor_count=10):
    donors = choose_partner_donors(data_list, seed=seed, neighbor_count=neighbor_count)
    mismatched = []
    mapping = []
    for i, sample in enumerate(data_list):
        donor_idx = int(donors[i])
        donor = data_list[donor_idx]
        new_sample = copy.copy(sample)
        for key in PARTNER_FIELDS:
            if key in donor:
                new_sample[key] = donor[key]
            elif key in new_sample:
                new_sample.pop(key)
        # Pair-contact labels become invalid after swapping partners.
        new_sample["partner_pair_contact_index"] = np.empty((2, 0), dtype=np.int64)
        new_sample["partner_pair_contact_cutoff"] = sample.get("partner_pair_contact_cutoff", None)
        mapping.append({
            "target_index": int(i),
            "target": sample_name(sample, i),
            "donor_index": donor_idx,
            "donor": sample_name(donor, donor_idx),
            "target_partner_length": partner_length(sample),
            "donor_partner_length": partner_length(donor),
        })
        mismatched.append(new_sample)
    return mismatched, mapping


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


def predict_fold_scores(
    model,
    checkpoint,
    data_list,
    checkpoint_plm_dim,
    checkpoint_aux_plm_dim,
    batch_size,
    device,
    rank_fusion,
):
    dataset = make_dataset(data_list, checkpoint, checkpoint_plm_dim, checkpoint_aux_plm_dim)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_proteins,
        num_workers=0,
    )
    labels, probs, aux_probs, cls_probs, rank_probs = predict_scores_with_heads(
        model,
        loader,
        device,
        rank_fusion=rank_fusion,
    )
    return labels, probs, aux_probs, cls_probs, rank_probs


def add_metrics(rows, summary, group, test_name, condition, metrics):
    key = f"{condition}_{group}"
    summary.setdefault(key, {})[test_name] = metrics
    rows.append({
        "group": key,
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


def summarize_drop(drop_summary):
    print("\n------------ Mismatched Partner Drop Summary ------------")
    ranked = sorted(
        drop_summary.items(),
        key=lambda item: (item[1]["avg_aupr_drop"], item[1]["avg_mcc_drop"]),
        reverse=True,
    )
    for name, values in ranked:
        print(
            f"{name:<18} AvgAUPRCdrop={values['avg_aupr_drop']:.4f} "
            f"AvgMCCdrop={values['avg_mcc_drop']:.4f} "
            f"MinAUPRCdrop={values['min_aupr_drop']:.4f} "
            f"MinMCCdrop={values['min_mcc_drop']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Mismatched-partner control for PC-BIND saved checkpoints."
    )
    parser.add_argument(
        "--run-dir",
        default="outputs_pcbind_pairmlp_w005_seed2073",
        help="Directory containing fold*_best.pt checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for control outputs. Defaults to --run-dir.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Fold seed. Defaults to checkpoint metadata.")
    parser.add_argument("--mismatch-seed", type=int, default=9917)
    parser.add_argument("--neighbor-count", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
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
    checkpoint_test_files = first_checkpoint.get("test_sets")
    if checkpoint_test_files:
        test_sets = {
            os.path.splitext(str(filename))[0]: load_pickle(dataset_path(str(filename)))
            for filename in checkpoint_test_files
        }
    else:
        test_sets = {
            name: load_pickle(path)
            for name, path in available_test_datasets(require_geo=True)
        }
    in_dim = train_list[0]["residue_graph_node"].shape[1]
    atom_dim = train_list[0]["atom_graph_node"].shape[1]
    plm_mode, plm_dim = configure_plm_feature_config([train_list, test_sets])

    print(f"Device: {device}")
    print(f"Run dir: {run_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Fold seed: {seed}")
    print(f"Grouped CV: {grouped_cv} (key={cv_group_key})")
    print(f"Mismatch seed: {args.mismatch_seed}")
    print(f"Partner donor neighbor count: {args.neighbor_count}")
    print(f"PLM feature mode: {plm_mode}")
    print(f"PLM feature dim: {plm_dim}")

    validation_folds = make_cv_folds(
        train_list,
        seed,
        num_folds=5,
        grouped=grouped_cv,
        group_key=cv_group_key,
    )
    calibration = load_saved_ensemble_calibration(run_dir, len(validation_folds))

    artifacts = []
    oof_labels = []
    oof_probs = []
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
        val_metrics = None
        if calibration is None:
            val_idx = validation_folds[fold - 1]
            val_list = [train_list[i] for i in val_idx]
            labels, probs, _, _, _ = predict_fold_scores(
                model,
                checkpoint,
                val_list,
                checkpoint_plm_dim,
                checkpoint_aux_plm_dim,
                args.batch_size,
                device,
                rank_fusion=float(checkpoint.get("two_head_rank_fusion", train_cfg.TWO_HEAD_RANK_FUSION)),
            )
            val_metrics = metrics_from_scores(
                labels,
                probs,
                split_name=f"MismatchControl-Fold{fold}-TruePartner-Val",
            )
            oof_labels.append(labels)
            oof_probs.append(probs)
        artifacts.append({
            "checkpoint": checkpoint,
            "checkpoint_plm_dim": checkpoint_plm_dim,
            "checkpoint_aux_plm_dim": checkpoint_aux_plm_dim,
            "val_mcc": None if val_metrics is None else val_metrics["mcc"],
            "val_auc_pr": None if val_metrics is None else val_metrics["auc_pr"],
            "rank_fusion": float(checkpoint.get("two_head_rank_fusion", train_cfg.TWO_HEAD_RANK_FUSION)),
        })

    if calibration is None:
        oof_y = np.concatenate(oof_labels, axis=0)
        oof_score = np.concatenate(oof_probs, axis=0)
        calibration = {
            "source": "recomputed_oof",
            "prob_threshold": best_mcc_threshold(oof_y, oof_score),
            "rank_threshold": best_mcc_threshold(oof_y, rank_normalize(oof_score)),
            "val_mcc_weights": normalized_positive_weights([artifact["val_mcc"] for artifact in artifacts]),
            "val_aupr_weights": normalized_positive_weights([artifact["val_auc_pr"] for artifact in artifacts]),
        }
    prob_threshold = calibration["prob_threshold"]
    rank_threshold = calibration["rank_threshold"]
    val_mcc_weights = calibration["val_mcc_weights"]
    val_aupr_weights = calibration["val_aupr_weights"]
    strategies = {
        "mean": {"mode": "mean", "weights": None, "threshold": prob_threshold},
        "weighted_mcc": {"mode": "weighted", "weights": val_mcc_weights, "threshold": prob_threshold},
        "weighted_aupr": {"mode": "weighted", "weights": val_aupr_weights, "threshold": prob_threshold},
        "rank": {"mode": "rank", "weights": None, "threshold": rank_threshold},
    }
    print("\n===== OOF True-Partner Calibration =====")
    print(f"Calibration source: {calibration['source']}")
    print(f"Probability threshold: {prob_threshold:.3f}")
    print(f"Rank threshold: {rank_threshold:.3f}")

    rows = []
    summary = {}
    drop_summary = {}
    mismatch_maps = {}

    for test_offset, (test_name, test_list) in enumerate(test_sets.items()):
        print(f"\nEvaluating mismatched-partner control on {test_name}", flush=True)
        mismatch_list, mapping = make_mismatched_partner_list(
            test_list,
            seed=args.mismatch_seed + test_offset,
            neighbor_count=args.neighbor_count,
        )
        mismatch_maps[test_name] = mapping
        true_fold_scores = []
        mismatch_fold_scores = []
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
            labels, true_probs, _, _, _ = predict_fold_scores(
                model,
                checkpoint,
                test_list,
                checkpoint_plm_dim,
                checkpoint_aux_plm_dim,
                args.batch_size,
                device,
                rank_fusion=artifact["rank_fusion"],
            )
            mismatch_labels, mismatch_probs, _, _, _ = predict_fold_scores(
                model,
                checkpoint,
                mismatch_list,
                checkpoint_plm_dim,
                checkpoint_aux_plm_dim,
                args.batch_size,
                device,
                rank_fusion=artifact["rank_fusion"],
            )
            if not np.array_equal(labels, mismatch_labels):
                raise RuntimeError(f"Label mismatch after partner swap on {test_name}, fold {fold}")
            y_true = labels if y_true is None else y_true
            true_fold_scores.append(true_probs)
            mismatch_fold_scores.append(mismatch_probs)

        for strategy_name, strategy in strategies.items():
            true_score = combine_fold_scores(true_fold_scores, strategy["mode"], strategy["weights"])
            mismatch_score = combine_fold_scores(mismatch_fold_scores, strategy["mode"], strategy["weights"])
            true_metrics = metrics_from_scores(
                y_true,
                true_score,
                split_name=f"MismatchControl-true_{strategy_name}-{test_name}",
                threshold=strategy["threshold"],
            )
            mismatch_metrics = metrics_from_scores(
                y_true,
                mismatch_score,
                split_name=f"MismatchControl-mismatch_{strategy_name}-{test_name}",
                threshold=strategy["threshold"],
            )
            add_metrics(rows, summary, strategy_name, test_name, "true", true_metrics)
            add_metrics(rows, summary, strategy_name, test_name, "mismatch", mismatch_metrics)
            drop_summary.setdefault(strategy_name, {
                "test_sets": {},
                "aupr_drops": [],
                "mcc_drops": [],
            })
            aupr_drop = true_metrics["auc_pr"] - mismatch_metrics["auc_pr"]
            mcc_drop = true_metrics["mcc"] - mismatch_metrics["mcc"]
            drop_summary[strategy_name]["test_sets"][test_name] = {
                "true_auc_pr": true_metrics["auc_pr"],
                "mismatch_auc_pr": mismatch_metrics["auc_pr"],
                "aupr_drop": aupr_drop,
                "true_mcc": true_metrics["mcc"],
                "mismatch_mcc": mismatch_metrics["mcc"],
                "mcc_drop": mcc_drop,
            }
            drop_summary[strategy_name]["aupr_drops"].append(aupr_drop)
            drop_summary[strategy_name]["mcc_drops"].append(mcc_drop)
            print(
                f"Drop {strategy_name}-{test_name}: "
                f"AUPRC {aupr_drop:+.4f}, MCC {mcc_drop:+.4f}",
                flush=True,
            )

    for strategy_name, values in drop_summary.items():
        aupr_drops = np.asarray(values.pop("aupr_drops"), dtype=np.float64)
        mcc_drops = np.asarray(values.pop("mcc_drops"), dtype=np.float64)
        values["avg_aupr_drop"] = float(aupr_drops.mean())
        values["avg_mcc_drop"] = float(mcc_drops.mean())
        values["min_aupr_drop"] = float(aupr_drops.min())
        values["min_mcc_drop"] = float(mcc_drops.min())

    metrics_csv = os.path.join(output_dir, "mismatched_partner_metrics.csv")
    summary_json = os.path.join(output_dir, "mismatched_partner_summary.json")
    write_metrics_csv(metrics_csv, rows)
    save_json(summary_json, {
        "run_dir": run_dir,
        "seed": seed,
        "mismatch_seed": args.mismatch_seed,
        "neighbor_count": args.neighbor_count,
        "grouped_cv": grouped_cv,
        "cv_group_key": cv_group_key,
        "calibration_source": calibration["source"],
        "geometry_feature_mode": GEOMETRY_FEATURE_MODE,
        "surface_feature_mode": SURFACE_FEATURE_MODE,
        "sequence_feature_mode": SEQUENCE_FEATURE_MODE,
        "plm_feature_mode": plm_mode,
        "plm_feature_dim": plm_dim,
        "prob_threshold": prob_threshold,
        "rank_threshold": rank_threshold,
        "val_mcc_weights": val_mcc_weights,
        "val_aupr_weights": val_aupr_weights,
        "summary": summary,
        "drop_summary": drop_summary,
        "mismatch_maps": mismatch_maps,
    })
    summarize_drop(drop_summary)
    print(f"\nSaved mismatched-partner metrics to: {metrics_csv}")
    print(f"Saved mismatched-partner summary to: {summary_json}")


if __name__ == "__main__":
    main()
