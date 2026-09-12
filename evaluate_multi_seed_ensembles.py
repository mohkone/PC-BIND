import argparse
import csv
import glob
from itertools import combinations
import os
import pickle
import re

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import CROSS5FOLD_multi_test as train_cfg
from CROSS5FOLD_multi_test import (
    SEED,
    GEOMETRY_FEATURE_MODE,
    SURFACE_FEATURE_MODE,
    SEQUENCE_FEATURE_MODE,
    PLM_FEATURE_MODE,
    PARTNER_CONTACT_AUX,
    PARTNER_CONTACT_AUX_WEIGHT,
    ProteinDataset,
    available_test_datasets,
    best_mcc_threshold,
    build_model,
    collate_proteins,
    combine_fold_scores,
    configure_plm_feature_config,
    dataset_path,
    metrics_from_scores,
    normalized_positive_weights,
    predict_scores_with_heads,
    rank_normalize,
    save_json,
    write_metrics_csv,
)
from evaluate_saved_ensembles import (
    apply_checkpoint_runtime_config,
    blend_head_scores,
    checkpoint_view_logit_fusion,
    load_compatible_state,
    stack_features,
)
from metrics import compute_auc_pr


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_run_spec(spec):
    if ":" in spec:
        out_dir, seed_text = spec.rsplit(":", 1)
        return out_dir, int(seed_text)
    return spec, None


def infer_seed(run_dir, checkpoint):
    if "seed" in checkpoint:
        return int(checkpoint["seed"])
    match = re.search(r"seed[_-]?(\d+)", os.path.basename(run_dir), flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return SEED


def discover_runs():
    candidates = ["outputs"] + sorted(glob.glob("outputs_seed*")) + sorted(glob.glob("outputs_*seed*"))
    runs = []
    seen = set()
    for path in candidates:
        if path in seen or not os.path.isdir(path):
            continue
        if os.path.exists(os.path.join(path, "fold1_best.pt")):
            runs.append(path)
            seen.add(path)
    return runs


def residue_offsets(protein_list):
    lengths = [int(p["residue_graph_node"].shape[0]) for p in protein_list]
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    return offsets


def residue_positions_for_complexes(offsets, complex_indices):
    pieces = [np.arange(offsets[i], offsets[i + 1], dtype=np.int64) for i in complex_indices]
    return np.concatenate(pieces, axis=0) if pieces else np.empty((0,), dtype=np.int64)


def fold_complex_indices(num_complexes, seed, fold):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(num_complexes)
    fold_ids = np.arange(num_complexes) % 5
    return indices[fold_ids == fold]


def build_dataset(samples, checkpoint, checkpoint_plm_dim, checkpoint_aux_plm_dim):
    return ProteinDataset(
        samples,
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


def load_model_from_checkpoint(checkpoint, in_dim, atom_dim, device, fallback_plm_dim, fallback_aux_plm_dim):
    checkpoint_plm_dim = checkpoint.get("plm_feature_dim")
    checkpoint_aux_plm_dim = checkpoint.get("aux_plm_feature_dim")
    checkpoint_plm_dim = int(fallback_plm_dim if checkpoint_plm_dim is None else checkpoint_plm_dim)
    checkpoint_aux_plm_dim = int((fallback_aux_plm_dim or 0) if checkpoint_aux_plm_dim is None else checkpoint_aux_plm_dim)
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
    return model, checkpoint_plm_dim, checkpoint_aux_plm_dim


def checkpoint_rank_fusion(checkpoint):
    return float(checkpoint.get("two_head_rank_fusion", train_cfg.TWO_HEAD_RANK_FUSION))


def load_run_artifacts(run_dir, explicit_seed, train_list, offsets, in_dim, atom_dim, device, plm_dim, aux_plm_dim, batch_size):
    first_checkpoint = torch.load(os.path.join(run_dir, "fold1_best.pt"), map_location=device, weights_only=False)
    seed = int(explicit_seed if explicit_seed is not None else infer_seed(run_dir, first_checkpoint))
    total_residues = int(offsets[-1])
    oof_labels = np.full((total_residues,), np.nan, dtype=np.float32)
    oof_main = np.full((total_residues,), np.nan, dtype=np.float32)
    oof_aux = np.full((total_residues,), np.nan, dtype=np.float32)
    oof_cls = np.full((total_residues,), np.nan, dtype=np.float32)
    oof_rank = np.full((total_residues,), np.nan, dtype=np.float32)

    artifacts = []
    print(f"\nLoading run {run_dir} with seed {seed}", flush=True)
    for fold in range(5):
        ckpt_path = os.path.join(run_dir, f"fold{fold + 1}_best.pt")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        ckpt_geometry_mode = checkpoint.get("geometry_feature_mode", "legacy_or_unknown")
        if ckpt_geometry_mode != GEOMETRY_FEATURE_MODE:
            print(
                f"Warning: {ckpt_path} geometry mode is {ckpt_geometry_mode}, "
                f"current code expects {GEOMETRY_FEATURE_MODE}.",
                flush=True,
            )
        model, checkpoint_plm_dim, checkpoint_aux_plm_dim = load_model_from_checkpoint(
            checkpoint, in_dim, atom_dim, device, plm_dim, aux_plm_dim
        )
        val_idx = fold_complex_indices(len(train_list), seed, fold)
        val_list = [train_list[i] for i in val_idx]
        val_dataset = build_dataset(checkpoint=checkpoint, samples=val_list,
                                    checkpoint_plm_dim=checkpoint_plm_dim,
                                    checkpoint_aux_plm_dim=checkpoint_aux_plm_dim)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_proteins,
            num_workers=0,
        )
        rank_fusion = checkpoint_rank_fusion(checkpoint)
        val_y, val_main, val_aux, val_cls, val_rank = predict_scores_with_heads(
            model, val_loader, device, rank_fusion=rank_fusion
        )
        dest = residue_positions_for_complexes(offsets, val_idx)
        if dest.size != val_y.size:
            raise RuntimeError(f"{ckpt_path}: residue alignment mismatch {dest.size} vs {val_y.size}")
        oof_labels[dest] = val_y
        oof_main[dest] = val_main
        oof_aux[dest] = val_aux
        oof_cls[dest] = val_cls
        oof_rank[dest] = val_rank
        val_metrics = metrics_from_scores(val_y, val_main, split_name=f"Run-{os.path.basename(run_dir)}-Fold{fold + 1}-Val")
        artifacts.append({
            "run_dir": run_dir,
            "seed": seed,
            "fold": fold + 1,
            "checkpoint": checkpoint,
            "val_mcc": val_metrics["mcc"],
            "val_auc_pr": val_metrics["auc_pr"],
            "plm_dim": checkpoint_plm_dim,
            "aux_plm_dim": checkpoint_aux_plm_dim,
            "rank_fusion": rank_fusion,
            "view_logit_fusion": checkpoint_view_logit_fusion(checkpoint),
        })

    for name, values in {
        "labels": oof_labels,
        "main": oof_main,
        "aux": oof_aux,
        "cls": oof_cls,
        "rank": oof_rank,
    }.items():
        if np.isnan(values).any():
            raise RuntimeError(f"{run_dir}: incomplete OOF {name} predictions")

    return {
        "run_dir": run_dir,
        "seed": seed,
        "artifacts": artifacts,
        "oof_labels": oof_labels,
        "oof_main": oof_main,
        "oof_aux": oof_aux,
        "oof_cls": oof_cls,
        "oof_rank": oof_rank,
    }


def combine_model_scores(scores, mode, weights=None):
    return combine_fold_scores(scores, mode=mode, weights=weights)


def run_stack_features(main_scores, aux_scores, cls_scores=None, rank_scores=None):
    cols = []
    for scores in (main_scores, aux_scores, cls_scores, rank_scores):
        if scores is None:
            continue
        for score in scores:
            score = np.asarray(score, dtype=np.float64)
            cols.append(score)
            cols.append(rank_normalize(score))
    return np.vstack(cols).T


def run_stack_feature_names(run_infos):
    names = []
    for family in ("main", "aux", "cls", "rank"):
        for info in run_infos:
            run_name = os.path.basename(info["run_dir"])
            label = f"{family}:{run_name}:seed{info['seed']}"
            names.append(label)
            names.append(f"{label}:ranknorm")
    return names


def fit_sparse_blender(features, labels, max_terms=6, candidate_count=18):
    """Fast OOF-selected sparse convex blend over run/family feature columns."""
    features = np.asarray(features, dtype=np.float64)
    single_auc = np.asarray(
        [compute_auc_pr(labels, features[:, idx]) for idx in range(features.shape[1])],
        dtype=np.float64,
    )
    candidates = np.argsort(single_auc)[::-1][: min(candidate_count, features.shape[1])]
    best_idx = int(candidates[0])
    weights = {best_idx: 1.0}
    score = features[:, best_idx].copy()
    auc_pr = float(single_auc[best_idx])
    alpha_grid = np.linspace(0.05, 0.75, 15)

    for _ in range(max_terms - 1):
        best = None
        for idx in candidates:
            idx = int(idx)
            if idx in weights:
                continue
            candidate = features[:, idx]
            for alpha in alpha_grid:
                blended = (1.0 - alpha) * score + alpha * candidate
                blended_auc = compute_auc_pr(labels, blended)
                key = (blended_auc, -len(weights), -alpha)
                if best is None or key > best["key"]:
                    best = {
                        "key": key,
                        "idx": idx,
                        "alpha": float(alpha),
                        "score": blended,
                        "auc_pr": float(blended_auc),
                    }
        if best is None or best["auc_pr"] <= auc_pr + 1e-5:
            break
        for idx in list(weights):
            weights[idx] *= 1.0 - best["alpha"]
        weights[best["idx"]] = weights.get(best["idx"], 0.0) + best["alpha"]
        score = best["score"]
        auc_pr = best["auc_pr"]

    ordered_weights = sorted(weights.items(), key=lambda item: abs(item[1]), reverse=True)
    return {
        "score": score,
        "auc_pr": auc_pr,
        "weights": ordered_weights,
        "candidate_indices": [int(idx) for idx in candidates],
        "single_auc_pr": single_auc,
    }


def predict_sparse_blender(blender, features):
    features = np.asarray(features, dtype=np.float64)
    score = np.zeros(features.shape[0], dtype=np.float64)
    for idx, weight in blender["weights"]:
        score += float(weight) * features[:, int(idx)]
    return score


def selected_sparse_features(blender, feature_names):
    return [
        {
            "feature": feature_names[int(idx)],
            "weight": float(weight),
            "single_auc_pr": float(blender["single_auc_pr"][int(idx)]),
        }
        for idx, weight in blender["weights"]
    ]


def run_subset_label(indices, run_infos):
    return "+".join(str(run_infos[idx]["seed"]) for idx in indices)


def all_run_subsets(run_count):
    for size in range(1, run_count + 1):
        yield from combinations(range(run_count), size)


def mean_subset(scores, indices):
    return np.mean([np.asarray(scores[idx], dtype=np.float64) for idx in indices], axis=0)


def rank_mean_subset(scores, indices):
    return np.mean([rank_normalize(scores[idx]) for idx in indices], axis=0)


def select_subset_strategy(labels, run_infos, scorer):
    best = None
    for indices in all_run_subsets(len(run_infos)):
        score, params = scorer(indices)
        auc_pr = compute_auc_pr(labels, score)
        key = (auc_pr, -len(indices))
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "indices": tuple(int(idx) for idx in indices),
                "score": score,
                "auc_pr": float(auc_pr),
                "params": dict(params),
            }
    best["threshold"] = best_mcc_threshold(labels, best["score"])
    best["label"] = run_subset_label(best["indices"], run_infos)
    return best


def build_subset_strategies(oof_y, run_infos, oof_run_main, oof_run_aux, oof_run_cls, oof_run_rank):
    alpha_grid = np.linspace(0.0, 1.0, 21)

    def main_scorer(indices):
        return mean_subset(oof_run_main, indices), {}

    def rank_scorer(indices):
        return rank_mean_subset(oof_run_main, indices), {}

    def teacher_scorer(indices):
        main = mean_subset(oof_run_main, indices)
        aux = mean_subset(oof_run_aux, indices)
        best = None
        for alpha in alpha_grid:
            score = (1.0 - alpha) * main + alpha * aux
            auc_pr = compute_auc_pr(oof_y, score)
            if best is None or auc_pr > best["auc_pr"]:
                best = {"score": score, "auc_pr": auc_pr, "alpha": float(alpha)}
        return best["score"], {"alpha": best["alpha"]}

    def head_scorer(indices):
        cls = mean_subset(oof_run_cls, indices)
        rank = mean_subset(oof_run_rank, indices)
        best = None
        for alpha in alpha_grid:
            score = blend_head_scores(cls, rank, alpha)
            auc_pr = compute_auc_pr(oof_y, score)
            if best is None or auc_pr > best["auc_pr"]:
                best = {"score": score, "auc_pr": auc_pr, "alpha": float(alpha)}
        return best["score"], {"alpha": best["alpha"]}

    return {
        "subset_main": select_subset_strategy(oof_y, run_infos, main_scorer),
        "subset_rank": select_subset_strategy(oof_y, run_infos, rank_scorer),
        "subset_teacher": select_subset_strategy(oof_y, run_infos, teacher_scorer),
        "subset_head": select_subset_strategy(oof_y, run_infos, head_scorer),
    }


def subset_strategy_summary(strategy):
    return {
        "runs": strategy["label"],
        "indices": list(strategy["indices"]),
        "auc_pr": strategy["auc_pr"],
        "threshold": strategy["threshold"],
        "params": strategy["params"],
    }


def evaluate_strategy(rows, summary, y_true, score, name, test_name, threshold):
    metrics = metrics_from_scores(
        y_true,
        score,
        split_name=f"MultiSeed-{name}-{test_name}",
        threshold=threshold,
    )
    summary.setdefault(name, {})[test_name] = metrics
    rows.append({
        "group": f"multiseed_{name}",
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
    return metrics


def summarize_strategy_rows(rows):
    groups = sorted({row["group"] for row in rows})
    ranking = []
    for group in groups:
        group_rows = [row for row in rows if row["group"] == group]
        auc_pr = np.asarray([row["auc_pr"] for row in group_rows], dtype=np.float64)
        mcc = np.asarray([row["mcc"] for row in group_rows], dtype=np.float64)
        f1 = np.asarray([row["f1"] for row in group_rows], dtype=np.float64)
        ranking.append({
            "group": group,
            "test_count": len(group_rows),
            "avg_auc_pr": float(np.mean(auc_pr)),
            "avg_mcc": float(np.mean(mcc)),
            "avg_f1": float(np.mean(f1)),
            "min_auc_pr": float(np.min(auc_pr)),
            "min_mcc": float(np.min(mcc)),
        })
    ranking.sort(key=lambda row: (row["avg_auc_pr"], row["avg_mcc"], row["min_auc_pr"]), reverse=True)

    best_by_test = []
    for test_name in sorted({row["test_set"] for row in rows}):
        test_rows = [row for row in rows if row["test_set"] == test_name]
        best_auc_pr = max(test_rows, key=lambda row: (row["auc_pr"], row["mcc"]))
        best_mcc = max(test_rows, key=lambda row: (row["mcc"], row["auc_pr"]))
        best_by_test.append({
            "test_set": test_name,
            "best_auc_pr_group": best_auc_pr["group"],
            "best_auc_pr": best_auc_pr["auc_pr"],
            "best_auc_pr_mcc": best_auc_pr["mcc"],
            "best_mcc_group": best_mcc["group"],
            "best_mcc": best_mcc["mcc"],
            "best_mcc_auc_pr": best_mcc["auc_pr"],
        })
    return ranking, best_by_test


def write_table_csv(path, rows, fieldnames):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_strategy_report(ranking, best_by_test, top_k=8):
    print("\n------------ Multi-seed Strategy Ranking ------------", flush=True)
    for row in ranking[:top_k]:
        print(
            f"{row['group']:<38} "
            f"AvgAUPRC={row['avg_auc_pr']:.4f} AvgMCC={row['avg_mcc']:.4f} "
            f"MinAUPRC={row['min_auc_pr']:.4f} MinMCC={row['min_mcc']:.4f}",
            flush=True,
        )

    print("\n------------ Best Strategy By Test Set ------------", flush=True)
    for row in best_by_test:
        print(
            f"{row['test_set']:<8} "
            f"best AUPRC: {row['best_auc_pr_group']}={row['best_auc_pr']:.4f} "
            f"(MCC={row['best_auc_pr_mcc']:.4f}); "
            f"best MCC: {row['best_mcc_group']}={row['best_mcc']:.4f} "
            f"(AUPRC={row['best_mcc_auc_pr']:.4f})",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate ensembles across multiple seed output directories.")
    parser.add_argument(
        "--runs",
        nargs="*",
        help="Output dirs, optionally with explicit seed as DIR:SEED. Default discovers outputs and outputs_seed*.",
    )
    parser.add_argument("--output-dir", default="outputs_multiseed")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    run_specs = args.runs if args.runs else discover_runs()
    if not run_specs:
        raise SystemExit("No runs found. Expected outputs/fold1_best.pt or outputs_seed*/fold1_best.pt")
    parsed_runs = [parse_run_spec(spec) for spec in run_specs]

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_list = load_pickle(dataset_path("Train335.pkl"))
    test_sets = {
        name: load_pickle(path)
        for name, path in available_test_datasets(require_geo=True)
    }
    in_dim = train_list[0]["residue_graph_node"].shape[1]
    atom_dim = train_list[0]["atom_graph_node"].shape[1]
    plm_mode, plm_dim = configure_plm_feature_config([train_list, test_sets])
    aux_plm_dim = train_cfg.AUX_PLM_FEATURE_DIM
    offsets = residue_offsets(train_list)

    run_infos = [
        load_run_artifacts(
            run_dir=run_dir,
            explicit_seed=seed,
            train_list=train_list,
            offsets=offsets,
            in_dim=in_dim,
            atom_dim=atom_dim,
            device=device,
            plm_dim=plm_dim,
            aux_plm_dim=aux_plm_dim,
            batch_size=args.batch_size,
        )
        for run_dir, seed in parsed_runs
    ]

    oof_y = run_infos[0]["oof_labels"]
    for info in run_infos[1:]:
        if not np.array_equal(oof_y, info["oof_labels"]):
            raise RuntimeError("OOF labels differ across runs after residue alignment.")

    oof_main = np.mean([info["oof_main"] for info in run_infos], axis=0)
    oof_aux = np.mean([info["oof_aux"] for info in run_infos], axis=0)
    oof_cls = np.mean([info["oof_cls"] for info in run_infos], axis=0)
    oof_rank = np.mean([info["oof_rank"] for info in run_infos], axis=0)
    oof_run_main = [info["oof_main"] for info in run_infos]
    oof_run_aux = [info["oof_aux"] for info in run_infos]
    oof_run_cls = [info["oof_cls"] for info in run_infos]
    oof_run_rank = [info["oof_rank"] for info in run_infos]

    head_alphas = np.linspace(0.0, 1.0, 21)
    head_auc_pr = [
        compute_auc_pr(oof_y, blend_head_scores(oof_cls, oof_rank, alpha))
        for alpha in head_alphas
    ]
    head_alpha = float(head_alphas[int(np.argmax(head_auc_pr))])
    oof_head = blend_head_scores(oof_cls, oof_rank, head_alpha)

    teacher_alphas = np.linspace(0.0, 1.0, 21)
    teacher_auc_pr = [
        compute_auc_pr(oof_y, (1.0 - alpha) * oof_main + alpha * oof_aux)
        for alpha in teacher_alphas
    ]
    teacher_alpha = float(teacher_alphas[int(np.argmax(teacher_auc_pr))])
    oof_teacher = (1.0 - teacher_alpha) * oof_main + teacher_alpha * oof_aux

    prob_threshold = best_mcc_threshold(oof_y, oof_main)
    rank_threshold = best_mcc_threshold(oof_y, rank_normalize(oof_main))
    head_threshold = best_mcc_threshold(oof_y, oof_head)
    head_rank_threshold = best_mcc_threshold(oof_y, rank_normalize(oof_head))
    teacher_threshold = best_mcc_threshold(oof_y, oof_teacher)
    teacher_rank_threshold = best_mcc_threshold(oof_y, rank_normalize(oof_teacher))

    stacker = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
    )
    stacker.fit(stack_features(oof_main, oof_aux), oof_y)
    oof_stack = stacker.predict_proba(stack_features(oof_main, oof_aux))[:, 1]
    stack_threshold = best_mcc_threshold(oof_y, oof_stack)

    run_features = run_stack_features(oof_run_main, oof_run_aux, oof_run_cls, oof_run_rank)
    run_stacker = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
    )
    run_stacker.fit(run_features, oof_y)
    oof_run_stack = run_stacker.predict_proba(run_features)[:, 1]
    run_stack_threshold = best_mcc_threshold(oof_y, oof_run_stack)

    sparse_run_stack = fit_sparse_blender(run_features, oof_y)
    oof_sparse_run_stack = sparse_run_stack["score"]
    sparse_run_stack_threshold = best_mcc_threshold(oof_y, oof_sparse_run_stack)
    sparse_run_stack_features = selected_sparse_features(
        sparse_run_stack,
        run_stack_feature_names(run_infos),
    )
    subset_strategies = build_subset_strategies(
        oof_y,
        run_infos,
        oof_run_main,
        oof_run_aux,
        oof_run_cls,
        oof_run_rank,
    )

    print(
        f"\nOOF multi-seed main AUPRC={compute_auc_pr(oof_y, oof_main):.4f}, "
        f"threshold={prob_threshold:.3f}",
        flush=True,
    )
    print(
        f"OOF head alpha={head_alpha:.2f}, AUPRC={max(head_auc_pr):.4f}, "
        f"threshold={head_threshold:.3f}",
        flush=True,
    )
    print(
        f"OOF teacher alpha={teacher_alpha:.2f}, AUPRC={max(teacher_auc_pr):.4f}, "
        f"threshold={teacher_threshold:.3f}",
        flush=True,
    )
    print(
        f"OOF teacher stack AUPRC={compute_auc_pr(oof_y, oof_stack):.4f}, "
        f"threshold={stack_threshold:.3f}",
        flush=True,
    )
    print(
        f"OOF run/family stack AUPRC={compute_auc_pr(oof_y, oof_run_stack):.4f}, "
        f"threshold={run_stack_threshold:.3f}",
        flush=True,
    )
    print(
        f"OOF sparse run/family stack AUPRC={compute_auc_pr(oof_y, oof_sparse_run_stack):.4f}, "
        f"threshold={sparse_run_stack_threshold:.3f}, "
        f"selected={len(sparse_run_stack['weights'])}/{run_features.shape[1]}",
        flush=True,
    )
    for name, strategy in subset_strategies.items():
        params = ", ".join(f"{key}={value:.2f}" for key, value in strategy["params"].items())
        params_text = f", {params}" if params else ""
        print(
            f"OOF {name} AUPRC={strategy['auc_pr']:.4f}, "
            f"threshold={strategy['threshold']:.3f}, runs={strategy['label']}{params_text}",
            flush=True,
        )

    artifacts = [artifact for info in run_infos for artifact in info["artifacts"]]
    val_mcc_weights = normalized_positive_weights([artifact["val_mcc"] for artifact in artifacts])
    val_aupr_weights = normalized_positive_weights([artifact["val_auc_pr"] for artifact in artifacts])

    rows = []
    summary = {}
    for test_name, test_list in test_sets.items():
        print(f"\nEvaluating multi-seed ensemble on {test_name}", flush=True)
        y_true = None
        main_scores = []
        aux_scores = []
        cls_scores = []
        rank_scores = []

        for artifact in artifacts:
            checkpoint = artifact["checkpoint"]
            model, checkpoint_plm_dim, checkpoint_aux_plm_dim = load_model_from_checkpoint(
                checkpoint,
                in_dim,
                atom_dim,
                device,
                plm_dim,
                aux_plm_dim,
            )
            dataset = build_dataset(
                samples=test_list,
                checkpoint=checkpoint,
                checkpoint_plm_dim=checkpoint_plm_dim,
                checkpoint_aux_plm_dim=checkpoint_aux_plm_dim,
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_proteins,
                num_workers=0,
            )
            labels, main, aux, cls, rank = predict_scores_with_heads(
                model, loader, device, rank_fusion=artifact["rank_fusion"]
            )
            y_true = labels if y_true is None else y_true
            main_scores.append(main)
            aux_scores.append(aux)
            cls_scores.append(cls)
            rank_scores.append(rank)

        strategy_defs = {
            "mean": ("mean", None, prob_threshold),
            "weighted_mcc": ("weighted", val_mcc_weights, prob_threshold),
            "weighted_aupr": ("weighted", val_aupr_weights, prob_threshold),
            "rank": ("rank", None, rank_threshold),
        }
        for name, (mode, weights, threshold) in strategy_defs.items():
            main_score = combine_model_scores(main_scores, mode, weights)
            aux_score = combine_model_scores(aux_scores, mode, weights)
            evaluate_strategy(rows, summary, y_true, main_score, name, test_name, threshold)

            teacher_score = (1.0 - teacher_alpha) * main_score + teacher_alpha * aux_score
            teacher_threshold_for_strategy = teacher_rank_threshold if mode == "rank" else teacher_threshold
            evaluate_strategy(
                rows,
                summary,
                y_true,
                teacher_score,
                f"teacher_blend_{name}",
                test_name,
                teacher_threshold_for_strategy,
            )

            stack_score = stacker.predict_proba(stack_features(main_score, aux_score))[:, 1]
            evaluate_strategy(
                rows,
                summary,
                y_true,
                stack_score,
                f"teacher_stack_{name}",
                test_name,
                stack_threshold,
            )

        cls_mean = combine_model_scores(cls_scores, "mean", None)
        rank_mean = combine_model_scores(rank_scores, "mean", None)
        head_score = blend_head_scores(cls_mean, rank_mean, head_alpha)
        evaluate_strategy(rows, summary, y_true, head_score, "head_blend", test_name, head_threshold)

        cls_rank = combine_model_scores(cls_scores, "rank", None)
        rank_rank = combine_model_scores(rank_scores, "rank", None)
        head_rank_score = blend_head_scores(cls_rank, rank_rank, head_alpha)
        evaluate_strategy(rows, summary, y_true, head_rank_score, "head_blend_rank", test_name, head_rank_threshold)

        run_main_scores = []
        run_aux_scores = []
        run_cls_scores = []
        run_rank_scores = []
        start = 0
        for info in run_infos:
            end = start + len(info["artifacts"])
            run_main_scores.append(combine_model_scores(main_scores[start:end], "mean", None))
            run_aux_scores.append(combine_model_scores(aux_scores[start:end], "mean", None))
            run_cls_scores.append(combine_model_scores(cls_scores[start:end], "mean", None))
            run_rank_scores.append(combine_model_scores(rank_scores[start:end], "mean", None))
            start = end
        run_stack_score = run_stacker.predict_proba(
            run_stack_features(run_main_scores, run_aux_scores, run_cls_scores, run_rank_scores)
        )[:, 1]
        evaluate_strategy(rows, summary, y_true, run_stack_score, "run_stack", test_name, run_stack_threshold)

        sparse_run_stack_score = predict_sparse_blender(
            sparse_run_stack,
            run_stack_features(run_main_scores, run_aux_scores, run_cls_scores, run_rank_scores),
        )
        evaluate_strategy(
            rows,
            summary,
            y_true,
            sparse_run_stack_score,
            "run_stack_sparse",
            test_name,
            sparse_run_stack_threshold,
        )

        subset_main = subset_strategies["subset_main"]
        subset_main_score = mean_subset(run_main_scores, subset_main["indices"])
        evaluate_strategy(
            rows,
            summary,
            y_true,
            subset_main_score,
            "subset_main",
            test_name,
            subset_main["threshold"],
        )

        subset_rank = subset_strategies["subset_rank"]
        subset_rank_score = rank_mean_subset(run_main_scores, subset_rank["indices"])
        evaluate_strategy(
            rows,
            summary,
            y_true,
            subset_rank_score,
            "subset_rank",
            test_name,
            subset_rank["threshold"],
        )

        subset_teacher = subset_strategies["subset_teacher"]
        subset_teacher_main = mean_subset(run_main_scores, subset_teacher["indices"])
        subset_teacher_aux = mean_subset(run_aux_scores, subset_teacher["indices"])
        subset_teacher_alpha = float(subset_teacher["params"].get("alpha", 0.0))
        subset_teacher_score = (
            (1.0 - subset_teacher_alpha) * subset_teacher_main
            + subset_teacher_alpha * subset_teacher_aux
        )
        evaluate_strategy(
            rows,
            summary,
            y_true,
            subset_teacher_score,
            "subset_teacher",
            test_name,
            subset_teacher["threshold"],
        )

        subset_head = subset_strategies["subset_head"]
        subset_head_cls = mean_subset(run_cls_scores, subset_head["indices"])
        subset_head_rank = mean_subset(run_rank_scores, subset_head["indices"])
        subset_head_alpha = float(subset_head["params"].get("alpha", 0.0))
        subset_head_score = blend_head_scores(subset_head_cls, subset_head_rank, subset_head_alpha)
        evaluate_strategy(
            rows,
            summary,
            y_true,
            subset_head_score,
            "subset_head",
            test_name,
            subset_head["threshold"],
        )

    strategy_ranking, best_by_test = summarize_strategy_rows(rows)
    write_metrics_csv(os.path.join(args.output_dir, "multi_seed_metrics.csv"), rows)
    write_table_csv(
        os.path.join(args.output_dir, "strategy_ranking.csv"),
        strategy_ranking,
        ["group", "test_count", "avg_auc_pr", "avg_mcc", "avg_f1", "min_auc_pr", "min_mcc"],
    )
    write_table_csv(
        os.path.join(args.output_dir, "best_by_test.csv"),
        best_by_test,
        [
            "test_set",
            "best_auc_pr_group",
            "best_auc_pr",
            "best_auc_pr_mcc",
            "best_mcc_group",
            "best_mcc",
            "best_mcc_auc_pr",
        ],
    )
    save_json(os.path.join(args.output_dir, "multi_seed_summary.json"), {
        "runs": [{"run_dir": info["run_dir"], "seed": info["seed"]} for info in run_infos],
        "model_count": len(artifacts),
        "geometry_feature_mode": GEOMETRY_FEATURE_MODE,
        "surface_feature_mode": SURFACE_FEATURE_MODE,
        "sequence_feature_mode": SEQUENCE_FEATURE_MODE,
        "plm_feature_mode": plm_mode,
        "plm_feature_dim": plm_dim,
        "aux_plm_feature_mode": train_cfg.AUX_PLM_FEATURE_MODE,
        "aux_plm_feature_dim": train_cfg.AUX_PLM_FEATURE_DIM,
        "partner_contact_aux": PARTNER_CONTACT_AUX,
        "partner_contact_aux_weight": PARTNER_CONTACT_AUX_WEIGHT,
        "prob_threshold": prob_threshold,
        "rank_threshold": rank_threshold,
        "head_blend_rank_alpha": head_alpha,
        "head_blend_threshold": head_threshold,
        "head_blend_rank_threshold": head_rank_threshold,
        "head_blend_oof_auc_pr": float(max(head_auc_pr)),
        "teacher_blend_alpha": teacher_alpha,
        "teacher_blend_threshold": teacher_threshold,
        "teacher_blend_rank_threshold": teacher_rank_threshold,
        "teacher_blend_oof_auc_pr": float(max(teacher_auc_pr)),
        "teacher_stack_threshold": stack_threshold,
        "teacher_stack_oof_auc_pr": compute_auc_pr(oof_y, oof_stack),
        "run_stack_threshold": run_stack_threshold,
        "run_stack_oof_auc_pr": compute_auc_pr(oof_y, oof_run_stack),
        "run_stack_sparse_threshold": sparse_run_stack_threshold,
        "run_stack_sparse_oof_auc_pr": compute_auc_pr(oof_y, oof_sparse_run_stack),
        "run_stack_sparse_method": "greedy_convex_feature_blend",
        "run_stack_sparse_selected_count": len(sparse_run_stack["weights"]),
        "run_stack_sparse_candidate_indices": sparse_run_stack["candidate_indices"],
        "run_stack_sparse_features": sparse_run_stack_features,
        "subset_strategies": {
            name: subset_strategy_summary(strategy)
            for name, strategy in subset_strategies.items()
        },
        "checkpoint_rank_fusions": [artifact["rank_fusion"] for artifact in artifacts],
        "checkpoint_view_logit_fusions": [artifact["view_logit_fusion"] for artifact in artifacts],
        "val_mcc_weights": val_mcc_weights,
        "val_aupr_weights": val_aupr_weights,
        "strategy_ranking": strategy_ranking,
        "best_by_test": best_by_test,
        "summary": summary,
    })
    print_strategy_report(strategy_ranking, best_by_test)
    print(f"\nSaved multi-seed outputs to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
