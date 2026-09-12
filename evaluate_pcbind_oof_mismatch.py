import argparse
import json
import os
import re

import numpy as np
import torch

import CROSS5FOLD_multi_test as train_cfg
from check_pcbind_prereqs import valid_partner_encoder_sample
from CROSS5FOLD_multi_test import (
    best_mcc_threshold,
    configure_plm_feature_config,
    dataset_path,
    make_cv_folds,
    metrics_from_scores,
    save_json,
)
from evaluate_mismatched_partner_control import make_mismatched_partner_list
from evaluate_pcbind_marginal import infer_run_seed, load_fold_model, load_pickle
from evaluate_pcbind_marginal_mismatch import evaluate_condition
from metrics import compute_auc_pr


def available_folds(run_dir):
    folds = []
    for name in os.listdir(run_dir):
        match = re.fullmatch(r"fold(\d+)_best\.pt", name)
        if match:
            folds.append(int(match.group(1)))
    return sorted(fold for fold in folds if 1 <= fold <= 5)


def cluster_bootstrap_auc_pr(
    labels,
    true_scores,
    mismatch_scores,
    cluster_indices,
    replicates,
    seed,
):
    observed = float(
        compute_auc_pr(labels, true_scores)
        - compute_auc_pr(labels, mismatch_scores)
    )
    if replicates <= 0:
        return {"observed_drop": observed, "replicates": 0}

    rng = np.random.RandomState(seed)
    deltas = []
    cluster_count = len(cluster_indices)
    for _ in range(replicates):
        selected = rng.randint(0, cluster_count, size=cluster_count)
        indices = np.concatenate([cluster_indices[idx] for idx in selected])
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size < 2:
            continue
        deltas.append(
            compute_auc_pr(sampled_labels, true_scores[indices])
            - compute_auc_pr(sampled_labels, mismatch_scores[indices])
        )
    if not deltas:
        return {"observed_drop": observed, "replicates": 0}
    deltas = np.asarray(deltas, dtype=np.float64)
    return {
        "observed_drop": observed,
        "bootstrap_mean_drop": float(deltas.mean()),
        "ci95_low": float(np.percentile(deltas, 2.5)),
        "ci95_high": float(np.percentile(deltas, 97.5)),
        "positive_fraction": float(np.mean(deltas > 0.0)),
        "replicates": int(deltas.size),
    }


def cluster_bootstrap_contact_gap(differences, cluster_indices, replicates, seed):
    finite = np.isfinite(differences)
    observed_values = differences[finite]
    observed = {
        "mean_d": float(np.mean(observed_values)),
        "median_d": float(np.median(observed_values)),
        "positive_fraction": float(np.mean(observed_values > 0.0)),
        "eligible_residues": int(observed_values.size),
    }
    if replicates <= 0:
        return {**observed, "replicates": 0}

    rng = np.random.RandomState(seed)
    means = []
    medians = []
    cluster_count = len(cluster_indices)
    for _ in range(replicates):
        selected = rng.randint(0, cluster_count, size=cluster_count)
        sampled = np.concatenate([differences[cluster_indices[idx]] for idx in selected])
        sampled = sampled[np.isfinite(sampled)]
        if sampled.size == 0:
            continue
        means.append(float(np.mean(sampled)))
        medians.append(float(np.median(sampled)))
    if not means:
        return {**observed, "replicates": 0}
    means = np.asarray(means, dtype=np.float64)
    medians = np.asarray(medians, dtype=np.float64)
    return {
        **observed,
        "mean_ci95_low": float(np.percentile(means, 2.5)),
        "mean_ci95_high": float(np.percentile(means, 97.5)),
        "median_ci95_low": float(np.percentile(medians, 2.5)),
        "median_ci95_high": float(np.percentile(medians, 97.5)),
        "mean_positive_fraction": float(np.mean(means > 0.0)),
        "replicates": int(means.size),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate true-versus-mismatched partner sensitivity only on grouped "
            "Train335 out-of-fold validation predictions."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mismatch-seed", type=int, default=12017)
    parser.add_argument("--neighbor-count", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--blend-alpha", type=float, default=0.10)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--contact-hard-k", type=int, default=None)
    parser.add_argument("--control-summary", default="")
    args = parser.parse_args()

    if not 0.0 <= args.blend_alpha <= 1.0:
        raise ValueError("--blend-alpha must be between 0 and 1")
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
        args.seed if args.seed is not None else int(first_checkpoint.get("seed", train_cfg.SEED)),
    )
    grouped_cv = bool(first_checkpoint.get("grouped_cv", False))
    group_key = str(first_checkpoint.get("cv_group_key", "complex_code"))
    contact_hard_k = int(
        args.contact_hard_k
        if args.contact_hard_k is not None
        else first_checkpoint.get("pair_contact_contrast_hard_k", 10)
    )
    if not grouped_cv:
        raise ValueError("OOF mismatch evaluation requires grouped_cv=True checkpoints")

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
    print(f"Fixed marginal blend alpha: {args.blend_alpha:.3f}")
    print(f"Contact hard-negative K: {contact_hard_k}")

    labels_parts = []
    true_site_parts = []
    mismatch_site_parts = []
    true_marginal_parts = []
    mismatch_marginal_parts = []
    true_positive_contact_parts = []
    mismatch_hard_contact_parts = []
    sample_records = []
    residue_offset = 0
    excluded = []

    for fold in folds:
        raw_indices = cv_folds[fold - 1]
        fold_samples = []
        for sample_idx in raw_indices:
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

        mismatch_samples, _ = make_mismatched_partner_list(
            fold_samples,
            seed=args.mismatch_seed + fold - 1,
            neighbor_count=args.neighbor_count,
        )
        model, checkpoint, checkpoint_plm_dim, checkpoint_aux_plm_dim = load_fold_model(
            run_dir,
            fold,
            in_dim,
            atom_dim,
            plm_dim,
            device,
        )
        (
            labels,
            true_site,
            true_marginal,
            true_positive_contact,
            _true_hard_contact,
        ) = evaluate_condition(
            model,
            checkpoint,
            fold_samples,
            checkpoint_plm_dim,
            checkpoint_aux_plm_dim,
            args.batch_size,
            device,
            return_contact_scores=True,
            contact_hard_k=contact_hard_k,
        )
        (
            mismatch_labels,
            mismatch_site,
            mismatch_marginal,
            _mismatch_positive_contact,
            mismatch_hard_contact,
        ) = evaluate_condition(
            model,
            checkpoint,
            mismatch_samples,
            checkpoint_plm_dim,
            checkpoint_aux_plm_dim,
            args.batch_size,
            device,
            return_contact_scores=True,
            contact_hard_k=contact_hard_k,
        )
        if not np.array_equal(labels, mismatch_labels):
            raise RuntimeError(f"Fold {fold} label mismatch after partner swap")

        labels_parts.append(labels)
        true_site_parts.append(true_site)
        mismatch_site_parts.append(mismatch_site)
        true_marginal_parts.append(true_marginal)
        mismatch_marginal_parts.append(mismatch_marginal)
        true_positive_contact_parts.append(true_positive_contact)
        mismatch_hard_contact_parts.append(mismatch_hard_contact)
        for sample in fold_samples:
            length = int(len(sample["label"]))
            sample_records.append({
                "fold": fold,
                "complex_code": str(sample.get(group_key, sample.get("complex_code", "?"))),
                "start": residue_offset,
                "end": residue_offset + length,
            })
            residue_offset += length
        print(
            f"Fold {fold}: samples={len(fold_samples)}, residues={labels.size}, "
            f"marginal AUPRC drop="
            f"{compute_auc_pr(labels, true_marginal) - compute_auc_pr(labels, mismatch_marginal):+.4f}"
        )

    labels = np.concatenate(labels_parts)
    true_site = np.concatenate(true_site_parts)
    mismatch_site = np.concatenate(mismatch_site_parts)
    true_marginal = np.concatenate(true_marginal_parts)
    mismatch_marginal = np.concatenate(mismatch_marginal_parts)
    true_positive_contact = np.concatenate(true_positive_contact_parts)
    mismatch_hard_contact = np.concatenate(mismatch_hard_contact_parts)
    contact_differences = true_positive_contact - mismatch_hard_contact
    true_blend = (1.0 - args.blend_alpha) * true_site + args.blend_alpha * true_marginal
    mismatch_blend = (
        (1.0 - args.blend_alpha) * mismatch_site
        + args.blend_alpha * mismatch_marginal
    )

    grouped_segments = {}
    for record in sample_records:
        grouped_segments.setdefault(record["complex_code"], []).append(
            np.arange(record["start"], record["end"], dtype=np.int64)
        )
    cluster_codes = sorted(grouped_segments)
    cluster_indices = [np.concatenate(grouped_segments[code]) for code in cluster_codes]

    contact_gap = cluster_bootstrap_contact_gap(
        contact_differences,
        cluster_indices,
        args.bootstrap_replicates,
        args.mismatch_seed + 2000,
    )
    positive_interface_residues = int(np.sum(labels > 0.5))
    contact_gap["positive_interface_residues"] = positive_interface_residues
    contact_gap["eligibility_coverage"] = (
        float(contact_gap["eligible_residues"] / positive_interface_residues)
        if positive_interface_residues
        else 0.0
    )
    per_complex_contact_gap = []
    for code, indices in zip(cluster_codes, cluster_indices):
        values = contact_differences[indices]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        per_complex_contact_gap.append({
            "complex_code": code,
            "eligible_residues": int(values.size),
            "mean_d": float(np.mean(values)),
            "median_d": float(np.median(values)),
            "positive_fraction": float(np.mean(values > 0.0)),
        })

    control_comparison = None
    if args.control_summary:
        control_path = os.path.abspath(args.control_summary)
        with open(control_path, "r", encoding="utf-8") as handle:
            control_output = json.load(handle)
        control_gap = control_output.get("contact_gap")
        if not control_gap:
            raise ValueError(f"Control summary has no contact_gap section: {control_path}")
        control_comparison = {
            "control_summary": control_path,
            "mean_d_change": float(contact_gap["mean_d"] - control_gap["mean_d"]),
            "median_d_change": float(contact_gap["median_d"] - control_gap["median_d"]),
            "positive_fraction_change": float(
                contact_gap["positive_fraction"] - control_gap["positive_fraction"]
            ),
        }

    summary = {}
    bootstrap = {}
    for strategy, true_scores, mismatch_scores in (
        ("site", true_site, mismatch_site),
        ("marginal", true_marginal, mismatch_marginal),
        ("blend", true_blend, mismatch_blend),
    ):
        threshold = best_mcc_threshold(labels, true_scores)
        true_metrics = metrics_from_scores(
            labels,
            true_scores,
            split_name=f"OOF-TruePartner-{strategy}",
            threshold=threshold,
        )
        mismatch_metrics = metrics_from_scores(
            labels,
            mismatch_scores,
            split_name=f"OOF-MismatchedPartner-{strategy}",
            threshold=threshold,
        )
        summary[strategy] = {
            "threshold": threshold,
            "true": true_metrics,
            "mismatch": mismatch_metrics,
            "aupr_drop": float(true_metrics["auc_pr"] - mismatch_metrics["auc_pr"]),
            "mcc_drop": float(true_metrics["mcc"] - mismatch_metrics["mcc"]),
            "mean_abs_score_delta": float(np.mean(np.abs(true_scores - mismatch_scores))),
            "score_correlation": float(np.corrcoef(true_scores, mismatch_scores)[0, 1]),
        }
        bootstrap[strategy] = cluster_bootstrap_auc_pr(
            labels,
            true_scores,
            mismatch_scores,
            cluster_indices,
            args.bootstrap_replicates,
            args.mismatch_seed + 1000 + len(bootstrap),
        )

    np.savez_compressed(
        os.path.join(output_dir, "pcbind_oof_mismatch_predictions.npz"),
        labels=labels.astype(np.float32),
        true_site=true_site.astype(np.float32),
        mismatch_site=mismatch_site.astype(np.float32),
        true_marginal=true_marginal.astype(np.float32),
        mismatch_marginal=mismatch_marginal.astype(np.float32),
        true_blend=true_blend.astype(np.float32),
        mismatch_blend=mismatch_blend.astype(np.float32),
        true_positive_contact=true_positive_contact.astype(np.float32),
        mismatch_hard_contact=mismatch_hard_contact.astype(np.float32),
        contact_differences=contact_differences.astype(np.float32),
    )
    output = {
        "run_dir": run_dir,
        "seed": seed,
        "mismatch_seed": args.mismatch_seed,
        "folds": folds,
        "grouped_cv": True,
        "cv_group_key": group_key,
        "blend_alpha": args.blend_alpha,
        "contact_hard_k": contact_hard_k,
        "evaluated_samples": len(sample_records),
        "evaluated_clusters": len(cluster_codes),
        "evaluated_residues": int(labels.size),
        "excluded": excluded,
        "summary": summary,
        "cluster_bootstrap_auc_pr": bootstrap,
        "contact_gap": contact_gap,
        "per_complex_contact_gap": per_complex_contact_gap,
        "control_comparison": control_comparison,
    }
    summary_path = os.path.join(output_dir, "pcbind_oof_mismatch_summary.json")
    save_json(summary_path, output)

    print("\n------------ OOF Mismatched-Partner Summary ------------")
    for strategy in ("site", "marginal", "blend"):
        item = summary[strategy]
        ci = bootstrap[strategy]
        ci_text = (
            f"95% CI [{ci['ci95_low']:+.4f}, {ci['ci95_high']:+.4f}]"
            if "ci95_low" in ci
            else "CI not requested"
        )
        print(
            f"{strategy:<8} AUPRCdrop={item['aupr_drop']:+.4f} "
            f"MCCdrop={item['mcc_drop']:+.4f} {ci_text}"
        )
    print("\n------------ Contact-Level Partner Gap ------------")
    mean_ci = (
        f"95% CI [{contact_gap['mean_ci95_low']:+.4f}, "
        f"{contact_gap['mean_ci95_high']:+.4f}]"
        if "mean_ci95_low" in contact_gap
        else "CI not requested"
    )
    print(
        f"eligible={contact_gap['eligible_residues']}/"
        f"{contact_gap['positive_interface_residues']} "
        f"({100.0 * contact_gap['eligibility_coverage']:.1f}%) "
        f"mean_D={contact_gap['mean_d']:+.4f} "
        f"median_D={contact_gap['median_d']:+.4f} "
        f"P(D>0)={contact_gap['positive_fraction']:.3f} {mean_ci}"
    )
    if control_comparison is not None:
        print(
            "Change from control: "
            f"mean_D={control_comparison['mean_d_change']:+.4f}, "
            f"median_D={control_comparison['median_d_change']:+.4f}, "
            f"P(D>0)={control_comparison['positive_fraction_change']:+.3f}"
        )
    print(f"Saved OOF mismatch summary to: {summary_path}")


if __name__ == "__main__":
    main()
