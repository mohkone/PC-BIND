import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import CROSS5FOLD_multi_test as train_cfg
from check_pcbind_prereqs import valid_partner_encoder_sample
from CROSS5FOLD_multi_test import (
    collate_proteins,
    combine_fold_scores,
    configure_plm_feature_config,
    dataset_path,
    metrics_from_scores,
    save_json,
    write_metrics_csv,
)
from evaluate_mismatched_partner_control import make_mismatched_partner_list
from evaluate_pcbind_marginal import (
    infer_run_seed,
    load_fold_model,
    load_pickle,
    make_dataset,
    predict_scores_with_marginal,
)
from evaluate_saved_ensembles import checkpoint_rank_fusion


def evaluate_condition(
    model,
    checkpoint,
    data_list,
    checkpoint_plm_dim,
    checkpoint_aux_plm_dim,
    batch_size,
    device,
    return_contact_scores=False,
    contact_hard_k=10,
):
    dataset = make_dataset(
        data_list,
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
    prediction = predict_scores_with_marginal(
        model,
        loader,
        device,
        rank_fusion=checkpoint_rank_fusion(checkpoint),
        smooth_marginal=False,
        return_contact_scores=return_contact_scores,
        contact_hard_k=contact_hard_k,
    )
    labels, site, _, _, _, marginal, _ = prediction[:7]
    if return_contact_scores:
        positive_scores, hard_scores = prediction[7:9]
        return labels, site, marginal, positive_scores, hard_scores
    return labels, site, marginal


def append_metrics(rows, summary, strategy, test_name, condition, metrics):
    summary.setdefault(strategy, {}).setdefault(test_name, {})[condition] = metrics
    rows.append({
        "group": f"{condition}_{strategy}",
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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Mismatched-partner control for PC-BIND site, raw contact-marginal, "
            "and OOF-selected direct-blend readouts."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--marginal-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mismatch-seed", type=int, default=9917)
    parser.add_argument("--neighbor-count", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--test-sets",
        default="Test60.pkl,Test287.pkl,TestB25.pkl",
        help=(
            "Comma-separated partner-capable test files. TestUB25 is excluded by "
            "default because the GraphPPIS benchmark contains monomeric targets, "
            "not paired partner structures."
        ),
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    summary_path = os.path.abspath(args.marginal_summary)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with open(summary_path, "r", encoding="utf-8") as f:
        marginal_summary = json.load(f)
    if marginal_summary.get("smooth_marginal", True):
        raise ValueError("Marginal summary must come from a raw (--no-smooth-marginal) evaluation.")

    first_checkpoint = torch.load(
        os.path.join(run_dir, "fold1_best.pt"),
        map_location="cpu",
        weights_only=False,
    )
    seed = infer_run_seed(run_dir, args.seed if args.seed is not None else train_cfg.SEED)
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
    partner_coverage = {}
    for test_name, samples in list(test_sets.items()):
        valid_samples = [sample for sample in samples if valid_partner_encoder_sample(sample)]
        excluded = [
            {
                "index": idx,
                "complex_code": str(sample.get("complex_code", "?")),
            }
            for idx, sample in enumerate(samples)
            if not valid_partner_encoder_sample(sample)
        ]
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modes = {
        "mean": {
            "mode": "mean",
            "alpha": float(marginal_summary["marginal_blend_alpha"]),
            "site_threshold": float(marginal_summary["prob_threshold"]),
            "marginal_threshold": float(marginal_summary["marginal_threshold"]),
            "blend_threshold": float(marginal_summary["marginal_blend_threshold"]),
        },
        "rank": {
            "mode": "rank",
            "alpha": float(marginal_summary["marginal_rank_blend_alpha"]),
            "site_threshold": float(marginal_summary["rank_threshold"]),
            "marginal_threshold": float(marginal_summary["marginal_rank_threshold"]),
            "blend_threshold": float(marginal_summary["marginal_rank_blend_threshold"]),
        },
    }

    print(f"Device: {device}")
    print(f"Run dir: {run_dir}")
    print(f"Marginal summary: {summary_path}")
    print(f"Output dir: {output_dir}")
    print(f"Fold seed: {seed}")
    print(f"Mismatch seed: {args.mismatch_seed}")
    print(f"Test sets: {', '.join(test_sets)}")
    for test_name, coverage in partner_coverage.items():
        print(
            f"Partner-complete {test_name}: "
            f"{coverage['evaluated']}/{coverage['total']}"
        )
    print(f"PLM feature mode: {plm_mode} ({plm_dim})")
    print(
        "OOF-selected marginal weights: "
        f"mean={modes['mean']['alpha']:.3f}, rank={modes['rank']['alpha']:.3f}"
    )

    rows = []
    result_summary = {}
    sensitivity_summary = {}
    mismatch_maps = {}
    drops = {}
    metrics_path = os.path.join(output_dir, "pcbind_marginal_mismatch_metrics.csv")
    summary_output_path = os.path.join(output_dir, "pcbind_marginal_mismatch_summary.json")
    mapping_path = os.path.join(output_dir, "pcbind_marginal_mismatch_mapping.json")
    progress_path = os.path.join(output_dir, "pcbind_marginal_mismatch_progress.json")

    for test_offset, (test_name, test_list) in enumerate(test_sets.items()):
        print(f"\nEvaluating marginal mismatch control on {test_name}", flush=True)
        mismatch_list, mapping = make_mismatched_partner_list(
            test_list,
            seed=args.mismatch_seed + test_offset,
            neighbor_count=args.neighbor_count,
        )
        mismatch_maps[test_name] = mapping
        true_site_folds = []
        true_marginal_folds = []
        mismatch_site_folds = []
        mismatch_marginal_folds = []
        y_true = None

        for fold in range(1, 6):
            model, checkpoint, checkpoint_plm_dim, checkpoint_aux_plm_dim = load_fold_model(
                run_dir,
                fold,
                in_dim,
                atom_dim,
                plm_dim,
                device,
            )
            labels, true_site, true_marginal = evaluate_condition(
                model,
                checkpoint,
                test_list,
                checkpoint_plm_dim,
                checkpoint_aux_plm_dim,
                args.batch_size,
                device,
            )
            mismatch_labels, mismatch_site, mismatch_marginal = evaluate_condition(
                model,
                checkpoint,
                mismatch_list,
                checkpoint_plm_dim,
                checkpoint_aux_plm_dim,
                args.batch_size,
                device,
            )
            if not np.array_equal(labels, mismatch_labels):
                raise RuntimeError(f"Label mismatch after partner swap on {test_name}, fold {fold}")
            y_true = labels if y_true is None else y_true
            true_site_folds.append(true_site)
            true_marginal_folds.append(true_marginal)
            mismatch_site_folds.append(mismatch_site)
            mismatch_marginal_folds.append(mismatch_marginal)

        for mode_name, config in modes.items():
            mode = config["mode"]
            alpha = config["alpha"]
            true_site = combine_fold_scores(true_site_folds, mode=mode)
            true_marginal = combine_fold_scores(true_marginal_folds, mode=mode)
            mismatch_site = combine_fold_scores(mismatch_site_folds, mode=mode)
            mismatch_marginal = combine_fold_scores(mismatch_marginal_folds, mode=mode)
            true_blend = (1.0 - alpha) * true_site + alpha * true_marginal
            mismatch_blend = (1.0 - alpha) * mismatch_site + alpha * mismatch_marginal

            readouts = {
                f"site_{mode_name}": (true_site, mismatch_site, config["site_threshold"]),
                f"marginal_{mode_name}": (
                    true_marginal,
                    mismatch_marginal,
                    config["marginal_threshold"],
                ),
                f"blend_{mode_name}": (true_blend, mismatch_blend, config["blend_threshold"]),
            }
            for strategy, (true_score, mismatch_score, threshold) in readouts.items():
                abs_delta = np.abs(true_score - mismatch_score)
                if np.std(true_score) > 0.0 and np.std(mismatch_score) > 0.0:
                    score_correlation = float(np.corrcoef(true_score, mismatch_score)[0, 1])
                else:
                    score_correlation = 1.0 if np.array_equal(true_score, mismatch_score) else 0.0
                sensitivity_summary.setdefault(strategy, {})[test_name] = {
                    "mean_abs_score_delta": float(np.mean(abs_delta)),
                    "median_abs_score_delta": float(np.median(abs_delta)),
                    "max_abs_score_delta": float(np.max(abs_delta)),
                    "score_correlation": score_correlation,
                }
                true_metrics = metrics_from_scores(
                    y_true,
                    true_score,
                    split_name=f"MarginalMismatch-true_{strategy}-{test_name}",
                    threshold=threshold,
                )
                mismatch_metrics = metrics_from_scores(
                    y_true,
                    mismatch_score,
                    split_name=f"MarginalMismatch-mismatch_{strategy}-{test_name}",
                    threshold=threshold,
                )
                append_metrics(rows, result_summary, strategy, test_name, "true", true_metrics)
                append_metrics(rows, result_summary, strategy, test_name, "mismatch", mismatch_metrics)
                aupr_drop = true_metrics["auc_pr"] - mismatch_metrics["auc_pr"]
                mcc_drop = true_metrics["mcc"] - mismatch_metrics["mcc"]
                drops.setdefault(strategy, {"aupr": [], "mcc": []})
                drops[strategy]["aupr"].append(aupr_drop)
                drops[strategy]["mcc"].append(mcc_drop)
                print(
                    f"Drop {strategy}-{test_name}: "
                    f"AUPRC {aupr_drop:+.4f}, MCC {mcc_drop:+.4f}, "
                    f"mean|score delta| {np.mean(abs_delta):.6f}, corr {score_correlation:.4f}",
                    flush=True,
                )

        # Preserve completed test sets even if a later inference or output step fails.
        write_metrics_csv(metrics_path, rows)
        save_json(progress_path, {
            "run_dir": run_dir,
            "marginal_summary": summary_path,
            "seed": seed,
            "mismatch_seed": args.mismatch_seed,
            "neighbor_count": args.neighbor_count,
            "completed_test_sets": list(mismatch_maps),
            "partner_coverage": partner_coverage,
            "smooth_marginal": False,
            "modes": modes,
            "summary": result_summary,
            "sensitivity": sensitivity_summary,
        })
        save_json(mapping_path, mismatch_maps)

    drop_summary = {}
    print("\n------------ Marginal Mismatched-Partner Drop Summary ------------")
    for strategy, values in drops.items():
        drop_summary[strategy] = {
            "avg_aupr_drop": float(np.mean(values["aupr"])),
            "avg_mcc_drop": float(np.mean(values["mcc"])),
            "min_aupr_drop": float(np.min(values["aupr"])),
            "min_mcc_drop": float(np.min(values["mcc"])),
        }
        item = drop_summary[strategy]
        print(
            f"{strategy:<18} AvgAUPRCdrop={item['avg_aupr_drop']:.4f} "
            f"AvgMCCdrop={item['avg_mcc_drop']:.4f} "
            f"MinAUPRCdrop={item['min_aupr_drop']:.4f} "
            f"MinMCCdrop={item['min_mcc_drop']:.4f}"
        )

    write_metrics_csv(metrics_path, rows)
    save_json(summary_output_path, {
        "run_dir": run_dir,
        "marginal_summary": summary_path,
        "seed": seed,
        "mismatch_seed": args.mismatch_seed,
        "neighbor_count": args.neighbor_count,
        "test_sets": list(test_sets),
        "partner_coverage": partner_coverage,
        "smooth_marginal": False,
        "modes": modes,
        "summary": result_summary,
        "sensitivity": sensitivity_summary,
        "drop_summary": drop_summary,
    })
    save_json(mapping_path, mismatch_maps)
    print(f"\nSaved marginal mismatch metrics to: {metrics_path}")
    print(f"Saved marginal mismatch summary to: {summary_output_path}")
    print(f"Saved mismatch mapping to: {mapping_path}")


if __name__ == "__main__":
    main()
