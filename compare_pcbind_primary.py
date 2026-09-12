import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, matthews_corrcoef, precision_recall_curve, roc_auc_score


DEFAULT_TESTS = ("Test60", "Test287", "TestB25", "TestUB25")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare the frozen PC-BIND v5 model with its matched grouped control."
    )
    parser.add_argument("--control-dir", default="outputs_grouped_full_seed2101")
    parser.add_argument("--model-dir", default="outputs_pcbind_v5_shared_seed2101")
    parser.add_argument("--data-dir", default=str(Path("data") / "geo"))
    parser.add_argument("--tests", nargs="*", default=list(DEFAULT_TESTS))
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2101)
    parser.add_argument("--output-json", default="pcbind_primary_comparison.json")
    parser.add_argument("--output-md", default="pcbind_primary_comparison.md")
    return parser.parse_args()


def load_predictions(run_dir, test_set):
    path = Path(run_dir) / f"ensemble_{test_set}_predictions.npz"
    archive = np.load(path)
    return (
        np.asarray(archive["labels"], dtype=np.int64),
        np.asarray(archive["probs"], dtype=np.float64),
        float(np.asarray(archive["threshold"]).reshape(-1)[0]),
    )


def complex_indices(data_path, expected_residues):
    with open(data_path, "rb") as handle:
        samples = pickle.load(handle)
    groups = defaultdict(list)
    offset = 0
    for sample_index, sample in enumerate(samples):
        count = int(sample["residue_graph_node"].shape[0])
        code = str(sample.get("complex_code", sample_index)).strip().upper()
        groups[code].append(np.arange(offset, offset + count, dtype=np.int64))
        offset += count
    if offset != expected_residues:
        raise ValueError(
            f"{data_path}: expected {expected_residues} residues but reconstructed {offset}"
        )
    return [np.concatenate(groups[key]) for key in sorted(groups)]


def metrics(labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int64)
    return {
        "auc_pr": auc_pr(labels, probabilities),
        "auc_roc": float(roc_auc_score(labels, probabilities)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
    }


def auc_pr(labels, probabilities):
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    return float(auc(recall, precision))


def cluster_bootstrap(labels, control_probs, model_probs, groups, replicates, seed):
    rng = np.random.default_rng(seed)
    auc_differences = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in selected])
        sampled_labels = labels[indices]
        auc_differences[replicate] = auc_pr(
            sampled_labels, model_probs[indices]
        ) - auc_pr(sampled_labels, control_probs[indices])
    low, high = np.percentile(auc_differences, [2.5, 97.5])
    return {
        "mean": float(np.mean(auc_differences)),
        "ci_low": float(low),
        "ci_high": float(high),
        "probability_positive": float(np.mean(auc_differences > 0.0)),
    }


def benchmark_bootstrap(differences, replicates=100000, seed=9102):
    rng = np.random.default_rng(seed)
    values = np.asarray(differences, dtype=np.float64)
    indices = rng.integers(0, values.size, size=(replicates, values.size))
    sampled = values[indices].mean(axis=1)
    low, high = np.percentile(sampled, [2.5, 97.5])
    return {
        "mean_difference": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def write_markdown(path, report):
    lines = [
        "# Frozen PC-BIND Primary Comparison",
        "",
        "PC-BIND v5 shared-partner encoding is compared with the exact grouped target-only control. Both use seed 2101, identical grouped five-fold splits, the same feature and optimization recipe, and a five-checkpoint probability mean. Test70 is excluded because of Train335 sequence overlap.",
        "",
        "| Test set | Control AUPRC | PC-BIND AUPRC | Difference | Clustered 95% CI | Control MCC | PC-BIND MCC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for test_set, row in report["tests"].items():
        lines.append(
            f"| {test_set} | {row['control']['auc_pr']:.4f} | "
            f"{row['model']['auc_pr']:.4f} | {row['auc_pr_difference']:+.4f} | "
            f"[{row['cluster_bootstrap']['ci_low']:+.4f}, {row['cluster_bootstrap']['ci_high']:+.4f}] | "
            f"{row['control']['mcc']:.4f} | {row['model']['mcc']:.4f} |"
        )
    summary = report["summary"]
    lines.extend(
        [
            "",
            f"Across the four retained tests, average AUPRC was {summary['control_avg_auc_pr']:.4f} for the control and {summary['model_avg_auc_pr']:.4f} for PC-BIND (difference {summary['auc_pr_benchmark_bootstrap']['mean_difference']:+.4f}; benchmark-bootstrap 95% CI [{summary['auc_pr_benchmark_bootstrap']['ci_low']:+.4f}, {summary['auc_pr_benchmark_bootstrap']['ci_high']:+.4f}]). Average MCC changed from {summary['control_avg_mcc']:.4f} to {summary['model_avg_mcc']:.4f}.",
            "",
            "The per-test clustered intervals quantify uncertainty over complexes within each dataset. The benchmark-bootstrap interval resamples only four datasets and should be interpreted cautiously.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    report = {
        "control_dir": str(Path(args.control_dir).resolve()),
        "model_dir": str(Path(args.model_dir).resolve()),
        "cluster_bootstrap_replicates": args.bootstrap,
        "tests": {},
    }
    auc_differences = []
    for test_index, test_set in enumerate(args.tests):
        labels_control, control_probs, control_threshold = load_predictions(
            args.control_dir, test_set
        )
        labels_model, model_probs, model_threshold = load_predictions(
            args.model_dir, test_set
        )
        if not np.array_equal(labels_control, labels_model):
            raise ValueError(f"{test_set}: control and model labels are not aligned")
        groups = complex_indices(
            Path(args.data_dir) / f"{test_set}.pkl", labels_control.size
        )
        control_metrics = metrics(labels_control, control_probs, control_threshold)
        model_metrics = metrics(labels_model, model_probs, model_threshold)
        difference = model_metrics["auc_pr"] - control_metrics["auc_pr"]
        auc_differences.append(difference)
        report["tests"][test_set] = {
            "complexes": len(groups),
            "residues": int(labels_control.size),
            "control_threshold": control_threshold,
            "model_threshold": model_threshold,
            "control": control_metrics,
            "model": model_metrics,
            "auc_pr_difference": difference,
            "mcc_difference": model_metrics["mcc"] - control_metrics["mcc"],
            "cluster_bootstrap": cluster_bootstrap(
                labels_control,
                control_probs,
                model_probs,
                groups,
                args.bootstrap,
                args.seed + test_index,
            ),
        }
        print(
            f"{test_set}: AUPRC {control_metrics['auc_pr']:.4f} -> "
            f"{model_metrics['auc_pr']:.4f} ({difference:+.4f})"
        )

    rows = list(report["tests"].values())
    report["summary"] = {
        "control_avg_auc_pr": float(np.mean([row["control"]["auc_pr"] for row in rows])),
        "model_avg_auc_pr": float(np.mean([row["model"]["auc_pr"] for row in rows])),
        "control_avg_mcc": float(np.mean([row["control"]["mcc"] for row in rows])),
        "model_avg_mcc": float(np.mean([row["model"]["mcc"] for row in rows])),
        "auc_pr_benchmark_bootstrap": benchmark_bootstrap(auc_differences),
    }
    Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(args.output_md, report)
    print(f"Wrote {Path(args.output_json).resolve()}")
    print(f"Wrote {Path(args.output_md).resolve()}")


if __name__ == "__main__":
    main()
