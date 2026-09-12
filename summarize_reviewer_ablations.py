import csv
import pathlib
import statistics as stats


ROOT = pathlib.Path(__file__).resolve().parent
PRIMARY_TEST_SETS = ["Test60", "Test287", "TestB25", "TestUB25"]

GROUPS = {
    "Full model": {
        "2036": "outputs_regdrop_rankwarm3_seed2036",
        "2037": "outputs_regdrop_rankwarm3_seed2037",
        "2038": "outputs_regdrop_rankwarm3_seed2038",
        "2039": "outputs_regdrop_rankwarm3_seed2039",
    },
    "No PLM": {
        "2041": "outputs_no_plm_seed2041",
        "2042": "outputs_no_plm_seed2042",
        "2043": "outputs_no_plm_seed2043",
    },
    "No auxiliary objective": {
        "2051": "outputs_no_aux_seed2051",
        "2052": "outputs_no_aux_seed2052",
        "2053": "outputs_no_aux_seed2053",
    },
}

ENSEMBLES = {
    "Full locked run-stack": "outputs_multiseed_1234_2025_2026_regdrop_rankwarm3_2036_2037_2038_2039_ckptfusion",
    "No PLM run-stack": "outputs_multiseed_no_plm_2041_2042_2043_ckptfusion",
    "No auxiliary run-stack": "outputs_multiseed_no_aux_2051_2052_2053_ckptfusion",
}


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def metric_for_test(rows, test_set):
    ensemble_rows = [
        r for r in rows
        if r.get("test_set") == test_set and r.get("group") == "ensemble_mean"
    ]
    if ensemble_rows:
        r = ensemble_rows[0]
        return float(r["auc_pr"]), float(r["mcc"])

    fold_rows = [
        r for r in rows
        if r.get("test_set") == test_set and r.get("group", "").startswith("fold")
    ]
    if not fold_rows:
        return None
    return (
        stats.mean(float(r["auc_pr"]) for r in fold_rows),
        stats.mean(float(r["mcc"]) for r in fold_rows),
    )


def seed_summary(run_dir):
    path = ROOT / run_dir / "metrics_summary.csv"
    if not path.exists():
        return None
    rows = read_csv(path)
    values = []
    for test_set in PRIMARY_TEST_SETS:
        metric = metric_for_test(rows, test_set)
        if metric is None:
            return None
        values.append(metric)
    aucs = [v[0] for v in values]
    mccs = [v[1] for v in values]
    return {
        "avg_auc_pr": stats.mean(aucs),
        "avg_mcc": stats.mean(mccs),
        "min_auc_pr": min(aucs),
        "min_mcc": min(mccs),
    }


def fmt_mean_sd(values):
    if not values:
        return "To run"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{stats.mean(values):.4f} +/- {stats.stdev(values):.4f}"


def group_summary(group):
    seed_rows = []
    missing = []
    for seed, run_dir in GROUPS[group].items():
        summary = seed_summary(run_dir)
        if summary is None:
            missing.append(seed)
        else:
            seed_rows.append((seed, summary))
    return seed_rows, missing


def ensemble_run_stack_summary(run_dir):
    path = ROOT / run_dir / "multi_seed_metrics.csv"
    if not path.exists():
        return None
    rows = [
        row for row in read_csv(path)
        if row.get("group") == "multiseed_run_stack"
        and row.get("test_set") in PRIMARY_TEST_SETS
    ]
    if len(rows) != len(PRIMARY_TEST_SETS):
        return None
    aucs = [float(row["auc_pr"]) for row in rows]
    mccs = [float(row["mcc"]) for row in rows]
    return {
        "avg_auc_pr": stats.mean(aucs),
        "avg_mcc": stats.mean(mccs),
        "min_auc_pr": min(aucs),
        "min_mcc": min(mccs),
    }


def main():
    lines = []
    lines.append("# Reviewer Ablation Summary")
    lines.append("")
    lines.append("Test70 is excluded from the primary summary because 53 target sequences are exact duplicates of Train335 sequences. The primary benchmark set is Test60, Test287, TestB25, and TestUB25.")
    lines.append("")
    lines.append("## Matched-Protocol Seed Ablations")
    lines.append("")
    lines.append("| Model | Complete seeds | AvgAUPRC | AvgMCC | MinAUPRC | MinMCC | Status |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for group in GROUPS:
        seed_rows, missing = group_summary(group)
        summaries = [s for _, s in seed_rows]
        status = "Complete" if not missing else f"Missing seeds: {', '.join(missing)}"
        lines.append(
            "| {group} | {n} | {avg_auc} | {avg_mcc} | {min_auc} | {min_mcc} | {status} |".format(
                group=group,
                n=len(seed_rows),
                avg_auc=fmt_mean_sd([s["avg_auc_pr"] for s in summaries]),
                avg_mcc=fmt_mean_sd([s["avg_mcc"] for s in summaries]),
                min_auc=fmt_mean_sd([s["min_auc_pr"] for s in summaries]),
                min_mcc=fmt_mean_sd([s["min_mcc"] for s in summaries]),
                status=status,
            )
        )

    lines.append("")
    lines.append("## Optional Run-Stack Ensembles")
    lines.append("")
    lines.append("| Model | AvgAUPRC | AvgMCC | MinAUPRC | MinMCC | Status |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for label, run_dir in ENSEMBLES.items():
        summary = ensemble_run_stack_summary(run_dir)
        if summary is None:
            lines.append(f"| {label} | To run | To run | To run | To run | Missing strategy_ranking.csv |")
        else:
            lines.append(
                f"| {label} | {summary['avg_auc_pr']:.4f} | {summary['avg_mcc']:.4f} | "
                f"{summary['min_auc_pr']:.4f} | {summary['min_mcc']:.4f} | Complete |"
            )

    lines.append("")
    lines.append("The run-stack comparison is exploratory: the full stack has seven heterogeneous seeds, whereas each ablation stack has three seeds. It therefore does not isolate one component as cleanly as the matched-protocol seed analysis.")
    lines.append("")
    lines.append("## Paper Table Fill-In")
    lines.append("")
    lines.append("Use the matched-protocol seed table for the primary reviewer response. Full-model seeds 2036-2039 use the same regularized rank-warmup recipe as the ablations. Older seeds 2025 and 2026 are excluded from this replication summary because their checkpoint metadata do not record the same regularization and warmup protocol. Use the heterogeneous run-stack table only as a secondary peak-performance ensemble analysis.")

    text = "\n".join(lines) + "\n"
    out_path = ROOT / "reviewer_ablation_summary.md"
    out_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
