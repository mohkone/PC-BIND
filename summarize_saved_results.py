import csv
import json
import os
from collections import defaultdict


OUTPUT_DIR = "outputs"
METRICS_PATH = os.path.join(OUTPUT_DIR, "saved_ensemble_metrics.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "saved_ensemble_summary.json")
TABLE_PATH = os.path.join(OUTPUT_DIR, "paper_ready_saved_ensemble_table.csv")
MARKDOWN_PATH = os.path.join(OUTPUT_DIR, "paper_ready_saved_ensemble_table.md")

METRIC_FIELDS = ["auc_pr", "auc_roc", "mcc", "f1", "acc", "precision", "recall"]
PREFERRED_TEST_ORDER = ["Test60", "Test287", "Test70", "TestB25", "TestUB25"]
REPORT_STRATEGIES = [
    "teacher_stack_weighted_aupr",
    "teacher_stack_mean",
    "teacher_blend_weighted_aupr",
    "rank",
    "head_blend",
    "head_blend_rank",
]


def load_metrics(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["strategy"] = row["group"].replace("saved_ensemble_", "", 1)
        for key in METRIC_FIELDS + ["threshold"]:
            row[key] = float(row[key])
    return rows


def load_summary(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ordered_tests(rows):
    present = {row["test_set"] for row in rows}
    preferred = [test for test in PREFERRED_TEST_ORDER if test in present]
    extras = sorted(present - set(preferred))
    return preferred + extras


def best_rows(rows, metric, tests=None):
    tests = set(tests or ordered_tests(rows))
    by_test = {}
    for test in tests:
        candidates = [r for r in rows if r["test_set"] == test]
        by_test[test] = max(candidates, key=lambda r: r[metric])
    return by_test


def aggregate_by_strategy(rows):
    tests = ordered_tests(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["strategy"]].append(row)

    aggregate = []
    for strategy, items in grouped.items():
        by_test = {row["test_set"]: row for row in items}
        if not all(test in by_test for test in tests):
            continue
        avg_auc_pr = sum(by_test[test]["auc_pr"] for test in tests) / len(tests)
        avg_mcc = sum(by_test[test]["mcc"] for test in tests) / len(tests)
        hard_auc_pr = (by_test["TestB25"]["auc_pr"] + by_test["TestUB25"]["auc_pr"]) / 2.0
        hard_mcc = (by_test["TestB25"]["mcc"] + by_test["TestUB25"]["mcc"]) / 2.0
        min_auc_pr = min(by_test[test]["auc_pr"] for test in tests)
        aggregate.append({
            "strategy": strategy,
            "avg_auc_pr": avg_auc_pr,
            "avg_mcc": avg_mcc,
            "hard_auc_pr": hard_auc_pr,
            "hard_mcc": hard_mcc,
            "min_auc_pr": min_auc_pr,
        })
    return aggregate


def write_table(rows, path):
    tests = ordered_tests(rows)
    report = [r for r in rows if r["strategy"] in REPORT_STRATEGIES]
    report.sort(key=lambda r: (REPORT_STRATEGIES.index(r["strategy"]), tests.index(r["test_set"])))
    fields = ["strategy", "test_set"] + METRIC_FIELDS + ["threshold"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in report:
            writer.writerow({field: row[field] for field in fields})
    return report


def write_markdown(report, aggregate, summary, path):
    tests = ordered_tests(report)
    best_auc = best_rows(report, "auc_pr")
    best_mcc = best_rows(report, "mcc")
    best_avg_auc = max(aggregate, key=lambda r: r["avg_auc_pr"])
    best_hard_mcc = max(aggregate, key=lambda r: r["hard_mcc"])

    lines = []
    lines.append("# Saved Ensemble Summary")
    lines.append("")
    lines.append("## OOF-Calibrated Signals")
    lines.append("")
    for key in [
        "teacher_stack_oof_auc_pr",
        "teacher_blend_oof_auc_pr",
        "head_blend_oof_auc_pr",
    ]:
        if key in summary:
            lines.append(f"- {key}: {summary[key]:.4f}")
    if "teacher_blend_alpha" in summary:
        lines.append(f"- teacher_blend_alpha: {summary['teacher_blend_alpha']:.2f}")
    if "head_blend_rank_alpha" in summary:
        lines.append(f"- head_blend_rank_alpha: {summary['head_blend_rank_alpha']:.2f}")

    lines.append("")
    lines.append("## Robust Picks")
    lines.append("")
    lines.append(
        f"- Best average AUPRC: {best_avg_auc['strategy']} "
        f"(avg AUPRC={best_avg_auc['avg_auc_pr']:.4f}, avg MCC={best_avg_auc['avg_mcc']:.4f})"
    )
    lines.append(
        f"- Best hard-set MCC: {best_hard_mcc['strategy']} "
        f"(hard MCC={best_hard_mcc['hard_mcc']:.4f}, hard AUPRC={best_hard_mcc['hard_auc_pr']:.4f})"
    )

    lines.append("")
    lines.append("## Best Within Reported Strategies")
    lines.append("")
    lines.append("| Test | Best AUPRC | AUPRC | Best MCC | MCC |")
    lines.append("| --- | --- | ---: | --- | ---: |")
    for test in tests:
        auc_row = best_auc[test]
        mcc_row = best_mcc[test]
        lines.append(
            f"| {test} | {auc_row['strategy']} | {auc_row['auc_pr']:.4f} | "
            f"{mcc_row['strategy']} | {mcc_row['mcc']:.4f} |"
        )

    lines.append("")
    lines.append("## Compact Table")
    lines.append("")
    lines.append("| Strategy | Test | AUPRC | AUROC | MCC | F1 | ACC | Recall |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in report:
        lines.append(
            f"| {row['strategy']} | {row['test_set']} | {row['auc_pr']:.4f} | "
            f"{row['auc_roc']:.4f} | {row['mcc']:.4f} | {row['f1']:.4f} | "
            f"{row['acc']:.4f} | {row['recall']:.4f} |"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    rows = load_metrics(METRICS_PATH)
    summary = load_summary(SUMMARY_PATH)
    aggregate = aggregate_by_strategy(rows)
    report = write_table(rows, TABLE_PATH)
    write_markdown(report, aggregate, summary, MARKDOWN_PATH)

    best_avg_auc = max(aggregate, key=lambda r: r["avg_auc_pr"])
    best_hard_mcc = max(aggregate, key=lambda r: r["hard_mcc"])
    print(f"Wrote {TABLE_PATH}")
    print(f"Wrote {MARKDOWN_PATH}")
    print(
        f"Best average AUPRC: {best_avg_auc['strategy']} "
        f"avg_auc_pr={best_avg_auc['avg_auc_pr']:.4f}, avg_mcc={best_avg_auc['avg_mcc']:.4f}"
    )
    print(
        f"Best hard-set MCC: {best_hard_mcc['strategy']} "
        f"hard_mcc={best_hard_mcc['hard_mcc']:.4f}, hard_auc_pr={best_hard_mcc['hard_auc_pr']:.4f}"
    )


if __name__ == "__main__":
    main()
