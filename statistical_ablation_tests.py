import csv
import itertools
import pathlib
import statistics as stats

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parent
TEST_SETS = ["Test60", "Test287", "TestB25", "TestUB25"]
BOOTSTRAP_SAMPLES = 100_000

SINGLE_SEED_GROUPS = {
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

RUN_STACK_GROUPS = {
    "Full locked run-stack": "outputs_multiseed_1234_2025_2026_regdrop_rankwarm3_2036_2037_2038_2039_ckptfusion",
    "No PLM run-stack": "outputs_multiseed_no_plm_2041_2042_2043_ckptfusion",
    "No auxiliary run-stack": "outputs_multiseed_no_aux_2051_2052_2053_ckptfusion",
}


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def auc_pr_for_test(rows, test_set):
    ensemble_rows = [
        r for r in rows
        if r.get("test_set") == test_set and r.get("group") == "ensemble_mean"
    ]
    if ensemble_rows:
        return float(ensemble_rows[0]["auc_pr"])

    fold_rows = [
        r for r in rows
        if r.get("test_set") == test_set and r.get("group", "").startswith("fold")
    ]
    if not fold_rows:
        return None
    return stats.mean(float(r["auc_pr"]) for r in fold_rows)


def single_seed_per_test_means(group_name):
    values_by_test = {test_set: [] for test_set in TEST_SETS}
    for seed, run_dir in SINGLE_SEED_GROUPS[group_name].items():
        path = ROOT / run_dir / "metrics_summary.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics for {group_name} seed {seed}: {path}")
        rows = read_csv(path)
        for test_set in TEST_SETS:
            auc_pr = auc_pr_for_test(rows, test_set)
            if auc_pr is None:
                raise RuntimeError(f"{path} does not contain {test_set}")
            values_by_test[test_set].append(auc_pr)
    return {
        test_set: stats.mean(values)
        for test_set, values in values_by_test.items()
    }


def run_stack_per_test(group_name):
    path = ROOT / RUN_STACK_GROUPS[group_name] / "multi_seed_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing multi-seed metrics: {path}")
    rows = read_csv(path)
    out = {}
    for test_set in TEST_SETS:
        matches = [
            r for r in rows
            if r.get("test_set") == test_set and r.get("group") == "multiseed_run_stack"
        ]
        if not matches:
            raise RuntimeError(f"{path} does not contain multiseed_run_stack for {test_set}")
        out[test_set] = float(matches[0]["auc_pr"])
    return out


def average_ranks(values):
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def wilcoxon_exact_two_sided(differences):
    nonzero = [float(d) for d in differences if abs(float(d)) > 1e-12]
    if not nonzero:
        return 1.0, 0.0, 0

    abs_values = [abs(d) for d in nonzero]
    ranks = average_ranks(abs_values)
    w_plus = sum(rank for rank, diff in zip(ranks, nonzero) if diff > 0)
    rank_total = sum(ranks)
    observed = min(w_plus, rank_total - w_plus)

    possible = []
    for signs in itertools.product([0, 1], repeat=len(nonzero)):
        candidate = sum(rank for rank, sign in zip(ranks, signs) if sign)
        possible.append(min(candidate, rank_total - candidate))

    p_value = sum(1 for value in possible if value <= observed + 1e-12) / len(possible)
    return p_value, w_plus, len(nonzero)


def paired_bootstrap_ci(differences, seed=12345):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(differences, dtype=np.float64)
    indices = rng.integers(0, diffs.size, size=(BOOTSTRAP_SAMPLES, diffs.size))
    boot_means = diffs[indices].mean(axis=1)
    lower, upper = np.percentile(boot_means, [2.5, 97.5])
    return float(lower), float(upper)


def paired_test(full_by_test, ablation_by_test):
    rows = []
    differences = []
    for test_set in TEST_SETS:
        full = float(full_by_test[test_set])
        ablation = float(ablation_by_test[test_set])
        diff = full - ablation
        differences.append(diff)
        rows.append((test_set, full, ablation, diff))
    mean_diff = stats.mean(differences)
    ci_low, ci_high = paired_bootstrap_ci(differences)
    p_value, w_plus, n = wilcoxon_exact_two_sided(differences)
    return rows, {
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "wilcoxon_p": p_value,
        "wilcoxon_w_plus": w_plus,
        "n": n,
    }


def fmt(value):
    return f"{value:.4f}"


def add_analysis(lines, title, full, ablations):
    lines.append(f"## {title}")
    lines.append("")
    for label, values in ablations.items():
        rows, summary = paired_test(full, values)
        lines.append(f"### Full vs {label}")
        lines.append("")
        lines.append("| Test set | Full AUPRC | Ablation AUPRC | Difference |")
        lines.append("|---|---:|---:|---:|")
        for test_set, full_auc, ablation_auc, diff in rows:
            lines.append(f"| {test_set} | {fmt(full_auc)} | {fmt(ablation_auc)} | {fmt(diff)} |")
        lines.append("")
        lines.append(
            "Mean paired difference = {mean}; 95% paired benchmark-bootstrap CI = [{low}, {high}]; "
            "two-sided exact Wilcoxon signed-rank p = {p} (n={n}).".format(
                mean=fmt(summary["mean_diff"]),
                low=fmt(summary["ci_low"]),
                high=fmt(summary["ci_high"]),
                p=f"{summary['wilcoxon_p']:.4f}",
                n=summary["n"],
            )
        )
        lines.append("")


def main():
    single_full = single_seed_per_test_means("Full model")
    single_no_plm = single_seed_per_test_means("No PLM")
    single_no_aux = single_seed_per_test_means("No auxiliary objective")

    run_stack_full = run_stack_per_test("Full locked run-stack")
    run_stack_no_plm = run_stack_per_test("No PLM run-stack")
    run_stack_no_aux = run_stack_per_test("No auxiliary run-stack")

    lines = [
        "# Statistical Ablation Tests",
        "",
        "AUPRC differences are paired by benchmark dataset. Test70 is excluded because 53 target sequences exactly duplicate Train335 sequences. The bootstrap resamples the four retained benchmark datasets with replacement, so the interval should be interpreted as benchmark-level uncertainty rather than residue-level uncertainty. With only four paired datasets, Wilcoxon p-values are conservative. Full-model values use matched-protocol seeds 2036-2039 only.",
        "",
        "The secondary run-stack comparisons are exploratory because the full stack contains seven heterogeneous seeds while each ablation stack contains three seeds. Those comparisons confound model component, seed composition, and ensemble size.",
        "",
    ]
    add_analysis(
        lines,
        "Primary Matched-Protocol Seed Ablation Analysis",
        single_full,
        {
            "No PLM": single_no_plm,
            "No auxiliary objective": single_no_aux,
        },
    )
    add_analysis(
        lines,
        "Secondary Run-Stack Ensemble Analysis",
        run_stack_full,
        {
            "No PLM run-stack": run_stack_no_plm,
            "No auxiliary run-stack": run_stack_no_aux,
        },
    )

    text = "\n".join(lines).rstrip() + "\n"
    out_path = ROOT / "statistical_ablation_tests.md"
    out_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
