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
    if not ensemble_rows:
        return None
    if len(ensemble_rows) != 1:
        raise ValueError(f"{test_set}: expected exactly one ensemble_mean row")
    # A mean of fold metrics is not the metric of the mean predictions.
    # Requiring the intended row prevents mixing these estimands across runs.
    value = float(ensemble_rows[0]["auc_pr"])
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{test_set}: invalid ensemble AUPRC {value}")
    return value


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
                raise RuntimeError(f"{path} does not contain ensemble_mean for {test_set}")
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
        if len(matches) != 1:
            raise RuntimeError(f"{path} must contain exactly one multiseed_run_stack row for {test_set}")
        value = float(matches[0]["auc_pr"])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{path}: invalid AUPRC for {test_set}: {value}")
        out[test_set] = value
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
    """Enumerate sign assignments conditional on nonzero absolute ranks.

    This handles tied ranks exactly under independent, symmetric differences.
    Values within 1e-12 of zero are treated as numerical zeros.
    """
    differences = finite_differences(differences)
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
    diffs = finite_differences(differences)
    indices = rng.integers(0, diffs.size, size=(BOOTSTRAP_SAMPLES, diffs.size))
    boot_means = diffs[indices].mean(axis=1)
    lower, upper = np.percentile(boot_means, [2.5, 97.5])
    return float(lower), float(upper)


def finite_differences(differences):
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("differences must be a finite nonempty vector")
    return values


def holm_adjust(p_values):
    """Holm family-wise adjustment, returned in the original order."""
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be finite and in [0, 1]")
    order = np.argsort(values)
    adjusted = np.empty(values.size)
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, (values.size - position) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def paired_test(full_by_test, ablation_by_test):
    rows = []
    differences = []
    for test_set in TEST_SETS:
        full = float(full_by_test[test_set])
        ablation = float(ablation_by_test[test_set])
        if not np.isfinite([full, ablation]).all() or not (0 <= full <= 1 and 0 <= ablation <= 1):
            raise ValueError(f"{test_set}: AUPRC values must be finite and in [0, 1]")
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
    analyses = [(label, *paired_test(full, values)) for label, values in ablations.items()]
    adjusted_p_values = holm_adjust([summary["wilcoxon_p"] for _, _, summary in analyses])
    for (label, rows, summary), adjusted_p in zip(analyses, adjusted_p_values):
        lines.append(f"### Full vs {label}")
        lines.append("")
        lines.append("| Test set | Full AUPRC | Ablation AUPRC | Difference |")
        lines.append("|---|---:|---:|---:|")
        for test_set, full_auc, ablation_auc, diff in rows:
            lines.append(f"| {test_set} | {fmt(full_auc)} | {fmt(ablation_auc)} | {fmt(diff)} |")
        lines.append("")
        lines.append(
            "Mean paired difference = {mean}; 95% paired benchmark-bootstrap CI = [{low}, {high}]; "
            "two-sided exact Wilcoxon signed-rank p = {p}; Holm-adjusted p = {adjusted} "
            "within this analysis family (n={n} nonzero benchmark differences).".format(
                mean=fmt(summary["mean_diff"]),
                low=fmt(summary["ci_low"]),
                high=fmt(summary["ci_high"]),
                p=f"{summary['wilcoxon_p']:.4f}",
                adjusted=f"{adjusted_p:.4f}",
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
        "AUPRC is trapezoidal precision-recall area. Differences are paired by benchmark dataset after averaging the ensemble_mean metric across training runs in each condition. Test70 is excluded because of reported Train335 overlap; this script does not perform a homology audit. Full-model values use seeds 2036-2039, while ablations use different seeds and three rather than four runs. These groups are intended to share a training protocol, but use different seed sets; the script does not independently validate run configurations or estimate training-seed uncertainty.",
        "",
        "The bootstrap resamples four benchmark differences. Its percentile interval assumes independent, exchangeable benchmarks and is exploratory, especially for related datasets such as bound/unbound variants. The exact signed-rank test additionally assumes symmetry under its null. With four nonzero differences its smallest possible two-sided p-value is 0.125; no result can reach 0.05. Reported confidence intervals are unadjusted; p-values receive Holm adjustment separately within the primary and secondary comparison families.",
        "",
        "The secondary run-stack comparisons are exploratory because the full stack contains seven heterogeneous seeds while each ablation stack contains three seeds. Those comparisons confound model component, seed composition, and ensemble size.",
        "",
    ]
    add_analysis(
        lines,
        "Across-Run Means, Paired by Benchmark",
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
