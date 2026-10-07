"""Validate the fresh projection-policy pairs and compute the registered contrast."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from compare_multiseed_dev_v3 import analyze_pairs as _frozen_analyze_pairs
from compare_multiseed_dev_v3 import validate_pairs as _frozen_validate_pairs
from frozen_development_protocol import require_equal
from research_provenance import file_record


TRAINING_SEEDS = (2102, 2103, 2104)
ARMS = ("projected", "rowsoftmax")
METRICS = ("auc_pr_trapezoidal", "mcc", "auroc")
BOOTSTRAP_METRICS = ("auc_pr_trapezoidal", "mcc")
ATTEMPTS = 5000
BOOTSTRAP_SEED = 2101
PRIMARY_ESTIMAND = (
    "Arithmetic mean of three per-training-seed complete pooled-OOF "
    "projected-minus-rowsoftmax trapezoidal PR-AUC differences."
)


def _math_pairs(pairs):
    """Adapt only prediction data after checking the actual projection arm metadata.

    The immutable arithmetic helper names its first and second arms ``ot`` and
    ``control``. Those private slots express subtraction order here; no metadata
    or transport flag is changed, and those names never appear in this report.
    """
    if not isinstance(pairs, (list, tuple)) or len(pairs) != 3:
        raise ValueError("Exactly three complete projection seed pairs are required")
    adapted = []
    for pair, seed in zip(pairs, TRAINING_SEEDS):
        if type(pair.get("training_seed")) is not int or pair["training_seed"] != seed:
            raise ValueError("Projection analysis requires ordered training seeds 2102, 2103, 2104")
        if any(key in pair for key in ("ot", "control", "noot")) or any(arm not in pair for arm in ARMS):
            raise ValueError("Every seed requires the projected and rowsoftmax arms")
        slots = {}
        for arm, slot, iterations in (("projected", "ot", 5), ("rowsoftmax", "control", 0)):
            record = pair[arm]
            if not isinstance(record, dict):
                raise ValueError("Projection arm record must be a dictionary")
            run = record.get("run")
            if run is not None:
                if (not isinstance(run, dict) or type(run.get("training_seed")) is not int
                        or run["training_seed"] != seed or type(run.get("fold_seed")) is not int
                        or run["fold_seed"] != 2101 or run.get("partner_transport") is not True
                        or type(run.get("transport_sinkhorn_iters")) is not int
                        or run["transport_sinkhorn_iters"] != iterations):
                    raise ValueError("Actual run metadata disagrees with the projection seed/arm")
                if "projection_arm" in run and run["projection_arm"] != arm:
                    raise ValueError("Projection arm identity disagrees with its paired slot")
            # Do not pass run metadata to the historical arithmetic helper: it
            # intentionally validates a different architecture contrast.
            slots[slot] = {key: record[key] for key in ("labels", "probs", "threshold", "identity")}
        adapted.append({"training_seed": seed, **slots})
    return adapted


def validate_projection_pairs(pairs):
    """Require six aligned, finite complete prediction populations before analysis."""
    return _frozen_validate_pairs(_math_pairs(pairs))


def analyze_projection_pairs(pairs, attempts=ATTEMPTS, bootstrap_seed=BOOTSTRAP_SEED):
    """Reuse frozen arithmetic; overrides exist only for small synthetic tests."""
    adapted = _math_pairs(pairs)
    result = _frozen_analyze_pairs(adapted, attempts=attempts, bootstrap_seed=bootstrap_seed)
    result["primary_estimand"] = PRIMARY_ESTIMAND
    result["contrast"] = "projected (five iterations) minus rowsoftmax (zero iterations)"
    result["both_arms_partner_transport_enabled"] = True
    result["historical_comparisons_in_primary_or_bootstrap"] = False
    for record in result["per_seed"]:
        scores = record["metrics"]
        record["metrics"] = {"projected": scores["ot"], "rowsoftmax": scores["control"]}
        record["projected_minus_rowsoftmax"] = record.pop("ot_minus_control")
    result["paired_complex_bootstrap"]["statistic"] = (
        "Compute each seed projected-minus-rowsoftmax metric difference on the "
        "common sampled complexes, then average those three differences."
    )
    return result


def validate_registered_analysis(protocol):
    """Refuse any departure from the frozen production RNG and metric policy."""
    require_equal(protocol["training_seeds"], list(TRAINING_SEEDS), "registered analysis training seeds")
    require_equal(protocol["fixed_fold_seed"], 2101, "registered analysis fixed fold seed")
    analysis = protocol["analysis"]
    require_equal(analysis["primary_estimand"], PRIMARY_ESTIMAND, "registered primary estimand")
    require_equal(analysis["delta_direction"], "projected (five iterations) minus rowsoftmax (zero iterations)",
                  "registered subtraction direction")
    bootstrap = analysis["bootstrap"]
    expected = {
        "replicates": ATTEMPTS, "seed": BOOTSTRAP_SEED, "unit": "complex_code",
        "replicates_are_attempts": True, "replenish_excluded_draws": False,
        "complex_draws_per_attempt": 209,
        "rng": "numpy.random.Generator(numpy.random.PCG64(2101))",
        "group_order": "sorted unique complex codes after stripping whitespace and uppercasing",
        "draw_operation": "rng.integers(209, size=209)",
        "paired_draws_shared_across_both_arms_and_all_training_seeds": True,
        "thresholds": "hold every fitted arm/seed saved threshold fixed",
        "single_class_draws": "exclude and count", "interval": "percentile 95%",
        "quantile_method": "numpy.quantile(values, [0.025, 0.975], method='linear')",
        "interval_metrics": ["trapezoidal PR-AUC mean difference", "MCC mean difference"],
    }
    for key, value in expected.items():
        require_equal(bootstrap[key], value, "registered bootstrap " + key)
    require_equal(analysis["p_value"], None, "no bootstrap-tail p-value")
    require_equal(analysis["seed_hypothesis_test"], None, "no seed hypothesis test")


def render_markdown(result):
    analysis = result["analysis"]
    lines = ["# Five-versus-zero projection development comparison", "",
             "Fresh training seeds 2102, 2103 and 2104 use fixed grouped folds from seed 2101. "
             "Both arms retain the learned transport architecture. Test287 was not evaluated.", "",
             "| Seed | Arm | Trapezoidal PR-AUC | MCC | AUROC | Saved threshold |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for pair in analysis["per_seed"]:
        for arm in ARMS:
            score = pair["metrics"][arm]
            lines.append(f"| {pair['training_seed']} | {arm} | {score['auc_pr_trapezoidal']:.6f} | "
                         f"{score['mcc']:.6f} | {score['auroc']:.6f} | {score['threshold']:.6f} |")
    lines += ["", "| Seed | Projected − rowsoftmax PR-AUC | Projected − rowsoftmax MCC | "
              "Projected − rowsoftmax AUROC |", "| --- | ---: | ---: | ---: |"]
    for pair in analysis["per_seed"]:
        delta = pair["projected_minus_rowsoftmax"]
        lines.append(f"| {pair['training_seed']} | {delta['auc_pr_trapezoidal']:+.6f} | "
                     f"{delta['mcc']:+.6f} | {delta['auroc']:+.6f} |")
    lines += ["", "| Difference | Mean | Sample SD (ddof=1) | Range | Positive / zero / negative |",
              "| --- | ---: | ---: | --- | --- |"]
    for key in METRICS:
        summary = analysis["seed_difference_summary"][key]
        signs = summary["sign_counts"]
        lines.append(f"| {key} | {summary['mean']:+.6f} | {summary['sample_standard_deviation']:.6f} | "
                     f"[{summary['range'][0]:+.6f}, {summary['range'][1]:+.6f}] | "
                     f"{signs['positive']} / {signs['zero']} / {signs['negative']} |")
    bootstrap = analysis["paired_complex_bootstrap"]
    lines += ["", f"The {bootstrap['requested_attempts']:,} shared whole-complex attempts produced "
              f"{bootstrap['valid_draws']:,} valid draws and excluded "
              f"{bootstrap['single_class_draws_excluded']:,} single-class draws without replenishment. "
              "Conditional paired 95% percentile intervals for the mean difference:", ""]
    for key in BOOTSTRAP_METRICS:
        low, high = bootstrap["paired_percentile_95_ci"][key]
        lines.append(f"- {key}: [{low:+.6f}, {high:+.6f}]")
    lines += ["", "All five folds passed parameter-schema, trainability, count and optimizer-group matching. "
              "First-fold initial tensors matched within each seed pair. The saved initialization evidence "
              "reports each later fold's tensor and RNG equality or divergence; later equality is not assumed.", "",
              "Previously observed seed-2101 and three-seed architecture comparisons are separate descriptive "
              "context, linked by recorded artifact hashes. Their predictions, checkpoints, scores and thresholds "
              "do not enter this prospective comparison.", "",
              "The contrast covers the complete existing projection/postprocessing policy, including dustbin "
              "mass and gradient effects. It does not isolate column capping alone.", ""]
    lines.extend("- " + limitation for limitation in result["limitations"])
    return "\n".join(lines) + "\n"


def compare(manifest_path, validate_only=False):
    """Read-only validation or one complete preregistered aggregate; never overwrite."""
    from projection_protocol_v1 import verify_manifest, require_launch_authorization
    from validate_projection_dev_v1 import validate_arm, validate_pair_records

    manifest_path = Path(manifest_path).resolve(strict=True)
    before = file_record(manifest_path)
    manifest, protocol = verify_manifest(manifest_path, require_acceptance=True)
    acceptance = file_record(manifest_path.with_name(manifest_path.stem + ".acceptance.json"))
    validate_registered_analysis(protocol)
    output = Path(manifest["aggregate_output_dir"])
    if not validate_only and output.exists():
        raise ValueError("Projection aggregate output already exists; refusing rerun or overwrite")
    authorization = None
    if not validate_only:
        authorization = require_launch_authorization(manifest_path, manifest=manifest, protocol=protocol)
    require_equal(manifest["jobs"], protocol["seed_and_arm_order"], "registered six-arm sequence")
    pairs, artifacts, matching = [], [], []
    for index, seed in enumerate(TRAINING_SEEDS):
        records = {}
        for offset, arm in enumerate(ARMS):
            job = manifest["jobs"][2 * index + offset]
            if (type(job.get("training_seed")) is not int or job["training_seed"] != seed
                    or job.get("arm") != arm or job.get("partner_transport") is not True
                    or type(job.get("transport_sinkhorn_iters")) is not int
                    or job["transport_sinkhorn_iters"] != (5 if arm == "projected" else 0)):
                raise ValueError("Declared job differs from the registered projection pair")
            record = validate_arm(manifest_path, job)
            records[arm] = record
            artifacts.append({"job": job, "run": record["run"], "artifacts": record["artifacts"],
                              "initializations": record["initializations"]})
        evidence = validate_pair_records(records["projected"], records["rowsoftmax"])
        matching.append({"training_seed": seed, "evidence": evidence})
        pairs.append({"training_seed": seed, **records})
    labels, identity, complexes, _ = validate_projection_pairs(pairs)
    if (labels.size != 66208 or complexes.size != 209 or np.unique(identity["sample_index"]).size != 334
            or not np.array_equal(np.unique(identity["fold_index"]), np.arange(5))):
        raise ValueError("Projection population must contain all 334 samples, 209 complexes and 66,208 residues")
    verify_manifest(manifest_path, require_acceptance=True)
    require_equal(file_record(manifest_path), before, "projection analysis manifest unchanged")
    if validate_only:
        return {"status": "validated_complete_six_runs", "manifest": before,
                "training_seeds": list(TRAINING_SEEDS), "bootstrap_run": False,
                "outputs_written": False, "external_evaluation": "skipped", "artifacts": artifacts,
                "initialization_matching": matching}
    analysis = analyze_projection_pairs(pairs, attempts=ATTEMPTS, bootstrap_seed=BOOTSTRAP_SEED)
    observed = protocol["observed_references"]
    result = {
        "schema_version": 1, "analysis_version": "capacity_projection_dev_v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "filtered-cohort, fixed-fold projection-policy development comparison; conditional complex bootstrap",
        "contrast_scope": protocol["contrast_scope"], "external_evaluation": "skipped",
        "execution_manifest": before, "protocol": manifest["protocol"],
        "acceptance": acceptance,
        "launch_authorization": authorization,
        "accepted_ancestry": protocol["accepted_ancestry"], "source_files": manifest["source_files"],
        "runtime": manifest["runtime"], "dataset_selection": manifest["dataset_selection"],
        "cv_splits": manifest["cv_splits"], "artifacts": artifacts,
        "initialization_matching": matching, "analysis": analysis,
        "historical_descriptive_context": {
            "included_in_primary_estimate_or_bootstrap": False,
            "prior_observation_status": protocol["prior_observation_status"],
            "source_records": {key: value for key, value in observed.items()
                               if isinstance(value, dict) and {"path", "sha256", "size_bytes"} <= value.keys()},
            "reporting_rule": protocol["analysis"]["historical_reference_reporting"],
        },
        "limitations": protocol["limitations"],
    }
    # Full artifact validation repeats after analysis: changed bytes cannot be
    # silently published under hashes collected before the resampling work.
    verify_manifest(manifest_path, require_acceptance=True)
    require_equal(require_launch_authorization(manifest_path, manifest=manifest, protocol=protocol),
                  authorization, "launch authorization changed during projection analysis")
    require_equal(file_record(manifest_path), before, "projection manifest unchanged after bootstrap")
    require_equal(file_record(acceptance["path"]), acceptance, "projection acceptance unchanged after bootstrap")
    again_records = []
    for prior in artifacts:
        again = validate_arm(manifest_path, prior["job"])
        require_equal(again["run"], prior["run"], "run changed during projection analysis")
        require_equal(again["artifacts"], prior["artifacts"], "artifacts changed during projection analysis")
        require_equal(again["initializations"], prior["initializations"], "initializations changed during analysis")
        again_records.append(again)
    for index, prior in enumerate(matching):
        again = validate_pair_records(again_records[2 * index], again_records[2 * index + 1])
        require_equal(again, prior["evidence"], "parameter/initialization pairing changed during analysis")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "comparison.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with (output / "RESULT.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(render_markdown(result))
    return {"status": "complete", "manifest": before, "comparison": file_record(output / "comparison.json"),
            "result_markdown": file_record(output / "RESULT.md"), "training_seeds": list(TRAINING_SEEDS)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = compare(args.manifest, validate_only=args.validate_only)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("artifacts", "initialization_matching")}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
