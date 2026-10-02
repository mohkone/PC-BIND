"""Validate six completed fixed-fold runs and perform the registered seed analysis."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import matthews_corrcoef

from compare_oof_development import score_metrics
from compare_pcbind_primary import validate_predictions
from filtered_cohort_protocol import read_json_with_record
from frozen_development_protocol import require_equal
from metrics import compute_auc_pr
from research_provenance import file_record


TRAINING_SEEDS = (2102, 2103, 2104)
IDENTITY_FIELDS = ("sample_index", "residue_index", "complex_id", "fold_index")
METRICS = ("auc_pr_trapezoidal", "mcc", "auroc")
BOOTSTRAP_METRICS = ("auc_pr_trapezoidal", "mcc")
ATTEMPTS = 5000
BOOTSTRAP_SEED = 2101


def validate_pairs(pairs):
    """Require three ordered complete pairs with the same OOF row population."""
    if not isinstance(pairs, (list, tuple)) or len(pairs) != 3:
        raise ValueError("Exactly three complete new-seed pairs are required")
    reference_labels, reference_identity = None, None
    for pair, expected_seed in zip(pairs, TRAINING_SEEDS):
        if type(pair.get("training_seed")) is not int or pair["training_seed"] != expected_seed:
            raise ValueError("Only ordered training seeds 2102, 2103, 2104 enter the primary analysis")
        for arm in ("ot", "control"):
            if arm not in pair:
                raise ValueError("Both arms are required for every training seed")
            record = pair[arm]
            labels = np.asarray(record["labels"])
            probs = np.asarray(record["probs"], dtype=np.float64)
            validate_predictions(labels, probs)
            if np.unique(labels).size != 2:
                raise ValueError("The complete OOF population must contain both classes")
            threshold = np.asarray(record["threshold"])
            if (threshold.ndim != 0 or threshold.dtype.kind not in "fiu"
                    or not np.isfinite(threshold) or not 0 <= threshold <= 1):
                raise ValueError("Every saved threshold must be a finite scalar in [0, 1]")
            identity = record.get("identity")
            if not isinstance(identity, dict) or any(key not in identity for key in IDENTITY_FIELDS):
                raise ValueError("Complete sample/residue/complex/fold identity is required")
            for key in IDENTITY_FIELDS:
                values = np.asarray(identity[key])
                if values.shape != labels.shape:
                    raise ValueError("OOF identity is not row-aligned: " + key)
                if key != "complex_id" and (values.dtype.kind not in "iu" or np.any(values < 0)):
                    raise ValueError("OOF indices must be nonnegative integers: " + key)
                if key == "complex_id" and values.dtype.kind not in "US":
                    raise ValueError("Complex identity must contain strings")
            if reference_labels is None:
                reference_labels = labels
                reference_identity = {key: np.asarray(identity[key]) for key in IDENTITY_FIELDS}
            else:
                if not np.array_equal(labels, reference_labels):
                    raise ValueError("Labels/order/coverage differ across the six runs")
                for key in IDENTITY_FIELDS:
                    if not np.array_equal(identity[key], reference_identity[key]):
                        raise ValueError("OOF identity/order/coverage differs across runs: " + key)
            run = record.get("run")
            if run is not None:
                if (run.get("training_seed") != expected_seed or run.get("fold_seed") != 2101
                        or run.get("partner_transport") is not (arm == "ot")):
                    raise ValueError("Run metadata disagrees with its seed/arm pairing")
    identities = np.column_stack([reference_identity[k] for k in ("sample_index", "residue_index")])
    if np.unique(identities, axis=0).shape[0] != reference_labels.size:
        raise ValueError("Each source sample/residue may appear only once in the OOF population")
    groups = np.char.upper(np.char.strip(reference_identity["complex_id"].astype(str)))
    if np.any(np.isin(groups, ["", "NONE", "NAN", "NULL"])):
        raise ValueError("Missing complex identities are validation failures")
    unique = np.unique(groups)
    if unique.size < 2:
        raise ValueError("Whole-complex resampling requires at least two complexes")
    return reference_labels.astype(np.int64), reference_identity, unique, [
        np.flatnonzero(groups == group) for group in unique
    ]


def _bootstrap_scores(labels, probs, threshold):
    # Same PR/MCC definitions as score_metrics; AUROC is descriptive only and
    # does not need to be sorted/recomputed inside every bootstrap draw.
    return {"auc_pr_trapezoidal": float(compute_auc_pr(labels, probs)),
            "mcc": float(matthews_corrcoef(labels, probs >= threshold))}


def analyze_pairs(pairs, attempts=ATTEMPTS, bootstrap_seed=BOOTSTRAP_SEED):
    """Analyze validated aligned pairs; parameter overrides are for synthetic tests."""
    if type(attempts) is not int or attempts < 1 or type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError("Bootstrap requires positive integer attempts and a nonnegative integer seed")
    labels, identity, unique, rows = validate_pairs(pairs)
    per_seed = []
    for pair in pairs:
        scores = {arm: score_metrics(labels, np.asarray(pair[arm]["probs"], dtype=np.float64),
                                    float(pair[arm]["threshold"])) for arm in ("ot", "control")}
        if any(not np.isfinite(value) for arm in scores.values() for value in arm.values()):
            raise ValueError("Nonfinite complete-population metrics")
        per_seed.append({"training_seed": pair["training_seed"], "metrics": scores,
                         "ot_minus_control": {key: scores["ot"][key] - scores["control"][key]
                                              for key in METRICS}})
    summaries = {}
    for key in METRICS:
        values = np.asarray([item["ot_minus_control"][key] for item in per_seed], dtype=np.float64)
        summaries[key] = {"mean": float(values.mean()), "sample_standard_deviation": float(values.std(ddof=1)),
                          "standard_deviation_ddof": 1, "range": [float(values.min()), float(values.max())],
                          "sign_counts": {"positive": int(np.count_nonzero(values > 0)),
                                          "zero": int(np.count_nonzero(values == 0)),
                                          "negative": int(np.count_nonzero(values < 0))}}
    rng = np.random.Generator(np.random.PCG64(bootstrap_seed))
    draws = {key: [] for key in BOOTSTRAP_METRICS}
    excluded = 0
    for _ in range(attempts):
        indices = np.concatenate([rows[index] for index in rng.integers(len(rows), size=len(rows))])
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size != 2:
            excluded += 1
            continue
        seed_deltas = {key: [] for key in BOOTSTRAP_METRICS}
        for pair in pairs:
            scores = {arm: _bootstrap_scores(sampled_labels, np.asarray(pair[arm]["probs"])[indices],
                                             float(pair[arm]["threshold"])) for arm in ("ot", "control")}
            for key in BOOTSTRAP_METRICS:
                value = scores["ot"][key] - scores["control"][key]
                if not np.isfinite(value):
                    raise ValueError("Nonfinite bootstrap metric; this is not an excluded draw")
                seed_deltas[key].append(value)
        for key in BOOTSTRAP_METRICS:
            draws[key].append(float(np.mean(seed_deltas[key])))
    valid = attempts - excluded
    if valid == 0:
        raise ValueError("No bootstrap draw contains both classes; refusing an interval")
    return {"primary_training_seeds": list(TRAINING_SEEDS), "reference_seed2101_in_primary": False,
            "primary_estimand": "mean of three per-training-seed pooled-OOF OT-minus-control trapezoidal PR-AUC differences",
            "sample_count": int(np.unique(identity["sample_index"]).size), "residue_count": int(labels.size),
            "complex_count": int(unique.size), "fold_indices": np.unique(identity["fold_index"]).tolist(),
            "per_seed": per_seed, "seed_difference_summary": summaries,
            "paired_complex_bootstrap": {
                "resampling_unit": "complex_code", "complex_count": int(unique.size),
                "sorted_normalized_complex_ids": unique.tolist(), "draws_per_attempt": int(unique.size),
                "requested_attempts": attempts, "valid_draws": valid, "single_class_draws_excluded": excluded,
                "replenish_excluded_draws": False, "seed": bootstrap_seed,
                "rng": "numpy.random.Generator(numpy.random.PCG64)",
                "draw_operation": f"rng.integers({len(rows)}, size={len(rows)})",
                "same_draw_for_all_six_runs": True, "whole_complex_rows_and_multiplicity_preserved": True,
                "all_six_thresholds_held_fixed": True,
                "statistic": "arithmetic mean of three per-seed metric differences on the same draw",
                "quantile_method": "linear", "paired_percentile_95_ci": {
                    key: np.quantile(values, [0.025, 0.975], method="linear").tolist()
                    for key, values in draws.items()},
                "scope": "conditional on fitted predictions and selected thresholds; excludes training-seed uncertainty"}}


def render_markdown(result):
    analysis = result["analysis"]
    lines = ["# Three-seed fixed-fold development comparison", "",
             "Primary analysis includes training seeds 2102, 2103 and 2104. Fixed fold seed: 2101. "
             "Test287 was not evaluated.", "",
             "| Seed | Arm | Trapezoidal PR-AUC | MCC | AUROC | Saved threshold |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for pair in analysis["per_seed"]:
        for arm in ("ot", "control"):
            score = pair["metrics"][arm]
            lines.append(f"| {pair['training_seed']} | {'OT' if arm == 'ot' else 'no-OT'} | "
                         f"{score['auc_pr_trapezoidal']:.6f} | {score['mcc']:.6f} | "
                         f"{score['auroc']:.6f} | {score['threshold']:.6f} |")
    lines += ["", "| Seed | OT − no-OT PR-AUC | OT − no-OT MCC | OT − no-OT AUROC |",
              "| --- | ---: | ---: | ---: |"]
    for pair in analysis["per_seed"]:
        delta = pair["ot_minus_control"]
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
    lines += ["", f"The {bootstrap['requested_attempts']:,} shared whole-complex bootstrap attempts produced "
              f"{bootstrap['valid_draws']:,} valid draws; {bootstrap['single_class_draws_excluded']:,} "
              "single-class draws were excluded without replacement attempts. Conditional 95% percentile intervals "
              "for the mean paired difference:", ""]
    for key in BOOTSTRAP_METRICS:
        low, high = bootstrap["paired_percentile_95_ci"][key]
        lines.append(f"- {key}: [{low:+.6f}, {high:+.6f}]")
    reference = result["historical_seed2101_reference"]
    delta = reference["ot_minus_control"]
    lines += ["", f"Previously observed seed 2101 (excluded from the primary mean and intervals): "
              f"PR-AUC difference {delta['auc_pr_trapezoidal']:+.6f}; MCC difference {delta['mcc']:+.6f}.", "",
              "These intervals condition on all fitted predictions and saved thresholds; they do not measure "
              "training-seed uncertainty. The three-seed SD and range describe only the observed repetitions.", ""]
    lines.extend("- " + limitation for limitation in result["limitations"])
    return "\n".join(lines) + "\n"


def compare(manifest_path, validate_only=False):
    # Lazy imports keep synthetic analysis tests independent of training/runtime gates.
    from multiseed_infrastructure_v3 import verify_manifest
    from validate_multiseed_dev_v3 import validate_arm

    manifest_path = Path(manifest_path).resolve(strict=True)
    before = file_record(manifest_path)
    manifest, protocol, v2 = verify_manifest(manifest_path, require_acceptance=not validate_only)
    output = Path(manifest["aggregate_output_dir"])
    if not validate_only and output.exists():
        raise ValueError("Aggregate output already exists; refusing overwrite")
    jobs = manifest["jobs"]
    if len(jobs) != 6:
        raise ValueError("Six declared jobs required")
    pairs, artifacts = [], []
    for index, seed in enumerate(TRAINING_SEEDS):
        records = {}
        for offset, arm in enumerate(("ot", "noot")):
            job = jobs[2 * index + offset]
            if type(job.get("training_seed")) is not int or job["training_seed"] != seed or job.get("arm") != arm:
                raise ValueError("Declared jobs must preserve all three ordered seed pairs")
            record = validate_arm(manifest_path, job)
            records["ot" if arm == "ot" else "control"] = record
            artifacts.append({"job": job, "run": record["run"], "artifacts": record["artifacts"]})
        pairs.append({"training_seed": seed, **records})
    labels, identity, unique, _ = validate_pairs(pairs)
    if (labels.size != 66208 or unique.size != 209 or np.unique(identity["sample_index"]).size != 334
            or not np.array_equal(np.unique(identity["fold_index"]), np.arange(5))):
        raise ValueError("Primary population must cover the original 334 samples, 209 complexes and 66,208 residues")
    verify_manifest(manifest_path, require_acceptance=not validate_only)
    require_equal(file_record(manifest_path), before, "analysis manifest unchanged")
    if validate_only:
        return {"status": "validated_complete_six_runs", "manifest": before, "training_seeds": list(TRAINING_SEEDS),
                "artifact_validation": "passed", "bootstrap_run": False, "outputs_written": False,
                "external_evaluation": "skipped", "artifacts": artifacts}
    analysis = analyze_pairs(pairs, attempts=ATTEMPTS, bootstrap_seed=BOOTSTRAP_SEED)
    reference, reference_record = read_json_with_record(protocol["reference_comparison"]["path"])
    require_equal(reference_record, protocol["reference_comparison"], "historical comparison bytes")
    result = {"schema_version": 1, "analysis_version": "multiseed_dev_v3", "status": "complete",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "filtered-cohort, fixed-fold development comparison; conditional complex bootstrap",
              "external_evaluation": "skipped", "execution_manifest": before,
              "protocol": manifest["protocol"], "accepted_v2": manifest["accepted_v2"],
              "accepted_v2_source_files": v2["source_files"], "reference_source_files": v2["reference_source_files"],
              "v3_source_files": manifest["source_files"], "runtime": manifest["runtime"],
              "dataset_selection": manifest["dataset_selection"], "cv_splits": v2["cv_splits"],
              "artifacts": artifacts, "analysis": analysis,
              "historical_seed2101_reference": {"label": "previously observed development reference",
                  "training_seed": 2101, "included_in_primary_estimate_or_intervals": False,
                  "source": reference_record, "metrics": reference["metrics"],
                  "ot_minus_control": reference["ot_minus_control"]},
              "limitations": [
                  "Checkpoint and threshold selection used development labels; the OOF scores remain selection-optimistic.",
                  "Fold training populations overlap; complex resampling and seed repetition do not remove this fitting dependence.",
                  "Only three new training seeds were observed; their variation is not a precise estimate of the full distribution of training randomness.",
                  "The filtered cohort has unverified biological residue mappings and unestablished sequence-homology independence.",
                  "Removing transport changes parameter count; the control is not capacity matched.",
                  "Fixed folds leave sampling, mapping, homology, architecture and hyperparameter variation unprobed.",
                  "Numeric seed pairing does not synchronize RNG histories across unequal architectures.",
                  "Test287 remains unevaluated; these development results do not establish external generalization or biological partner specificity.",
                  "No seed-level hypothesis test or bootstrap-tail p-value is reported."]}
    # Verify frozen bytes after the complete resampling computation, before publishing.
    verify_manifest(manifest_path, require_acceptance=True)
    require_equal(file_record(manifest_path), before, "analysis manifest unchanged after bootstrap")
    for record in artifacts:
        # Re-validation includes every checkpoint, metric, OOF identity and source binding.
        again = validate_arm(manifest_path, record["job"])
        require_equal(again["run"], record["run"], "run changed during analysis")
        require_equal(again["artifacts"], record["artifacts"], "artifacts changed during analysis")
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
    result = compare(args.manifest, args.validate_only)
    # Validation artifacts stay in the returned Python record; stdout is status only.
    print(json.dumps({key: value for key, value in result.items() if key != "artifacts"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
