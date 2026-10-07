"""Read-only completion audit; never fits models or reruns bootstrap draws."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent
EXPECTED_V3 = "c181e3206996c5674ed33e7d0c0e91a83d42a3cc870a839c1b4819a470ddc73a"
EXPECTED_V2 = "ca69112d633de00ee80d23a7c7913a939d780cf2561f403393588f95a5eb0519"
TOLERANCE = 1e-12


def record(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), str(path)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": after.st_size}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def collect_records(value, collection):
    if isinstance(value, dict):
        if all(key in value for key in ("path", "sha256", "size_bytes")):
            item = {key: value[key] for key in ("path", "sha256", "size_bytes")}
            key = str(Path(item["path"]).resolve(strict=True))
            if key in collection:
                assert collection[key] == item, "Inconsistent recorded hash: " + key
            collection[key] = item
        for child in value.values():
            collect_records(child, collection)
    elif isinstance(value, list):
        for child in value:
            collect_records(child, collection)


def close(actual, expected, label):
    import numpy as np
    assert np.isfinite(actual) and np.isfinite(expected), label
    assert abs(float(actual) - float(expected)) <= TOLERANCE, (label, actual, expected)


def audit(report):
    # Queue state is the first experiment input read, before any metrics.
    queue_path = ROOT / "outputs_multiseed_dev_v3_process/queue_state.json"
    queue = read(queue_path)
    queue_record = record(queue_path)
    assert queue["status"] == "complete"
    assert queue["comparison"]["status"] == "complete"
    assert len(queue["jobs"]) == 6
    assert queue["external_evaluation"] == "skipped in every arm"
    order = [(seed, arm) for seed in (2102, 2103, 2104) for arm in ("ot", "noot")]
    for job, expected in zip(queue["jobs"], order):
        assert (job["training_seed"], job["arm"]) == expected
        assert job["status"] == "complete" and job["exit_code"] == 0
        assert record(job["validation"]["path"]) == job["validation"]
        evidence = read(job["validation"]["path"])
        assert evidence["status"] == "complete" and evidence["external_evaluation"] == "skipped"
        assert len(evidence["artifacts"]["checkpoints"]) == 5
    assert queue["execution_manifest_v3"]["sha256"] == EXPECTED_V3
    assert queue["accepted_v2"]["sha256"] == EXPECTED_V2
    assert record(queue["execution_manifest_v3"]["path"]) == queue["execution_manifest_v3"]
    assert record(queue["accepted_v2"]["path"]) == queue["accepted_v2"]
    for artifact in queue["comparison"]["artifacts"].values():
        assert record(artifact["path"]) == artifact
    manifest = read(queue["execution_manifest_v3"]["path"])
    final = read(queue["comparison"]["artifacts"]["comparison"]["path"])
    assert final["status"] == "complete" and final["external_evaluation"] == "skipped"
    assert final["execution_manifest"] == queue["execution_manifest_v3"]
    assert final["accepted_v2"] == queue["accepted_v2"]
    bound_records = {str(queue_path.resolve()): queue_record}
    for value in (queue, manifest, final):
        collect_records(value, bound_records)
    for expected in bound_records.values():
        assert record(expected["path"]) == expected, "Recorded artifact mismatch: " + expected["path"]
    report.update(queue=queue_record, execution_manifest_v3=queue["execution_manifest_v3"],
                  accepted_v2=queue["accepted_v2"], final_artifacts=queue["comparison"]["artifacts"])

    command = [str(ROOT / ".venv/Scripts/python.exe"), "-B",
               str(ROOT / "compare_multiseed_dev_v3.py"), "--manifest",
               queue["execution_manifest_v3"]["path"], "--validate-only"]
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    stdout_path = EVIDENCE / "validate_only_stdout.txt"
    stderr_path = EVIDENCE / "validate_only_stderr.txt"
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, check=False)
    report["validate_only"] = {"command": command, "exit_code": completed.returncode,
                             "stdout": record(stdout_path), "stderr": record(stderr_path)}
    assert completed.returncode == 0, "Frozen validate-only CLI failed; inspect recorded logs"
    validation = read(stdout_path)
    assert validation["status"] == "validated_complete_six_runs"
    assert validation["artifact_validation"] == "passed"
    assert validation["training_seeds"] == [2102, 2103, 2104]
    assert validation["bootstrap_run"] is False and validation["outputs_written"] is False
    assert validation["external_evaluation"] == "skipped"

    # Independent whole-population metrics use public sklearn definitions,
    # without importing or calling the frozen bootstrap analyzer.
    import numpy as np
    from sklearn.metrics import auc, matthews_corrcoef, precision_recall_curve, roc_auc_score
    analysis = final["analysis"]
    assert analysis["primary_training_seeds"] == [2102, 2103, 2104]
    assert analysis["reference_seed2101_in_primary"] is False
    assert final["historical_seed2101_reference"]["included_in_primary_estimate_or_intervals"] is False
    assert final["historical_seed2101_reference"]["training_seed"] == 2101
    assert (analysis["sample_count"], analysis["complex_count"], analysis["residue_count"]) == (334, 209, 66208)
    assert analysis["fold_indices"] == [0, 1, 2, 3, 4]
    first_labels, first_identity = None, None
    recomputed = []
    checkpoint_count = 0
    for index, seed in enumerate((2102, 2103, 2104)):
        saved_pair = analysis["per_seed"][index]
        assert saved_pair["training_seed"] == seed
        scores = {}
        for offset, role in enumerate(("ot", "control")):
            artifact = final["artifacts"][2 * index + offset]
            assert artifact["job"] == manifest["jobs"][2 * index + offset]
            assert artifact["run"]["training_seed"] == seed and artifact["run"]["fold_seed"] == 2101
            assert artifact["run"]["completed_fold_indices"] == [0, 1, 2, 3, 4]
            assert artifact["run"]["skip_test_eval"] is True
            assert artifact["run"]["partner_transport"] is (role == "ot")
            assert artifact["run"]["runtime"] == manifest["runtime"]
            checkpoint_count += len(artifact["artifacts"]["checkpoints"])
            with np.load(artifact["artifacts"]["oof_predictions"]["path"], allow_pickle=False) as archive:
                labels = archive["labels"].astype(np.int64)
                probabilities = archive["probs"].astype(np.float64)
                threshold = float(archive["threshold"].item())
                identity = {key: archive[key] for key in ("sample_index", "residue_index", "complex_id", "fold_index")}
            assert labels.shape == probabilities.shape == (66208,)
            assert np.isfinite(probabilities).all() and np.isin(labels, [0, 1]).all()
            assert np.all((probabilities >= 0) & (probabilities <= 1))
            if first_labels is None:
                first_labels, first_identity = labels, identity
            else:
                assert np.array_equal(labels, first_labels)
                for key in first_identity:
                    assert np.array_equal(identity[key], first_identity[key]), key
            precision, recall, _ = precision_recall_curve(labels, probabilities)
            scores[role] = {"auc_pr_trapezoidal": float(auc(recall, precision)),
                            "mcc": float(matthews_corrcoef(labels, probabilities >= threshold)),
                            "auroc": float(roc_auc_score(labels, probabilities)), "threshold": threshold}
            for metric, value in scores[role].items():
                close(value, saved_pair["metrics"][role][metric], f"{seed}/{role}/{metric}")
        deltas = {metric: scores["ot"][metric] - scores["control"][metric]
                  for metric in ("auc_pr_trapezoidal", "mcc", "auroc")}
        for metric, value in deltas.items():
            close(value, saved_pair["ot_minus_control"][metric], f"{seed}/delta/{metric}")
        recomputed.append({"training_seed": seed, "metrics": scores, "ot_minus_control": deltas})
    assert checkpoint_count == 30
    assert np.unique(first_identity["sample_index"]).size == 334
    assert np.unique(np.column_stack((first_identity["sample_index"], first_identity["residue_index"])), axis=0).shape[0] == 66208
    summaries = {}
    for metric in ("auc_pr_trapezoidal", "mcc", "auroc"):
        values = np.asarray([pair["ot_minus_control"][metric] for pair in recomputed])
        summaries[metric] = {"mean": float(values.mean()), "sample_standard_deviation": float(values.std(ddof=1)),
                             "standard_deviation_ddof": 1, "range": [float(values.min()), float(values.max())],
                             "sign_counts": {"positive": int(np.count_nonzero(values > 0)), "zero": int(np.count_nonzero(values == 0)),
                                             "negative": int(np.count_nonzero(values < 0))}}
        expected = analysis["seed_difference_summary"][metric]
        for field in ("mean", "sample_standard_deviation"):
            close(summaries[metric][field], expected[field], metric + "/" + field)
        for value, saved in zip(summaries[metric]["range"], expected["range"]):
            close(value, saved, metric + "/range")
        assert summaries[metric]["standard_deviation_ddof"] == expected["standard_deviation_ddof"]
        assert summaries[metric]["sign_counts"] == expected["sign_counts"]
    bootstrap = analysis["paired_complex_bootstrap"]
    required = {"requested_attempts": 5000, "valid_draws": 5000, "single_class_draws_excluded": 0,
                "seed": 2101, "rng": "numpy.random.Generator(numpy.random.PCG64)",
                "draw_operation": "rng.integers(209, size=209)", "complex_count": 209, "draws_per_attempt": 209,
                "resampling_unit": "complex_code", "same_draw_for_all_six_runs": True,
                "whole_complex_rows_and_multiplicity_preserved": True, "all_six_thresholds_held_fixed": True,
                "replenish_excluded_draws": False, "quantile_method": "linear"}
    for key, value in required.items():
        assert bootstrap[key] == value, key
    groups = np.unique(np.char.upper(np.char.strip(first_identity["complex_id"].astype(str)))).tolist()
    assert len(groups) == 209 and bootstrap["sorted_normalized_complex_ids"] == groups
    for interval in bootstrap["paired_percentile_95_ci"].values():
        assert len(interval) == 2 and np.isfinite(interval).all() and interval[0] <= interval[1]
    report.update(status="passed", completion_checks={"six_arms_completed_exit_zero": True,
        "recorded_artifact_hashes_verified": len(bound_records), "selected_checkpoints_validated": checkpoint_count,
        "oof_archives_validated": 6, "samples_per_arm": 334, "complexes_per_arm": 209, "residues_per_arm": 66208,
        "fold_indices": [0, 1, 2, 3, 4], "external_evaluation": "skipped", "historical_seed2101_excluded": True},
        recomputed_per_seed_metrics=recomputed, recomputed_seed_difference_summary=summaries,
        recorded_bootstrap=bootstrap, metric_absolute_tolerance=TOLERANCE,
        bound_original_artifacts=list(bound_records.values()))
    for expected in bound_records.values():
        assert record(expected["path"]) == expected, "Original changed during completion audit: " + expected["path"]


def main():
    report = {"schema_version": 1, "audit_version": "completion_verification_v1", "status": "started",
              "created_utc": datetime.now(timezone.utc).isoformat(), "client_date": "2026-10-07",
              "client_timezone": "Asia/Shanghai", "verification_source": record(__file__),
              "bootstrap_recomputed": False, "training": False, "output_originals_modified": False}
    destination = EVIDENCE / "verification.json"
    assert not destination.exists(), "Refusing evidence overwrite"
    try:
        audit(report)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["ended_utc"] = datetime.now(timezone.utc).isoformat()
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": report["status"], "verification": record(destination)}, indent=2))


if __name__ == "__main__":
    main()
