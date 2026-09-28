import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import matthews_corrcoef

from metrics import compute_auc_pr, compute_auc_roc
from research_provenance import load_dataset_with_provenance, prediction_identity


DEFAULT_TESTS = ("Test60", "Test287", "TestB25", "TestUB25")
IDENTITY_ARRAY_FIELDS = ("sample_index", "residue_index", "complex_id")
IDENTITY_SCALAR_FIELDS = ("dataset_sha256", "dataset_name", "provenance_id", "identity_schema_version")
PROTOCOL_FIELDS = (
    "seed", "model_mode", "grouped_cv", "cv_group_key", "max_folds",
    "geometry_feature_mode", "surface_feature_mode", "sequence_feature_mode",
    "plm_feature_mode", "plm_feature_dim", "aux_plm_feature_mode",
    "aux_plm_feature_dim", "batch_size", "model_dropout", "model_edge_dropout",
    "weight_decay", "checkpoint_selection_metric", "two_head_binding",
    "two_head_rank_loss_weight", "two_head_consistency_weight",
    "two_head_rank_fusion", "two_head_rank_warmup_epochs",
    "partner_contact_aux", "partner_contact_aux_weight", "patch_label_distribution",
    "use_plm_features", "use_aux_plm_features",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare saved prediction arrays with a paired complex bootstrap."
    )
    parser.add_argument("--control-dir", default="outputs_grouped_full_seed2101")
    parser.add_argument("--model-dir", default="outputs_pcbind_v5_shared_seed2101")
    parser.add_argument("--data-dir", default=str(Path("data") / "geo"))
    parser.add_argument("--tests", nargs="+", default=list(DEFAULT_TESTS))
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2101)
    parser.add_argument("--output-json", default="pcbind_primary_comparison.json")
    parser.add_argument("--output-md", default="pcbind_primary_comparison.md")
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    if len(set(args.tests)) != len(args.tests):
        parser.error("--tests must not contain duplicates")
    return args


def load_predictions(run_dir, test_set):
    path = Path(run_dir) / f"ensemble_{test_set}_predictions.npz"
    with np.load(path, allow_pickle=False) as archive:
        labels = np.asarray(archive["labels"])
        probabilities = np.asarray(archive["probs"], dtype=np.float64)
        threshold = np.asarray(archive["threshold"], dtype=np.float64)
        _archive_identity(archive, path)
    validate_predictions(labels, probabilities)
    if threshold.size != 1 or not np.isfinite(threshold).all():
        raise ValueError(f"{path}: threshold must be one finite scalar")
    threshold = float(threshold.reshape(-1)[0])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{path}: threshold must be in [0, 1]")
    return labels.astype(np.int64), probabilities, threshold


def _valid_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _archive_identity(archive, path):
    fields = IDENTITY_ARRAY_FIELDS + IDENTITY_SCALAR_FIELDS
    present = set(archive.files).intersection(fields + ("fold_index",))
    if not present:
        return None
    missing = set(fields).difference(present)
    if missing:
        raise ValueError(f"{path}: partial prediction identity; missing {', '.join(sorted(missing))}")
    identity = {key: np.asarray(archive[key]) for key in fields}
    expected_shape = archive["labels"].shape
    for key in IDENTITY_ARRAY_FIELDS:
        values = identity[key]
        if values.ndim != 1 or values.shape != expected_shape:
            raise ValueError(f"{path}: {key} must align with every prediction")
        if key != "complex_id" and (values.dtype.kind not in "iu" or np.any(values < 0)):
            raise ValueError(f"{path}: {key} must contain nonnegative integer indices")
    if identity["complex_id"].dtype.kind != "U":
        raise ValueError(f"{path}: complex_id must contain unicode strings")
    for key in IDENTITY_SCALAR_FIELDS:
        if identity[key].ndim != 0:
            raise ValueError(f"{path}: {key} must be a scalar")
        if key != "identity_schema_version" and identity[key].dtype.kind != "U":
            raise ValueError(f"{path}: {key} must be a unicode string")
    if identity["identity_schema_version"].dtype.kind not in "iu" or identity["identity_schema_version"].item() != 1:
        raise ValueError(f"{path}: unsupported prediction identity schema")
    for key in IDENTITY_SCALAR_FIELDS:
        identity[key] = identity[key].item()
    if not _valid_sha256(identity["dataset_sha256"]):
        raise ValueError(f"{path}: invalid dataset_sha256")
    if not identity["dataset_name"].strip() or not identity["provenance_id"].strip():
        raise ValueError(f"{path}: dataset_name and provenance_id must be nonempty")
    if "fold_index" in archive.files:
        folds = np.asarray(archive["fold_index"])
        if folds.shape != expected_shape or folds.dtype.kind not in "iu" or np.any(folds < -1):
            raise ValueError(f"{path}: invalid fold_index vector")
        identity["fold_index"] = folds
    return identity


def load_prediction_identity(run_dir, test_set):
    path = Path(run_dir) / f"ensemble_{test_set}_predictions.npz"
    with np.load(path, allow_pickle=False) as archive:
        return _archive_identity(archive, path)


def _validate_manifest(manifest, path):
    required = {"schema_version", "provenance_id", "datasets", "cv_splits",
                "scheduled_fold_indices", "completed_fold_indices"}
    if not isinstance(manifest, dict) or not required.issubset(manifest):
        raise ValueError(f"{path}: incomplete run provenance")
    if manifest["schema_version"] != 1 or not isinstance(manifest["provenance_id"], str) or not manifest["provenance_id"].strip():
        raise ValueError(f"{path}: unsupported or invalid run provenance")
    training = manifest["datasets"].get("Train335", {})
    count = training.get("sample_count")
    if not _valid_sha256(training.get("sha256")) or not isinstance(count, int) or count < 1:
        raise ValueError(f"{path}: missing training dataset hash or sample count")
    splits = manifest["cv_splits"]
    if not isinstance(splits, list) or len(splits) < 2:
        raise ValueError(f"{path}: cv_splits must record the full cross-validation split")
    canonical = {}
    validation_members = []
    for split in splits:
        fold = split.get("fold_index")
        if not isinstance(fold, int) or fold < 0 or fold in canonical:
            raise ValueError(f"{path}: invalid or duplicate fold_index")
        item = {}
        for kind in ("train", "val"):
            indices = split.get(f"{kind}_indices", [])
            complexes = split.get(f"{kind}_complex_ids", [])
            groups = split.get(f"{kind}_group_ids", [])
            if (not indices or any(type(index) is not int or not 0 <= index < count for index in indices)
                    or len(set(indices)) != len(indices)
                    or len(complexes) != len(indices) or len(groups) != len(indices)):
                raise ValueError(f"{path}: invalid {kind} identities in fold {fold}")
            item[kind] = sorted(zip(indices, complexes, groups))
        train_indices = [row[0] for row in item["train"]]
        val_indices = [row[0] for row in item["val"]]
        if sorted(train_indices + val_indices) != list(range(count)):
            raise ValueError(f"{path}: train/validation indices must partition the training dataset")
        if manifest.get("grouped_cv") and ({row[2] for row in item["train"]} & {row[2] for row in item["val"]}):
            raise ValueError(f"{path}: grouped CV has overlapping train/validation groups")
        canonical[fold] = item
        validation_members.extend(val_indices)
    if sorted(canonical) != list(range(len(splits))) or sorted(validation_members) != list(range(count)):
        raise ValueError(f"{path}: held-out folds must partition the training dataset exactly once")
    for key in ("scheduled_fold_indices", "completed_fold_indices"):
        indices = manifest[key]
        if (not isinstance(indices, list) or not indices or any(type(i) is not int or i not in canonical for i in indices)
                or len(indices) != len(set(indices))):
            raise ValueError(f"{path}: invalid {key}")
    if not set(manifest["completed_fold_indices"]).issubset(manifest["scheduled_fold_indices"]):
        raise ValueError(f"{path}: completed folds were not scheduled")
    return canonical


def validate_predictions(labels, probabilities):
    if labels.ndim != 1 or labels.size == 0 or probabilities.shape != labels.shape:
        raise ValueError("labels and probabilities must be aligned nonempty 1D arrays")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must contain only 0 and 1")
    if not np.isfinite(probabilities).all() or not ((0 <= probabilities) & (probabilities <= 1)).all():
        raise ValueError("probabilities must be finite and in [0, 1]")


def run_provenance(control_dir, model_dir):
    """Expose saved configuration without certifying undocumented provenance."""
    metadata = {}
    warnings = []
    for role, directory in (("control", control_dir), ("model", model_dir)):
        path = Path(directory) / "ensemble_summary.json"
        if not path.exists():
            metadata[role] = {}
            warnings.append(f"{role}: no ensemble_summary.json; training protocol is unverified.")
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        metadata[role] = {key: value for key, value in summary.items()
                          if key not in {"ensemble", "single_fold_summary"}}
        weights = summary.get("val_mcc_weights", [])
        if len(weights) != 5:
            warnings.append(f"{role}: summary records {len(weights)} checkpoint weights, not five.")
    differences = {
        field: {"control": metadata["control"].get(field), "model": metadata["model"].get(field)}
        for field in PROTOCOL_FIELDS
        if metadata["control"].get(field) != metadata["model"].get(field)
    }
    if differences:
        warnings.append("Protocol differences: " + ", ".join(differences) + ".")
    manifests = {}
    split_manifests = {}
    for role, directory in (("control", control_dir), ("model", model_dir)):
        path = Path(directory) / "run_provenance.json"
        if not path.exists():
            manifests[role] = None
            warnings.append(f"{role}: no run_provenance.json; training dataset hash and actual folds are unverified.")
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        split_manifests[role] = _validate_manifest(manifest, path)
        manifests[role] = manifest
        if manifest.get("status") != "complete":
            warnings.append(f"{role}: run provenance is not marked complete.")
    training_pairing = "unverified"
    if all(manifests.values()):
        if manifests["control"]["datasets"]["Train335"]["sha256"] != manifests["model"]["datasets"]["Train335"]["sha256"]:
            raise ValueError("control and model training dataset hashes differ")
        if split_manifests["control"] != split_manifests["model"]:
            raise ValueError("control and model actual cross-validation folds differ")
        for key in ("scheduled_fold_indices", "completed_fold_indices"):
            if sorted(manifests["control"][key]) != sorted(manifests["model"][key]):
                raise ValueError(f"control and model {key} differ")
        training_pairing = "matching recorded Train335 hash and actual fold manifests"
    warnings.append("Threshold-selection provenance is not independently certified by this comparison.")
    return {"metadata": metadata, "protocol_differences": differences, "warnings": warnings,
            "run_manifests": manifests, "training_pairing": training_pairing}


def complex_indices(data_path, expected_residues, expected_labels=None, identities=None):
    samples, dataset_record = load_dataset_with_provenance(data_path)
    groups = defaultdict(list)
    data_labels = []
    offset = 0
    for sample_index, sample in enumerate(samples):
        count = int(sample["residue_graph_node"].shape[0])
        raw_code = sample.get("complex_code")
        code = str(raw_code).strip().upper() if raw_code is not None else ""
        if not code or code in {"NAN", "NONE"} or count <= 0:
            raise ValueError(f"{data_path}: sample {sample_index} needs a complex_code and residues")
        labels = np.asarray(sample["label"]).reshape(-1)
        if labels.size != count or not np.isin(labels, [0, 1]).all():
            raise ValueError(f"{data_path}: invalid labels in sample {sample_index}")
        data_labels.append(labels)
        groups[code].append(np.arange(offset, offset + count, dtype=np.int64))
        offset += count
    if offset != expected_residues:
        raise ValueError(
            f"{data_path}: expected {expected_residues} residues but reconstructed {offset}"
        )
    if expected_labels is not None and not np.array_equal(np.concatenate(data_labels), expected_labels):
        raise ValueError(f"{data_path}: prediction labels do not match dataset order")
    if identities:
        expected_identity = prediction_identity(samples)
        for role, identity in identities.items():
            if identity is None:
                continue
            if identity["dataset_name"] != Path(data_path).stem:
                raise ValueError(f"{role}: prediction dataset_name does not match {data_path}")
            if identity["dataset_sha256"] != dataset_record["sha256"]:
                raise ValueError(f"{role}: prediction dataset hash does not match {data_path}")
            for key in IDENTITY_ARRAY_FIELDS:
                if not np.array_equal(identity[key], expected_identity[key]):
                    raise ValueError(f"{role}: {key} does not match actual dataset residue order")
    return [np.concatenate(groups[key]) for key in sorted(groups)]


def verify_prediction_sources(identities, provenance, test_set):
    """Bind new arrays to their own run manifest; never equate treatment UUIDs."""
    status = {}
    for role, identity in identities.items():
        if identity is None:
            status[role] = "legacy: label order only; residue identity and archive-to-run binding unverified"
            provenance["warnings"].append(f"{test_set} {role}: {status[role]}.")
            continue
        manifest = provenance["run_manifests"].get(role)
        if manifest is not None:
            if identity["provenance_id"] != manifest["provenance_id"]:
                raise ValueError(f"{test_set} {role}: archive provenance_id does not match its run manifest")
            source = manifest["datasets"].get(test_set, {})
            if source.get("sha256") != identity["dataset_sha256"]:
                raise ValueError(f"{test_set} {role}: archive dataset hash does not match its run manifest")
            status[role] = "dataset hash/residue order verified; bound to saved run manifest"
        else:
            status[role] = "dataset hash/residue order verified; archive-to-run binding unverified"
    if all(identity is not None for identity in identities.values()):
        control, model = identities["control"], identities["model"]
        for key in IDENTITY_ARRAY_FIELDS + ("dataset_name", "dataset_sha256"):
            if not np.array_equal(control[key], model[key]):
                raise ValueError(f"{test_set}: control and model prediction identity differs ({key})")
    return status


def metrics(labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int64)
    return {
        "auc_pr": auc_pr(labels, probabilities),
        "auc_roc": float(compute_auc_roc(labels, probabilities)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
    }


def auc_pr(labels, probabilities):
    return float(compute_auc_pr(labels, probabilities))


def cluster_bootstrap(labels, control_probs, model_probs, groups, replicates, seed):
    validate_predictions(labels, control_probs)
    validate_predictions(labels, model_probs)
    if replicates < 1 or not groups:
        raise ValueError("bootstrap requires positive replicates and nonempty groups")
    for group in groups:
        if np.asarray(group).ndim != 1 or len(group) == 0 or np.asarray(group).dtype.kind not in "iu":
            raise ValueError("each group must be a nonempty integer index vector")
    all_indices = np.concatenate(groups)
    if not np.array_equal(np.sort(all_indices), np.arange(labels.size)):
        raise ValueError("groups must partition every residue exactly once")
    rng = np.random.default_rng(seed)
    auc_differences = []
    invalid_replicates = 0
    for replicate in range(replicates):
        selected = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in selected])
        sampled_labels = labels[indices]
        if not np.any(sampled_labels == 1):
            invalid_replicates += 1
            continue
        auc_differences.append(auc_pr(
            sampled_labels, model_probs[indices]
        ) - auc_pr(sampled_labels, control_probs[indices]))
    if not auc_differences:
        raise ValueError("no bootstrap sample contained positive labels; AUPRC is undefined")
    auc_differences = np.asarray(auc_differences)
    low, high = np.percentile(auc_differences, [2.5, 97.5])
    return {
        "mean": float(np.mean(auc_differences)),
        "ci_low": float(low),
        "ci_high": float(high),
        "probability_positive": float(np.mean(auc_differences > 0.0)),
        "valid_replicates": len(auc_differences),
        "invalid_no_positive_replicates": invalid_replicates,
        "interpretation": "Percentile interval conditional on samples containing positives; probability_positive is a bootstrap fraction, not a p-value or posterior probability.",
    }


def benchmark_bootstrap(differences, replicates=100000, seed=9102):
    rng = np.random.default_rng(seed)
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all() or replicates < 1:
        raise ValueError("benchmark bootstrap requires finite differences and positive replicates")
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
        "# PC-BIND Saved-Prediction Comparison",
        "",
        f"Control: `{report['control_dir']}`. Model: `{report['model_dir']}`. The saved probability-mean arrays are compared at their saved thresholds.",
        f"Resolved dataset directory: `{report['data_dir']}`. Any filesystem junctions or symbolic links are resolved in this path.",
        "",
        "AUPRC denotes trapezoidal precision-recall area (not average precision). It is computed over pooled residues, with paired resampling of entire complexes. Larger complexes therefore contribute more residues to each metric.",
        f"Training pairing: {report['provenance']['training_pairing']}.",
        "",
        *[f"- {warning}" for warning in report["provenance"]["warnings"]],
        "",
        "| Test set | Control AUPRC | Model AUPRC | Difference | Clustered 95% CI | Control MCC | Model MCC |",
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
            f"Across {len(report['tests'])} selected tests, average AUPRC was {summary['control_avg_auc_pr']:.4f} for the control and {summary['model_avg_auc_pr']:.4f} for the model (difference {summary['auc_pr_benchmark_bootstrap']['mean_difference']:+.4f}; exploratory benchmark-bootstrap 95% CI [{summary['auc_pr_benchmark_bootstrap']['ci_low']:+.4f}, {summary['auc_pr_benchmark_bootstrap']['ci_high']:+.4f}]). Average MCC changed from {summary['control_avg_mcc']:.4f} to {summary['model_avg_mcc']:.4f}.",
            "",
            "The per-test intervals describe complex sampling conditional on these fitted models; they do not include training-seed variability. Bootstrap samples without positive labels are excluded and counted in the JSON report. The benchmark interval assumes independent, exchangeable datasets; with few or related benchmarks it is exploratory and cannot establish generalization to a population of tasks. Intervals are unadjusted for multiple comparisons. The bootstrap fraction above zero is not a p-value.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    report = {
        "control_dir": str(Path(args.control_dir).resolve()),
        "model_dir": str(Path(args.model_dir).resolve()),
        "data_dir": str(Path(args.data_dir).resolve()),
        "cluster_bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
        "metric_definition": "trapezoidal precision-recall AUC over pooled residues",
        "provenance": run_provenance(args.control_dir, args.model_dir),
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
        identities = {
            "control": load_prediction_identity(args.control_dir, test_set),
            "model": load_prediction_identity(args.model_dir, test_set),
        }
        groups = complex_indices(
            Path(args.data_dir) / f"{test_set}.pkl", labels_control.size, labels_control,
            identities=identities,
        )
        identity_status = verify_prediction_sources(identities, report["provenance"], test_set)
        if np.unique(labels_control).size < 2:
            raise ValueError(f"{test_set}: primary comparison requires both binary classes")
        control_metrics = metrics(labels_control, control_probs, control_threshold)
        model_metrics = metrics(labels_model, model_probs, model_threshold)
        difference = model_metrics["auc_pr"] - control_metrics["auc_pr"]
        auc_differences.append(difference)
        report["tests"][test_set] = {
            "complexes": len(groups),
            "residues": int(labels_control.size),
            "identity_verification": identity_status,
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
