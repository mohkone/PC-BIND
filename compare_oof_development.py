"""Compare only complete, frozen development OOF predictions, paired by complex."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import matthews_corrcoef

from compare_pcbind_primary import _archive_identity, validate_predictions
from filtered_cohort_protocol import validate_fixed_folds
from frozen_development_protocol import (
    read_freeze, require_equal, run_configuration, validate_completed_run, verify_environment,
)
from metrics import compute_auc_pr, compute_auc_roc
from research_provenance import file_record, load_dataset_with_provenance, prediction_identity


def expected_oof(proteins, folds):
    identities, labels = [], []
    for fold in folds:
        indices = fold["val_indices"]
        identities.append(prediction_identity(proteins, indices, fold["fold_index"]))
        labels.extend(np.asarray(proteins[index]["label"]) for index in indices)
    return np.concatenate(labels), {
        key: np.concatenate([item[key] for item in identities]) for key in identities[0]
    }


def load_oof(path, run, expected_labels, expected_identity):
    before = file_record(path)
    with np.load(path, allow_pickle=False) as archive:
        labels = np.asarray(archive["labels"])
        probs = np.asarray(archive["probs"], dtype=np.float64)
        identity = _archive_identity(archive, path)
        threshold = np.asarray(archive["threshold"])
    validate_predictions(labels, probs)
    if identity is None or "fold_index" not in identity:
        raise ValueError("Complete OOF residue and fold identity is required")
    if not np.array_equal(labels, expected_labels):
        raise ValueError("OOF labels/order/coverage differ from the full fixed validation population")
    for key, expected in expected_identity.items():
        if not np.array_equal(identity[key], expected):
            raise ValueError(f"OOF identity/order/coverage mismatch: {key}")
    for key, expected in {
        "dataset_name": "Train335", "dataset_sha256": run["datasets"]["Train335"]["sha256"],
        "provenance_id": run["provenance_id"], "identity_schema_version": 1,
    }.items():
        require_equal(identity[key], expected, f"prediction {key}")
    if threshold.shape != (1,) or not np.isfinite(threshold).all() or not 0 <= threshold[0] <= 1:
        raise ValueError("OOF threshold must contain one finite value in [0, 1]")
    require_equal(file_record(path), before, "OOF archive changed during reading")
    return labels.astype(np.int64), probs, float(threshold[0]), before


def score_metrics(labels, probs, threshold):
    return {
        "auc_pr_trapezoidal": float(compute_auc_pr(labels, probs)),
        "mcc": float(matthews_corrcoef(labels, probs >= threshold)),
        "auroc": float(compute_auc_roc(labels, probs)), "threshold": threshold,
    }


def validate_checkpoint(checkpoint, run, summary, number, labels, probs):
    import torch

    fold = number - 1
    require_equal(checkpoint.get("fold_index"), fold, "checkpoint fold index")
    require_equal(checkpoint.get("cv_split"), run["cv_splits"][fold], "checkpoint split membership")
    saved_run = checkpoint.get("run_provenance", {})
    require_equal(run_configuration(saved_run), run_configuration(run), "checkpoint run configuration")
    require_equal(saved_run.get("provenance_id"), run["provenance_id"], "checkpoint run identity")
    require_equal(saved_run.get("completed_fold_indices"), list(range(number)), "checkpoint completed folds")
    if saved_run.get("partner_transport") is not run["partner_transport"]:
        raise ValueError("Checkpoint transport provenance differs from its run")
    for key in ("partner_transport", "skip_test_eval", "use_plm_features", "use_aux_plm_features"):
        expected = run["partner_transport"] if key == "partner_transport" else True
        if checkpoint.get(key) is not expected:
            raise ValueError(f"Invalid checkpoint {key}")
    for key in summary:
        if key in checkpoint and key not in {"run_provenance", "partner_transport"}:
            require_equal(checkpoint[key], summary[key], f"checkpoint recipe {key}")
    threshold = checkpoint.get("threshold")
    if not isinstance(threshold, (float, int)) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid checkpoint threshold")
    for key, metric in (("val_mcc", "mcc"), ("val_auc_pr", "auc_pr_trapezoidal")):
        actual = score_metrics(labels, probs, threshold)[metric]
        if not np.isclose(checkpoint.get(key, np.nan), actual, rtol=0, atol=1e-6):
            raise ValueError(f"Checkpoint {key} does not reproduce its OOF fold predictions")
    state = checkpoint.get("model_state")
    if not isinstance(state, dict) or not state:
        raise ValueError("Missing checkpoint model state")
    if any(torch.is_floating_point(value) and not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Nonfinite checkpoint model state")


def paired_complex_bootstrap(labels, model_probs, control_probs, model_threshold,
                             control_threshold, complex_ids, replicates=5000, seed=2101):
    validate_predictions(labels, model_probs)
    validate_predictions(labels, control_probs)
    groups = np.char.upper(np.char.strip(np.asarray(complex_ids, dtype=str)))
    if groups.shape != labels.shape or np.any(np.isin(groups, ["", "NONE", "NAN", "NULL"])):
        raise ValueError("Every residue requires a nonmissing complex bootstrap group")
    unique = np.unique(groups)
    if unique.size < 2 or type(replicates) is not int or replicates < 1:
        raise ValueError("Bootstrap requires multiple complexes and positive replicates")
    rows = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(seed)
    deltas = {"auc_pr_trapezoidal": [], "mcc": []}
    skipped = 0
    # A single draw is shared by both arms, preserving all targets/residues in
    # each selected complex. Duplicate draws repeat the entire complex.
    for _ in range(replicates):
        indices = np.concatenate([rows[index] for index in rng.integers(len(rows), size=len(rows))])
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size != 2:
            skipped += 1
            continue
        model = model_probs[indices]
        control = control_probs[indices]
        deltas["auc_pr_trapezoidal"].append(float(
            compute_auc_pr(sampled_labels, model) - compute_auc_pr(sampled_labels, control)))
        deltas["mcc"].append(float(
            matthews_corrcoef(sampled_labels, model >= model_threshold)
            - matthews_corrcoef(sampled_labels, control >= control_threshold)))
    if not deltas["mcc"]:
        raise ValueError("No bootstrap draws contain both classes")
    return {
        "resampling_unit": "complex_code", "complex_count": len(rows),
        "requested_replicates": replicates, "valid_replicates": len(deltas["mcc"]),
        "single_class_draws_excluded": skipped, "seed": seed,
        "paired_percentile_95_ci": {
            key: [float(value) for value in np.quantile(values, [0.025, 0.975])]
            for key, values in deltas.items()
        },
    }


def validate_pair(model_dir, control_dir, freeze):
    folds_json = verify_environment(freeze)
    runs, summaries = {}, {}
    for role, directory, transport in (("ot", model_dir, True), ("control", control_dir, False)):
        runs[role], summaries[role] = validate_completed_run(directory, freeze, transport)
    training = freeze["run_configuration"]["datasets"]["Train335"]
    proteins, record = load_dataset_with_provenance(training["path"])
    require_equal(record, training, "loaded training pickle")
    folds = validate_fixed_folds(proteins, folds_json, 2101, training["sha256"])
    labels, identity = expected_oof(proteins, folds)
    # Free the large feature matrices before resampling; retain only identities.
    del proteins
    arrays, artifacts = {}, {}
    for role, directory in (("ot", model_dir), ("control", control_dir)):
        array = load_oof(Path(directory) / "oof_predictions.npz", runs[role], labels, identity)
        arrays[role] = array[:3]
        artifacts[role] = {"oof_predictions": array[3]}
        if not np.isclose(array[2], summaries[role]["ensemble_threshold"], rtol=0, atol=1e-7):
            raise ValueError(f"{role}: OOF threshold differs from saved summary")
        for name in ("run_provenance.json", "ensemble_summary.json", "launcher_provenance.json",
                     "prerequisite_provenance.json"):
            artifacts[role][name] = file_record(Path(directory) / name)
        import torch
        artifacts[role]["checkpoints"] = []
        for number in range(1, 6):
            path = Path(directory) / f"fold{number}_best.pt"
            before = file_record(path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            mask = identity["fold_index"] == number - 1
            validate_checkpoint(checkpoint, runs[role], summaries[role], number, array[0][mask], array[1][mask])
            require_equal(file_record(path), before, "checkpoint changed during reading")
            artifacts[role]["checkpoints"].append(before)
            del checkpoint
    return arrays, identity, artifacts


def compare(model_dir, control_dir, freeze_path, output_dir, validate_only=False):
    freeze_record = file_record(freeze_path)
    freeze = read_freeze(freeze_path)
    arrays, identity, artifacts = validate_pair(model_dir, control_dir, freeze)
    require_equal(file_record(freeze_path), freeze_record, "freeze manifest changed")
    labels, model_probs, model_threshold = arrays["ot"]
    _, control_probs, control_threshold = arrays["control"]
    result = {
        "status": "validated" if validate_only else "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "filtered-cohort analysis; complete five-fold development OOF only",
        "freeze_manifest": freeze_record,
        "sample_count": len(np.unique(identity["sample_index"])), "residue_count": len(labels),
        "complex_count": len(np.unique(np.char.upper(np.char.strip(identity["complex_id"])))),
        "fold_indices": [int(value) for value in np.unique(identity["fold_index"])],
        "external_evaluation": "skipped in both arms", "artifacts": artifacts,
    }
    if validate_only:
        return result
    model = score_metrics(labels, model_probs, model_threshold)
    control = score_metrics(labels, control_probs, control_threshold)
    result.update({
        "metrics": {"ot": model, "control": control},
        "ot_minus_control": {key: model[key] - control[key] for key in
                              ("auc_pr_trapezoidal", "mcc", "auroc")},
        "paired_complex_bootstrap": paired_complex_bootstrap(
            labels, model_probs, control_probs, model_threshold, control_threshold,
            identity["complex_id"], freeze["comparison"]["bootstrap_replicates"],
            freeze["comparison"]["bootstrap_seed"],
        ),
        "limitations": [
            "Checkpoint and threshold selection used these development labels; results are selection-optimistic.",
            "Bootstrap holds fitted predictions and each selected threshold fixed; it does not quantify seed or training variability.",
            "Training sets overlap across folds; complex resampling does not remove dependence caused by model fitting.",
            "This filtered sensitivity cohort has no verified original PDB residue mappings or established homology independence.",
            "The no-OT branch removes transport parameters; this is not a parameter-count-matched capacity ablation.",
            "No Test287 performance or external generalization claim is evaluated here.",
        ],
    })
    output = Path(output_dir)
    # Do not overwrite a previous analysis or emit a partial-cohort result.
    output.mkdir(parents=True, exist_ok=False)
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    table = ["| Complete OOF metric | OT | no-OT | OT minus no-OT | Paired 95% interval |",
             "| --- | ---: | ---: | ---: | --- |"]
    intervals = result["paired_complex_bootstrap"]["paired_percentile_95_ci"]
    for key in ("auc_pr_trapezoidal", "mcc", "auroc"):
        interval = intervals.get(key)
        text = f"[{interval[0]:.4f}, {interval[1]:.4f}]" if interval else "Descriptive only"
        table.append(f"| {key} | {model[key]:.4f} | {control[key]:.4f} | "
                     f"{result['ot_minus_control'][key]:+.4f} | {text} |")
    markdown = (
        "# Frozen five-fold development comparison\n\n"
        f"Both arms completed the same five fixed folds: {result['sample_count']} samples, "
        f"{result['complex_count']} complexes and {len(labels)} residues. Test287 was skipped. "
        "All source, runtime, dataset and split identities passed the frozen protocol gate.\n\n"
        + "\n".join(table) + "\n\n"
        + "PR-AUC is trapezoidal area under the precision-recall curve, not average precision. "
        "Primary PR-AUC and secondary MCC intervals use the same 5,000 whole-complex draws "
        "for both arms, with their saved pooled OOF thresholds held fixed.\n\n"
        + "\n".join("- " + limitation for limitation in result["limitations"]) + "\n"
    )
    (output / "RESULT.md").write_text(markdown, encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--control-dir", required=True)
    parser.add_argument("--freeze-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = compare(args.model_dir, args.control_dir, args.freeze_manifest, args.output_dir, args.validate_only)
    print(json.dumps({key: result[key] for key in
                      ("status", "sample_count", "residue_count", "complex_count", "fold_indices")}, indent=2))


if __name__ == "__main__":
    main()
