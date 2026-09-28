"""Recompute descriptive results and inventory run provenance without base data.

This never selects a strategy, optimizes a threshold, or treats residues as
independent bootstrap units. Original result files are read-only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    auc, average_precision_score, matthews_corrcoef, precision_recall_curve,
    precision_recall_fscore_support, roc_auc_score,
)


TESTS = ("Test60", "Test287", "TestB25", "TestUB25")
CONFIG_KEYS = (
    "seed", "max_folds", "grouped_cv", "cv_group_key", "partner_conditioning",
    "partner_direct_fusion", "partner_residue_encoder", "partner_transport",
    "transport_implementation", "plm_feature_mode", "plm_feature_dim",
    "aux_plm_feature_mode", "aux_plm_feature_dim", "partner_contact_aux",
)


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def audit_run(path):
    summary_path = path / "ensemble_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run = {
        "path": str(path.resolve()),
        "summary_sha256": sha256(summary_path),
        "config": {key: summary.get(key) for key in CONFIG_KEYS},
        "checkpoint_files": sorted(p.name for p in path.glob("fold*_best.pt")),
        "tests": {},
        "warnings": [],
    }
    if summary.get("max_folds") != 5 or len(run["checkpoint_files"]) != 5:
        run["warnings"].append("Incomplete five-fold evidence; treat as a pilot.")
    if summary.get("partner_transport") is False and "ot" in path.name.lower():
        run["warnings"].append("Transport is disabled in serialized metadata; folder name is not treatment evidence.")
    if not summary.get("grouped_cv"):
        run["warnings"].append("Run metadata does not establish complex-grouped cross-validation.")
    for test in TESTS:
        archive_path = path / f"ensemble_{test}_predictions.npz"
        if not archive_path.exists():
            run["warnings"].append(f"Missing probability-mean predictions for {test}.")
            continue
        with np.load(archive_path, allow_pickle=False) as saved:
            labels = np.asarray(saved["labels"])
            scores = np.asarray(saved["probs"], dtype=float)
            threshold_values = np.asarray(saved["threshold"])
            identity_present = all(key in saved for key in (
                "sample_index", "residue_index", "complex_id", "dataset_sha256",
            ))
        if (labels.ndim != 1 or scores.shape != labels.shape or labels.size == 0
                or not np.isin(labels, [0, 1]).all()
                or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))
                or threshold_values.size != 1 or not np.isfinite(threshold_values).all()):
            raise ValueError(f"Invalid predictions: {archive_path}")
        threshold = float(threshold_values.item())
        if not 0 <= threshold <= 1:
            raise ValueError(f"Invalid threshold: {archive_path}")
        labels = labels.astype(int)
        predicted = scores >= threshold
        precision, recall, _ = precision_recall_curve(labels, scores)
        binary_p, binary_r, binary_f, _ = precision_recall_fscore_support(
            labels, predicted, average="binary", zero_division=0,
        )
        values = {
            "auc_pr": float(auc(recall, precision)) if labels.any() else None,
            "average_precision": float(average_precision_score(labels, scores)) if labels.any() else None,
            "auc_roc": float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else None,
            "mcc": float(matthews_corrcoef(labels, predicted)),
            "interface_precision": float(binary_p),
            "interface_recall": float(binary_r),
            "interface_f1": float(binary_f),
        }
        recorded = summary.get("ensemble", {}).get("mean", {}).get(test, {})
        errors = {key: abs(values[key] - recorded[key]) for key in ("auc_pr", "auc_roc", "mcc")
                  if values[key] is not None and key in recorded}
        run["tests"][test] = {
            "prediction_sha256": sha256(archive_path),
            "residues": int(labels.size), "positives": int(labels.sum()),
            "prevalence": float(labels.mean()), "threshold": threshold,
            "explicit_residue_identity_present": identity_present,
            "metrics": values, "absolute_error_vs_summary": errors,
        }
        if not identity_present:
            run["warnings"].append(f"{test}: residue identity absent; dataset ordering cannot be certified from NPZ alone.")
        if len(errors) != 3:
            run["warnings"].append(f"{test}: incomplete recorded metrics for verification.")
        elif max(errors.values()) > 1e-6:
            run["warnings"].append(f"{test}: recomputed metrics differ from summary.")
    return run


def write_report(report, path):
    lines = ["# Saved artifact audit", "", report["scope"], "",
             "Only the saved probability mean and stored threshold are evaluated. "
             "No test-set strategy selection or threshold refitting is performed.", "",
             "| Run | Seed | Folds present | Grouped CV | Transport | Main/aux PLM width | Contact auxiliary |",
             "|---|---:|---:|---|---|---|---|"]
    for name, run in report["runs"].items():
        config = run["config"]
        lines.append(f"| {name} | {config['seed']} | {len(run['checkpoint_files'])} | "
                     f"{config['grouped_cv']} | {config['partner_transport']} | "
                     f"{config['plm_feature_dim']}/{config['aux_plm_feature_dim']} | {config['partner_contact_aux']} |")
    lines += ["", "AUPRC below is trapezoidal area; AP is non-interpolated average precision. "
              "Interface precision/recall/F1 refer only to the positive class, whereas the legacy CSV uses class-macro values."]
    for name, run in report["runs"].items():
        lines += ["", f"## {name}", "", "| Dataset | Residues | Prevalence | AUPRC | AP | MCC | Interface precision | Interface recall | Interface F1 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for test, row in run["tests"].items():
            m = row["metrics"]
            cells = [test, str(row['residues']), f"{row['prevalence']:.4f}"]
            cells += ["undefined" if m[k] is None else f"{m[k]:.4f}" for k in (
                "auc_pr", "average_precision", "mcc", "interface_precision", "interface_recall", "interface_f1")]
            lines.append("| " + " | ".join(cells) + " |")
        lines += [""] + ["- " + warning for warning in run["warnings"]]
    lines += ["", "These descriptive results do not establish an OT benefit, a matched causal contrast, "
              "homology-independent generalization, or variation across independent training seeds. "
              "The JSON records hashes and differences from the original metric summaries.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("research_review"))
    args = parser.parse_args()
    paths = args.runs or sorted(path.parent for path in Path(".").glob("outputs*/ensemble_summary.json"))
    if not paths:
        parser.error("No saved run summaries found")
    report = {
        "scope": "Read-only audit of saved arrays and serialized metadata. Base datasets, chain identities, "
                 "fold membership, and homology separation are not verified by this audit.",
        "runs": {str(path): audit_run(path) for path in paths},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "artifact_audit.json"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(report, args.output_dir / "artifact_audit.md")
    print(f"Audited {len(paths)} runs; wrote {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
