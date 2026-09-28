"""Read-only input coverage and exact-identity audit; does not run homology search."""
import argparse
import gc
import hashlib
import json
import pickle
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from check_pcbind_prereqs import (
    PLM_SPECS, count_positive_pairs, finite_matrix, partner_residue_count,
    valid_partner_encoder_sample, valid_target_plm,
)

DEFAULT_FILES = ["Train335.pkl", "Test60.pkl", "Test287.pkl", "TestB25.pkl", "TestUB25.pkl", "Test70.pkl"]


def target_failures(sample):
    reasons = []
    rows = len(sample["residue_graph_node"])
    for key, (width, model_key, model_name) in PLM_SPECS.items():
        if not valid_target_plm(sample, key):
            if key not in sample:
                reasons.append(f"{key}: missing")
            elif not finite_matrix(sample[key], (rows, width)):
                reasons.append(f"{key}: wrong shape/nonfinite, expected ({rows}, {width})")
            if sample.get(model_key, model_name) != model_name:
                reasons.append(f"{model_key}: {sample.get(model_key)!r}")
    return reasons


def partner_failures(sample):
    if valid_partner_encoder_sample(sample):
        return []
    reasons = []
    required = ["partner_residue_surface_features", "partner_residue_sequence_features",
                "partner_residue_coords", "partner_residue_frames", "partner_residue_geo_edge",
                "partner_residue_sequences", "partner_chain_slices",
                "partner_residue_plm_embedding", "partner_residue_plm_embedding_8m"]
    missing = [key for key in required if key not in sample]
    if missing:
        reasons.append("missing fields: " + ", ".join(missing))
    rows = partner_residue_count(sample)
    if rows == 0:
        reasons.append("zero partner residues (target-pathway fallback)")
    if sample.get("pdb_augmentation_status") == "failed":
        reasons.append("pdb_augmentation_status=failed")
    for key, shape in [("partner_residue_surface_features", (rows, 14)),
                       ("partner_residue_sequence_features", (rows, 39)),
                       ("partner_residue_coords", (rows, 3)),
                       ("partner_residue_frames", (rows, 3, 3)),
                       ("partner_residue_plm_embedding", (rows, 480)),
                       ("partner_residue_plm_embedding_8m", (rows, 320))]:
        if key in sample and not finite_matrix(sample[key], shape):
            reasons.append(f"{key}: wrong shape/nonfinite, expected {shape}")
    for _, (_, model_key, model_name) in PLM_SPECS.items():
        if sample.get("partner_" + model_key, model_name) != model_name:
            reasons.append(f"partner_{model_key}: {sample.get('partner_' + model_key)!r}")
    if not reasons:
        reasons.append("invalid chain slices, per-chain sequence lengths, or edge indices; see check_pcbind_prereqs.py")
    return reasons


def read_dataset(path):
    before = path.stat()
    with path.open("rb") as handle:
        samples = pickle.load(handle)
    records = []
    failures = []
    styles = Counter()
    for index, sample in enumerate(samples):
        code = str(sample.get("complex_code", "")).strip().upper()
        match = re.fullmatch(r"([0-9][A-Z0-9]{3})(?:[_:](.+))?", code)
        pdb_id = match.group(1) if match else None
        styles["bare_pdb" if match and not match.group(2) else "chain_suffix" if match else "other"] += 1
        sequence = str(sample.get("residue_sequence", "")).strip().upper()
        row = {"index": index, "complex_code": code, "pdb_id": pdb_id,
               "pdb_chain": sample.get("pdb_chain"), "target_residues": len(sample["residue_graph_node"]),
               "sequence_length": len(sequence), "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest() if sequence else None,
               "partner_residues": partner_residue_count(sample)}
        row["partner_sequence_sha256"] = [
            hashlib.sha256(str(sequence).strip().upper().encode()).hexdigest()
            for sequence in sample.get("partner_residue_sequences", [])
            if str(sequence).strip()
        ]
        records.append(row)
        target_reasons = target_failures(sample)
        partner_reasons = partner_failures(sample)
        if target_reasons or partner_reasons or len(sequence) != row["target_residues"]:
            failures.append({**row, "target_plm_failures": target_reasons,
                             "target_sequence_failure": None if len(sequence) == row["target_residues"] else "missing or wrong-length target sequence",
                             "partner_encoder_failures": partner_reasons})
    codes = Counter(row["complex_code"] for row in records)
    pair_records, positive_pairs = count_positive_pairs(samples)
    summary = {"workspace_path": str(path.absolute()), "resolved_path": str(path.resolve()),
               "bytes": before.st_size, "modified_utc": datetime.fromtimestamp(before.st_mtime, timezone.utc).isoformat(),
               "samples": len(samples), "unique_complex_codes": len(codes),
               "unique_pdb_ids": len({row["pdb_id"] for row in records if row["pdb_id"]}),
               "multi_sample_complex_codes": {key: count for key, count in codes.items() if count > 1},
               "complex_code_styles": dict(styles),
               "target_sequences_present": sum(row["sequence_length"] > 0 for row in records),
               "samples_with_partner_chain_sequences": sum(bool(row["partner_sequence_sha256"]) for row in records),
               "valid_target_main_plm": sum(bool(valid_target_plm(s, "residue_plm_embedding")) for s in samples),
               "valid_target_aux_plm": sum(bool(valid_target_plm(s, "residue_plm_embedding_8m")) for s in samples),
               "valid_partner_encoder": sum(valid_partner_encoder_sample(s) for s in samples),
               "nonempty_partner_samples": sum(row["partner_residues"] > 0 for row in records),
               "valid_pair_label_samples": pair_records, "positive_pair_labels": positive_pairs,
               "mapping_verification_metadata_present": sum("pdb_mapping_verified" in s for s in samples),
               "mapping_verified_true": sum(s.get("pdb_mapping_verified") is True for s in samples),
               "failed_or_fallback_samples": failures}
    del samples
    gc.collect()
    with path.open("rb") as handle:
        summary["sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed during read-only audit: {path}")
    return summary, records


def exact_overlap(train, test):
    train_codes = {r["complex_code"] for r in train}
    train_pdb = {r["pdb_id"] for r in train if r["pdb_id"]}
    matches = []
    for row in test:
        sequence_hash = row["sequence_sha256"]
        matching = [r for r in train if sequence_hash and r["sequence_sha256"] == sequence_hash]
        if matching:
            matches.append({"test": row, "train": matching})
    return {"shared_exact_complex_codes": sorted(train_codes & {r["complex_code"] for r in test}),
            "shared_normalized_pdb_ids": sorted(train_pdb & {r["pdb_id"] for r in test}),
            "test_samples_with_exact_train_sequence": len(matches),
            "shared_unique_exact_sequences": len({m["test"]["sequence_sha256"] for m in matches}),
            "exact_sequence_matches": matches}


def cross_side_overlap(train, test, train_side, test_side):
    """Compare exact whole chain sequences, counting each affected sample once."""
    def hashes(row, side):
        return row.get("partner_sequence_sha256", []) if side == "partner" else ([row["sequence_sha256"]] if row["sequence_sha256"] else [])

    source = {value for row in train for value in hashes(row, train_side)}
    affected = []
    matched_sequences = set()
    matched_chains = 0
    for row in test:
        query = hashes(row, test_side)
        matches = sorted(source & set(query))
        if matches:
            matched_sequences.update(matches)
            matched_chains += sum(value in source for value in query)
            affected.append({"index": row["index"], "complex_code": row["complex_code"],
                             "pdb_chain": row.get("pdb_chain"), "matching_sequence_sha256": matches})
    return {"comparison": f"test {test_side} versus Train335 {train_side}",
            "available_test_samples": sum(bool(hashes(row, test_side)) for row in test),
            "available_train_samples": sum(bool(hashes(row, train_side)) for row in train),
            "distinct_train_sequences": len(source),
            "affected_test_samples": len(affected), "distinct_matched_sequences": len(matched_sequences),
            "matching_test_chain_occurrences": matched_chains, "affected_samples": affected}


def write_markdown(path, report):
    lines = ["# Read-only input integrity audit", "",
             f"Generated: {report['generated_utc']}. Input directory resolves to `{report['resolved_data_dir']}`.", "",
             "This audit reads the linked pickle files without altering them or downloading anything. It checks input coverage and exact target-sequence/PDB identity only. It is not a sequence homology, structural-independence, partner-independence, or label-alignment proof. Indices below are zero-based. File hashes and every affected sample are in `input_audit.json`.", "",
             "| Dataset | Samples | Complex codes / PDB IDs | Valid target main / aux PLM | Valid partner encoder | Nonempty partner |", "|---|---:|---:|---:|---:|---:|"]
    for name, data in report["datasets"].items():
        lines.append(f"| {name} | {data['samples']} | {data['unique_complex_codes']} / {data['unique_pdb_ids']} | {data['valid_target_main_plm']} / {data['valid_target_aux_plm']} | {data['valid_partner_encoder']} | {data['nonempty_partner_samples']} |")
    lines += ["", "## Target input failures", ""]
    for name, data in report["datasets"].items():
        for row in data["failed_or_fallback_samples"]:
            if row["target_plm_failures"] or row["target_sequence_failure"]:
                reasons = row["target_plm_failures"] + ([row["target_sequence_failure"]] if row["target_sequence_failure"] else [])
                lines.append(f"- `{name}[{row['index']}]`, `{row['complex_code']}`: " + "; ".join(reasons) + ".")
    lines += ["", "## Partner coverage and provenance", ""]
    for name, data in report["datasets"].items():
        invalid = [r for r in data["failed_or_fallback_samples"] if r["partner_encoder_failures"]]
        if invalid:
            causes = Counter(reason for row in invalid for reason in row["partner_encoder_failures"])
            lines.append(f"- `{name}`: {len(invalid)} invalid or absent partner encoders. " + "; ".join(f"{count} × {reason}" for reason, count in causes.items()) + ".")
    lines.append("- Existing mapping-verification metadata counts by dataset: " + "; ".join(f"{name}: {data['mapping_verification_metadata_present']}/{data['samples']}" for name, data in report["datasets"].items()) + ". A successful shape check does not verify original residue/label or atom ordering.")
    lines += ["", "## Exact overlap with Train335", "", "| Test dataset | Shared full complex codes | Shared normalized PDB IDs | Test samples with identical target sequence |", "|---|---:|---:|---:|"]
    for name, data in report["train_test_exact_overlap"].items():
        lines.append(f"| {name} | {len(data['shared_exact_complex_codes'])} | {len(data['shared_normalized_pdb_ids'])} | {data['test_samples_with_exact_train_sequence']} |")
    lines += ["", "Zero exact matches does not exclude close homologs. Missing sequences are not treated as identical. The sequence comparison uses SHA-256 fingerprints of normalized whole target strings; no local alignment or BLAST search was performed.", "", "## Identifier formats", ""]
    lines[-2:] = ["## Exact sequence overlap across input sides", "",
                  "Each cell reports affected test samples / distinct matched sequences. Partner comparisons use stored chain-separated partner sequences only; missing partner sequence fields are not reconstructed. These are input-identity overlaps, not evidence by themselves of label contamination.", "",
                  "| Test dataset | Test target → train partner | Test partner → train target | Test partner → train partner | Test samples with partner sequences |",
                  "|---|---:|---:|---:|---:|"]
    for name, data in report["cross_side_exact_overlap"].items():
        cells = [f"{data[key]['affected_test_samples']} / {data[key]['distinct_matched_sequences']}" for key in ("test_target_train_partner", "test_partner_train_target", "test_partner_train_partner")]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {data['test_partner_train_target']['available_test_samples']} |")
    lines += ["", "## Identifier formats", ""]
    for name, data in report["datasets"].items():
        lines.append(f"- `{name}`: " + ", ".join(f"{count} {style}" for style, count in data["complex_code_styles"].items()) + ".")
    lines += ["", "Normalize PDB identity separately from full complex codes when auditing overlap: a chain suffix must not hide a shared source structure. Preserve the original complex grouping key in recorded experiments.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/geo")
    parser.add_argument("--output-dir", default="research_review")
    parser.add_argument("files", nargs="*", default=DEFAULT_FILES)
    args = parser.parse_args()
    if "Train335.pkl" not in args.files:
        parser.error("Train335.pkl is required for train-test overlap")
    data_dir = Path(args.data_dir)
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              "resolved_data_dir": str(data_dir.resolve()),
              "scope": "Read-only schema/coverage and exact target sequence/PDB identity; not homology proof.",
              "datasets": {}, "train_test_exact_overlap": {}, "cross_side_exact_overlap": {}}
    records = {}
    for name in args.files:
        report["datasets"][name], records[name] = read_dataset(data_dir / name)
        print(f"Audited {name}", flush=True)
    for name in args.files:
        if name != "Train335.pkl":
            report["train_test_exact_overlap"][name] = exact_overlap(records["Train335.pkl"], records[name])
            report["cross_side_exact_overlap"][name] = {
                "test_target_train_partner": cross_side_overlap(records["Train335.pkl"], records[name], "partner", "target"),
                "test_partner_train_target": cross_side_overlap(records["Train335.pkl"], records[name], "target", "partner"),
                "test_partner_train_partner": cross_side_overlap(records["Train335.pkl"], records[name], "partner", "partner"),
            }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "input_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(output / "input_audit.md", report)
    print(f"Wrote {output / 'input_audit.json'} and {output / 'input_audit.md'}")


if __name__ == "__main__":
    main()
