"""Build the explicitly filtered PC-BIND sensitivity cohort without editing sources.

Only trusted project pickles should be supplied. Failed records are selected by
original index AND code; a second, valid sample from the same complex is retained.
"""
import argparse
import gc
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit_input_integrity import partner_failures, target_failures
from check_pcbind_prereqs import PLM_SPECS, valid_partner_encoder_sample, valid_target_plm
from research_provenance import cv_split_manifest, file_record, load_dataset_with_provenance


ANALYSIS_LABEL = "filtered-cohort analysis"
COHORT_SPEC = {
    "Train335.pkl": {"original_count": 335, "exclude": {106: "2J3R"}},
    "Test287.pkl": {"original_count": 287, "exclude": {55: "4M0W", 161: "6MAV"}},
}


def write_json_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def content_hash(value):
    """Order-sensitive, typed digest of every sample field, including array bytes."""
    digest = hashlib.sha256()

    def atom(tag, data):
        digest.update(tag)
        digest.update(str(len(data)).encode("ascii") + b":")
        digest.update(data)

    def visit(item):
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise TypeError("Object arrays need an explicit provenance representation")
            atom(b"array-dtype", item.dtype.str.encode())
            atom(b"array-shape", repr(item.shape).encode())
            raw = memoryview(np.ascontiguousarray(item)).cast("B") if item.size else b""
            atom(b"array", raw)
        elif isinstance(item, np.generic):
            atom(b"numpy-dtype", item.dtype.str.encode())
            atom(b"numpy-scalar", item.tobytes())
        elif isinstance(item, dict):
            atom(b"dict", str(len(item)).encode())
            for key in sorted(item):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            atom(type(item).__name__.encode(), str(len(item)).encode())
            for child in item:
                visit(child)
        elif isinstance(item, str):
            atom(b"str", item.encode("utf-8"))
        elif isinstance(item, bytes):
            atom(b"bytes", item)
        elif item is None or isinstance(item, (bool, int, float)):
            atom(type(item).__name__.encode(), repr(item).encode("ascii"))
        else:
            raise TypeError(f"Unsupported provenance type: {type(item).__name__}")

    visit(value)
    return digest.hexdigest()


def select_samples(samples, exclusions, expected_count):
    if len(samples) != expected_count:
        raise ValueError(f"Source population changed: expected {expected_count}, got {len(samples)}")
    for index, code in exclusions.items():
        if index < 0 or index >= len(samples) or samples[index].get("complex_code") != code:
            raise ValueError(f"Original index/code mismatch at {index}: expected {code}")
    retained = [index for index in range(len(samples)) if index not in exclusions]
    return [samples[index] for index in retained], retained


def validate_retained(samples):
    """Fail rather than silently drop another sample or synthesize a representation."""
    for index, sample in enumerate(samples):
        rows = len(sample["residue_graph_node"])
        sequence = sample.get("residue_sequence")
        if not isinstance(sequence, str) or len(sequence) != rows:
            raise ValueError(f"Retained sample {index}: missing or wrong-length target sequence")
        if len(sample.get("label", [])) != rows:
            raise ValueError(f"Retained sample {index}: label length mismatch")
        if not all(valid_target_plm(sample, key) for key in PLM_SPECS):
            raise ValueError(f"Retained sample {index}: invalid target PLM")
        if not valid_partner_encoder_sample(sample):
            raise ValueError(f"Retained sample {index}: invalid partner encoder")
        for prefix in ("", "partner_"):
            for key in PLM_SPECS:
                if np.any(~np.any(np.asarray(sample[prefix + key]) != 0, axis=1)):
                    raise ValueError(f"Retained sample {index}: zero-filled {prefix + key} row")


def validate_folds(samples, folds):
    indices = [int(index) for fold in folds for index in fold]
    if sorted(indices) != list(range(len(samples))) or any(len(fold) == 0 for fold in folds):
        raise ValueError("Validation folds must partition the filtered training population")
    owner = {}
    for fold_index, fold in enumerate(folds):
        for index in fold:
            code = str(samples[int(index)]["complex_code"]).strip().upper()
            if code in owner and owner[code] != fold_index:
                raise ValueError(f"Complex {code} crosses folds")
            owner[code] = fold_index


def build_cohort(source_dir, output_dir, report_dir, seeds=(2101,), num_folds=5):
    # Import only when building; this is exactly the training implementation.
    from CROSS5FOLD_multi_test import make_cv_folds

    source = Path(source_dir).resolve(strict=True)
    destination = Path(output_dir).resolve()
    reports = Path(report_dir).resolve()
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError("Output must be a fresh directory outside the resolved source tree")
    if reports == source or source in reports.parents:
        raise ValueError("Reports must not be written inside the resolved source tree")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite versioned cohort: {destination}")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide one or more unique seeds")

    reports.mkdir(parents=True, exist_ok=True)
    originals = {path.name: file_record(path) for path in sorted(source.glob("*.pkl"))}
    if not set(COHORT_SPEC).issubset(originals):
        raise ValueError("Source is missing a primary dataset")
    original_record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "resolved_source_dir": str(source), "files": originals,
        "policy": "All source pickles are read-only; hashes recorded before derived writes.",
    }
    preserved_path = reports / "original_pickle_hashes.json"
    if preserved_path.exists():
        preserved = json.loads(preserved_path.read_text(encoding="utf-8"))
        if preserved.get("files") != originals or preserved.get("resolved_source_dir") != str(source):
            raise RuntimeError("Previously preserved source hashes differ; use a new version and investigate")
    else:
        write_json_new(preserved_path, original_record)
    destination.mkdir(parents=True, exist_ok=False)
    incomplete = destination / "INCOMPLETE.txt"
    incomplete.write_text("Do not train: build not yet validated.\n", encoding="utf-8")
    manifest = {
        "schema_version": 1, "analysis_label": ANALYSIS_LABEL,
        "cohort_version": destination.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "content_verified",
        "reason": "Original target chain and residue identity/order cannot be recovered for three failed samples.",
        "scope": "Sensitivity analysis only; not an exactly repaired full benchmark.",
        "mapping_verification": "Existing sample fields unchanged; no new PDB mappings claimed verified.",
        "participating_arms": ["OT", "no-OT"],
        "arm_policy": "Both arms must use these same dataset hashes, grouped folds, and external sample order.",
        "external_set": "Test287.pkl",
        "excluded_secondary_sets": {
            "Test60.pkl": "Overlap-contaminated relative to Train335",
            "TestB25.pkl": "Overlap-contaminated relative to Train335",
            "Test70.pkl": "Overlap-contaminated relative to Train335",
            "TestUB25.pkl": "Not in this primary comparison; partner coverage and cross-side overlap limitations",
        },
        "filename_note": "Legacy filenames retained for loader compatibility; actual filtered counts recorded below.",
        "original_pickles": originals, "datasets": {}, "split_manifests": {},
        "builder": file_record(__file__),
    }
    for filename, spec in COHORT_SPEC.items():
        samples, record = load_dataset_with_provenance(source / filename)
        if record["sha256"] != originals[filename]["sha256"]:
            raise RuntimeError(f"Source changed after hash preservation: {filename}")
        filtered, retained = select_samples(samples, spec["exclude"], spec["original_count"])
        validate_retained(filtered)
        excluded = [{
            "original_index": index, "complex_code": code,
            "content_sha256": content_hash(samples[index]),
            "target_plm_failures": target_failures(samples[index]),
            "partner_encoder_failures": partner_failures(samples[index]),
            "reason": "No declared target chain or verifiable original residue order; preparation failed",
        } for index, code in spec["exclude"].items()]
        rows = [{"filtered_index": index, "original_index": original_index,
                 "complex_code": str(sample["complex_code"]),
                 "sample_content_sha256": content_hash(sample)}
                for index, (original_index, sample) in enumerate(zip(retained, filtered))]
        output_path = destination / filename
        with output_path.open("xb") as stream:
            pickle.dump(filtered, stream, protocol=pickle.HIGHEST_PROTOCOL)
        # Release large source arrays before checking the on-disk derivative.
        del samples, filtered
        gc.collect()
        restored, derived_record = load_dataset_with_provenance(output_path)
        if [content_hash(sample) for sample in restored] != [row["sample_content_sha256"] for row in rows]:
            raise RuntimeError(f"Round-trip changed a retained sample: {filename}")
        manifest["datasets"][filename] = {
            **derived_record, "original_sample_count": spec["original_count"],
            "excluded_samples": excluded, "sample_index_map": rows,
            "retained_sample_content_unchanged": True,
            "mapping_verified_true": sum(s.get("pdb_mapping_verified") is True for s in restored),
        }
        if filename == "Train335.pkl":
            codes = [str(sample["complex_code"]).strip().upper() for sample in restored]
            if any(len(code) != 4 or not code.isalnum() for code in codes):
                raise ValueError("This cohort requires bare PDB complex codes; revise grouping explicitly")
            for seed in seeds:
                folds = make_cv_folds(restored, seed, num_folds, grouped=True, group_key="complex_code")
                validate_folds(restored, folds)
                split_name = f"grouped_folds_seed{seed}.json"
                split = {
                    "schema_version": 1, "analysis_label": ANALYSIS_LABEL,
                    "train_sha256": derived_record["sha256"], "seed": seed,
                    "num_folds": num_folds, "group_key": "complex_code",
                    "index_convention": "zero-based indices into filtered Train335.pkl",
                    "complex_count": len(set(codes)), "sample_count": len(restored),
                    "applies_to_arms": ["OT", "no-OT"],
                    "folds": cv_split_manifest(restored, folds),
                    "split_code": file_record(Path(__file__).with_name("CROSS5FOLD_multi_test.py")),
                }
                write_json_new(destination / split_name, split)
                write_json_new(reports / split_name, split)
                manifest["split_manifests"][str(seed)] = file_record(destination / split_name)
        del restored
        gc.collect()
        print(f"Verified {filename}: {derived_record['sample_count']} retained samples", flush=True)

    # Re-hash every original, including secondary sets, after all derived writes.
    after = {name: file_record(source / name) for name in originals}
    if after != originals:
        raise RuntimeError("An original pickle changed during cohort construction")
    manifest["all_original_pickle_hashes_unchanged"] = True
    write_json_new(destination / "cohort_manifest.json", manifest)
    write_json_new(reports / "cohort_manifest.json", manifest)
    incomplete.unlink()
    print(f"Built {ANALYSIS_LABEL}: {destination}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="data/geo")
    parser.add_argument("--output-dir", default="data/geo_filtered_v1")
    parser.add_argument("--report-dir", default="research_review/data_repair_v1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2101])
    parser.add_argument("--num-folds", type=int, default=5)
    args = parser.parse_args()
    build_cohort(args.source_dir, args.output_dir, args.report_dir, args.seeds, args.num_folds)


if __name__ == "__main__":
    main()
