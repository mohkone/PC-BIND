"""Read-only, fail-closed contract for the versioned filtered sensitivity cohort."""
import hashlib
import json
import os
from pathlib import Path

from research_provenance import file_record


DECLARED_FILES = ("Train335.pkl", "Test287.pkl")
EXPECTED_COUNTS = {"Train335.pkl": 334, "Test287.pkl": 285}


def read_json_with_record(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read()
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"JSON changed while reading: {path}")
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                   "size_bytes": len(raw)}


def resolve_cohort_file(directory, name):
    if (not isinstance(name, str) or not name or Path(name).name != name
            or any(char in name for char in '/\\:') or name in {".", ".."}):
        raise ValueError(f"Invalid cohort leaf filename: {name!r}")
    path = (directory / name).resolve(strict=True)
    if path.parent != directory or not path.is_file():
        raise ValueError(f"Cohort file escapes the resolved directory: {name}")
    return path


def verify_filtered_cohort(data_dir, cohort_manifest_path, fold_manifest_path, seed):
    """Verify BOTH pickle byte streams, even during training-only development."""
    directory = Path(data_dir).resolve(strict=True)
    if not directory.is_dir() or (directory / "INCOMPLETE.txt").exists():
        raise ValueError("Filtered cohort directory is absent or construction is incomplete")
    if type(seed) is not int or seed != 2101:
        raise ValueError("geo_filtered_v1 has a fixed seed-2101 fold manifest")
    manifest_path = Path(cohort_manifest_path).resolve(strict=True)
    fold_path = Path(fold_manifest_path).resolve(strict=True)
    if manifest_path != resolve_cohort_file(directory, "cohort_manifest.json"):
        raise ValueError("Cohort manifest must be the selected directory's cohort_manifest.json")
    if fold_path != resolve_cohort_file(directory, "grouped_folds_seed2101.json"):
        raise ValueError("Fold manifest must be the selected directory's fixed seed-2101 manifest")
    cohort, manifest_record = read_json_with_record(manifest_path)
    if (cohort.get("schema_version") != 1 or cohort.get("cohort_version") != "geo_filtered_v1"
            or cohort.get("analysis_label") != "filtered-cohort analysis"
            or cohort.get("status") != "content_verified"
            or cohort.get("external_set") != "Test287.pkl"):
        raise ValueError("Invalid or incomplete filtered-cohort identity")
    declarations = cohort.get("datasets")
    if not isinstance(declarations, dict) or set(declarations) != set(DECLARED_FILES):
        raise ValueError("Filtered cohort must declare exactly Train335.pkl and Test287.pkl")
    records = {}
    for name in DECLARED_FILES:
        expected = declarations[name]
        if not isinstance(expected, dict) or expected.get("sample_count") != EXPECTED_COUNTS[name]:
            raise ValueError(f"Unexpected filtered sample count: {name}")
        actual = file_record(resolve_cohort_file(directory, name))
        if not isinstance(expected.get("path"), str) or Path(expected["path"]).resolve() != Path(actual["path"]):
            raise ValueError(f"Dataset path differs from the bound cohort directory: {name}")
        if actual["sha256"] != expected.get("sha256") or actual["size_bytes"] != expected.get("size_bytes"):
            raise ValueError(f"Dataset hash/size differs from cohort manifest: {name}")
        records[name] = {**actual, "sample_count": expected["sample_count"]}
    folds, fold_record = read_json_with_record(fold_path)
    expected_fold = cohort.get("split_manifests", {}).get(str(seed), {})
    if not isinstance(expected_fold.get("path"), str) or Path(expected_fold["path"]).resolve() != fold_path:
        raise ValueError("Grouped fold path differs from the bound cohort directory")
    if (fold_record["sha256"] != expected_fold.get("sha256")
            or fold_record["size_bytes"] != expected_fold.get("size_bytes")):
        raise ValueError("Grouped fold manifest hash/size differs from cohort manifest")
    if (folds.get("schema_version") != 1 or folds.get("analysis_label") != "filtered-cohort analysis"
            or type(folds.get("seed")) is not int
            or folds["seed"] != seed or folds.get("num_folds") != 5
            or folds.get("group_key") != "complex_code"
            or folds.get("sample_count") != EXPECTED_COUNTS["Train335.pkl"]
            or folds.get("train_sha256") != records["Train335.pkl"]["sha256"]):
        raise ValueError("Fixed fold metadata does not describe the selected training cohort")
    if file_record(manifest_path) != manifest_record or file_record(fold_path) != fold_record:
        raise RuntimeError("A manifest changed during cohort verification")
    provenance = {
        "resolved_data_dir": str(directory), "analysis_label": cohort["analysis_label"],
        "cohort_version": cohort["cohort_version"], "cohort_manifest": manifest_record,
        "declared_files": list(DECLARED_FILES), "datasets": records,
        "grouped_fold_manifest": fold_record, "seed": seed, "hashes_verified": True,
    }
    return provenance, cohort, folds


def validate_fixed_folds(proteins, folds_json, seed, train_sha256, group_key="complex_code"):
    """Validate, then return saved splits unchanged; never generate or reorder them."""
    count = len(proteins)
    folds = folds_json.get("folds")
    if (folds_json.get("schema_version") != 1 or type(folds_json.get("seed")) is not int
            or folds_json["seed"] != seed or folds_json.get("train_sha256") != train_sha256
            or folds_json.get("sample_count") != count or folds_json.get("group_key") != group_key
            or not isinstance(folds, list) or not 2 <= len(folds) <= count
            or folds_json.get("num_folds") != len(folds)):
        raise ValueError("Fixed fold metadata does not match the loaded training population")
    complex_ids = [str(sample.get("complex_code", "")) for sample in proteins]
    groups = [str(sample.get(group_key, "")).strip().upper() for sample in proteins]
    if any(group in {"", "NONE", "NAN", "NULL"} for group in groups):
        raise ValueError("Fixed folds require nonmissing complex group identifiers")
    if folds_json.get("complex_count") != len(set(groups)):
        raise ValueError("Fixed fold complex count does not match the loaded training population")
    all_indices = set(range(count))
    validation_indices = []
    group_owner = {}
    for number, fold in enumerate(folds):
        if not isinstance(fold, dict) or type(fold.get("fold_index")) is not int or fold["fold_index"] != number:
            raise ValueError("Fixed fold indices must be sequential and zero-based")
        index_sets = {}
        for split in ("train", "val"):
            indices = fold.get(f"{split}_indices")
            if (not isinstance(indices, list) or not indices
                    or any(type(index) is not int or not 0 <= index < count for index in indices)
                    or len(set(indices)) != len(indices)):
                raise ValueError(f"Invalid or duplicate {split} indices in fixed fold {number}")
            if fold.get(f"{split}_complex_ids") != [complex_ids[index] for index in indices]:
                raise ValueError(f"Complex identity/order mismatch in fixed fold {number} {split}")
            if fold.get(f"{split}_group_ids") != [groups[index] for index in indices]:
                raise ValueError(f"Group identity/order mismatch in fixed fold {number} {split}")
            index_sets[split] = set(indices)
        if index_sets["train"] & index_sets["val"] or index_sets["train"] | index_sets["val"] != all_indices:
            raise ValueError(f"Fixed fold {number} does not partition the training population")
        train_groups = {groups[index] for index in fold["train_indices"]}
        val_groups = {groups[index] for index in fold["val_indices"]}
        if train_groups & val_groups:
            raise ValueError(f"Complex crosses training and validation in fixed fold {number}")
        for group in val_groups:
            if group in group_owner:
                raise ValueError(f"Complex {group} validates in multiple fixed folds")
            group_owner[group] = number
        validation_indices.extend(fold["val_indices"])
    if sorted(validation_indices) != list(range(count)):
        raise ValueError("Fixed validation folds must cover every sample exactly once")
    return folds
