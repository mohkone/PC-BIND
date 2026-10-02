"""Reproducible file, split, and residue identity for future research runs."""
import hashlib
import importlib.metadata
import os
import pickle
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _hash_stream(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def file_record(path):
    resolved = Path(path).resolve(strict=True)
    with resolved.open("rb") as stream:
        before = os.fstat(stream.fileno())
        digest = _hash_stream(stream)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"File changed while computing provenance: {resolved}")
    return {"path": str(resolved), "sha256": digest, "size_bytes": after.st_size}


def load_dataset_with_provenance(path):
    """Hash and load the same open, trusted pickle file; reject concurrent edits."""
    resolved = Path(path).resolve(strict=True)
    with resolved.open("rb") as stream:
        before = os.fstat(stream.fileno())
        digest = _hash_stream(stream)
        stream.seek(0)
        data = pickle.load(stream)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Dataset changed while loading: {resolved}")
    return data, {"path": str(resolved), "sha256": digest, "size_bytes": after.st_size,
                  "sample_count": len(data)}


def cv_split_manifest(proteins, folds, group_key="complex_code"):
    manifest = []
    for fold_index, val_indices in enumerate(folds):
        train_indices = np.concatenate([indices for i, indices in enumerate(folds) if i != fold_index])
        item = {"fold_index": fold_index}
        for split, indices in [("train", train_indices), ("val", val_indices)]:
            item[f"{split}_indices"] = [int(index) for index in indices]
            item[f"{split}_complex_ids"] = [str(proteins[int(index)].get("complex_code", "")) for index in indices]
            item[f"{split}_group_ids"] = [str(proteins[int(index)].get(group_key, "")).strip().upper() for index in indices]
        manifest.append(item)
    return manifest


def build_run_provenance(datasets, code_paths, cv_splits, device, scheduled_folds):
    import torch

    versions = {}
    for package in ["numpy", "torch", "scikit-learn", "tqdm", "transformers"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    runtime = {
        "python": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "packages": versions, "device": str(device),
        "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "torch_num_threads": torch.get_num_threads(),
    }
    if torch.cuda.is_available():
        runtime["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {
        "schema_version": 1, "provenance_id": str(uuid.uuid4()),
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "index_convention": "zero_based; residue_index is local to source sample, not PDB numbering",
        "datasets": datasets, "code_files": [file_record(path) for path in code_paths],
        "runtime": runtime, "cv_splits": cv_splits,
        "scheduled_fold_indices": list(range(scheduled_folds)), "completed_fold_indices": [],
    }


def prediction_identity(proteins, sample_indices=None, fold_index=None, expected_labels=None):
    """Identity follows loader sample order and within-sample label order exactly."""
    indices = list(range(len(proteins))) if sample_indices is None else [int(i) for i in sample_indices]
    lengths = [len(proteins[index]["label"]) for index in indices]
    sample_index = np.repeat(np.asarray(indices, dtype=np.int64), lengths)
    residue_index = np.concatenate([np.arange(length, dtype=np.int64) for length in lengths]) if lengths else np.empty(0, dtype=np.int64)
    complex_id = np.repeat(np.asarray([str(proteins[index].get("complex_code", "")) for index in indices], dtype=str), lengths)
    identity = {"sample_index": sample_index, "residue_index": residue_index, "complex_id": complex_id}
    if fold_index is not None:
        identity["fold_index"] = np.full(len(sample_index), fold_index, dtype=np.int64)
    if expected_labels is not None:
        labels = np.concatenate([np.asarray(proteins[index]["label"]) for index in indices]) if indices else np.empty(0)
        if not np.array_equal(labels, np.asarray(expected_labels)):
            raise ValueError("Prediction labels do not match the declared sample/residue order")
    return identity


def prediction_source(provenance, dataset_name):
    """Scalar unicode arrays remain loadable with numpy allow_pickle=False."""
    return {
        "provenance_id": np.asarray(provenance["provenance_id"]),
        "dataset_name": np.asarray(dataset_name),
        "dataset_sha256": np.asarray(provenance["datasets"][dataset_name]["sha256"]),
        "identity_schema_version": np.asarray(1, dtype=np.int64),
    }
