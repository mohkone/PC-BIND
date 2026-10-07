"""Read-only initialization evidence for the frozen projection ablation.

The evidence hooks observe initialization without reseeding or consuming RNG.
Dry runs construct models and optimizers, but never fit, forward, or step them.
"""
import hashlib
import json
import math
from pathlib import Path
import random
import re

import numpy as np
import torch

from research_provenance import file_record


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError(f"Unsupported initialization evidence value: {type(value).__name__}")


def capture_rng_states():
    """Copy the three CPU RNG states; this operation does not draw a sample."""
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone()}


def rng_fingerprints(states):
    if set(states) != {"python", "numpy", "torch_cpu"}:
        raise ValueError("Initialization RNG states must contain exactly Python, NumPy and Torch CPU")
    python_state, numpy_state, torch_state = states["python"], states["numpy"], states["torch_cpu"]
    if (not isinstance(python_state, tuple) or len(python_state) != 3
            or python_state[0] != 3 or len(python_state[1]) != 625):
        raise ValueError("Malformed Python RNG state")
    if (not isinstance(numpy_state, tuple) or len(numpy_state) != 5
            or numpy_state[0] != "MT19937"
            or not isinstance(numpy_state[1], np.ndarray)
            or numpy_state[1].dtype != np.uint32 or numpy_state[1].shape != (624,)):
        raise ValueError("Malformed NumPy RNG state")
    if (not isinstance(torch_state, torch.Tensor) or torch_state.dtype != torch.uint8
            or torch_state.device.type != "cpu" or torch_state.ndim != 1
            or not torch_state.numel()):
        raise ValueError("Malformed Torch CPU RNG state")
    return {"python_sha256": _sha(_canonical(_json_value(python_state))),
            "numpy_sha256": _sha(_canonical(_json_value(numpy_state))),
            "torch_cpu_sha256": _sha(torch_state.numpy().tobytes())}


def tensor_snapshot(model):
    """Clone state tensors without changing the model, gradients, or RNG state."""
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _tensor_schema(name, tensor):
    if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
        raise ValueError(f"Initialization state must contain ordinary tensors: {name}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"Nonfinite initialization tensor: {name}")
    return {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "numel": tensor.numel()}


def _tensor_digest(tensors, names):
    digest = hashlib.sha256()
    for name in sorted(names):
        tensor = tensors[name]
        description = _tensor_schema(name, tensor)
        header = _canonical(description)
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def optimizer_descriptor(model, optimizer):
    names = {id(value): name for name, value in model.named_parameters()}
    seen = []
    groups = []
    for group in optimizer.param_groups:
        group_names = []
        for parameter in group["params"]:
            if id(parameter) not in names:
                raise ValueError("Optimizer parameter is absent from model parameter schema")
            group_names.append(names[id(parameter)])
        seen.extend(group_names)
        settings = {key: _json_value(value) for key, value in group.items() if key != "params"}
        _canonical(settings)
        groups.append({"parameter_names": group_names, "settings": settings})
    if len(set(seen)) != len(seen) or set(seen) != set(names.values()):
        raise ValueError("Optimizer groups must cover all model parameters exactly once")
    if optimizer.state:
        raise ValueError("Initialization evidence requires an optimizer before its first step")
    return {"class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
            "groups": groups, "state_is_empty": True}


def initialization_snapshot(model, optimizer, fold_index, training_seed, fold_seed,
                            arm, projection_iterations, rng_before, rng_after):
    state = tensor_snapshot(model)
    parameters = []
    for name, value in sorted(model.named_parameters()):
        entry = _tensor_schema(name, value)
        entry["requires_grad"] = bool(value.requires_grad)
        parameters.append(entry)
    buffers = [_tensor_schema(name, value) for name, value in sorted(model.named_buffers())]
    snapshot = {"schema_version": 1, "training_seed": training_seed, "fold_seed": fold_seed,
                "fold_index": fold_index, "arm": arm, "projection_iterations": projection_iterations,
                "parameter_schema": parameters, "buffer_schema": buffers,
                "state_schema": [_tensor_schema(name, value) for name, value in sorted(state.items())],
                "optimizer": optimizer_descriptor(model, optimizer), "model_state": state,
                "rng_before": rng_before, "rng_after": rng_after}
    _record_from_snapshot(snapshot)
    return snapshot


def _record_from_snapshot(snapshot):
    keys = ("schema_version", "training_seed", "fold_seed", "fold_index", "arm", "projection_iterations",
            "parameter_schema", "buffer_schema", "state_schema", "optimizer")
    record = {key: snapshot[key] for key in keys}
    state = snapshot["model_state"]
    actual_schema = [_tensor_schema(name, value) for name, value in sorted(state.items())]
    if actual_schema != record["state_schema"]:
        raise ValueError("Initialization snapshot tensor schema mismatch")
    for entry in record["parameter_schema"] + record["buffer_schema"]:
        schema = {key: entry[key] for key in ("name", "shape", "dtype", "numel")}
        if entry["name"] not in state or schema != _tensor_schema(entry["name"], state[entry["name"]]):
            raise ValueError("Initialization parameter or buffer schema differs from state")
    parameter_names = [entry["name"] for entry in record["parameter_schema"]]
    record.update({"parameter_count": sum(entry["numel"] for entry in record["parameter_schema"]),
                   "trainable_parameter_count": sum(entry["numel"] for entry in record["parameter_schema"]
                                                    if entry["requires_grad"]),
                   "parameter_sha256": _tensor_digest(state, parameter_names),
                   "state_sha256": _tensor_digest(state, list(state)),
                   "rng_before": rng_fingerprints(snapshot["rng_before"]),
                   "rng_after": rng_fingerprints(snapshot["rng_after"])})
    validate_initialization(record)
    return record


def capture_initialization(model, optimizer, fold_index, training_seed, fold_seed,
                           arm, projection_iterations, rng_before, rng_after):
    return _record_from_snapshot(initialization_snapshot(
        model, optimizer, fold_index, training_seed, fold_seed, arm,
        projection_iterations, rng_before, rng_after))


def validate_initialization(record, snapshot=None):
    expected_keys = {"schema_version", "training_seed", "fold_seed", "fold_index", "arm",
                     "projection_iterations", "parameter_schema", "buffer_schema", "state_schema", "optimizer",
                     "parameter_count", "trainable_parameter_count", "parameter_sha256", "state_sha256",
                     "rng_before", "rng_after"}
    if set(record) != expected_keys or record["schema_version"] != 1:
        raise ValueError("Unexpected initialization record schema")
    if (type(record["training_seed"]) is not int or record["training_seed"] not in (2102, 2103, 2104)
            or type(record["fold_seed"]) is not int or record["fold_seed"] != 2101
            or type(record["fold_index"]) is not int or record["fold_index"] not in range(5)
            or record["arm"] not in ("projected", "rowsoftmax")
            or type(record["projection_iterations"]) is not int
            or record["projection_iterations"] != {"projected": 5, "rowsoftmax": 0}[record["arm"]]):
        raise ValueError("Initialization seed, fold or projection arm mismatch")
    for key in ("parameter_schema", "buffer_schema", "state_schema"):
        entries = record[key]
        if not isinstance(entries, list) or (key != "buffer_schema" and not entries):
            raise ValueError("Initialization tensor schema must be a nonempty list")
        names = [entry.get("name") for entry in entries]
        if any(not isinstance(name, str) or not name for name in names) or names != sorted(set(names)):
            raise ValueError("Initialization tensor names must be unique and sorted")
        for entry in entries:
            entry_keys = {"name", "shape", "dtype", "numel"}
            if key == "parameter_schema":
                entry_keys.add("requires_grad")
            if set(entry) != entry_keys or not isinstance(entry["shape"], list):
                raise ValueError("Initialization tensor descriptor is malformed")
            if any(type(value) is not int or value < 0 for value in entry["shape"]):
                raise ValueError("Initialization tensor shape is malformed")
            if type(entry["numel"]) is not int or entry["numel"] != math.prod(entry["shape"]):
                raise ValueError("Initialization tensor element count differs from shape")
            if not isinstance(entry["dtype"], str) or not entry["dtype"].startswith("torch."):
                raise ValueError("Initialization tensor dtype is malformed")
            if key == "parameter_schema" and type(entry["requires_grad"]) is not bool:
                raise ValueError("Initialization trainability must be boolean")
    parameters = record["parameter_schema"]
    if (type(record["parameter_count"]) is not int or type(record["trainable_parameter_count"]) is not int
            or record["parameter_count"] != sum(entry["numel"] for entry in parameters)
            or record["trainable_parameter_count"] != sum(entry["numel"] for entry in parameters
                                                        if entry["requires_grad"])):
        raise ValueError("Initialization parameter counts are inconsistent")
    state_schema = {entry["name"]: entry for entry in record["state_schema"]}
    schema_names = [entry["name"] for entry in parameters + record["buffer_schema"]]
    if len(set(schema_names)) != len(schema_names) or set(schema_names) != set(state_schema):
        raise ValueError("Initialization state must contain exactly declared parameters and buffers")
    for entry in parameters + record["buffer_schema"]:
        if {key: entry[key] for key in ("name", "shape", "dtype", "numel")} != state_schema[entry["name"]]:
            raise ValueError("Initialization parameter/buffer schema mismatch")
    optimizer = record["optimizer"]
    if (set(optimizer) != {"class", "groups", "state_is_empty"}
            or optimizer["class"] != "torch.optim.adamw.AdamW" or optimizer["state_is_empty"] is not True
            or not isinstance(optimizer["groups"], list) or not optimizer["groups"]):
        raise ValueError("Initialization optimizer must be empty-state AdamW")
    optimizer_names = []
    for group in optimizer["groups"]:
        if set(group) != {"parameter_names", "settings"} or not isinstance(group["parameter_names"], list):
            raise ValueError("Malformed initialization optimizer group")
        optimizer_names.extend(group["parameter_names"])
        _canonical(group["settings"])
    if len(set(optimizer_names)) != len(optimizer_names) or set(optimizer_names) != {entry["name"] for entry in parameters}:
        raise ValueError("Initialization optimizer groups are not a parameter partition")
    hashes = [record["parameter_sha256"], record["state_sha256"]]
    for key in ("rng_before", "rng_after"):
        if set(record[key]) != {"python_sha256", "numpy_sha256", "torch_cpu_sha256"}:
            raise ValueError("Malformed initialization RNG fingerprint")
        hashes.extend(record[key].values())
    if any(not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None for value in hashes):
        raise ValueError("Malformed initialization SHA-256 fingerprint")
    if snapshot is not None:
        if _record_from_snapshot(snapshot) != record:
            raise ValueError("Initialization record does not reproduce from saved tensors/RNG states")
    return record


def validate_initialization_pair(projected, rowsoftmax, require_initial_values=True):
    validate_initialization(projected)
    validate_initialization(rowsoftmax)
    if projected["arm"] != "projected" or rowsoftmax["arm"] != "rowsoftmax":
        raise ValueError("Initialization pair must be projected then row-softmax")
    allowed = {"arm", "projection_iterations", "parameter_sha256", "state_sha256", "rng_before", "rng_after"}
    if {key: value for key, value in projected.items() if key not in allowed} != {
            key: value for key, value in rowsoftmax.items() if key not in allowed}:
        raise ValueError("Initialization pair differs in parameter schema, trainability, counts, optimizer or seed/fold")
    equal_values = (projected["parameter_sha256"] == rowsoftmax["parameter_sha256"]
                    and projected["state_sha256"] == rowsoftmax["state_sha256"])
    equal_rng = (projected["rng_before"] == rowsoftmax["rng_before"]
                 and projected["rng_after"] == rowsoftmax["rng_after"])
    if require_initial_values and not (equal_values and equal_rng):
        raise ValueError("First-fold paired initialization tensors or RNG fingerprints differ")
    return {"fold_index": projected["fold_index"], "training_seed": projected["training_seed"],
            "schema_trainability_counts_optimizer_equal": True,
            "initial_values_equal": equal_values, "rng_fingerprints_equal": equal_rng,
            "initial_value_equality_required": bool(require_initial_values)}


def _arguments(namespace, fold_index, rng_before, rng_after):
    return (fold_index, namespace["SEED"], namespace["FOLD_SEED"], namespace["cli_args"].arm,
            namespace["TRANSPORT_SINKHORN_ITERS"], rng_before, rng_after)


def dry_run_initializations(namespace):
    """Observe five model/optimizer initializations; no fitting, output files, or reseeding."""
    if namespace["MODEL_MODE"] != "innovative_triview" or namespace["folds_to_run"] != 5:
        raise ValueError("Projection dry-run initialization requires the fixed five-fold innovative recipe")
    records = []
    for fold in range(namespace["folds_to_run"]):
        before = capture_rng_states()
        model = namespace["build_model"](namespace["in_dim"], namespace["atom_dim"], namespace["device"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,
                                     weight_decay=namespace["OPT_WEIGHT_DECAY"])
        after = capture_rng_states()
        records.append(capture_initialization(model, optimizer, *_arguments(namespace, fold, before, after)))
        del optimizer, model
    return records


def record_initialization(namespace, fold_index, model, optimizer, run_provenance,
                          output_dir, rng_before, rng_after):
    """Save exclusive initialization evidence before any optimization of this fold."""
    records = run_provenance.setdefault("initialization_records", [])
    if len(records) != fold_index:
        raise ValueError("Initialization evidence is not in original fold order")
    snapshot = initialization_snapshot(model, optimizer, *_arguments(namespace, fold_index, rng_before, rng_after))
    record = _record_from_snapshot(snapshot)
    target = Path(output_dir)
    json_path = target / f"initialization_fold{fold_index + 1}.json"
    tensor_path = target / f"initialization_fold{fold_index + 1}.pt"
    if json_path.exists() or tensor_path.exists():
        raise ValueError("Refusing to overwrite initialization evidence")
    with tensor_path.open("xb") as stream:
        torch.save(snapshot, stream)
    with json_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    entry = {"fold_index": fold_index, "record": file_record(json_path), "snapshot": file_record(tensor_path)}
    records.append(entry)
    return entry
