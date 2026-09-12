
import os
import math
import pickle
import copy
import random
import json
import csv
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from metrics import *   # compute_auc_roc, compute_auc_pr, compute_acc, compute_performance


def env_flag(name, default="1"):
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}


SEED = int(os.environ.get("PPI_SEED", "1234"))
MODEL_MODE = "innovative_triview"  # "innovative_triview" or "checkpoint_finetune"
CHECKPOINT_PATH = os.path.join("data", "bib_triview_best.pt")
ADD_REVERSE_EDGES = MODEL_MODE != "checkpoint_finetune"
EVAL_SMOOTH_STEPS = 2
EVAL_SMOOTH_ALPHA = 0.75
OUTPUT_DIR = os.environ.get("PPI_OUTPUT_DIR", "outputs")
TOP_K_CHECKPOINTS = 3
GEO_DATA_DIR = os.path.join("data", "geo")
TEST_PKL_FILES = [
    name.strip()
    for name in os.environ.get(
        "PPI_TEST_SETS",
        "Test60.pkl,Test287.pkl,Test70.pkl,TestB25.pkl,TestUB25.pkl",
    ).split(",")
    if name.strip()
]
USE_ATOM_GEO_EDGES = False
GEOMETRY_FEATURE_MODE = "residue_local_frame_rbf_v1"
SURFACE_FEATURE_MODE = "surface_exposure_v1"
SURFACE_FEATURE_DIM = 14
SEQUENCE_FEATURE_MODE = "residue_biochemical_v1"
SEQUENCE_FEATURE_DIM = 39
USE_PLM_FEATURES = env_flag("PPI_USE_PLM_FEATURES", "1") and not env_flag("PPI_DISABLE_PLM", "0")
USE_AUX_PLM_FEATURES = env_flag("PPI_USE_AUX_PLM_FEATURES", "1") and not env_flag("PPI_DISABLE_AUX_PLM", "0")
PLM_FEATURE_MODE = "unknown_plm"
PLM_FEATURE_DIM = int(os.environ.get("PPI_PLM_DIM", "320")) if USE_PLM_FEATURES else 0
PLM_FEATURE_KEY = os.environ.get("PPI_PLM_FEATURE_KEY", "residue_plm_embedding")
PLM_MODEL_KEY = os.environ.get("PPI_PLM_MODEL_KEY", "residue_plm_model")
AUX_PLM_FEATURE_MODE = "none"
AUX_PLM_FEATURE_DIM = int(os.environ.get("PPI_AUX_PLM_DIM", "0")) if USE_AUX_PLM_FEATURES else 0
AUX_PLM_FEATURE_KEY = os.environ.get("PPI_AUX_PLM_FEATURE_KEY", "residue_plm_embedding_8m")
AUX_PLM_MODEL_KEY = os.environ.get("PPI_AUX_PLM_MODEL_KEY", "residue_plm_model_8m")
PATCH_LABEL_DISTRIBUTION = os.environ.get("PPI_PATCH_LABEL_DISTRIBUTION", "0").lower() in {"1", "true", "yes", "on"}
PATCH_LABEL_STEPS = int(os.environ.get("PPI_PATCH_LABEL_STEPS", "2"))
PATCH_LABEL_ALPHA = float(os.environ.get("PPI_PATCH_LABEL_ALPHA", "0.45"))
PATCH_LABEL_MAX_UNLABELED = float(os.environ.get("PPI_PATCH_LABEL_MAX_UNLABELED", "0.45"))
PARTNER_CONTACT_AUX = env_flag("PPI_PARTNER_CONTACT_AUX", "1")
PARTNER_CONTACT_AUX_WEIGHT = float(os.environ.get("PPI_PARTNER_CONTACT_AUX_WEIGHT", "0.12"))
PARTNER_CONTACT_POS_WEIGHT = 3.0
PAIR_CONTACT_LOSS = env_flag("PPI_PAIR_CONTACT_LOSS", "0")
PAIR_CONTACT_LOSS_WEIGHT = float(os.environ.get("PPI_PAIR_CONTACT_LOSS_WEIGHT", "0.06"))
PAIR_CONTACT_POS_WEIGHT = float(os.environ.get("PPI_PAIR_CONTACT_POS_WEIGHT", "8.0"))
PAIR_CONTACT_HEAD = os.environ.get("PPI_PAIR_CONTACT_HEAD", "attention").lower()
PAIR_CONTACT_WARMUP_EPOCHS = int(os.environ.get("PPI_PAIR_CONTACT_WARMUP_EPOCHS", "0"))
PAIR_MARGINAL_CONSISTENCY = env_flag("PPI_PAIR_MARGINAL_CONSISTENCY", "0")
PAIR_MARGINAL_CONSISTENCY_WEIGHT = float(
    os.environ.get("PPI_PAIR_MARGINAL_CONSISTENCY_WEIGHT", "0.02")
)
PAIR_MARGINAL_CONSISTENCY_WARMUP_EPOCHS = int(
    os.environ.get("PPI_PAIR_MARGINAL_CONSISTENCY_WARMUP_EPOCHS", "5")
)
PAIR_MARGINAL_CONTRAST = env_flag("PPI_PAIR_MARGINAL_CONTRAST", "0")
PAIR_MARGINAL_CONTRAST_WEIGHT = float(
    os.environ.get("PPI_PAIR_MARGINAL_CONTRAST_WEIGHT", "0.02")
)
PAIR_MARGINAL_CONTRAST_MARGIN = float(
    os.environ.get("PPI_PAIR_MARGINAL_CONTRAST_MARGIN", "0.03")
)
PAIR_MARGINAL_CONTRAST_WARMUP_EPOCHS = int(
    os.environ.get("PPI_PAIR_MARGINAL_CONTRAST_WARMUP_EPOCHS", "5")
)
PAIR_CONTACT_CONTRAST = env_flag("PPI_PAIR_CONTACT_CONTRAST", "0")
PAIR_CONTACT_CONTRAST_WEIGHT = float(
    os.environ.get("PPI_PAIR_CONTACT_CONTRAST_WEIGHT", "0.005")
)
PAIR_CONTACT_CONTRAST_MARGIN = float(
    os.environ.get("PPI_PAIR_CONTACT_CONTRAST_MARGIN", "0.03")
)
PAIR_CONTACT_CONTRAST_HARD_K = max(
    1,
    int(os.environ.get("PPI_PAIR_CONTACT_CONTRAST_HARD_K", "10")),
)
PAIR_CONTACT_CONTRAST_WARMUP_EPOCHS = int(
    os.environ.get("PPI_PAIR_CONTACT_CONTRAST_WARMUP_EPOCHS", "5")
)
TWO_HEAD_BINDING = os.environ.get("PPI_TWO_HEAD_BINDING", "1") != "0"
TWO_HEAD_RANK_LOSS_WEIGHT = float(os.environ.get("PPI_TWO_HEAD_RANK_LOSS_WEIGHT", "0.55"))
TWO_HEAD_CONSISTENCY_WEIGHT = float(os.environ.get("PPI_TWO_HEAD_CONSISTENCY_WEIGHT", "0.04"))
TWO_HEAD_RANK_FUSION = float(os.environ.get("PPI_TWO_HEAD_RANK_FUSION", "0.35"))
TWO_HEAD_RANK_WARMUP_EPOCHS = int(os.environ.get("PPI_TWO_HEAD_RANK_WARMUP_EPOCHS", "0"))
VIEW_LOGIT_FUSION = float(os.environ.get("PPI_VIEW_LOGIT_FUSION", "0.0"))
MODEL_DROPOUT = float(os.environ.get("PPI_MODEL_DROPOUT", "0.20"))
MODEL_EDGE_DROPOUT = float(os.environ.get("PPI_MODEL_EDGE_DROPOUT", "0.04"))
OPT_WEIGHT_DECAY = float(os.environ.get("PPI_WEIGHT_DECAY", "2e-4"))
CHECKPOINT_SELECTION_METRIC = os.environ.get("PPI_CHECKPOINT_SELECTION_METRIC", "mcc").lower()
CHECKPOINT_SELECTION_AUPR_WEIGHT = float(os.environ.get("PPI_CHECKPOINT_SELECTION_AUPR_WEIGHT", "0.35"))
PARTNER_CONDITIONING = env_flag("PPI_PARTNER_CONDITIONING", "0")
PARTNER_TOP_K = int(os.environ.get("PPI_PARTNER_TOP_K", "16"))
PARTNER_DIRECT_FUSION = env_flag("PPI_PARTNER_DIRECT_FUSION", "0")
PARTNER_CONTRAST = env_flag("PPI_PARTNER_CONTRAST", "0")
PARTNER_CONTRAST_WEIGHT = float(os.environ.get("PPI_PARTNER_CONTRAST_WEIGHT", "0.03"))
PARTNER_CONTRAST_MARGIN = float(os.environ.get("PPI_PARTNER_CONTRAST_MARGIN", "0.05"))
PARTNER_CONTRAST_WARMUP_EPOCHS = int(os.environ.get("PPI_PARTNER_CONTRAST_WARMUP_EPOCHS", "0"))
PARTNER_LOGIT_MODE = os.environ.get("PPI_PARTNER_LOGIT_MODE", "standard").lower()
PARTNER_DELTA_SCALE = float(os.environ.get("PPI_PARTNER_DELTA_SCALE", "1.0"))
PARTNER_RESIDUE_ENCODER = env_flag("PPI_PARTNER_RESIDUE_ENCODER", "0")
PARTNER_ENCODER_LAYERS = int(os.environ.get("PPI_PARTNER_ENCODER_LAYERS", "1"))
PARTNER_TARGET_FUSION = float(os.environ.get("PPI_PARTNER_TARGET_FUSION", "0.25"))
PARTNER_PLM_FEATURE_KEY = os.environ.get(
    "PPI_PARTNER_PLM_FEATURE_KEY",
    "partner_residue_plm_embedding",
)
PARTNER_AUX_PLM_FEATURE_KEY = os.environ.get(
    "PPI_PARTNER_AUX_PLM_FEATURE_KEY",
    "partner_residue_plm_embedding_8m",
)
BATCH_SIZE = int(os.environ.get("PPI_BATCH_SIZE", "1"))
MAX_FOLDS = max(1, min(5, int(os.environ.get("PPI_MAX_FOLDS", "5"))))
SKIP_TEST_EVAL = env_flag("PPI_SKIP_TEST_EVAL", "0")
PARTNER_FEATURE_DIM = SURFACE_FEATURE_DIM + SEQUENCE_FEATURE_DIM
GROUPED_CV = env_flag("PPI_GROUPED_CV", "0")
CV_GROUP_KEY = os.environ.get("PPI_CV_GROUP_KEY", "complex_code")


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_cv_folds(protein_list, seed, num_folds=5, grouped=False, group_key="complex_code"):
    """Return validation indices while optionally keeping each complex in one fold."""
    rng = np.random.RandomState(seed)
    num_samples = len(protein_list)
    if not grouped:
        indices = rng.permutation(num_samples)
        fold_ids = np.arange(num_samples) % num_folds
        return [indices[fold_ids == fold] for fold in range(num_folds)]

    grouped_indices = {}
    for sample_idx, protein in enumerate(protein_list):
        value = str(protein.get(group_key, "")).strip().upper()
        group_id = value if value else f"__sample_{sample_idx}"
        grouped_indices.setdefault(group_id, []).append(sample_idx)

    group_ids = np.asarray(list(grouped_indices), dtype=object)
    rng.shuffle(group_ids)
    folds = [[] for _ in range(num_folds)]
    fold_sizes = np.zeros((num_folds,), dtype=np.int64)
    for group_id in group_ids:
        members = grouped_indices[str(group_id)]
        smallest_fold = int(np.argmin(fold_sizes))
        folds[smallest_fold].extend(members)
        fold_sizes[smallest_fold] += len(members)

    return [np.asarray(sorted(indices), dtype=np.int64) for indices in folds]


def fit_feature_stats(protein_list, eps=1e-6):
    """Fit node-feature normalization on the training split only."""
    total = 0
    feat_sum = None
    feat_sq_sum = None

    for p in protein_list:
        x = p["residue_graph_node"].astype(np.float64, copy=False)
        if feat_sum is None:
            feat_sum = np.zeros(x.shape[1], dtype=np.float64)
            feat_sq_sum = np.zeros(x.shape[1], dtype=np.float64)
        feat_sum += x.sum(axis=0)
        feat_sq_sum += np.square(x).sum(axis=0)
        total += x.shape[0]

    mean = feat_sum / total
    var = np.maximum(feat_sq_sum / total - np.square(mean), eps)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def fit_atom_feature_stats(protein_list, eps=1e-6):
    """Fit atom-feature normalization on the training split only."""
    total = 0
    feat_sum = None
    feat_sq_sum = None

    for p in protein_list:
        x = p["atom_graph_node"].astype(np.float64, copy=False)
        if feat_sum is None:
            feat_sum = np.zeros(x.shape[1], dtype=np.float64)
            feat_sq_sum = np.zeros(x.shape[1], dtype=np.float64)
        feat_sum += x.sum(axis=0)
        feat_sq_sum += np.square(x).sum(axis=0)
        total += x.shape[0]

    mean = feat_sum / total
    var = np.maximum(feat_sq_sum / total - np.square(mean), eps)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def get_surface_features(protein, dim=SURFACE_FEATURE_DIM):
    features = protein.get("residue_surface_features")
    n_res = protein["residue_graph_node"].shape[0]
    if features is None:
        return np.zeros((n_res, dim), dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != n_res:
        return np.zeros((n_res, dim), dtype=np.float32)
    if features.shape[1] == dim:
        return features
    out = np.zeros((n_res, dim), dtype=np.float32)
    width = min(dim, features.shape[1])
    out[:, :width] = features[:, :width]
    return out


def fit_surface_feature_stats(protein_list, eps=1e-6):
    """Fit surface-feature normalization on the training split only."""
    total = 0
    feat_sum = np.zeros(SURFACE_FEATURE_DIM, dtype=np.float64)
    feat_sq_sum = np.zeros(SURFACE_FEATURE_DIM, dtype=np.float64)

    for p in protein_list:
        x = get_surface_features(p).astype(np.float64, copy=False)
        feat_sum += x.sum(axis=0)
        feat_sq_sum += np.square(x).sum(axis=0)
        total += x.shape[0]

    mean = feat_sum / max(total, 1)
    var = np.maximum(feat_sq_sum / max(total, 1) - np.square(mean), eps)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def get_sequence_features(protein, dim=SEQUENCE_FEATURE_DIM):
    features = protein.get("residue_sequence_features")
    n_res = protein["residue_graph_node"].shape[0]
    if features is None:
        return np.zeros((n_res, dim), dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != n_res:
        return np.zeros((n_res, dim), dtype=np.float32)
    if features.shape[1] == dim:
        return features
    out = np.zeros((n_res, dim), dtype=np.float32)
    width = min(dim, features.shape[1])
    out[:, :width] = features[:, :width]
    return out


def get_partner_surface_features(protein, dim=SURFACE_FEATURE_DIM):
    features = protein.get("partner_residue_surface_features")
    if features is None:
        return np.zeros((0, dim), dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] == 0:
        return np.zeros((0, dim), dtype=np.float32)
    if features.shape[1] == dim:
        return features
    out = np.zeros((features.shape[0], dim), dtype=np.float32)
    width = min(dim, features.shape[1])
    out[:, :width] = features[:, :width]
    return out


def get_partner_sequence_features(protein, dim=SEQUENCE_FEATURE_DIM):
    features = protein.get("partner_residue_sequence_features")
    if features is None:
        return np.zeros((0, dim), dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] == 0:
        return np.zeros((0, dim), dtype=np.float32)
    if features.shape[1] == dim:
        return features
    out = np.zeros((features.shape[0], dim), dtype=np.float32)
    width = min(dim, features.shape[1])
    out[:, :width] = features[:, :width]
    return out


def get_partner_plm_features(protein, dim, key):
    """Return aligned partner PLM features without silently changing residue count."""
    surface = get_partner_surface_features(protein)
    n_res = int(surface.shape[0])
    if dim <= 0:
        return np.zeros((n_res, 0), dtype=np.float32)
    features = protein.get(key)
    if features is None:
        return np.zeros((n_res, dim), dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != n_res:
        return np.zeros((n_res, dim), dtype=np.float32)
    if features.shape[1] == dim:
        return features
    out = np.zeros((n_res, dim), dtype=np.float32)
    width = min(dim, features.shape[1])
    out[:, :width] = features[:, :width]
    return out


def fit_sequence_feature_stats(protein_list, eps=1e-6):
    """Fit residue biochemical-feature normalization on the training split only."""
    total = 0
    feat_sum = np.zeros(SEQUENCE_FEATURE_DIM, dtype=np.float64)
    feat_sq_sum = np.zeros(SEQUENCE_FEATURE_DIM, dtype=np.float64)

    for p in protein_list:
        x = get_sequence_features(p).astype(np.float64, copy=False)
        feat_sum += x.sum(axis=0)
        feat_sq_sum += np.square(x).sum(axis=0)
        total += x.shape[0]

    mean = feat_sum / max(total, 1)
    var = np.maximum(feat_sq_sum / max(total, 1) - np.square(mean), eps)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def plm_model_to_mode(model_name):
    if not model_name or model_name == "missing_sequence":
        return "missing_sequence"
    base = os.path.basename(str(model_name)).replace("-", "_")
    return f"{base}_v1"


def infer_plm_feature_config(protein_lists, feature_key=PLM_FEATURE_KEY, model_key=PLM_MODEL_KEY, default_dim=None):
    """Infer PLM dimension/model from loaded data; env vars may override."""
    all_lists = []
    for item in protein_lists:
        if isinstance(item, dict):
            all_lists.extend(item.values())
        else:
            all_lists.append(item)

    inferred_dim = None
    inferred_mode = None
    for protein_list in all_lists:
        for protein in protein_list:
            features = protein.get(feature_key)
            if features is None:
                continue
            features = np.asarray(features)
            if features.ndim == 2 and features.shape[1] > 0:
                inferred_dim = int(features.shape[1])
                model_name = protein.get(model_key)
                if model_name and model_name != "missing_sequence":
                    inferred_mode = plm_model_to_mode(model_name)
                break
        if inferred_dim is not None:
            break

    dim = int(inferred_dim if inferred_dim is not None else (default_dim or 0))
    mode = inferred_mode or "none"
    return mode, dim


def configure_plm_feature_config(protein_lists):
    global PLM_FEATURE_MODE, PLM_FEATURE_DIM, AUX_PLM_FEATURE_MODE, AUX_PLM_FEATURE_DIM
    if USE_PLM_FEATURES:
        main_mode, main_dim = infer_plm_feature_config(
            protein_lists,
            feature_key=PLM_FEATURE_KEY,
            model_key=PLM_MODEL_KEY,
            default_dim=PLM_FEATURE_DIM,
        )
        PLM_FEATURE_MODE = os.environ.get("PPI_PLM_FEATURE_MODE", main_mode)
        PLM_FEATURE_DIM = int(os.environ.get("PPI_PLM_DIM", main_dim))
    else:
        PLM_FEATURE_MODE = os.environ.get("PPI_PLM_FEATURE_MODE", "disabled")
        PLM_FEATURE_DIM = 0

    if USE_AUX_PLM_FEATURES:
        aux_mode, aux_dim = infer_plm_feature_config(
            protein_lists,
            feature_key=AUX_PLM_FEATURE_KEY,
            model_key=AUX_PLM_MODEL_KEY,
            default_dim=0,
        )
        AUX_PLM_FEATURE_MODE = os.environ.get("PPI_AUX_PLM_FEATURE_MODE", aux_mode)
        AUX_PLM_FEATURE_DIM = int(os.environ.get("PPI_AUX_PLM_DIM", aux_dim))
    else:
        AUX_PLM_FEATURE_MODE = os.environ.get("PPI_AUX_PLM_FEATURE_MODE", "disabled")
        AUX_PLM_FEATURE_DIM = 0
    return PLM_FEATURE_MODE, PLM_FEATURE_DIM


def get_plm_features(protein, dim=None, key=PLM_FEATURE_KEY):
    if dim is None:
        dim = PLM_FEATURE_DIM
    features = protein.get(key)
    n_res = protein["residue_graph_node"].shape[0]
    if features is None:
        return np.zeros((n_res, dim), dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != n_res:
        return np.zeros((n_res, dim), dtype=np.float32)
    if features.shape[1] == dim:
        return features
    out = np.zeros((n_res, dim), dtype=np.float32)
    width = min(dim, features.shape[1])
    out[:, :width] = features[:, :width]
    return out


def fit_plm_feature_stats(protein_list, dim=None, key=PLM_FEATURE_KEY, eps=1e-6):
    """Fit PLM embedding normalization on the training split only."""
    if dim is None:
        dim = PLM_FEATURE_DIM
    total = 0
    feat_sum = np.zeros(dim, dtype=np.float64)
    feat_sq_sum = np.zeros(dim, dtype=np.float64)

    for p in protein_list:
        x = get_plm_features(p, dim=dim, key=key).astype(np.float64, copy=False)
        feat_sum += x.sum(axis=0)
        feat_sq_sum += np.square(x).sum(axis=0)
        total += x.shape[0]

    mean = feat_sum / max(total, 1)
    var = np.maximum(feat_sq_sum / max(total, 1) - np.square(mean), eps)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def compute_pos_weight(protein_list):
    labels = np.concatenate([p["label"].reshape(-1) for p in protein_list])
    positives = float(labels.sum())
    negatives = float(labels.size - positives)
    return negatives / max(positives, 1.0)


def global_mean_max_pool(x, batch):
    """Return per-node graph context from mean and max pooled graph embeddings."""
    num_graphs = int(batch.max().item()) + 1
    pooled = []

    for gid in range(num_graphs):
        idx = (batch == gid).nonzero(as_tuple=False).squeeze(-1)
        x_g = x[idx]
        mean_g = x_g.mean(dim=0)
        max_g = x_g.max(dim=0).values
        pooled.append(torch.cat((mean_g, max_g), dim=-1))

    graph_context = torch.stack(pooled, dim=0)
    return graph_context[batch]


def atom_to_residue_pool(atom_h, atom2res, num_residues):
    res_sum = atom_h.new_zeros((num_residues, atom_h.size(-1)))
    res_count = atom_h.new_zeros((num_residues, 1))
    res_sum.index_add_(0, atom2res, atom_h)
    res_count.index_add_(0, atom2res, torch.ones((atom_h.size(0), 1), device=atom_h.device, dtype=atom_h.dtype))
    return res_sum / res_count.clamp_min(1.0)


def residue_topology_features(res_edge, atom_edge, atom2res, num_residues):
    """Derived residue descriptors from residue and atom graph topology."""
    device = atom2res.device
    dtype = torch.float32

    res_deg = torch.zeros(num_residues, device=device, dtype=dtype)
    if res_edge.numel() > 0:
        _, res_dst = res_edge
        res_deg.index_add_(0, res_dst, torch.ones_like(res_dst, dtype=dtype))

    atom_deg = torch.zeros(atom2res.size(0), device=device, dtype=dtype)
    if atom_edge.numel() > 0:
        _, atom_dst = atom_edge
        atom_deg.index_add_(0, atom_dst, torch.ones_like(atom_dst, dtype=dtype))

    atom_count = torch.zeros(num_residues, device=device, dtype=dtype)
    atom_count.index_add_(0, atom2res, torch.ones_like(atom2res, dtype=dtype))

    atom_deg_sum = torch.zeros(num_residues, device=device, dtype=dtype)
    atom_deg_sum.index_add_(0, atom2res, atom_deg)
    atom_deg_mean = atom_deg_sum / atom_count.clamp_min(1.0)

    topo = torch.stack(
        (
            torch.log1p(res_deg),
            torch.log1p(atom_count),
            torch.log1p(atom_deg_mean),
        ),
        dim=-1,
    )
    return (topo - topo.mean(dim=0, keepdim=True)) / topo.std(dim=0, keepdim=True).clamp_min(1e-6)


def smooth_node_scores(scores, edge_index, steps=EVAL_SMOOTH_STEPS, alpha=EVAL_SMOOTH_ALPHA):
    """Topology-guided score refinement for spatially clustered binding residues."""
    if steps <= 0 or edge_index.numel() == 0:
        return scores

    out = scores
    src, dst = edge_index
    for _ in range(steps):
        neigh = torch.zeros_like(out)
        deg = torch.zeros_like(out)
        neigh.index_add_(0, dst, out[src])
        deg.index_add_(0, dst, torch.ones_like(out[src]))
        neigh = neigh / deg.clamp_min(1.0)
        out = alpha * out + (1.0 - alpha) * neigh
    return out


def geometry_patch_soft_targets(targets, edge_index):
    """Diffuse positive supervision over the local interface patch during training only."""
    if (not PATCH_LABEL_DISTRIBUTION) or PATCH_LABEL_STEPS <= 0 or edge_index.numel() == 0:
        return targets

    soft = targets.float()
    frontier = targets.float()
    src, dst = edge_index
    for step in range(PATCH_LABEL_STEPS):
        neigh = torch.zeros_like(soft)
        deg = torch.zeros_like(soft)
        neigh.index_add_(0, dst, frontier[src])
        deg.index_add_(0, dst, torch.ones_like(frontier[src]))
        frontier = neigh / deg.clamp_min(1.0)
        soft = torch.maximum(soft, (PATCH_LABEL_ALPHA ** (step + 1)) * frontier)

    unlabeled = targets <= 0.5
    soft = torch.where(unlabeled, soft.clamp_max(PATCH_LABEL_MAX_UNLABELED), soft)
    return soft


def best_mcc_threshold(y_true, y_score):
    thresholds = np.arange(0.0, 1.0, 0.001)
    mccs = np.zeros_like(thresholds)
    for i, t in enumerate(thresholds):
        y_pred_bin = (y_score > t).astype("int")
        mccs[i] = matthews_corrcoef(y_true, y_pred_bin)
    best_idx = int(np.argmax(mccs))
    return float(np.round(thresholds[best_idx], 4))


def metrics_from_scores(y_true, y_score, split_name="Eval", threshold=None):
    auc_roc = compute_auc_roc(y_true, y_score)
    auc_pr = compute_auc_pr(y_true, y_score)
    t_opt = best_mcc_threshold(y_true, y_score) if threshold is None else float(threshold)

    y_pred_bin = (y_score >= t_opt).astype("int")
    acc = compute_acc(y_true, y_pred_bin)
    mcc, recall, precision, f1, sc, sp = compute_performance(y_true, y_pred_bin)

    print(f"\n===== {split_name} Results =====")
    print(f"Threshold: {t_opt:.3f}")
    print(f"AUROC: {auc_roc:.4f}")
    print(f"AUPRC: {auc_pr:.4f}")
    print(f"ACC:   {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"MCC:       {mcc:.4f}")
    print(f"Sensitivity: {sc:.4f}")
    print(f"Specificity: {sp:.4f}")

    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "t_opt": t_opt,
    }


def validation_selection_score(metrics):
    if CHECKPOINT_SELECTION_METRIC in {"aupr", "auc_pr"}:
        return metrics["auc_pr"]
    if CHECKPOINT_SELECTION_METRIC in {"mcc_aupr", "mcc+aupr", "blend"}:
        return metrics["mcc"] + CHECKPOINT_SELECTION_AUPR_WEIGHT * metrics["auc_pr"]
    if CHECKPOINT_SELECTION_METRIC == "f1":
        return metrics["f1"]
    return metrics["mcc"]


def save_json(path, obj):
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, default=convert, indent=2)


def write_metrics_csv(path, rows):
    if not rows:
        return
    fieldnames = ["group", "test_set", "acc", "precision", "recall", "f1", "mcc", "auc_roc", "auc_pr", "threshold"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalized_positive_weights(values, eps=1e-8):
    weights = np.asarray(values, dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    if float(weights.sum()) <= eps:
        weights = np.ones_like(weights, dtype=np.float64)
    return weights / weights.sum()


def rank_normalize(scores):
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(scores.size, dtype=np.float64)
    return ranks / max(scores.size - 1, 1)


def combine_fold_scores(fold_scores, mode="mean", weights=None):
    score_matrix = np.stack(fold_scores, axis=0)
    if mode == "weighted":
        weights = normalized_positive_weights(weights)
        return np.average(score_matrix, axis=0, weights=weights)
    if mode == "rank":
        ranked = np.stack([rank_normalize(scores) for scores in fold_scores], axis=0)
        return ranked.mean(axis=0)
    return score_matrix.mean(axis=0)


def clone_state_dict_cpu(model):
    return copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})


def average_state_dicts(states):
    if len(states) == 1:
        return copy.deepcopy(states[0])

    avg_state = {}
    for key in states[0].keys():
        values = [state[key] for state in states]
        if values[0].dtype.is_floating_point:
            avg_state[key] = torch.stack(values, dim=0).mean(dim=0)
        else:
            avg_state[key] = values[0].clone()
    return avg_state


def update_top_checkpoints(top_states, score, threshold, state, epoch, top_k=TOP_K_CHECKPOINTS):
    top_states.append({
        "score": float(score),
        "threshold": float(threshold),
        "state": state,
        "epoch": int(epoch),
    })
    top_states.sort(key=lambda item: item["score"], reverse=True)
    del top_states[top_k:]


def edge_softmax(logits, dst, num_nodes):
    """Stable softmax over incoming edges for each destination node."""
    max_per_dst = logits.new_full((num_nodes,), -1e9)
    if hasattr(max_per_dst, "scatter_reduce_"):
        max_per_dst.scatter_reduce_(0, dst, logits, reduce="amax", include_self=True)
    else:
        for i in range(logits.numel()):
            d = int(dst[i].item())
            max_per_dst[d] = torch.maximum(max_per_dst[d], logits[i])

    exp_logits = torch.exp(logits - max_per_dst[dst])
    denom = logits.new_zeros((num_nodes,))
    denom.index_add_(0, dst, exp_logits)
    return exp_logits / denom[dst].clamp_min(1e-8)


def rbf_encode(values, centers, gamma):
    return torch.exp(-gamma * torch.square(values.unsqueeze(-1) - centers))


def build_edge_features(x, edge_index, out_dim=32, max_distance=10.0):
    """
    Continuous edge descriptors for GEA.
    If true coordinates are absent, feature-space distance/similarity acts as a
    drop-in proxy; coordinate RBFs can replace the first block later.
    """
    if edge_index.numel() == 0:
        return x.new_zeros((0, out_dim))

    src, dst = edge_index
    h_src = x[src]
    h_dst = x[dst]
    diff = h_src - h_dst

    feat_dist = torch.norm(diff, dim=-1)
    feat_dist = feat_dist / feat_dist.detach().mean().clamp_min(1e-6)
    centers = torch.linspace(0.0, max_distance, 16, device=x.device, dtype=x.dtype)
    rbf = rbf_encode(feat_dist.clamp(max=max_distance), centers, gamma=0.5)

    cosine = F.cosine_similarity(h_src, h_dst, dim=-1).unsqueeze(-1)
    abs_diff = diff.abs()
    diff_stats = torch.stack(
        (
            abs_diff.mean(dim=-1),
            abs_diff.std(dim=-1),
            abs_diff.max(dim=-1).values,
        ),
        dim=-1,
    )

    deg = x.new_zeros((x.size(0),))
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
    degree_stats = torch.stack(
        (
            torch.log1p(deg[src]),
            torch.log1p(deg[dst]),
        ),
        dim=-1,
    )

    edge_feat = torch.cat((rbf, cosine, diff_stats, degree_stats), dim=-1)
    if edge_feat.size(-1) < out_dim:
        pad = x.new_zeros((edge_feat.size(0), out_dim - edge_feat.size(-1)))
        edge_feat = torch.cat((edge_feat, pad), dim=-1)
    return edge_feat[:, :out_dim]


def build_coordinate_edge_features(coords, edge_index, out_dim=32, max_distance=10.0, frames=None):
    """Rotation-aware edge geometry from distance and optional residue-local frames."""
    if coords is None or edge_index.numel() == 0:
        return None

    src, dst = edge_index
    vec = coords[dst] - coords[src]
    dist = torch.norm(vec, dim=-1).clamp_min(1e-6)
    unit = vec / dist.unsqueeze(-1)
    centers = torch.linspace(0.0, max_distance, 16, device=coords.device, dtype=coords.dtype)
    rbf = rbf_encode(dist.clamp(max=max_distance), centers, gamma=0.5)
    log_dist = torch.log1p(dist).unsqueeze(-1)

    if frames is not None and frames.dim() == 3 and frames.size(-1) == 3:
        f_src = frames[src]
        f_dst = frames[dst]
        dir_src = torch.bmm(f_src.transpose(1, 2), unit.unsqueeze(-1)).squeeze(-1)
        dir_dst = torch.bmm(f_dst.transpose(1, 2), (-unit).unsqueeze(-1)).squeeze(-1)
        rel_orient = torch.bmm(f_src.transpose(1, 2), f_dst).reshape(edge_index.size(1), -1)
        edge_feat = torch.cat((rbf, dir_src, dir_dst, rel_orient, log_dist), dim=-1)
    else:
        edge_feat = torch.cat((rbf, unit, log_dist), dim=-1)
    if edge_feat.size(-1) < out_dim:
        pad = coords.new_zeros((edge_feat.size(0), out_dim - edge_feat.size(-1)))
        edge_feat = torch.cat((edge_feat, pad), dim=-1)
    return edge_feat[:, :out_dim]


class WeightedFocalLoss(nn.Module):
    """Focal BCE loss for sparse positive residue labels."""
    def __init__(self, pos_weight, gamma=1.5, label_smoothing=0.02):
        super().__init__()
        self.register_buffer("pos_weight", torch.as_tensor(pos_weight, dtype=torch.float32))
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        pt = torch.exp(-bce)
        focal = torch.pow(1.0 - pt, self.gamma)
        return (focal * bce).mean()


class CompositeBindingLoss(nn.Module):
    """Focal BCE plus pairwise ranking and soft Dice terms for better MCC/AUPRC."""
    def __init__(
        self,
        pos_weight,
        gamma=1.5,
        label_smoothing=0.02,
        rank_weight=0.15,
        dice_weight=0.10,
        max_pairs=4096,
    ):
        super().__init__()
        self.focal = WeightedFocalLoss(pos_weight, gamma=gamma, label_smoothing=label_smoothing)
        self.rank_weight = rank_weight
        self.dice_weight = dice_weight
        self.max_pairs = max_pairs

    def ranking_loss(self, logits, targets):
        pos = logits[targets > 0.5]
        neg = logits[targets <= 0.5]
        if pos.numel() == 0 or neg.numel() == 0:
            return logits.new_tensor(0.0)

        if pos.numel() * neg.numel() > self.max_pairs:
            pos_idx = torch.randint(pos.numel(), (self.max_pairs,), device=logits.device)
            neg_idx = torch.randint(neg.numel(), (self.max_pairs,), device=logits.device)
            pos = pos[pos_idx]
            neg = neg[neg_idx]
            return F.softplus(neg - pos).mean()

        return F.softplus(neg.unsqueeze(0) - pos.unsqueeze(1)).mean()

    def dice_loss(self, logits, targets):
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum()
        denom = probs.sum() + targets.sum()
        return 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)

    def forward(self, logits, targets, hard_targets=None, aux_logits=None, aux_targets=None, aux_mask=None):
        rank_targets = targets if hard_targets is None else hard_targets
        base_loss = (
            self.focal(logits, targets)
            + self.rank_weight * self.ranking_loss(logits, rank_targets)
            + self.dice_weight * self.dice_loss(logits, targets)
        )
        if aux_logits is None or aux_targets is None or aux_mask is None or not PARTNER_CONTACT_AUX:
            return base_loss

        aux_loss = F.binary_cross_entropy_with_logits(
            aux_logits,
            aux_targets,
            pos_weight=aux_logits.new_tensor(PARTNER_CONTACT_POS_WEIGHT),
            reduction="none",
        )
        aux_loss = (aux_loss * aux_mask).sum() / aux_mask.sum().clamp_min(1.0)
        return base_loss + PARTNER_CONTACT_AUX_WEIGHT * aux_loss


class TwoHeadBindingLoss(nn.Module):
    """Separate calibration and ranking heads while keeping their probabilities aligned."""
    def __init__(
        self,
        pos_weight,
        cls_rank_weight=0.12,
        cls_dice_weight=0.10,
        rank_rank_weight=0.45,
        rank_dice_weight=0.04,
        rank_loss_weight=TWO_HEAD_RANK_LOSS_WEIGHT,
        consistency_weight=TWO_HEAD_CONSISTENCY_WEIGHT,
    ):
        super().__init__()
        self.cls_loss = CompositeBindingLoss(
            pos_weight=pos_weight,
            gamma=1.5,
            label_smoothing=0.02,
            rank_weight=cls_rank_weight,
            dice_weight=cls_dice_weight,
        )
        self.rank_loss = CompositeBindingLoss(
            pos_weight=pos_weight,
            gamma=1.2,
            label_smoothing=0.0,
            rank_weight=rank_rank_weight,
            dice_weight=rank_dice_weight,
        )
        self.rank_loss_weight = rank_loss_weight
        self.consistency_weight = consistency_weight

    def forward(
        self,
        cls_logits,
        rank_logits,
        targets,
        hard_targets=None,
        aux_logits=None,
        aux_targets=None,
        aux_mask=None,
        rank_loss_scale=1.0,
    ):
        rank_targets = targets if hard_targets is None else hard_targets
        cls_loss = self.cls_loss(
            cls_logits,
            targets,
            hard_targets=hard_targets,
            aux_logits=aux_logits,
            aux_targets=aux_targets,
            aux_mask=aux_mask,
        )
        rank_loss = self.rank_loss(rank_logits, rank_targets, hard_targets=rank_targets)
        consistency = F.mse_loss(torch.sigmoid(cls_logits), torch.sigmoid(rank_logits))
        return cls_loss + self.rank_loss_weight * rank_loss_scale * rank_loss + self.consistency_weight * consistency


def sparse_pair_contact_loss(pair_output):
    if not PAIR_CONTACT_LOSS or pair_output is None:
        return None
    pair_logits, pair_targets = pair_output[:2]
    if pair_logits is None or pair_logits.numel() == 0:
        return None
    return F.binary_cross_entropy_with_logits(
        pair_logits,
        pair_targets.to(pair_logits.dtype),
        pos_weight=pair_logits.new_tensor(PAIR_CONTACT_POS_WEIGHT),
    )


def pair_contact_score_vectors(pair_output, site_targets, num_nodes, hard_k=10):
    """Return explicit-positive and hard-candidate pooled logits per target residue."""
    if pair_output is None or len(pair_output) < 4:
        return None, None
    pair_logits, pair_targets, target_indices, explicit_positive = pair_output
    if pair_logits is None or pair_logits.numel() == 0:
        return None, None

    positive_scores = pair_logits.new_full((num_nodes,), float("nan"))
    hard_scores = pair_logits.new_full((num_nodes,), float("nan"))
    explicit_positive = explicit_positive.bool()
    hard_k = max(1, int(hard_k))

    eligible_entries = (
        explicit_positive
        & (pair_targets > 0.5)
        & (site_targets[target_indices] > 0.5)
    )
    candidate_entries = ~explicit_positive
    eligible_targets = torch.unique(target_indices[eligible_entries], sorted=True)
    for target_idx in eligible_targets:
        positive = pair_logits[eligible_entries & (target_indices == target_idx)]
        if positive.numel() > 0:
            positive_scores[target_idx] = torch.logsumexp(positive, dim=0)
    candidate_targets = torch.unique(target_indices[candidate_entries], sorted=True)
    for target_idx in candidate_targets:
        candidates = pair_logits[candidate_entries & (target_indices == target_idx)]
        if candidates.numel() > 0:
            k = min(hard_k, candidates.numel())
            hard_scores[target_idx] = torch.logsumexp(
                torch.topk(candidates, k=k).values,
                dim=0,
            )
    return positive_scores, hard_scores


def pair_contact_hard_partner_loss(
    true_pair_output,
    mismatch_pair_output,
    site_targets,
    batch_vec,
):
    if not PAIR_CONTACT_CONTRAST:
        return None, None
    true_positive, _ = pair_contact_score_vectors(
        true_pair_output,
        site_targets,
        site_targets.numel(),
        hard_k=PAIR_CONTACT_CONTRAST_HARD_K,
    )
    if true_positive is None:
        return None, None

    _, mismatch_hard = pair_contact_score_vectors(
        mismatch_pair_output,
        site_targets,
        site_targets.numel(),
        hard_k=PAIR_CONTACT_CONTRAST_HARD_K,
    )
    if mismatch_hard is None:
        return None, None

    eligible = torch.isfinite(true_positive) & torch.isfinite(mismatch_hard)
    if not eligible.any():
        return None, None

    differences = true_positive - mismatch_hard
    margin = differences.new_tensor(PAIR_CONTACT_CONTRAST_MARGIN)
    graph_losses = []
    for gid in torch.unique(batch_vec[eligible], sorted=True):
        graph_mask = eligible & (batch_vec == gid)
        graph_losses.append(F.relu(margin - differences[graph_mask]).mean())
    return torch.stack(graph_losses).mean(), differences[eligible]


def pair_marginal_consistency_loss(
    pair_marginal,
    cls_logits,
    rank_logits,
    batch_vec,
    pair_contact_graphs,
):
    if not PAIR_MARGINAL_CONSISTENCY or pair_marginal is None:
        return None
    if pair_marginal.numel() == 0 or pair_contact_graphs is None or pair_contact_graphs.numel() == 0:
        return None

    valid_nodes = torch.isin(batch_vec, pair_contact_graphs)
    if not valid_nodes.any():
        return None

    site_logits = cls_logits
    if rank_logits is not None:
        site_logits = (
            (1.0 - TWO_HEAD_RANK_FUSION) * cls_logits
            + TWO_HEAD_RANK_FUSION * rank_logits
        )
    site_teacher = torch.sigmoid(site_logits.detach())
    marginal = pair_marginal.clamp(min=1e-6, max=1.0 - 1e-6)
    return F.l1_loss(marginal[valid_nodes], site_teacher[valid_nodes])


def make_mismatched_partner_graph_batch(
    partner_feats,
    partner_batch,
    partner_edge,
    partner_coords,
    partner_frames,
    batch_vec,
    length_match=False,
):
    if (
        partner_feats is None
        or partner_batch is None
        or partner_feats.numel() == 0
        or partner_batch.numel() == 0
    ):
        return (
            partner_feats,
            partner_batch,
            partner_edge,
            partner_coords,
            partner_frames,
            False,
        )

    graph_ids = torch.unique(batch_vec, sorted=True)
    valid_ids = [gid for gid in graph_ids if (partner_batch == gid).any()]
    if len(valid_ids) < 2:
        return (
            partner_feats,
            partner_batch,
            partner_edge,
            partner_coords,
            partner_frames,
            False,
        )

    if length_match:
        partner_lengths = {
            int(gid.item()): int((partner_batch == gid).sum().item())
            for gid in valid_ids
        }
        donor_ids = []
        for target_gid in valid_ids:
            target_value = int(target_gid.item())
            donor_ids.append(min(
                (gid for gid in valid_ids if int(gid.item()) != target_value),
                key=lambda gid: (
                    abs(partner_lengths[int(gid.item())] - partner_lengths[target_value]),
                    int(gid.item()),
                ),
            ))
    else:
        donor_ids = valid_ids[1:] + valid_ids[:1]
    mismatched_feats = []
    mismatched_batch = []
    mismatched_edges = []
    mismatched_coords = []
    mismatched_frames = []
    partner_offset = 0
    for target_gid, donor_gid in zip(valid_ids, donor_ids):
        donor_idx = (partner_batch == donor_gid).nonzero(as_tuple=False).squeeze(-1)
        if donor_idx.numel() == 0:
            continue
        donor_feats = partner_feats[donor_idx]
        mismatched_feats.append(donor_feats)
        mismatched_coords.append(partner_coords[donor_idx])
        mismatched_frames.append(partner_frames[donor_idx])
        mismatched_batch.append(torch.full(
            (donor_feats.size(0),),
            int(target_gid.item()),
            dtype=partner_batch.dtype,
            device=partner_batch.device,
        ))

        if partner_edge is not None and partner_edge.numel() > 0:
            edge_mask = (
                (partner_batch[partner_edge[0]] == donor_gid)
                & (partner_batch[partner_edge[1]] == donor_gid)
            )
            donor_edges = partner_edge[:, edge_mask]
            if donor_edges.numel() > 0:
                old_to_new = torch.full(
                    (partner_feats.size(0),),
                    -1,
                    dtype=torch.long,
                    device=partner_feats.device,
                )
                old_to_new[donor_idx] = torch.arange(
                    partner_offset,
                    partner_offset + donor_idx.numel(),
                    dtype=torch.long,
                    device=partner_feats.device,
                )
                mismatched_edges.append(old_to_new[donor_edges])
        partner_offset += donor_idx.numel()

    if not mismatched_feats:
        return (
            partner_feats,
            partner_batch,
            partner_edge,
            partner_coords,
            partner_frames,
            False,
        )
    edge = (
        torch.cat(mismatched_edges, dim=1)
        if mismatched_edges
        else torch.empty((2, 0), dtype=torch.long, device=partner_feats.device)
    )
    return (
        torch.cat(mismatched_feats, dim=0),
        torch.cat(mismatched_batch, dim=0),
        edge,
        torch.cat(mismatched_coords, dim=0),
        torch.cat(mismatched_frames, dim=0),
        True,
    )


def pair_marginal_partner_contrastive_loss(
    true_marginal,
    mismatch_marginal,
    targets,
    batch_vec,
    pair_contact_graphs,
):
    if (
        not PAIR_MARGINAL_CONTRAST
        or true_marginal is None
        or mismatch_marginal is None
        or true_marginal.numel() == 0
        or mismatch_marginal.numel() == 0
        or pair_contact_graphs is None
        or pair_contact_graphs.numel() == 0
    ):
        return None

    labelled_graphs = set(
        int(gid) for gid in pair_contact_graphs.detach().cpu().tolist()
    )

    graph_losses = []
    margin = true_marginal.new_tensor(PAIR_MARGINAL_CONTRAST_MARGIN)
    for gid in torch.unique(batch_vec, sorted=True):
        gid_value = int(gid.item())
        if gid_value not in labelled_graphs:
            continue
        positive = (batch_vec == gid) & (targets > 0.5)
        if not positive.any():
            continue
        graph_losses.append(
            F.relu(
                margin
                - true_marginal[positive]
                + mismatch_marginal[positive]
            ).mean()
        )
    if not graph_losses:
        return None
    return torch.stack(graph_losses).mean()


def site_separation_score(logits, targets, batch_vec):
    graph_scores = []
    for gid in torch.unique(batch_vec, sorted=True):
        mask = batch_vec == gid
        pos = logits[mask & (targets > 0.5)]
        neg = logits[mask & (targets <= 0.5)]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        graph_scores.append(pos.mean() - neg.mean())
    if not graph_scores:
        return None
    return torch.stack(graph_scores).mean()


def partner_contrastive_loss(true_logits, mismatch_logits, targets, batch_vec):
    if (
        not PARTNER_CONTRAST
        or true_logits is None
        or mismatch_logits is None
        or true_logits.numel() == 0
        or mismatch_logits.numel() == 0
    ):
        return None
    true_score = site_separation_score(true_logits, targets, batch_vec)
    mismatch_score = site_separation_score(mismatch_logits, targets, batch_vec)
    if true_score is None or mismatch_score is None:
        return None
    margin = true_logits.new_tensor(PARTNER_CONTRAST_MARGIN)
    return F.relu(margin - true_score + mismatch_score)


class ModelEMA:
    """Exponential moving average of model weights for steadier validation."""
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.state_dict().items()
            if param.dtype.is_floating_point
        }
        self.backup = None

    def update(self, model):
        with torch.no_grad():
            for name, param in model.state_dict().items():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def store(self, model):
        self.backup = {
            name: param.detach().clone()
            for name, param in model.state_dict().items()
            if name in self.shadow
        }

    def copy_to(self, model):
        state = model.state_dict()
        for name, value in self.shadow.items():
            state[name].copy_(value)

    def restore(self, model):
        if self.backup is None:
            return
        state = model.state_dict()
        for name, value in self.backup.items():
            state[name].copy_(value)
        self.backup = None


# ===================== Graph & Substructure Modules =====================

class GraphConv(nn.Module):
    """
    一个简单的 message passing：
    h_i^{(l+1)} = ReLU( W_self h_i^{(l)} + W_neigh * sum_{j in N(i)} h_j^{(l)} )
    """
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.neigh_lin = nn.Linear(in_dim, out_dim)
        self.gate_lin = nn.Linear(in_dim * 2, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        """
        x: [N, d]
        edge_index: [2, E]  (src, dst)
        """
        if x.size(0) == 0:
            return x

        if edge_index.numel() == 0:
            agg = torch.zeros_like(x)
        else:
            src, dst = edge_index  # [E], [E]
            agg = torch.zeros_like(x)
            deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            agg.index_add_(0, dst, x[src])
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
            agg = agg / deg.clamp_min(1.0).unsqueeze(-1)

        gate = torch.sigmoid(self.gate_lin(torch.cat((x, agg), dim=-1)))
        out = self.self_lin(x) + gate * self.neigh_lin(agg)
        out = F.relu(out)
        out = self.dropout(out)
        return self.norm(out)


class GeometricEdgeAttention(nn.Module):
    """
    Geometry/edge-enhanced attention message passing.
    Uses continuous edge descriptors instead of treating all edges uniformly.
    """
    def __init__(self, in_dim, hidden_dim, edge_feat_dim=32, dropout=0.1):
        super().__init__()
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * in_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.msg_lin = nn.Linear(in_dim, hidden_dim)
        self.out_lin = nn.Linear(in_dim + hidden_dim, in_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x, edge_index, edge_feat):
        if x.size(0) == 0 or edge_index.numel() == 0:
            return x

        src, dst = edge_index
        e = self.edge_proj(edge_feat)
        h_src = x[src]
        h_dst = x[dst]

        att_logits = self.attn_mlp(torch.cat((h_dst, h_src, e), dim=-1)).squeeze(-1)
        att = edge_softmax(att_logits, dst, x.size(0))

        msg = self.msg_lin(h_src) * (1.0 + torch.tanh(e))
        agg = x.new_zeros((x.size(0), msg.size(-1)))
        agg.index_add_(0, dst, att.unsqueeze(-1) * msg)

        out = self.out_lin(torch.cat((x, agg), dim=-1))
        out = self.dropout(F.relu(out))
        return self.norm(x + out)


class LegacyGraphConv(nn.Module):
    """Checkpoint-compatible graph convolution used by saved tri-view models."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.neigh_lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        if x.size(0) == 0:
            return x
        if edge_index.numel() == 0:
            agg = torch.zeros_like(x)
        else:
            src, dst = edge_index
            agg = torch.zeros_like(x)
            agg.index_add_(0, dst, x[src])
        return F.relu(self.self_lin(x) + self.neigh_lin(agg))


class SubstructureAttention(nn.Module):
    """
    交互注意力版 SubstructureAttention：
    - 不再用全局 query，而是让同一个图里的节点之间做多头自注意力：
        Q = W_q x, K = W_k x, V = W_v x
        head h 的注意力可以理解为一种“子结构模式”
    - 每个 head 对应一种 soft 子结构，最后 Residual + LayerNorm 稳定训练。
    """
    def __init__(self, hidden_dim, num_substructs=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_substructs   # 每个 head 看作一个子结构
        assert hidden_dim % self.num_heads == 0, \
            "hidden_dim 必须能被 num_substructs 整除"
        self.head_dim = hidden_dim // self.num_heads

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, batch):
        """
        x: [N, d]
        batch: [N]，每个节点属于哪一个 graph 的 id (0..B-1)
        返回：x_new: [N, d]
        """
        if x.size(0) == 0:
            return x

        B = int(batch.max().item()) + 1
        d = self.hidden_dim
        out = torch.zeros_like(x)

        for g in range(B):
            idx = (batch == g).nonzero(as_tuple=False).squeeze(-1)  # 当前图的节点下标
            if idx.numel() == 0:
                continue

            x_g = x[idx]                 # [Ng, d]
            Ng = x_g.size(0)

            # Q, K, V: [Ng, d]
            Q = self.W_q(x_g)
            K = self.W_k(x_g)
            V = self.W_v(x_g)

            # reshape -> [H, Ng, d_h]
            def split_heads(t):
                # t: [Ng, d] -> [H, Ng, d_h]
                return t.view(Ng, self.num_heads, self.head_dim).transpose(0, 1)

            Q_h = split_heads(Q)
            K_h = split_heads(K)
            V_h = split_heads(V)

            # scores: [H, Ng, Ng]
            scores = torch.bmm(Q_h, K_h.transpose(1, 2)) / math.sqrt(self.head_dim)
            attn = torch.softmax(scores, dim=-1)          # [H, Ng, Ng]
            attn = self.dropout(attn)

            # context: [H, Ng, d_h]
            context = torch.bmm(attn, V_h)

            # concat heads: [Ng, H, d_h] -> [Ng, d]
            context = context.transpose(0, 1).contiguous().view(Ng, d)

            # 输出映射 + 残差 + LayerNorm
            x_g_new = self.out_proj(context)
            x_g_new = self.dropout(x_g_new)
            x_g_new = self.norm(x_g + x_g_new)

            out[idx] = x_g_new

        return out


class ResidueSubstructNet(nn.Module):
    """
    只基于残基图的模型：
    - 输入：residue_graph_node [Nr, F]
    - 图卷积若干层
    - 交互注意力版 SubstructureAttention 在每个图内部做多头自注意力，
      每个 head 对应一种“子结构模式”
    - 输出：每个残基一个 logit（binding / non-binding）
    """
    def __init__(
        self,
        in_dim,
        hidden_dim=256,
        num_layers=2,
        num_substructs=8,
        dropout=0.1,
        edge_dropout=0.1,
    ):
        super().__init__()
        self.edge_dropout = edge_dropout
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.convs = nn.ModuleList([
            GraphConv(hidden_dim, hidden_dim, dropout=dropout) for _ in range(num_layers)
        ])
        self.sub_attn = SubstructureAttention(
            hidden_dim, num_substructs=num_substructs, dropout=dropout
        )
        self.jump_proj = nn.Sequential(
            nn.Linear(hidden_dim * (num_layers + 1), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def drop_edges(self, edge_index):
        if (not self.training) or self.edge_dropout <= 0 or edge_index.numel() == 0:
            return edge_index
        keep = torch.rand(edge_index.size(1), device=edge_index.device) > self.edge_dropout
        if keep.sum() == 0:
            return edge_index
        return edge_index[:, keep]

    def forward(self, x, edge_index, batch):
        """
        x: [N, F]   (所有 batch 里节点拼接到一起)
        edge_index: [2, E]
        batch: [N]  节点对应的 graph id
        return: logits [N]
        """
        edge_index = self.drop_edges(edge_index)
        h = self.in_proj(x)
        layer_outputs = [h]
        for conv in self.convs:
            h = h + conv(h, edge_index)
            layer_outputs.append(h)

        h = self.jump_proj(torch.cat(layer_outputs, dim=-1))
        h = self.sub_attn(h, batch)
        context = self.context_proj(global_mean_max_pool(h, batch))
        gate = self.fusion_gate(torch.cat((h, context), dim=-1))
        fused = torch.cat((h, context, gate * h + (1.0 - gate) * context), dim=-1)
        logits = self.classifier(fused).squeeze(-1)  # [N]
        return logits


class TriViewAtomResidueNet(nn.Module):
    """
    Residue-atom-sequence fusion model inspired by the paper setup:
    - residue graph captures residue-level spatial contacts
    - atom graph captures fine-grained structural chemistry
    - BiGRU captures ordered residue context
    - adaptive gates choose how much each view contributes per residue
    """
    def __init__(
        self,
        res_dim,
        atom_dim,
        surface_dim=SURFACE_FEATURE_DIM,
        sequence_dim=SEQUENCE_FEATURE_DIM,
        plm_dim=PLM_FEATURE_DIM,
        aux_plm_dim=AUX_PLM_FEATURE_DIM,
        hidden_dim=256,
        num_layers=2,
        num_substructs=8,
        dropout=0.2,
        edge_dropout=0.05,
        view_logit_fusion=VIEW_LOGIT_FUSION,
        partner_conditioning=PARTNER_CONDITIONING,
        partner_top_k=PARTNER_TOP_K,
        partner_direct_fusion=PARTNER_DIRECT_FUSION,
        partner_logit_mode=PARTNER_LOGIT_MODE,
        partner_delta_scale=PARTNER_DELTA_SCALE,
        partner_residue_encoder=PARTNER_RESIDUE_ENCODER,
        partner_encoder_layers=PARTNER_ENCODER_LAYERS,
        partner_target_fusion=PARTNER_TARGET_FUSION,
        pair_contact_head=PAIR_CONTACT_HEAD,
    ):
        super().__init__()
        self.edge_dropout = edge_dropout
        self.plm_dim = int(plm_dim or 0)
        self.aux_plm_dim = int(aux_plm_dim or 0)
        self.use_rank_head = TWO_HEAD_BINDING
        self.view_logit_fusion = float(view_logit_fusion)
        self.res_in = nn.Sequential(
            nn.Linear(res_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.atom_in = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.spatial_convs = nn.ModuleList([
            GeometricEdgeAttention(hidden_dim, hidden_dim, edge_feat_dim=32, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.atom_convs = nn.ModuleList([
            GeometricEdgeAttention(hidden_dim, hidden_dim, edge_feat_dim=32, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.seq_gru = nn.GRU(
            hidden_dim,
            hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.atom_to_res = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.topology_proj = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.surface_proj = nn.Sequential(
            nn.Linear(surface_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.biochem_proj = nn.Sequential(
            nn.Linear(sequence_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.plm_proj = None
        if self.plm_dim > 0:
            self.plm_proj = nn.Sequential(
                nn.Linear(self.plm_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        self.aux_plm_proj = None
        if self.aux_plm_dim > 0:
            self.aux_plm_proj = nn.Sequential(
                nn.Linear(self.aux_plm_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        self.num_views = 6
        if self.plm_proj is not None:
            self.num_views += 1
        if self.aux_plm_proj is not None:
            self.num_views += 1
        self.view_gate = nn.Sequential(
            nn.Linear(hidden_dim * self.num_views, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_views),
        )
        self.view_logit_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.fuse_norm = nn.LayerNorm(hidden_dim)
        self.sub_attn = SubstructureAttention(
            hidden_dim, num_substructs=num_substructs, dropout=dropout
        )
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cls = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.rank_cls = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.partner_conditioning = bool(partner_conditioning)
        self.partner_top_k = int(partner_top_k)
        self.partner_direct_fusion = bool(partner_direct_fusion)
        self.partner_logit_mode = str(partner_logit_mode or "standard").lower()
        self.partner_delta_scale = float(partner_delta_scale)
        self.partner_residue_encoder = bool(partner_residue_encoder)
        self.partner_encoder_layers = max(int(partner_encoder_layers), 1)
        self.partner_target_fusion = float(partner_target_fusion)
        self.pair_contact_head = str(pair_contact_head or "attention").lower()
        self.pair_scorer = None
        self.partner_interaction_cls = None
        self.partner_interaction_rank_cls = None
        self.partner_residual_norm = None
        self.partner_residual_gate = None
        self.partner_residual_cls = None
        self.partner_residual_rank_cls = None
        if self.partner_conditioning:
            if self.partner_residue_encoder:
                self.shared_residue_input_norm = nn.LayerNorm(hidden_dim)
                self.shared_residue_convs = nn.ModuleList([
                    GeometricEdgeAttention(
                        hidden_dim,
                        hidden_dim,
                        edge_feat_dim=32,
                        dropout=dropout,
                    )
                    for _ in range(self.partner_encoder_layers)
                ])
                self.shared_residue_output_norm = nn.LayerNorm(hidden_dim)
                self.shared_target_gate = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Sigmoid(),
                )
                self.partner_proj = None
            else:
                self.partner_proj = nn.Sequential(
                    nn.Linear(PARTNER_FEATURE_DIM, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            self.partner_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.partner_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.partner_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.partner_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
            self.partner_norm = nn.LayerNorm(hidden_dim)
            if self.partner_direct_fusion:
                self.partner_direct_proj = nn.Sequential(
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                )
            if self.pair_contact_head == "mlp":
                self.pair_scorer = nn.Sequential(
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1),
                )
            if self.partner_logit_mode in {"interaction_only", "delta"}:
                self.partner_interaction_cls = nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
                self.partner_interaction_rank_cls = nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
            if self.partner_logit_mode == "residual":
                # The residual branch receives only the partner message. Per-graph
                # centering in forward removes partner-global offsets and prevents
                # a direct target-only shortcut into the residual classifier.
                self.partner_residual_norm = nn.LayerNorm(hidden_dim)
                self.partner_residual_gate = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Sigmoid(),
                )
                self.partner_residual_cls = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1),
                )
                self.partner_residual_rank_cls = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1),
                )
        self.partner_cls = None
        if PARTNER_CONTACT_AUX:
            self.partner_cls = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def drop_edges(self, edge_index):
        if (not self.training) or self.edge_dropout <= 0 or edge_index.numel() == 0:
            return edge_index
        keep = torch.rand(edge_index.size(1), device=edge_index.device) > self.edge_dropout
        if keep.sum() == 0:
            return edge_index
        return edge_index[:, keep]

    def sequence_view(self, h, batch):
        out = torch.zeros_like(h)
        num_graphs = int(batch.max().item()) + 1
        for gid in range(num_graphs):
            idx = (batch == gid).nonzero(as_tuple=False).squeeze(-1)
            seq = h[idx].unsqueeze(0)
            seq_out, _ = self.seq_gru(seq)
            out[idx] = seq_out.squeeze(0)
        return out

    def shared_residue_context(self, view_list, edge_index, coords=None, frames=None):
        """Encode target or partner residues with exactly the same feature/geometry block."""
        if not view_list:
            raise ValueError("shared residue encoder requires at least one residue view")
        h = torch.stack(view_list, dim=0).mean(dim=0)
        h = self.shared_residue_input_norm(h)
        edge_index = self.drop_edges(edge_index)
        edge_feat = None
        if coords is not None and coords.dim() == 2 and coords.size(-1) == 3:
            edge_feat = build_coordinate_edge_features(
                coords,
                edge_index,
                out_dim=32,
                frames=frames,
            )
        if edge_feat is None:
            edge_feat = build_edge_features(h, edge_index, out_dim=32)
        for conv in self.shared_residue_convs:
            h = h + conv(h, edge_index, edge_feat)
            if coords is None or coords.size(-1) != 3:
                edge_feat = build_edge_features(h, edge_index, out_dim=32)
        return self.shared_residue_output_norm(h)

    def encode_partner_residues(
        self,
        partner_feats,
        partner_edge=None,
        partner_coords=None,
        partner_frames=None,
    ):
        if not self.partner_residue_encoder:
            return self.partner_proj(partner_feats[:, :PARTNER_FEATURE_DIM])

        expected_width = PARTNER_FEATURE_DIM + self.plm_dim + self.aux_plm_dim
        if partner_feats.size(-1) != expected_width:
            raise ValueError(
                f"partner feature width {partner_feats.size(-1)} does not match "
                f"configured width {expected_width}"
            )
        offset = 0
        partner_surface = partner_feats[:, offset : offset + SURFACE_FEATURE_DIM]
        offset += SURFACE_FEATURE_DIM
        partner_sequence = partner_feats[:, offset : offset + SEQUENCE_FEATURE_DIM]
        offset += SEQUENCE_FEATURE_DIM
        view_list = [
            self.surface_proj(partner_surface),
            self.biochem_proj(partner_sequence),
        ]
        if self.plm_proj is not None:
            partner_plm = partner_feats[:, offset : offset + self.plm_dim]
            offset += self.plm_dim
            view_list.append(self.plm_proj(partner_plm))
        if self.aux_plm_proj is not None:
            partner_aux_plm = partner_feats[:, offset : offset + self.aux_plm_dim]
            view_list.append(self.aux_plm_proj(partner_aux_plm))
        if partner_edge is None:
            partner_edge = torch.empty((2, 0), dtype=torch.long, device=partner_feats.device)
        return self.shared_residue_context(
            view_list,
            partner_edge,
            coords=partner_coords,
            frames=partner_frames,
        )

    def pair_logits_from_embeddings(self, target_h, partner_h):
        if self.pair_scorer is None:
            return None
        pair_features = torch.cat(
            (
                target_h,
                partner_h,
                target_h * partner_h,
                torch.abs(target_h - partner_h),
            ),
            dim=-1,
        )
        return self.pair_scorer(pair_features).squeeze(-1)

    def partner_message(
        self,
        fused,
        batch,
        partner_feats=None,
        partner_batch=None,
        partner_edge=None,
        partner_coords=None,
        partner_frames=None,
    ):
        if (
            not self.partner_conditioning
            or partner_feats is None
            or partner_batch is None
            or partner_feats.numel() == 0
        ):
            return fused.new_zeros(fused.shape)

        partner_h = self.encode_partner_residues(
            partner_feats,
            partner_edge=partner_edge,
            partner_coords=partner_coords,
            partner_frames=partner_frames,
        )
        message = fused.new_zeros(fused.shape)
        num_graphs = int(batch.max().item()) + 1
        scale = math.sqrt(float(fused.size(-1)))

        for gid in range(num_graphs):
            idx = (batch == gid).nonzero(as_tuple=False).squeeze(-1)
            pidx = (partner_batch == gid).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0 or pidx.numel() == 0:
                continue

            q = self.partner_q(fused[idx])
            k = self.partner_k(partner_h[pidx])
            v = self.partner_v(partner_h[pidx])
            scores = torch.matmul(q, k.transpose(0, 1)) / scale
            top_k = min(max(int(self.partner_top_k), 1), scores.size(1))
            top_scores, top_idx = torch.topk(scores, k=top_k, dim=1)
            attn = torch.softmax(top_scores, dim=1)
            top_v = v[top_idx]
            message[idx] = (attn.unsqueeze(-1) * top_v).sum(dim=1)

        return message

    def partner_condition(
        self,
        fused,
        batch,
        partner_feats=None,
        partner_batch=None,
        partner_edge=None,
        partner_coords=None,
        partner_frames=None,
        pair_contact_index=None,
        pair_contact_graphs=None,
        return_pair=False,
        return_pair_marginal=False,
        return_pair_logit_matrix=False,
        return_message=False,
    ):
        logit_matrix_cols = max(int(self.partner_top_k), 1)
        message = fused.new_zeros(fused.shape)

        def pack_outputs(conditioned_value, pair_value, marginal_value, matrix_value):
            outputs = [conditioned_value]
            if return_pair:
                outputs.append(pair_value)
            if return_pair_marginal:
                outputs.append(marginal_value)
            if return_pair_logit_matrix:
                outputs.append(matrix_value)
            if return_message:
                outputs.append(message)
            return outputs[0] if len(outputs) == 1 else tuple(outputs)

        if (
            not self.partner_conditioning
            or partner_feats is None
            or partner_batch is None
            or partner_feats.numel() == 0
        ):
            pair_marginal = fused.new_zeros((fused.size(0),))
            pair_logit_matrix = fused.new_full((fused.size(0), logit_matrix_cols), float("nan"))
            empty_pair = (
                fused.new_empty((0,)),
                fused.new_empty((0,)),
                torch.empty((0,), dtype=torch.long, device=fused.device),
                torch.empty((0,), dtype=torch.bool, device=fused.device),
            )
            return pack_outputs(fused, empty_pair, pair_marginal, pair_logit_matrix)

        partner_h = self.encode_partner_residues(
            partner_feats,
            partner_edge=partner_edge,
            partner_coords=partner_coords,
            partner_frames=partner_frames,
        )
        conditioned = fused.clone()
        pair_marginal = fused.new_zeros((fused.size(0),)) if return_pair_marginal else None
        pair_logit_matrix = (
            fused.new_full((fused.size(0), logit_matrix_cols), float("nan"))
            if return_pair_logit_matrix else None
        )
        num_graphs = int(batch.max().item()) + 1
        scale = math.sqrt(float(fused.size(-1)))
        pair_logits = []
        pair_targets = []
        pair_target_indices = []
        pair_explicit_positive = []
        labelled_graphs = None
        if return_pair and pair_contact_graphs is not None and pair_contact_graphs.numel() > 0:
            labelled_graphs = set(int(gid) for gid in pair_contact_graphs.detach().cpu().tolist())

        for gid in range(num_graphs):
            idx = (batch == gid).nonzero(as_tuple=False).squeeze(-1)
            pidx = (partner_batch == gid).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0 or pidx.numel() == 0:
                continue

            q = self.partner_q(fused[idx])
            k = self.partner_k(partner_h[pidx])
            v = self.partner_v(partner_h[pidx])
            scores = torch.matmul(q, k.transpose(0, 1)) / scale
            top_k = min(max(int(self.partner_top_k), 1), scores.size(1))
            top_scores, top_idx = torch.topk(scores, k=top_k, dim=1)
            attn = torch.softmax(top_scores, dim=1)
            top_v = v[top_idx]
            msg = (attn.unsqueeze(-1) * top_v).sum(dim=1)
            message[idx] = msg
            if getattr(self, "partner_direct_fusion", False):
                direct_input = torch.cat(
                    (
                        fused[idx],
                        msg,
                        fused[idx] * msg,
                        torch.abs(fused[idx] - msg),
                    ),
                    dim=-1,
                )
                conditioned[idx] = self.partner_norm(fused[idx] + self.partner_direct_proj(direct_input))
            else:
                gate = self.partner_gate(torch.cat((fused[idx], msg), dim=-1))
                conditioned[idx] = self.partner_norm(fused[idx] + gate * msg)

            selected_res = None
            selected_partner = None
            selected_logits = None
            if return_pair or return_pair_marginal or return_pair_logit_matrix:
                selected_res = idx.unsqueeze(1).expand(-1, top_k).reshape(-1)
                selected_partner = pidx[top_idx].reshape(-1)
                selected_logits = top_scores.reshape(-1)
                if self.pair_scorer is not None:
                    selected_logits = self.pair_logits_from_embeddings(
                        fused[selected_res],
                        partner_h[selected_partner],
                    )

            if return_pair_logit_matrix and selected_logits is not None:
                pair_logit_matrix[idx, :top_k] = selected_logits.view(idx.numel(), top_k)

            if return_pair_marginal and selected_logits is not None:
                pair_logit_rows = selected_logits.view(idx.numel(), top_k)
                log_no_contact = F.logsigmoid(-pair_logit_rows).sum(dim=1)
                pair_marginal[idx] = -torch.expm1(log_no_contact)

            if return_pair and labelled_graphs is not None and gid in labelled_graphs:
                selected_labels = selected_logits.new_zeros(selected_logits.shape)
                pos_edges = None
                if pair_contact_index is not None and pair_contact_index.numel() > 0:
                    pos_mask = (
                        (batch[pair_contact_index[0]] == gid)
                        & (partner_batch[pair_contact_index[1]] == gid)
                    )
                    if pos_mask.any():
                        pos_edges = pair_contact_index[:, pos_mask]
                        max_partner = int(
                            max(
                                selected_partner.max().item() if selected_partner.numel() else 0,
                                pos_edges[1].max().item() if pos_edges.numel() else 0,
                            )
                        ) + 1
                        selected_key = selected_res * max_partner + selected_partner
                        positive_key = pos_edges[0] * max_partner + pos_edges[1]
                        selected_labels = torch.isin(selected_key, positive_key).to(selected_logits.dtype)

                pair_logits.append(selected_logits)
                pair_targets.append(selected_labels)
                pair_target_indices.append(selected_res)
                pair_explicit_positive.append(torch.zeros_like(selected_labels, dtype=torch.bool))

                if pos_edges is not None and pos_edges.numel() > 0:
                    target_local = torch.empty((fused.size(0),), dtype=torch.long, device=fused.device)
                    partner_local = torch.empty((partner_h.size(0),), dtype=torch.long, device=fused.device)
                    target_local[idx] = torch.arange(idx.numel(), dtype=torch.long, device=fused.device)
                    partner_local[pidx] = torch.arange(pidx.numel(), dtype=torch.long, device=fused.device)
                    positive_logits = scores[target_local[pos_edges[0]], partner_local[pos_edges[1]]]
                    if self.pair_scorer is not None:
                        positive_logits = self.pair_logits_from_embeddings(
                            fused[pos_edges[0]],
                            partner_h[pos_edges[1]],
                        )
                    pair_logits.append(positive_logits)
                    pair_targets.append(torch.ones_like(positive_logits))
                    pair_target_indices.append(pos_edges[0])
                    pair_explicit_positive.append(torch.ones_like(positive_logits, dtype=torch.bool))

        pair_output = None
        if return_pair:
            if pair_logits:
                pair_output = (
                    torch.cat(pair_logits, dim=0),
                    torch.cat(pair_targets, dim=0),
                    torch.cat(pair_target_indices, dim=0),
                    torch.cat(pair_explicit_positive, dim=0),
                )
            else:
                pair_output = (
                    fused.new_empty((0,)),
                    fused.new_empty((0,)),
                    torch.empty((0,), dtype=torch.long, device=fused.device),
                    torch.empty((0,), dtype=torch.bool, device=fused.device),
                )
        return pack_outputs(conditioned, pair_output, pair_marginal, pair_logit_matrix)

    def forward(
        self,
        res_x,
        res_edge,
        atom_x,
        atom_edge,
        atom2res,
        batch,
        res_coords=None,
        atom_coords=None,
        res_frames=None,
        surface_feats=None,
        sequence_feats=None,
        plm_feats=None,
        aux_plm_feats=None,
        partner_feats=None,
        partner_batch=None,
        partner_edge=None,
        partner_coords=None,
        partner_frames=None,
        pair_contact_index=None,
        pair_contact_graphs=None,
        return_aux=False,
        return_heads=False,
        return_pair=False,
        return_pair_marginal=False,
        return_pair_logit_matrix=False,
    ):
        res_edge = self.drop_edges(res_edge)
        atom_edge = self.drop_edges(atom_edge)

        res_h = self.res_in(res_x)
        res_edge_feat = None
        if res_coords is not None and res_coords.size(-1) == 3:
            res_edge_feat = build_coordinate_edge_features(res_coords, res_edge, out_dim=32, frames=res_frames)
        if res_edge_feat is None:
            res_edge_feat = build_edge_features(res_h, res_edge, out_dim=32)
        for conv in self.spatial_convs:
            res_h = res_h + conv(res_h, res_edge, res_edge_feat)
            if res_coords is None or res_coords.size(-1) != 3:
                res_edge_feat = build_edge_features(res_h, res_edge, out_dim=32)

        seq_h = self.sequence_view(res_h, batch)

        atom_h = self.atom_in(atom_x)
        atom_edge_feat = None
        if atom_coords is not None and atom_coords.size(-1) == 3:
            atom_edge_feat = build_coordinate_edge_features(atom_coords, atom_edge, out_dim=32)
        if atom_edge_feat is None:
            atom_edge_feat = build_edge_features(atom_h, atom_edge, out_dim=32)
        for conv in self.atom_convs:
            atom_h = atom_h + conv(atom_h, atom_edge, atom_edge_feat)
            if atom_coords is None or atom_coords.size(-1) != 3:
                atom_edge_feat = build_edge_features(atom_h, atom_edge, out_dim=32)
        atom_h = self.atom_to_res(atom_to_residue_pool(atom_h, atom2res, res_x.size(0)))

        topo_h = self.topology_proj(
            residue_topology_features(res_edge, atom_edge, atom2res, res_x.size(0))
        )
        if surface_feats is None or surface_feats.size(-1) != SURFACE_FEATURE_DIM:
            surface_feats = res_x.new_zeros((res_x.size(0), SURFACE_FEATURE_DIM))
        surf_h = self.surface_proj(surface_feats)
        if sequence_feats is None or sequence_feats.size(-1) != SEQUENCE_FEATURE_DIM:
            sequence_feats = res_x.new_zeros((res_x.size(0), SEQUENCE_FEATURE_DIM))
        biochem_h = self.biochem_proj(sequence_feats)
        view_list = [res_h, seq_h, atom_h, topo_h, surf_h, biochem_h]
        shared_view_list = [surf_h, biochem_h]
        if self.plm_proj is not None:
            if plm_feats is None or plm_feats.size(-1) != self.plm_dim:
                plm_feats = res_x.new_zeros((res_x.size(0), self.plm_dim))
            plm_h = self.plm_proj(plm_feats)
            view_list.append(plm_h)
            shared_view_list.append(plm_h)
        if self.aux_plm_proj is not None:
            if aux_plm_feats is None or aux_plm_feats.size(-1) != self.aux_plm_dim:
                aux_plm_feats = res_x.new_zeros((res_x.size(0), self.aux_plm_dim))
            aux_plm_h = self.aux_plm_proj(aux_plm_feats)
            view_list.append(aux_plm_h)
            shared_view_list.append(aux_plm_h)

        views = torch.stack(view_list, dim=1)
        gate_input = torch.cat(view_list, dim=-1)
        gate = torch.softmax(
            self.view_gate(gate_input),
            dim=-1,
        )
        view_logit_fusion = float(getattr(self, "view_logit_fusion", VIEW_LOGIT_FUSION))
        view_logits = None
        if view_logit_fusion > 0.0:
            view_logits = (self.view_logit_head(views).squeeze(-1) * gate).sum(dim=1)

        fused = (views * gate.unsqueeze(-1)).sum(dim=1)
        fused = self.fuse_norm(fused + res_h)
        if self.partner_conditioning and self.partner_residue_encoder:
            target_shared_h = self.shared_residue_context(
                shared_view_list,
                res_edge,
                coords=res_coords,
                frames=res_frames,
            )
            target_gate = self.shared_target_gate(torch.cat((fused, target_shared_h), dim=-1))
            fused = self.fuse_norm(
                fused + self.partner_target_fusion * target_gate * target_shared_h
            )
        fused = self.sub_attn(fused, batch)
        target_only_fused = fused
        pair_output = None
        pair_marginal = None
        pair_logit_matrix = None
        partner_message = None
        residual_mode = getattr(self, "partner_logit_mode", "standard") == "residual"
        if residual_mode:
            partner_result = self.partner_condition(
                fused,
                batch,
                partner_feats,
                partner_batch,
                partner_edge=partner_edge,
                partner_coords=partner_coords,
                partner_frames=partner_frames,
                pair_contact_index=pair_contact_index,
                pair_contact_graphs=pair_contact_graphs,
                return_pair=return_pair,
                return_pair_marginal=return_pair_marginal,
                return_pair_logit_matrix=return_pair_logit_matrix,
                return_message=True,
            )
            partner_values = list(partner_result)
            _partner_conditioned_fused = partner_values.pop(0)
            if return_pair:
                pair_output = partner_values.pop(0)
            if return_pair_marginal:
                pair_marginal = partner_values.pop(0)
            if return_pair_logit_matrix:
                pair_logit_matrix = partner_values.pop(0)
            partner_message = partner_values.pop(0)
            if partner_values:
                raise RuntimeError("Unexpected residual partner outputs")
            # The intrinsic classifier is intentionally isolated from partner
            # conditioning. Partner information may affect logits only through
            # the bounded residual pathway below.
            fused = target_only_fused
        elif return_pair or return_pair_marginal or return_pair_logit_matrix:
            partner_result = self.partner_condition(
                fused,
                batch,
                partner_feats,
                partner_batch,
                partner_edge=partner_edge,
                partner_coords=partner_coords,
                partner_frames=partner_frames,
                pair_contact_index=pair_contact_index,
                pair_contact_graphs=pair_contact_graphs,
                return_pair=return_pair,
                return_pair_marginal=return_pair_marginal,
                return_pair_logit_matrix=return_pair_logit_matrix,
            )
            if return_pair and return_pair_marginal and return_pair_logit_matrix:
                fused, pair_output, pair_marginal, pair_logit_matrix = partner_result
            elif return_pair and return_pair_marginal:
                fused, pair_output, pair_marginal = partner_result
            elif return_pair and return_pair_logit_matrix:
                fused, pair_output, pair_logit_matrix = partner_result
            elif return_pair:
                fused, pair_output = partner_result
            elif return_pair_marginal and return_pair_logit_matrix:
                fused, pair_marginal, pair_logit_matrix = partner_result
            else:
                if return_pair_marginal:
                    fused, pair_marginal = partner_result
                else:
                    fused, pair_logit_matrix = partner_result
        else:
            fused = self.partner_condition(
                fused,
                batch,
                partner_feats,
                partner_batch,
                partner_edge=partner_edge,
                partner_coords=partner_coords,
                partner_frames=partner_frames,
            )

        context = self.context_proj(global_mean_max_pool(fused, batch))
        final_h = torch.cat((fused, context), dim=-1)
        cls_logits = self.cls(final_h).squeeze(-1)
        use_partner_interaction_logits = (
            getattr(self, "partner_logit_mode", "standard") in {"interaction_only", "delta"}
            and self.partner_interaction_cls is not None
            and partner_feats is not None
            and partner_batch is not None
            and partner_feats.numel() > 0
        )
        if view_logits is not None and not use_partner_interaction_logits:
            cls_logits = (1.0 - view_logit_fusion) * cls_logits + view_logit_fusion * view_logits
        if getattr(self, "use_rank_head", True):
            rank_logits = self.rank_cls(final_h).squeeze(-1)
        else:
            rank_logits = cls_logits
        if use_partner_interaction_logits:
            partner_msg = self.partner_message(
                target_only_fused,
                batch,
                partner_feats,
                partner_batch,
                partner_edge=partner_edge,
                partner_coords=partner_coords,
                partner_frames=partner_frames,
            )
            interaction_h = torch.cat(
                (
                    partner_msg,
                    target_only_fused * partner_msg,
                    torch.abs(target_only_fused - partner_msg),
                ),
                dim=-1,
            )
            interaction_cls_logits = self.partner_interaction_cls(interaction_h).squeeze(-1)
            interaction_rank_logits = self.partner_interaction_rank_cls(interaction_h).squeeze(-1)
            if getattr(self, "partner_logit_mode", "standard") == "interaction_only":
                cls_logits = interaction_cls_logits
                rank_logits = interaction_rank_logits
            else:
                scale = float(getattr(self, "partner_delta_scale", PARTNER_DELTA_SCALE))
                cls_logits = cls_logits + scale * interaction_cls_logits
                rank_logits = rank_logits + scale * interaction_rank_logits
        use_partner_residual = (
            residual_mode
            and self.partner_residual_cls is not None
            and self.partner_residual_rank_cls is not None
            and self.partner_residual_gate is not None
            and partner_message is not None
            and partner_feats is not None
            and partner_batch is not None
            and partner_feats.numel() > 0
        )
        if use_partner_residual:
            centered_message = partner_message.clone()
            num_graphs = int(batch.max().item()) + 1
            for gid in range(num_graphs):
                idx = (batch == gid).nonzero(as_tuple=False).squeeze(-1)
                if idx.numel() > 0:
                    centered_message[idx] = (
                        centered_message[idx] - centered_message[idx].mean(dim=0, keepdim=True)
                    )
            residual_h = self.partner_residual_norm(centered_message)
            residual_gate = self.partner_residual_gate(residual_h).squeeze(-1)
            cls_delta = torch.tanh(self.partner_residual_cls(residual_h).squeeze(-1))
            rank_delta = torch.tanh(self.partner_residual_rank_cls(residual_h).squeeze(-1))
            scale = float(getattr(self, "partner_delta_scale", PARTNER_DELTA_SCALE))
            cls_logits = cls_logits + scale * residual_gate * cls_delta
            rank_logits = rank_logits + scale * residual_gate * rank_delta
        logits = (1.0 - TWO_HEAD_RANK_FUSION) * cls_logits + TWO_HEAD_RANK_FUSION * rank_logits
        partner_logits = self.partner_cls(final_h).squeeze(-1) if return_aux and self.partner_cls is not None else None
        if return_heads:
            if return_aux:
                if return_pair and return_pair_marginal and return_pair_logit_matrix:
                    return cls_logits, rank_logits, partner_logits, pair_output, pair_marginal, pair_logit_matrix
                if return_pair and return_pair_marginal:
                    return cls_logits, rank_logits, partner_logits, pair_output, pair_marginal
                if return_pair and return_pair_logit_matrix:
                    return cls_logits, rank_logits, partner_logits, pair_output, pair_logit_matrix
                if return_pair:
                    return cls_logits, rank_logits, partner_logits, pair_output
                if return_pair_marginal and return_pair_logit_matrix:
                    return cls_logits, rank_logits, partner_logits, pair_marginal, pair_logit_matrix
                if return_pair_marginal:
                    return cls_logits, rank_logits, partner_logits, pair_marginal
                if return_pair_logit_matrix:
                    return cls_logits, rank_logits, partner_logits, pair_logit_matrix
                return cls_logits, rank_logits, partner_logits
            if return_pair and return_pair_marginal and return_pair_logit_matrix:
                return cls_logits, rank_logits, pair_output, pair_marginal, pair_logit_matrix
            if return_pair and return_pair_marginal:
                return cls_logits, rank_logits, pair_output, pair_marginal
            if return_pair and return_pair_logit_matrix:
                return cls_logits, rank_logits, pair_output, pair_logit_matrix
            if return_pair:
                return cls_logits, rank_logits, pair_output
            if return_pair_marginal and return_pair_logit_matrix:
                return cls_logits, rank_logits, pair_marginal, pair_logit_matrix
            if return_pair_marginal:
                return cls_logits, rank_logits, pair_marginal
            if return_pair_logit_matrix:
                return cls_logits, rank_logits, pair_logit_matrix
            return cls_logits, rank_logits
        if return_aux:
            if return_pair and return_pair_marginal and return_pair_logit_matrix:
                return logits, partner_logits, pair_output, pair_marginal, pair_logit_matrix
            if return_pair and return_pair_marginal:
                return logits, partner_logits, pair_output, pair_marginal
            if return_pair and return_pair_logit_matrix:
                return logits, partner_logits, pair_output, pair_logit_matrix
            if return_pair:
                return logits, partner_logits, pair_output
            if return_pair_marginal and return_pair_logit_matrix:
                return logits, partner_logits, pair_marginal, pair_logit_matrix
            if return_pair_marginal:
                return logits, partner_logits, pair_marginal
            if return_pair_logit_matrix:
                return logits, partner_logits, pair_logit_matrix
            return logits, partner_logits
        if return_pair and return_pair_marginal and return_pair_logit_matrix:
            return logits, pair_output, pair_marginal, pair_logit_matrix
        if return_pair and return_pair_marginal:
            return logits, pair_output, pair_marginal
        if return_pair and return_pair_logit_matrix:
            return logits, pair_output, pair_logit_matrix
        if return_pair:
            return logits, pair_output
        if return_pair_marginal and return_pair_logit_matrix:
            return logits, pair_marginal, pair_logit_matrix
        if return_pair_marginal:
            return logits, pair_marginal
        if return_pair_logit_matrix:
            return logits, pair_logit_matrix
        return logits


class CheckpointTriViewNet(nn.Module):
    """Architecture matching data/bib_triview_best.pt for warm-start or evaluation."""
    def __init__(self, res_dim=1024, atom_dim=37, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.res_in = nn.Linear(res_dim, hidden_dim)
        self.spatial_convs = nn.ModuleList([
            LegacyGraphConv(hidden_dim, hidden_dim),
            LegacyGraphConv(hidden_dim, hidden_dim),
        ])
        self.seq_gru = nn.GRU(
            hidden_dim,
            hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.atom_in = nn.Linear(atom_dim, hidden_dim)
        self.atom_convs = nn.ModuleList([
            LegacyGraphConv(hidden_dim, hidden_dim),
            LegacyGraphConv(hidden_dim, hidden_dim),
        ])
        self.atom_to_res = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.fuse_norm = nn.LayerNorm(hidden_dim)
        self.sub_attn = SubstructureAttention(hidden_dim, num_substructs=8, dropout=dropout)
        self.cls = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def sequence_view(self, h, batch):
        out = torch.zeros_like(h)
        num_graphs = int(batch.max().item()) + 1
        for gid in range(num_graphs):
            idx = (batch == gid).nonzero(as_tuple=False).squeeze(-1)
            seq_out, _ = self.seq_gru(h[idx].unsqueeze(0))
            out[idx] = seq_out.squeeze(0)
        return out

    def forward(
        self,
        res_x,
        res_edge,
        atom_x,
        atom_edge,
        atom2res,
        batch,
        res_coords=None,
        atom_coords=None,
        res_frames=None,
        surface_feats=None,
        sequence_feats=None,
        plm_feats=None,
        aux_plm_feats=None,
        return_aux=False,
        return_heads=False,
    ):
        spatial = F.relu(self.res_in(res_x))
        for conv in self.spatial_convs:
            spatial = conv(spatial, res_edge)

        seq = self.sequence_view(spatial, batch)

        atom = F.relu(self.atom_in(atom_x))
        for conv in self.atom_convs:
            atom = conv(atom, atom_edge)
        atom = F.relu(self.atom_to_res(atom_to_residue_pool(atom, atom2res, res_x.size(0))))

        view_weights = torch.softmax(self.gate(torch.cat((spatial, seq, atom), dim=-1)), dim=-1)
        fused = (
            view_weights[:, 0:1] * spatial
            + view_weights[:, 1:2] * seq
            + view_weights[:, 2:3] * atom
        )
        fused = self.fuse_norm(fused)
        fused = self.sub_attn(fused, batch)
        logits = self.cls(fused).squeeze(-1)
        if return_heads:
            if return_aux:
                return logits, logits, None
            return logits, logits
        if return_aux:
            return logits, None
        return logits


def load_checkpoint_if_available(model, checkpoint_path, device):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return False
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    print(f"Loaded checkpoint: {checkpoint_path}")
    return True


def find_existing_file(directory, filename):
    exact = os.path.join(directory, filename)
    if os.path.exists(exact):
        return exact
    target = filename.lower()
    if not os.path.isdir(directory):
        return None
    for name in os.listdir(directory):
        if name.lower() == target:
            return os.path.join(directory, name)
    return None


def dataset_path(filename):
    geo_path = find_existing_file(GEO_DATA_DIR, filename)
    geo_train_ready = find_existing_file(GEO_DATA_DIR, "Train335.pkl") is not None
    if geo_train_ready and geo_path is not None:
        return geo_path
    data_path = find_existing_file("data", filename)
    if data_path is not None:
        return data_path
    return os.path.join("data", filename)


def contact_contrast_dataset_coverage(data_list):
    positive_sites = 0
    eligible_sites = 0
    samples_with_contacts = 0
    for sample in data_list:
        labels = np.asarray(sample.get("label", []), dtype=np.float32).reshape(-1)
        positive_sites += int(np.sum(labels > 0.5))
        pair_index = np.asarray(
            sample.get("partner_pair_contact_index", np.empty((2, 0))),
            dtype=np.int64,
        )
        if pair_index.ndim != 2 or pair_index.shape[0] != 2 or pair_index.shape[1] == 0:
            continue
        target_indices = np.unique(pair_index[0])
        target_indices = target_indices[
            (target_indices >= 0) & (target_indices < labels.size)
        ]
        eligible = target_indices[labels[target_indices] > 0.5]
        if eligible.size > 0:
            samples_with_contacts += 1
            eligible_sites += int(eligible.size)
    return {
        "positive_interface_residues": int(positive_sites),
        "contact_contrast_eligible_residues": int(eligible_sites),
        "contact_contrast_coverage": (
            float(eligible_sites / positive_sites) if positive_sites else 0.0
        ),
        "samples_with_eligible_contacts": int(samples_with_contacts),
        "total_samples": int(len(data_list)),
    }


def available_test_datasets(require_geo=True):
    paths = []
    geo_train_ready = find_existing_file(GEO_DATA_DIR, "Train335.pkl") is not None
    for filename in TEST_PKL_FILES:
        path = None
        if require_geo and geo_train_ready:
            path = find_existing_file(GEO_DATA_DIR, filename)
        if path is None and not require_geo:
            path = find_existing_file("data", filename)
        if path is None:
            continue
        name = os.path.splitext(filename)[0]
        paths.append((name, path))
    return paths


def build_model(
    in_dim,
    atom_dim,
    device,
    plm_dim=None,
    aux_plm_dim=None,
    partner_conditioning=None,
    partner_top_k=None,
    partner_direct_fusion=None,
    partner_logit_mode=None,
    partner_delta_scale=None,
    partner_residue_encoder=None,
    partner_encoder_layers=None,
    partner_target_fusion=None,
    pair_contact_head=None,
):
    if plm_dim is None:
        plm_dim = PLM_FEATURE_DIM
    if aux_plm_dim is None:
        aux_plm_dim = AUX_PLM_FEATURE_DIM
    if partner_conditioning is None:
        partner_conditioning = PARTNER_CONDITIONING
    if partner_top_k is None:
        partner_top_k = PARTNER_TOP_K
    if partner_direct_fusion is None:
        partner_direct_fusion = PARTNER_DIRECT_FUSION
    if partner_logit_mode is None:
        partner_logit_mode = PARTNER_LOGIT_MODE
    if partner_delta_scale is None:
        partner_delta_scale = PARTNER_DELTA_SCALE
    if partner_residue_encoder is None:
        partner_residue_encoder = PARTNER_RESIDUE_ENCODER
    if partner_encoder_layers is None:
        partner_encoder_layers = PARTNER_ENCODER_LAYERS
    if partner_target_fusion is None:
        partner_target_fusion = PARTNER_TARGET_FUSION
    if pair_contact_head is None:
        pair_contact_head = PAIR_CONTACT_HEAD
    if MODEL_MODE == "checkpoint_finetune":
        model = CheckpointTriViewNet(
            res_dim=in_dim,
            atom_dim=atom_dim,
            hidden_dim=256,
            dropout=0.2,
        ).to(device)
        load_checkpoint_if_available(model, CHECKPOINT_PATH, device)
        return model

    return TriViewAtomResidueNet(
        res_dim=in_dim,
        atom_dim=atom_dim,
        surface_dim=SURFACE_FEATURE_DIM,
        sequence_dim=SEQUENCE_FEATURE_DIM,
        plm_dim=plm_dim,
        aux_plm_dim=aux_plm_dim,
        hidden_dim=256,
        num_layers=2,
        num_substructs=8,
        dropout=MODEL_DROPOUT,
        edge_dropout=MODEL_EDGE_DROPOUT,
        view_logit_fusion=VIEW_LOGIT_FUSION,
        partner_conditioning=partner_conditioning,
        partner_top_k=partner_top_k,
        partner_direct_fusion=partner_direct_fusion,
        partner_logit_mode=partner_logit_mode,
        partner_delta_scale=partner_delta_scale,
        partner_residue_encoder=partner_residue_encoder,
        partner_encoder_layers=partner_encoder_layers,
        partner_target_fusion=partner_target_fusion,
        pair_contact_head=pair_contact_head,
    ).to(device)


# ===================== Dataset & Collate =====================

class ProteinDataset(Dataset):
    """
    直接用原来的 train352.pkl / test60.pkl 里的残基图：
    每条样本对应一个蛋白复合体，里面是一个残基级图。
    """
    def __init__(
        self,
        protein_list,
        feature_mean=None,
        feature_std=None,
        atom_feature_mean=None,
        atom_feature_std=None,
        surface_feature_mean=None,
        surface_feature_std=None,
        sequence_feature_mean=None,
        sequence_feature_std=None,
        plm_feature_mean=None,
        plm_feature_std=None,
        plm_feature_dim=None,
        aux_plm_feature_mean=None,
        aux_plm_feature_std=None,
        aux_plm_feature_dim=None,
    ):
        self.proteins = protein_list
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.atom_feature_mean = atom_feature_mean
        self.atom_feature_std = atom_feature_std
        self.surface_feature_mean = surface_feature_mean
        self.surface_feature_std = surface_feature_std
        self.sequence_feature_mean = sequence_feature_mean
        self.sequence_feature_std = sequence_feature_std
        self.plm_feature_mean = plm_feature_mean
        self.plm_feature_std = plm_feature_std
        if plm_feature_dim is None and plm_feature_mean is not None:
            plm_feature_dim = int(np.asarray(plm_feature_mean).shape[0])
        self.plm_feature_dim = int(PLM_FEATURE_DIM if plm_feature_dim is None else plm_feature_dim)
        self.aux_plm_feature_mean = aux_plm_feature_mean
        self.aux_plm_feature_std = aux_plm_feature_std
        if aux_plm_feature_dim is None and aux_plm_feature_mean is not None:
            aux_plm_feature_dim = int(np.asarray(aux_plm_feature_mean).shape[0])
        self.aux_plm_feature_dim = int(AUX_PLM_FEATURE_DIM if aux_plm_feature_dim is None else aux_plm_feature_dim)

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, idx):
        p = self.proteins[idx]
        # residue_graph_node: (Nr, 1024) float64
        # residue_graph_edge: (2, Er)    int32
        # atom_graph_node: (Na, 37) float64
        # atom_graph_edge: (2, Ea) int32
        # a2r_map: len Na, atom index -> residue index
        # label: (Nr,) 0/1
        res_node = torch.from_numpy(p['residue_graph_node']).float()
        if self.feature_mean is not None and self.feature_std is not None:
            mean = torch.from_numpy(self.feature_mean).to(res_node.dtype)
            std = torch.from_numpy(self.feature_std).to(res_node.dtype)
            res_node = (res_node - mean) / std
        res_edge = torch.from_numpy(p['residue_graph_edge']).long()
        atom_node = torch.from_numpy(p['atom_graph_node']).float()
        if self.atom_feature_mean is not None and self.atom_feature_std is not None:
            atom_mean = torch.from_numpy(self.atom_feature_mean).to(atom_node.dtype)
            atom_std = torch.from_numpy(self.atom_feature_std).to(atom_node.dtype)
            atom_node = (atom_node - atom_mean) / atom_std
        if 'residue_geo_edge' in p:
            res_edge = torch.from_numpy(p['residue_geo_edge']).long()
        atom_edge_key = 'atom_geo_edge' if USE_ATOM_GEO_EDGES and 'atom_geo_edge' in p else 'atom_graph_edge'
        atom_edge = torch.from_numpy(p[atom_edge_key]).long()
        atom2res = torch.tensor(p['a2r_map'], dtype=torch.long)
        if 'residue_coords' in p:
            res_coords = torch.from_numpy(p['residue_coords']).float()
        else:
            res_coords = torch.zeros((res_node.size(0), 0), dtype=torch.float32)
        if 'residue_frames' in p:
            res_frames = torch.from_numpy(p['residue_frames']).float()
        else:
            res_frames = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(res_node.size(0), 1, 1)
        if 'atom_coords' in p:
            atom_coords = torch.from_numpy(p['atom_coords']).float()
        else:
            atom_coords = torch.zeros((atom_node.size(0), 0), dtype=torch.float32)
        if 'partner_contact_label' in p and 'partner_contact_mask' in p:
            partner_contact = torch.from_numpy(p['partner_contact_label']).float()
            partner_mask = torch.from_numpy(p['partner_contact_mask']).float()
        else:
            partner_contact = torch.zeros((res_node.size(0),), dtype=torch.float32)
            partner_mask = torch.zeros((res_node.size(0),), dtype=torch.float32)
        surface_feats = torch.from_numpy(get_surface_features(p)).float()
        if self.surface_feature_mean is not None and self.surface_feature_std is not None:
            surface_mean = torch.from_numpy(self.surface_feature_mean).to(surface_feats.dtype)
            surface_std = torch.from_numpy(self.surface_feature_std).to(surface_feats.dtype)
            surface_feats = (surface_feats - surface_mean) / surface_std
        sequence_feats = torch.from_numpy(get_sequence_features(p)).float()
        if self.sequence_feature_mean is not None and self.sequence_feature_std is not None:
            sequence_mean = torch.from_numpy(self.sequence_feature_mean).to(sequence_feats.dtype)
            sequence_std = torch.from_numpy(self.sequence_feature_std).to(sequence_feats.dtype)
            sequence_feats = (sequence_feats - sequence_mean) / sequence_std
        plm_feats = torch.from_numpy(get_plm_features(p, dim=self.plm_feature_dim)).float()
        if self.plm_feature_mean is not None and self.plm_feature_std is not None:
            plm_mean = torch.from_numpy(self.plm_feature_mean).to(plm_feats.dtype)
            plm_std = torch.from_numpy(self.plm_feature_std).to(plm_feats.dtype)
            plm_feats = (plm_feats - plm_mean) / plm_std
        aux_plm_feats = torch.from_numpy(
            get_plm_features(p, dim=self.aux_plm_feature_dim, key=AUX_PLM_FEATURE_KEY)
        ).float()
        if self.aux_plm_feature_mean is not None and self.aux_plm_feature_std is not None:
            aux_plm_mean = torch.from_numpy(self.aux_plm_feature_mean).to(aux_plm_feats.dtype)
            aux_plm_std = torch.from_numpy(self.aux_plm_feature_std).to(aux_plm_feats.dtype)
            aux_plm_feats = (aux_plm_feats - aux_plm_mean) / aux_plm_std
        partner_surface_feats = torch.from_numpy(get_partner_surface_features(p)).float()
        if (
            partner_surface_feats.numel() > 0
            and self.surface_feature_mean is not None
            and self.surface_feature_std is not None
        ):
            surface_mean = torch.from_numpy(self.surface_feature_mean).to(partner_surface_feats.dtype)
            surface_std = torch.from_numpy(self.surface_feature_std).to(partner_surface_feats.dtype)
            partner_surface_feats = (partner_surface_feats - surface_mean) / surface_std
        partner_sequence_feats = torch.from_numpy(get_partner_sequence_features(p)).float()
        if (
            partner_sequence_feats.numel() > 0
            and self.sequence_feature_mean is not None
            and self.sequence_feature_std is not None
        ):
            sequence_mean = torch.from_numpy(self.sequence_feature_mean).to(partner_sequence_feats.dtype)
            sequence_std = torch.from_numpy(self.sequence_feature_std).to(partner_sequence_feats.dtype)
            partner_sequence_feats = (partner_sequence_feats - sequence_mean) / sequence_std
        partner_plm_feats = torch.from_numpy(
            get_partner_plm_features(p, self.plm_feature_dim, PARTNER_PLM_FEATURE_KEY)
        ).float()
        if self.plm_feature_mean is not None and self.plm_feature_std is not None:
            partner_plm_mean = torch.from_numpy(self.plm_feature_mean).to(partner_plm_feats.dtype)
            partner_plm_std = torch.from_numpy(self.plm_feature_std).to(partner_plm_feats.dtype)
            partner_plm_feats = (partner_plm_feats - partner_plm_mean) / partner_plm_std
        partner_aux_plm_feats = torch.from_numpy(
            get_partner_plm_features(p, self.aux_plm_feature_dim, PARTNER_AUX_PLM_FEATURE_KEY)
        ).float()
        if self.aux_plm_feature_mean is not None and self.aux_plm_feature_std is not None:
            partner_aux_mean = torch.from_numpy(self.aux_plm_feature_mean).to(partner_aux_plm_feats.dtype)
            partner_aux_std = torch.from_numpy(self.aux_plm_feature_std).to(partner_aux_plm_feats.dtype)
            partner_aux_plm_feats = (partner_aux_plm_feats - partner_aux_mean) / partner_aux_std

        partner_width = PARTNER_FEATURE_DIM + self.plm_feature_dim + self.aux_plm_feature_dim
        partner_count = int(partner_surface_feats.size(0))
        partner_coords = torch.from_numpy(
            np.asarray(p.get("partner_residue_coords", np.zeros((0, 3))), dtype=np.float32)
        ).float()
        partner_frames = torch.from_numpy(
            np.asarray(p.get("partner_residue_frames", np.zeros((0, 3, 3))), dtype=np.float32)
        ).float()
        partner_edge = torch.from_numpy(
            np.asarray(p.get("partner_residue_geo_edge", np.empty((2, 0))), dtype=np.int64)
        ).long()
        partner_lengths = (
            partner_sequence_feats.size(0),
            partner_plm_feats.size(0),
            partner_aux_plm_feats.size(0),
            partner_coords.size(0) if partner_coords.dim() == 2 else -1,
            partner_frames.size(0) if partner_frames.dim() == 3 else -1,
        )
        partner_valid = (
            partner_count > 0
            and all(length == partner_count for length in partner_lengths)
            and partner_coords.shape == (partner_count, 3)
            and partner_frames.shape == (partner_count, 3, 3)
            and partner_edge.dim() == 2
            and partner_edge.size(0) == 2
        )
        if partner_valid and partner_edge.numel() > 0:
            partner_valid = bool(
                partner_edge.min().item() >= 0
                and partner_edge.max().item() < partner_count
            )
        if partner_valid:
            partner_feats = torch.cat(
                (
                    partner_surface_feats,
                    partner_sequence_feats,
                    partner_plm_feats,
                    partner_aux_plm_feats,
                ),
                dim=-1,
            )
        else:
            partner_feats = torch.zeros((0, partner_width), dtype=torch.float32)
            partner_coords = torch.zeros((0, 3), dtype=torch.float32)
            partner_frames = torch.zeros((0, 3, 3), dtype=torch.float32)
            partner_edge = torch.empty((2, 0), dtype=torch.long)
        partner_pair_contact_index = torch.empty((2, 0), dtype=torch.long)
        pair_contact_available = torch.tensor(False, dtype=torch.bool)
        if "partner_pair_contact_index" in p and partner_feats.size(0) > 0:
            pair_index = np.asarray(p["partner_pair_contact_index"], dtype=np.int64)
            if pair_index.ndim == 2 and pair_index.shape[0] == 2:
                pair_index = pair_index.copy()
                valid = (
                    (pair_index[0] >= 0)
                    & (pair_index[0] < res_node.size(0))
                    & (pair_index[1] >= 0)
                    & (pair_index[1] < partner_feats.size(0))
                )
                partner_pair_contact_index = torch.from_numpy(pair_index[:, valid]).long()
                pair_contact_available = torch.tensor(True, dtype=torch.bool)
        label = torch.from_numpy(p['label']).float()
        return (
            res_node, res_edge, atom_node, atom_edge, atom2res,
            res_coords, atom_coords, res_frames, surface_feats,
            sequence_feats, plm_feats, aux_plm_feats,
            partner_feats, partner_edge, partner_coords, partner_frames,
            partner_pair_contact_index, pair_contact_available,
            partner_contact, partner_mask, label
        )


def collate_proteins(batch):
    """
    把多个图打成一个 batch：
    - 节点特征拼接
    - 边索引做偏移
    - 生成 batch 向量，记录每个节点属于哪个图
    """
    node_feats = []
    edge_indices = []
    labels = []
    batch_vec = []
    atom_feats = []
    atom_edges = []
    atom2res_list = []
    res_coords_list = []
    atom_coords_list = []
    res_frames_list = []
    surface_feats_list = []
    sequence_feats_list = []
    plm_feats_list = []
    aux_plm_feats_list = []
    partner_feats_list = []
    partner_batch_list = []
    partner_edge_indices = []
    partner_coords_list = []
    partner_frames_list = []
    partner_pair_contact_indices = []
    pair_contact_graphs = []
    partner_contacts = []
    partner_masks = []

    node_offset = 0
    atom_offset = 0
    partner_offset = 0
    for gid, (
        res_node, res_edge, atom_node, atom_edge, atom2res,
        res_coords, atom_coords, res_frames, surface_feats, sequence_feats, plm_feats, aux_plm_feats,
        partner_feats, partner_edge, partner_coords, partner_frames,
        partner_pair_contact_index, pair_contact_available, partner_contact, partner_mask, label
    ) in enumerate(batch):
        N = res_node.size(0)
        A = atom_node.size(0)
        P = partner_feats.size(0)
        node_feats.append(res_node)
        atom_feats.append(atom_node)
        res_coords_list.append(res_coords)
        atom_coords_list.append(atom_coords)
        res_frames_list.append(res_frames)
        surface_feats_list.append(surface_feats)
        sequence_feats_list.append(sequence_feats)
        plm_feats_list.append(plm_feats)
        aux_plm_feats_list.append(aux_plm_feats)
        if P > 0:
            partner_feats_list.append(partner_feats)
            partner_batch_list.append(torch.full((P,), gid, dtype=torch.long))
            partner_coords_list.append(partner_coords)
            partner_frames_list.append(partner_frames)
            pei = partner_edge.clone()
            pei[0] += partner_offset
            pei[1] += partner_offset
            if ADD_REVERSE_EDGES:
                partner_rev = torch.stack((pei[1], pei[0]), dim=0)
                pei = torch.cat((pei, partner_rev), dim=1)
            partner_edge_indices.append(pei)
        if bool(pair_contact_available.item()) and P > 0:
            pair_contact_graphs.append(gid)
            if partner_pair_contact_index.numel() > 0:
                pair_idx = partner_pair_contact_index.clone()
                valid = (
                    (pair_idx[0] >= 0)
                    & (pair_idx[0] < N)
                    & (pair_idx[1] >= 0)
                    & (pair_idx[1] < P)
                )
                pair_idx = pair_idx[:, valid]
                if pair_idx.numel() > 0:
                    pair_idx[0] += node_offset
                    pair_idx[1] += partner_offset
                    partner_pair_contact_indices.append(pair_idx)
        partner_contacts.append(partner_contact)
        partner_masks.append(partner_mask)
        labels.append(label)
        batch_vec.append(torch.full((N,), gid, dtype=torch.long))

        ei = res_edge.clone()
        ei[0] += node_offset
        ei[1] += node_offset
        if ADD_REVERSE_EDGES:
            rev_ei = torch.stack((ei[1], ei[0]), dim=0)
            ei = torch.cat((ei, rev_ei), dim=1)
        edge_indices.append(ei)

        aei = atom_edge.clone()
        aei[0] += atom_offset
        aei[1] += atom_offset
        if ADD_REVERSE_EDGES:
            rev_aei = torch.stack((aei[1], aei[0]), dim=0)
            aei = torch.cat((aei, rev_aei), dim=1)
        atom_edges.append(aei)
        atom2res_list.append(atom2res + node_offset)

        node_offset += N
        atom_offset += A
        partner_offset += P

    node_feats = torch.cat(node_feats, dim=0)                     # [N_total, F]
    atom_feats = torch.cat(atom_feats, dim=0)                     # [A_total, F_atom]
    if all(coords.dim() == 2 and coords.size(-1) == 3 for coords in res_coords_list):
        res_coords = torch.cat(res_coords_list, dim=0)
    else:
        res_coords = torch.zeros((node_offset, 0), dtype=torch.float32)
    if all(coords.dim() == 2 and coords.size(-1) == 3 for coords in atom_coords_list):
        atom_coords = torch.cat(atom_coords_list, dim=0)
    else:
        atom_coords = torch.zeros((atom_offset, 0), dtype=torch.float32)
    res_frames = torch.cat(res_frames_list, dim=0)
    surface_feats = torch.cat(surface_feats_list, dim=0)
    sequence_feats = torch.cat(sequence_feats_list, dim=0)
    plm_feats = torch.cat(plm_feats_list, dim=0)
    aux_plm_feats = torch.cat(aux_plm_feats_list, dim=0)
    if partner_feats_list:
        partner_feats = torch.cat(partner_feats_list, dim=0)
        partner_batch = torch.cat(partner_batch_list, dim=0)
        partner_coords = torch.cat(partner_coords_list, dim=0)
        partner_frames = torch.cat(partner_frames_list, dim=0)
        partner_edge = (
            torch.cat(partner_edge_indices, dim=1)
            if partner_edge_indices
            else torch.empty((2, 0), dtype=torch.long)
        )
    else:
        partner_width = int(batch[0][12].size(1)) if batch else PARTNER_FEATURE_DIM
        partner_feats = torch.zeros((0, partner_width), dtype=torch.float32)
        partner_batch = torch.zeros((0,), dtype=torch.long)
        partner_coords = torch.zeros((0, 3), dtype=torch.float32)
        partner_frames = torch.zeros((0, 3, 3), dtype=torch.float32)
        partner_edge = torch.empty((2, 0), dtype=torch.long)
    if partner_pair_contact_indices:
        partner_pair_contact_index = torch.cat(partner_pair_contact_indices, dim=1)
    else:
        partner_pair_contact_index = torch.empty((2, 0), dtype=torch.long)
    pair_contact_graphs = torch.tensor(pair_contact_graphs, dtype=torch.long)
    partner_contact = torch.cat(partner_contacts, dim=0)
    partner_mask = torch.cat(partner_masks, dim=0)
    labels = torch.cat(labels, dim=0)                             # [N_total]
    batch_vec = torch.cat(batch_vec, dim=0)                       # [N_total]
    edge_index = torch.cat(edge_indices, dim=1) if edge_indices else torch.empty(2, 0, dtype=torch.long)
    atom_edge_index = torch.cat(atom_edges, dim=1) if atom_edges else torch.empty(2, 0, dtype=torch.long)
    atom2res = torch.cat(atom2res_list, dim=0)

    return (
        node_feats, edge_index, atom_feats, atom_edge_index, atom2res,
        res_coords, atom_coords, res_frames, surface_feats,
        sequence_feats, plm_feats, aux_plm_feats, partner_feats, partner_batch,
        partner_edge, partner_coords, partner_frames,
        partner_pair_contact_index, pair_contact_graphs,
        partner_contact, partner_mask, batch_vec, labels
    )


# ===================== Train & Evaluate =====================

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, fold_idx=None, max_grad_norm=5.0, ema=None):
    model.train()
    total_loss = 0.0
    total_nodes = 0
    rank_loss_scale = 1.0
    if TWO_HEAD_BINDING and TWO_HEAD_RANK_WARMUP_EPOCHS > 0:
        rank_loss_scale = min(1.0, float(epoch) / float(TWO_HEAD_RANK_WARMUP_EPOCHS))
    pair_loss_scale = 1.0
    if PAIR_CONTACT_LOSS and PAIR_CONTACT_WARMUP_EPOCHS > 0:
        pair_loss_scale = min(1.0, float(epoch) / float(PAIR_CONTACT_WARMUP_EPOCHS))
    marginal_consistency_scale = 1.0
    if PAIR_MARGINAL_CONSISTENCY and PAIR_MARGINAL_CONSISTENCY_WARMUP_EPOCHS > 0:
        marginal_consistency_scale = min(
            1.0,
            float(epoch) / float(PAIR_MARGINAL_CONSISTENCY_WARMUP_EPOCHS),
        )
    marginal_contrast_scale = 1.0
    if PAIR_MARGINAL_CONTRAST and PAIR_MARGINAL_CONTRAST_WARMUP_EPOCHS > 0:
        marginal_contrast_scale = min(
            1.0,
            float(epoch) / float(PAIR_MARGINAL_CONTRAST_WARMUP_EPOCHS),
        )
    pair_contact_contrast_scale = 1.0
    if PAIR_CONTACT_CONTRAST and PAIR_CONTACT_CONTRAST_WARMUP_EPOCHS > 0:
        pair_contact_contrast_scale = min(
            1.0,
            float(epoch) / float(PAIR_CONTACT_CONTRAST_WARMUP_EPOCHS),
        )
    contrast_loss_scale = 1.0
    if PARTNER_CONTRAST and PARTNER_CONTRAST_WARMUP_EPOCHS > 0:
        contrast_loss_scale = min(1.0, float(epoch) / float(PARTNER_CONTRAST_WARMUP_EPOCHS))

    if fold_idx is None:
        desc = f"Epoch {epoch}"
    else:
        desc = f"Fold {fold_idx+1} Epoch {epoch}"
    pbar = tqdm(loader, desc=desc, ncols=100)

    for (
        x, edge_index, atom_x, atom_edge_index, atom2res,
        res_coords, atom_coords, res_frames, surface_feats, sequence_feats, plm_feats, aux_plm_feats,
        partner_feats, partner_batch, partner_edge, partner_coords, partner_frames,
        partner_pair_contact_index, pair_contact_graphs,
        partner_contact, partner_mask, batch_vec, y
    ) in pbar:
        x = x.to(device)
        edge_index = edge_index.to(device)
        atom_x = atom_x.to(device)
        atom_edge_index = atom_edge_index.to(device)
        atom2res = atom2res.to(device)
        res_coords = res_coords.to(device)
        atom_coords = atom_coords.to(device)
        res_frames = res_frames.to(device)
        surface_feats = surface_feats.to(device)
        sequence_feats = sequence_feats.to(device)
        plm_feats = plm_feats.to(device)
        aux_plm_feats = aux_plm_feats.to(device)
        partner_feats = partner_feats.to(device)
        partner_batch = partner_batch.to(device)
        partner_edge = partner_edge.to(device)
        partner_coords = partner_coords.to(device)
        partner_frames = partner_frames.to(device)
        partner_pair_contact_index = partner_pair_contact_index.to(device)
        pair_contact_graphs = pair_contact_graphs.to(device)
        partner_contact = partner_contact.to(device)
        partner_mask = partner_mask.to(device)
        batch_vec = batch_vec.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pair_output = None
        pair_marginal = None
        mismatch_logits_for_contrast = None
        mismatch_pair_marginal = None
        mismatch_pair_output = None
        need_pair_output = PAIR_CONTACT_LOSS or PAIR_CONTACT_CONTRAST
        need_pair_marginal = PAIR_MARGINAL_CONSISTENCY or PAIR_MARGINAL_CONTRAST
        if TWO_HEAD_BINDING and hasattr(model, "rank_cls"):
            if need_pair_output:
                pair_result = model(
                    x, edge_index, atom_x, atom_edge_index, atom2res,
                    batch_vec, res_coords, atom_coords, res_frames, surface_feats, sequence_feats, plm_feats,
                    aux_plm_feats, partner_feats, partner_batch,
                    partner_edge=partner_edge,
                    partner_coords=partner_coords,
                    partner_frames=partner_frames,
                    pair_contact_index=partner_pair_contact_index,
                    pair_contact_graphs=pair_contact_graphs,
                    return_aux=True,
                    return_heads=True,
                    return_pair=True,
                    return_pair_marginal=need_pair_marginal,
                )
                if need_pair_marginal:
                    cls_logits, rank_logits, partner_logits, pair_output, pair_marginal = pair_result
                else:
                    cls_logits, rank_logits, partner_logits, pair_output = pair_result
            elif need_pair_marginal:
                cls_logits, rank_logits, partner_logits, pair_marginal = model(
                    x, edge_index, atom_x, atom_edge_index, atom2res,
                    batch_vec, res_coords, atom_coords, res_frames, surface_feats, sequence_feats, plm_feats,
                    aux_plm_feats, partner_feats, partner_batch,
                    partner_edge=partner_edge,
                    partner_coords=partner_coords,
                    partner_frames=partner_frames,
                    return_aux=True,
                    return_heads=True,
                    return_pair_marginal=True,
                )
            else:
                cls_logits, rank_logits, partner_logits = model(
                    x, edge_index, atom_x, atom_edge_index, atom2res,
                    batch_vec, res_coords, atom_coords, res_frames, surface_feats, sequence_feats, plm_feats,
                    aux_plm_feats, partner_feats, partner_batch,
                    partner_edge=partner_edge,
                    partner_coords=partner_coords,
                    partner_frames=partner_frames,
                    return_aux=True,
                    return_heads=True,
                )
        else:
            if need_pair_output:
                pair_result = model(
                    x, edge_index, atom_x, atom_edge_index, atom2res,
                    batch_vec, res_coords, atom_coords, res_frames, surface_feats, sequence_feats, plm_feats,
                    aux_plm_feats, partner_feats, partner_batch,
                    partner_edge=partner_edge,
                    partner_coords=partner_coords,
                    partner_frames=partner_frames,
                    pair_contact_index=partner_pair_contact_index,
                    pair_contact_graphs=pair_contact_graphs,
                    return_aux=True,
                    return_pair=True,
                    return_pair_marginal=need_pair_marginal,
                )
                if need_pair_marginal:
                    cls_logits, partner_logits, pair_output, pair_marginal = pair_result
                else:
                    cls_logits, partner_logits, pair_output = pair_result
            elif need_pair_marginal:
                cls_logits, partner_logits, pair_marginal = model(
                    x, edge_index, atom_x, atom_edge_index, atom2res,
                    batch_vec, res_coords, atom_coords, res_frames, surface_feats, sequence_feats, plm_feats,
                    aux_plm_feats, partner_feats, partner_batch,
                    partner_edge=partner_edge,
                    partner_coords=partner_coords,
                    partner_frames=partner_frames,
                    return_aux=True,
                    return_pair_marginal=True,
                )
            else:
                cls_logits, partner_logits = model(
                    x, edge_index, atom_x, atom_edge_index, atom2res,
                    batch_vec, res_coords, atom_coords, res_frames, surface_feats, sequence_feats, plm_feats,
                    aux_plm_feats, partner_feats, partner_batch,
                    partner_edge=partner_edge,
                    partner_coords=partner_coords,
                    partner_frames=partner_frames,
                    return_aux=True,
                )
            rank_logits = None
        if (
            PARTNER_CONTRAST
            or PAIR_MARGINAL_CONTRAST
            or PAIR_CONTACT_CONTRAST
        ) and PARTNER_CONDITIONING:
            (
                mismatch_partner_feats,
                mismatch_partner_batch,
                mismatch_partner_edge,
                mismatch_partner_coords,
                mismatch_partner_frames,
                has_mismatch,
            ) = make_mismatched_partner_graph_batch(
                partner_feats,
                partner_batch,
                partner_edge,
                partner_coords,
                partner_frames,
                batch_vec,
                length_match=PAIR_CONTACT_CONTRAST,
            )
            if has_mismatch:
                if TWO_HEAD_BINDING and hasattr(model, "rank_cls"):
                    mismatch_result = model(
                        x, edge_index, atom_x, atom_edge_index, atom2res,
                        batch_vec, res_coords, atom_coords, res_frames, surface_feats, sequence_feats, plm_feats,
                        aux_plm_feats, mismatch_partner_feats, mismatch_partner_batch,
                        partner_edge=mismatch_partner_edge,
                        partner_coords=mismatch_partner_coords,
                        partner_frames=mismatch_partner_frames,
                        pair_contact_graphs=pair_contact_graphs,
                        return_heads=True,
                        return_pair=PAIR_CONTACT_CONTRAST,
                        return_pair_marginal=PAIR_MARGINAL_CONTRAST,
                    )
                    if PAIR_CONTACT_CONTRAST and PAIR_MARGINAL_CONTRAST:
                        (
                            mismatch_cls_logits,
                            mismatch_rank_logits,
                            mismatch_pair_output,
                            mismatch_pair_marginal,
                        ) = mismatch_result
                    elif PAIR_CONTACT_CONTRAST:
                        mismatch_cls_logits, mismatch_rank_logits, mismatch_pair_output = mismatch_result
                    elif PAIR_MARGINAL_CONTRAST:
                        mismatch_cls_logits, mismatch_rank_logits, mismatch_pair_marginal = mismatch_result
                    else:
                        mismatch_cls_logits, mismatch_rank_logits = mismatch_result
                    mismatch_logits_for_contrast = (
                        (1.0 - TWO_HEAD_RANK_FUSION) * mismatch_cls_logits
                        + TWO_HEAD_RANK_FUSION * mismatch_rank_logits
                    )
                else:
                    mismatch_result = model(
                        x, edge_index, atom_x, atom_edge_index, atom2res,
                        batch_vec, res_coords, atom_coords, res_frames, surface_feats, sequence_feats, plm_feats,
                        aux_plm_feats, mismatch_partner_feats, mismatch_partner_batch,
                        partner_edge=mismatch_partner_edge,
                        partner_coords=mismatch_partner_coords,
                        partner_frames=mismatch_partner_frames,
                        pair_contact_graphs=pair_contact_graphs,
                        return_pair=PAIR_CONTACT_CONTRAST,
                        return_pair_marginal=PAIR_MARGINAL_CONTRAST,
                    )
                    if PAIR_CONTACT_CONTRAST and PAIR_MARGINAL_CONTRAST:
                        (
                            mismatch_logits_for_contrast,
                            mismatch_pair_output,
                            mismatch_pair_marginal,
                        ) = mismatch_result
                    elif PAIR_CONTACT_CONTRAST:
                        mismatch_logits_for_contrast, mismatch_pair_output = mismatch_result
                    elif PAIR_MARGINAL_CONTRAST:
                        mismatch_logits_for_contrast, mismatch_pair_marginal = mismatch_result
                    else:
                        mismatch_logits_for_contrast = mismatch_result
        train_targets = geometry_patch_soft_targets(y, edge_index)
        if rank_logits is None:
            loss = criterion(
                cls_logits,
                train_targets,
                hard_targets=y,
                aux_logits=partner_logits,
                aux_targets=partner_contact,
                aux_mask=partner_mask,
                rank_loss_scale=rank_loss_scale,
            )
        else:
            loss = criterion(
                cls_logits,
                rank_logits,
                train_targets,
                hard_targets=y,
                aux_logits=partner_logits,
                aux_targets=partner_contact,
                aux_mask=partner_mask,
            )
        pair_loss = sparse_pair_contact_loss(pair_output)
        if pair_loss is not None:
            loss = loss + PAIR_CONTACT_LOSS_WEIGHT * pair_loss_scale * pair_loss
        marginal_consistency = pair_marginal_consistency_loss(
            pair_marginal,
            cls_logits,
            rank_logits,
            batch_vec,
            pair_contact_graphs,
        )
        if marginal_consistency is not None:
            loss = (
                loss
                + PAIR_MARGINAL_CONSISTENCY_WEIGHT
                * marginal_consistency_scale
                * marginal_consistency
            )
        marginal_contrast = pair_marginal_partner_contrastive_loss(
            pair_marginal,
            mismatch_pair_marginal,
            y,
            batch_vec,
            pair_contact_graphs,
        )
        if marginal_contrast is not None:
            loss = (
                loss
                + PAIR_MARGINAL_CONTRAST_WEIGHT
                * marginal_contrast_scale
                * marginal_contrast
            )
        pair_contact_contrast, _ = pair_contact_hard_partner_loss(
            pair_output,
            mismatch_pair_output,
            y,
            batch_vec,
        )
        if pair_contact_contrast is not None:
            loss = (
                loss
                + PAIR_CONTACT_CONTRAST_WEIGHT
                * pair_contact_contrast_scale
                * pair_contact_contrast
            )
        true_logits_for_contrast = cls_logits
        if rank_logits is not None:
            true_logits_for_contrast = (
                (1.0 - TWO_HEAD_RANK_FUSION) * cls_logits
                + TWO_HEAD_RANK_FUSION * rank_logits
            )
        contrast_loss = partner_contrastive_loss(
            true_logits_for_contrast,
            mismatch_logits_for_contrast,
            y,
            batch_vec,
        )
        if contrast_loss is not None:
            loss = loss + PARTNER_CONTRAST_WEIGHT * contrast_loss_scale * contrast_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if ema is not None:
            ema.update(model)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_nodes += bs
        pbar.set_postfix(loss=total_loss / total_nodes)

    return total_loss / total_nodes


def evaluate(model, loader, device, split_name="Val", threshold=None):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for (
            x, edge_index, atom_x, atom_edge_index, atom2res,
            res_coords, atom_coords, res_frames, surface_feats, sequence_feats, plm_feats, aux_plm_feats,
            partner_feats, partner_batch, partner_edge, partner_coords, partner_frames,
            partner_pair_contact_index, pair_contact_graphs,
            partner_contact, partner_mask, batch_vec, y
        ) in loader:
            x = x.to(device)
            edge_index = edge_index.to(device)
            atom_x = atom_x.to(device)
            atom_edge_index = atom_edge_index.to(device)
            atom2res = atom2res.to(device)
            res_coords = res_coords.to(device)
            atom_coords = atom_coords.to(device)
            res_frames = res_frames.to(device)
            surface_feats = surface_feats.to(device)
            sequence_feats = sequence_feats.to(device)
            plm_feats = plm_feats.to(device)
            aux_plm_feats = aux_plm_feats.to(device)
            partner_feats = partner_feats.to(device)
            partner_batch = partner_batch.to(device)
            partner_edge = partner_edge.to(device)
            partner_coords = partner_coords.to(device)
            partner_frames = partner_frames.to(device)
            batch_vec = batch_vec.to(device)

            logits = model(
                x, edge_index, atom_x, atom_edge_index, atom2res,
                batch_vec, res_coords, atom_coords, res_frames, surface_feats, sequence_feats, plm_feats,
                aux_plm_feats, partner_feats, partner_batch,
                partner_edge=partner_edge,
                partner_coords=partner_coords,
                partner_frames=partner_frames,
            )      # [N_total]
            probs = torch.sigmoid(logits)
            probs = smooth_node_scores(probs, edge_index).cpu().numpy()   # [N_total]
            all_probs.append(probs)
            all_labels.append(y.numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_score = np.concatenate(all_probs, axis=0)
    return metrics_from_scores(y_true, y_score, split_name=split_name, threshold=threshold)


def evaluate_pair_contacts(model, loader, device, split_name="PairContact-Val"):
    if not PAIR_CONTACT_LOSS:
        return None
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for (
            x, edge_index, atom_x, atom_edge_index, atom2res,
            res_coords, atom_coords, res_frames, surface_feats, sequence_feats, plm_feats, aux_plm_feats,
            partner_feats, partner_batch, partner_edge, partner_coords, partner_frames,
            partner_pair_contact_index, pair_contact_graphs,
            partner_contact, partner_mask, batch_vec, y
        ) in loader:
            x = x.to(device)
            edge_index = edge_index.to(device)
            atom_x = atom_x.to(device)
            atom_edge_index = atom_edge_index.to(device)
            atom2res = atom2res.to(device)
            res_coords = res_coords.to(device)
            atom_coords = atom_coords.to(device)
            res_frames = res_frames.to(device)
            surface_feats = surface_feats.to(device)
            sequence_feats = sequence_feats.to(device)
            plm_feats = plm_feats.to(device)
            aux_plm_feats = aux_plm_feats.to(device)
            partner_feats = partner_feats.to(device)
            partner_batch = partner_batch.to(device)
            partner_edge = partner_edge.to(device)
            partner_coords = partner_coords.to(device)
            partner_frames = partner_frames.to(device)
            partner_pair_contact_index = partner_pair_contact_index.to(device)
            pair_contact_graphs = pair_contact_graphs.to(device)
            batch_vec = batch_vec.to(device)

            if TWO_HEAD_BINDING and hasattr(model, "rank_cls"):
                _, _, _, pair_output = model(
                    x, edge_index, atom_x, atom_edge_index, atom2res,
                    batch_vec, res_coords, atom_coords, res_frames, surface_feats, sequence_feats, plm_feats,
                    aux_plm_feats, partner_feats, partner_batch,
                    partner_edge=partner_edge,
                    partner_coords=partner_coords,
                    partner_frames=partner_frames,
                    pair_contact_index=partner_pair_contact_index,
                    pair_contact_graphs=pair_contact_graphs,
                    return_aux=True,
                    return_heads=True,
                    return_pair=True,
                )
            else:
                _, _, pair_output = model(
                    x, edge_index, atom_x, atom_edge_index, atom2res,
                    batch_vec, res_coords, atom_coords, res_frames, surface_feats, sequence_feats, plm_feats,
                    aux_plm_feats, partner_feats, partner_batch,
                    partner_edge=partner_edge,
                    partner_coords=partner_coords,
                    partner_frames=partner_frames,
                    pair_contact_index=partner_pair_contact_index,
                    pair_contact_graphs=pair_contact_graphs,
                    return_aux=True,
                    return_pair=True,
                )

            if pair_output is None:
                continue
            pair_logits, pair_targets = pair_output[:2]
            if pair_logits.numel() == 0:
                continue
            all_probs.append(torch.sigmoid(pair_logits).cpu().numpy())
            all_targets.append(pair_targets.cpu().numpy())

    if not all_targets:
        print(f"\n===== {split_name} Pair Contact =====")
        print("No labelled candidate pairs available.")
        return None

    targets = np.concatenate(all_targets, axis=0)
    probs = np.concatenate(all_probs, axis=0)
    if np.unique(targets).size < 2:
        auroc = float("nan")
        aupr = float("nan")
    else:
        auroc = compute_auc_roc(targets, probs)
        aupr = compute_auc_pr(targets, probs)
    print(f"\n===== {split_name} Pair Contact =====")
    print(f"Candidates: {targets.size}")
    print(f"Positive rate: {targets.mean():.4f}")
    print(f"Pair AUROC: {auroc:.4f}")
    print(f"Pair AUPRC: {aupr:.4f}")
    return {
        "pair_auc_roc": auroc,
        "pair_auc_pr": aupr,
        "pair_candidates": int(targets.size),
        "pair_positive_rate": float(targets.mean()),
    }


def predict_scores(model, loader, device):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for (
            x, edge_index, atom_x, atom_edge_index, atom2res,
            res_coords, atom_coords, res_frames, surface_feats, sequence_feats, plm_feats, aux_plm_feats,
            partner_feats, partner_batch, partner_edge, partner_coords, partner_frames,
            partner_pair_contact_index, pair_contact_graphs,
            partner_contact, partner_mask, batch_vec, y
        ) in loader:
            x = x.to(device)
            edge_index = edge_index.to(device)
            atom_x = atom_x.to(device)
            atom_edge_index = atom_edge_index.to(device)
            atom2res = atom2res.to(device)
            res_coords = res_coords.to(device)
            atom_coords = atom_coords.to(device)
            res_frames = res_frames.to(device)
            surface_feats = surface_feats.to(device)
            sequence_feats = sequence_feats.to(device)
            plm_feats = plm_feats.to(device)
            aux_plm_feats = aux_plm_feats.to(device)
            partner_feats = partner_feats.to(device)
            partner_batch = partner_batch.to(device)
            partner_edge = partner_edge.to(device)
            partner_coords = partner_coords.to(device)
            partner_frames = partner_frames.to(device)
            batch_vec = batch_vec.to(device)

            logits = model(
                x, edge_index, atom_x, atom_edge_index, atom2res,
                batch_vec, res_coords, atom_coords, res_frames, surface_feats, sequence_feats, plm_feats,
                aux_plm_feats, partner_feats, partner_batch,
                partner_edge=partner_edge,
                partner_coords=partner_coords,
                partner_frames=partner_frames,
            )
            probs = torch.sigmoid(logits)
            probs = smooth_node_scores(probs, edge_index).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())

    return np.concatenate(all_labels, axis=0), np.concatenate(all_probs, axis=0)


def predict_scores_with_aux(model, loader, device):
    model.eval()
    all_labels = []
    all_main_probs = []
    all_aux_probs = []

    with torch.no_grad():
        for (
            x, edge_index, atom_x, atom_edge_index, atom2res,
            res_coords, atom_coords, res_frames, surface_feats, sequence_feats, plm_feats, aux_plm_feats,
            partner_feats, partner_batch, partner_edge, partner_coords, partner_frames,
            partner_pair_contact_index, pair_contact_graphs,
            partner_contact, partner_mask, batch_vec, y
        ) in loader:
            x = x.to(device)
            edge_index = edge_index.to(device)
            atom_x = atom_x.to(device)
            atom_edge_index = atom_edge_index.to(device)
            atom2res = atom2res.to(device)
            res_coords = res_coords.to(device)
            atom_coords = atom_coords.to(device)
            res_frames = res_frames.to(device)
            surface_feats = surface_feats.to(device)
            sequence_feats = sequence_feats.to(device)
            plm_feats = plm_feats.to(device)
            aux_plm_feats = aux_plm_feats.to(device)
            partner_feats = partner_feats.to(device)
            partner_batch = partner_batch.to(device)
            partner_edge = partner_edge.to(device)
            partner_coords = partner_coords.to(device)
            partner_frames = partner_frames.to(device)
            batch_vec = batch_vec.to(device)

            logits, aux_logits = model(
                x,
                edge_index,
                atom_x,
                atom_edge_index,
                atom2res,
                batch_vec,
                res_coords,
                atom_coords,
                res_frames,
                surface_feats,
                sequence_feats,
                plm_feats,
                aux_plm_feats,
                partner_feats,
                partner_batch,
                partner_edge=partner_edge,
                partner_coords=partner_coords,
                partner_frames=partner_frames,
                return_aux=True,
            )
            main_probs = smooth_node_scores(torch.sigmoid(logits), edge_index).cpu().numpy()
            if aux_logits is None:
                aux_probs = main_probs
            else:
                aux_probs = smooth_node_scores(torch.sigmoid(aux_logits), edge_index).cpu().numpy()
            all_main_probs.append(main_probs)
            all_aux_probs.append(aux_probs)
            all_labels.append(y.numpy())

    return (
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_main_probs, axis=0),
        np.concatenate(all_aux_probs, axis=0),
    )


def predict_scores_with_heads(model, loader, device, rank_fusion=None):
    model.eval()
    if rank_fusion is None:
        rank_fusion = TWO_HEAD_RANK_FUSION
    all_labels = []
    all_fused_probs = []
    all_aux_probs = []
    all_cls_probs = []
    all_rank_probs = []

    with torch.no_grad():
        for (
            x, edge_index, atom_x, atom_edge_index, atom2res,
            res_coords, atom_coords, res_frames, surface_feats, sequence_feats, plm_feats, aux_plm_feats,
            partner_feats, partner_batch, partner_edge, partner_coords, partner_frames,
            partner_pair_contact_index, pair_contact_graphs,
            partner_contact, partner_mask, batch_vec, y
        ) in loader:
            x = x.to(device)
            edge_index = edge_index.to(device)
            atom_x = atom_x.to(device)
            atom_edge_index = atom_edge_index.to(device)
            atom2res = atom2res.to(device)
            res_coords = res_coords.to(device)
            atom_coords = atom_coords.to(device)
            res_frames = res_frames.to(device)
            surface_feats = surface_feats.to(device)
            sequence_feats = sequence_feats.to(device)
            plm_feats = plm_feats.to(device)
            aux_plm_feats = aux_plm_feats.to(device)
            partner_feats = partner_feats.to(device)
            partner_batch = partner_batch.to(device)
            partner_edge = partner_edge.to(device)
            partner_coords = partner_coords.to(device)
            partner_frames = partner_frames.to(device)
            batch_vec = batch_vec.to(device)

            cls_logits, rank_logits, aux_logits = model(
                x,
                edge_index,
                atom_x,
                atom_edge_index,
                atom2res,
                batch_vec,
                res_coords,
                atom_coords,
                res_frames,
                surface_feats,
                sequence_feats,
                plm_feats,
                aux_plm_feats,
                partner_feats,
                partner_batch,
                partner_edge=partner_edge,
                partner_coords=partner_coords,
                partner_frames=partner_frames,
                return_aux=True,
                return_heads=True,
            )
            if not getattr(model, "use_rank_head", False):
                rank_logits = cls_logits
            fused_logits = (1.0 - rank_fusion) * cls_logits + rank_fusion * rank_logits

            fused_probs = smooth_node_scores(torch.sigmoid(fused_logits), edge_index).cpu().numpy()
            cls_probs = smooth_node_scores(torch.sigmoid(cls_logits), edge_index).cpu().numpy()
            rank_probs = smooth_node_scores(torch.sigmoid(rank_logits), edge_index).cpu().numpy()
            if aux_logits is None:
                aux_probs = fused_probs
            else:
                aux_probs = smooth_node_scores(torch.sigmoid(aux_logits), edge_index).cpu().numpy()
            all_fused_probs.append(fused_probs)
            all_cls_probs.append(cls_probs)
            all_rank_probs.append(rank_probs)
            all_aux_probs.append(aux_probs)
            all_labels.append(y.numpy())

    return (
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_fused_probs, axis=0),
        np.concatenate(all_aux_probs, axis=0),
        np.concatenate(all_cls_probs, axis=0),
        np.concatenate(all_rank_probs, axis=0),
    )


# ===================== Main: 5-fold CV on train, evaluated on multiple held-out test sets =====================

if __name__ == "__main__":
    seed_everything()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Model mode:", MODEL_MODE)
    print("Seed:", SEED)
    print("Output dir:", OUTPUT_DIR)
    print("Two-head binding:", TWO_HEAD_BINDING)
    if TWO_HEAD_BINDING:
        print(
            "Two-head settings: "
            f"rank_loss_weight={TWO_HEAD_RANK_LOSS_WEIGHT:.2f}, "
            f"consistency_weight={TWO_HEAD_CONSISTENCY_WEIGHT:.2f}, "
            f"rank_fusion={TWO_HEAD_RANK_FUSION:.2f}, "
            f"rank_warmup_epochs={TWO_HEAD_RANK_WARMUP_EPOCHS}"
        )
    print(f"View-logit fusion: {VIEW_LOGIT_FUSION:.2f}")
    print(
        "Regularization: "
        f"dropout={MODEL_DROPOUT:.2f}, "
        f"edge_dropout={MODEL_EDGE_DROPOUT:.2f}, "
        f"weight_decay={OPT_WEIGHT_DECAY:.1e}"
    )
    print("Patch label distribution:", PATCH_LABEL_DISTRIBUTION)
    if PATCH_LABEL_DISTRIBUTION:
        print(
            "Patch label settings: "
            f"steps={PATCH_LABEL_STEPS}, "
            f"alpha={PATCH_LABEL_ALPHA:.2f}, "
            f"max_unlabeled={PATCH_LABEL_MAX_UNLABELED:.2f}"
        )
    print(
        "Checkpoint selection: "
        f"metric={CHECKPOINT_SELECTION_METRIC}, "
        f"aupr_weight={CHECKPOINT_SELECTION_AUPR_WEIGHT:.2f}"
    )

    # ---- 1) 读数据 ----
    train_path = dataset_path("Train335.pkl")

    # 多个测试集（按你的要求）；新增测试集只要放进 data/geo 即可自动纳入。
    test_paths = [] if SKIP_TEST_EVAL else available_test_datasets(require_geo=True)

    with open(train_path, "rb") as f:
        train_list = pickle.load(f)
    print("Train data:", train_path)
    pair_contact_contrast_coverage = contact_contrast_dataset_coverage(train_list)

    # 逐个载入测试集，键名使用规范名（不受原始文件大小写影响）
    test_sets = {}
    for name, p in test_paths:
        with open(p, "rb") as f:
            test_sets[name] = pickle.load(f)

    N = len(train_list)
    print("Total train complexes:", N)
    for name, lst in test_sets.items():
        print(f"Total test complexes ({name}):", len(lst))
    if SKIP_TEST_EVAL:
        print("External test evaluation: skipped (OOF development run)")

    plm_mode, plm_dim = configure_plm_feature_config([train_list, test_sets])
    print("PLM feature mode:", plm_mode)
    print("PLM feature dim:", plm_dim)
    print("Aux PLM feature mode:", AUX_PLM_FEATURE_MODE)
    print("Aux PLM feature dim:", AUX_PLM_FEATURE_DIM)
    print("Primary PLM enabled:", USE_PLM_FEATURES)
    print("Aux PLM enabled:", USE_AUX_PLM_FEATURES)
    print("Partner-contact auxiliary objective:", PARTNER_CONTACT_AUX)
    print("Partner-conditioned bipartite attention:", PARTNER_CONDITIONING)
    if PARTNER_CONDITIONING:
        print("Partner-conditioned top-k:", PARTNER_TOP_K)
        print("Partner direct fusion:", PARTNER_DIRECT_FUSION)
        print("Partner logit mode:", PARTNER_LOGIT_MODE)
        print("Shared partner residue encoder:", PARTNER_RESIDUE_ENCODER)
        if PARTNER_RESIDUE_ENCODER:
            print(
                "Shared partner encoder settings: "
                f"layers={PARTNER_ENCODER_LAYERS}, "
                f"target_fusion={PARTNER_TARGET_FUSION:.3f}"
            )
        if PARTNER_LOGIT_MODE in {"delta", "residual"}:
            label = "residual alpha" if PARTNER_LOGIT_MODE == "residual" else "delta scale"
            print(f"Partner {label}: {PARTNER_DELTA_SCALE:.3f}")
    print("Partner-contrast training:", PARTNER_CONTRAST)
    if PARTNER_CONTRAST:
        print(
            "Partner-contrast settings: "
            f"loss_weight={PARTNER_CONTRAST_WEIGHT:.3f}, "
            f"margin={PARTNER_CONTRAST_MARGIN:.3f}, "
            f"warmup_epochs={PARTNER_CONTRAST_WARMUP_EPOCHS}"
        )
        if BATCH_SIZE < 2:
            print("Warning: partner contrast needs batch_size >= 2; contrast loss will be skipped.")
    print("Pair-contact prediction loss:", PAIR_CONTACT_LOSS)
    if PAIR_CONTACT_LOSS:
        print(
            "Pair-contact settings: "
            f"loss_weight={PAIR_CONTACT_LOSS_WEIGHT:.3f}, "
            f"pos_weight={PAIR_CONTACT_POS_WEIGHT:.2f}, "
            f"head={PAIR_CONTACT_HEAD}, "
            f"warmup_epochs={PAIR_CONTACT_WARMUP_EPOCHS}"
        )
    print("Pair-marginal consistency:", PAIR_MARGINAL_CONSISTENCY)
    if PAIR_MARGINAL_CONSISTENCY:
        print(
            "Pair-marginal consistency settings: "
            f"loss_weight={PAIR_MARGINAL_CONSISTENCY_WEIGHT:.3f}, "
            f"warmup_epochs={PAIR_MARGINAL_CONSISTENCY_WARMUP_EPOCHS}, "
            "teacher=detached_site_probability"
        )
    print("Pair-marginal partner contrast:", PAIR_MARGINAL_CONTRAST)
    if PAIR_MARGINAL_CONTRAST:
        print(
            "Pair-marginal partner-contrast settings: "
            f"loss_weight={PAIR_MARGINAL_CONTRAST_WEIGHT:.3f}, "
            f"margin={PAIR_MARGINAL_CONTRAST_MARGIN:.3f}, "
            f"warmup_epochs={PAIR_MARGINAL_CONTRAST_WARMUP_EPOCHS}"
        )
        if BATCH_SIZE < 2:
            print(
                "Warning: pair-marginal partner contrast needs batch_size >= 2; "
                "contrast loss will be skipped."
            )
    print("Pair-contact hard partner discrimination:", PAIR_CONTACT_CONTRAST)
    if PAIR_CONTACT_CONTRAST:
        print(
            "Pair-contact hard partner settings: "
            f"loss_weight={PAIR_CONTACT_CONTRAST_WEIGHT:.3f}, "
            f"margin={PAIR_CONTACT_CONTRAST_MARGIN:.3f}, "
            f"hard_k={PAIR_CONTACT_CONTRAST_HARD_K}, "
            f"warmup_epochs={PAIR_CONTACT_CONTRAST_WARMUP_EPOCHS}, "
            "mismatch_policy=in_batch_nearest_length"
        )
        print(
            "Contact-contrast coverage: "
            f"eligible={pair_contact_contrast_coverage['contact_contrast_eligible_residues']}/"
            f"{pair_contact_contrast_coverage['positive_interface_residues']} "
            f"({100.0 * pair_contact_contrast_coverage['contact_contrast_coverage']:.1f}%), "
            f"samples={pair_contact_contrast_coverage['samples_with_eligible_contacts']}/"
            f"{pair_contact_contrast_coverage['total_samples']}"
        )
        if BATCH_SIZE < 2:
            print(
                "Warning: pair-contact hard partner discrimination needs batch_size >= 2; "
                "contrast loss will be skipped."
            )

    # ---- 2) 5-fold indices ----
    num_folds = 5
    cv_folds = make_cv_folds(
        train_list,
        seed=SEED,
        num_folds=num_folds,
        grouped=GROUPED_CV,
        group_key=CV_GROUP_KEY,
    )
    print(
        "CV split: "
        + (f"grouped by {CV_GROUP_KEY}" if GROUPED_CV else "legacy sample-level")
    )
    print("Validation fold sizes:", [int(len(indices)) for indices in cv_folds])

    # ---- 3) DataLoader 配置 ----
    epochs = 30
    patience = 8
    batch_size = max(1, BATCH_SIZE)
    print("Batch size:", batch_size)

    # 记录每折在各测试集上的指标
    fold_metrics_by_test = {name: [] for name in test_sets.keys()}
    fold_artifacts = []
    oof_labels = []
    oof_probs = []
    csv_rows = []

    # 为了初始化 in_dim：用任意一个样本
    in_dim = train_list[0]['residue_graph_node'].shape[1]
    atom_dim = train_list[0]['atom_graph_node'].shape[1]

    # ---- 4) 逐折训练 / 验证 / 测试 ----
    folds_to_run = min(MAX_FOLDS, num_folds)
    if folds_to_run < num_folds:
        print(f"Pilot fold limit: running {folds_to_run}/{num_folds} CV folds")
    for fold in range(folds_to_run):
        print("\n" + "=" * 30)
        print(f"========== Fold {fold+1}/{num_folds} ==========")

        val_idx = cv_folds[fold]
        train_idx = np.concatenate(
            [cv_folds[other_fold] for other_fold in range(num_folds) if other_fold != fold]
        )

        train_prots = [train_list[i] for i in train_idx]
        val_prots = [train_list[i] for i in val_idx]
        if MODEL_MODE == "checkpoint_finetune":
            feat_mean = feat_std = None
            atom_feat_mean = atom_feat_std = None
            surface_feat_mean = surface_feat_std = None
            sequence_feat_mean = sequence_feat_std = None
            plm_feat_mean = plm_feat_std = None
            aux_plm_feat_mean = aux_plm_feat_std = None
        else:
            feat_mean, feat_std = fit_feature_stats(train_prots)
            atom_feat_mean, atom_feat_std = fit_atom_feature_stats(train_prots)
            surface_feat_mean, surface_feat_std = fit_surface_feature_stats(train_prots)
            sequence_feat_mean, sequence_feat_std = fit_sequence_feature_stats(train_prots)
            if PLM_FEATURE_DIM > 0:
                plm_feat_mean, plm_feat_std = fit_plm_feature_stats(train_prots)
            else:
                plm_feat_mean = plm_feat_std = None
            if AUX_PLM_FEATURE_DIM > 0:
                aux_plm_feat_mean, aux_plm_feat_std = fit_plm_feature_stats(
                    train_prots,
                    dim=AUX_PLM_FEATURE_DIM,
                    key=AUX_PLM_FEATURE_KEY,
                )
            else:
                aux_plm_feat_mean = aux_plm_feat_std = None
        pos_weight = compute_pos_weight(train_prots)

        print(f"Fold {fold+1}: train complexes = {len(train_prots)}, val complexes = {len(val_prots)}")
        print(f"Fold {fold+1}: positive class weight = {pos_weight:.3f}")

        train_dataset = ProteinDataset(
            train_prots,
            feat_mean,
            feat_std,
            atom_feat_mean,
            atom_feat_std,
            surface_feat_mean,
            surface_feat_std,
            sequence_feat_mean,
            sequence_feat_std,
            plm_feat_mean,
            plm_feat_std,
            PLM_FEATURE_DIM,
            aux_plm_feat_mean,
            aux_plm_feat_std,
            AUX_PLM_FEATURE_DIM,
        )
        val_dataset = ProteinDataset(
            val_prots,
            feat_mean,
            feat_std,
            atom_feat_mean,
            atom_feat_std,
            surface_feat_mean,
            surface_feat_std,
            sequence_feat_mean,
            sequence_feat_std,
            plm_feat_mean,
            plm_feat_std,
            PLM_FEATURE_DIM,
            aux_plm_feat_mean,
            aux_plm_feat_std,
            AUX_PLM_FEATURE_DIM,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_proteins,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_proteins,
            num_workers=0
        )
        test_loaders = {}
        for name, lst in test_sets.items():
            test_dataset = ProteinDataset(
                lst,
                feat_mean,
                feat_std,
                atom_feat_mean,
                atom_feat_std,
                surface_feat_mean,
                surface_feat_std,
                sequence_feat_mean,
                sequence_feat_std,
                plm_feat_mean,
                plm_feat_std,
                PLM_FEATURE_DIM,
                aux_plm_feat_mean,
                aux_plm_feat_std,
                AUX_PLM_FEATURE_DIM,
            )
            test_loaders[name] = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_proteins,
                num_workers=0
        )

        # 每折重新初始化模型
        model = build_model(in_dim, atom_dim, device)

        if TWO_HEAD_BINDING and MODEL_MODE != "checkpoint_finetune":
            criterion = TwoHeadBindingLoss(
                pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device),
            )
        else:
            criterion = CompositeBindingLoss(
                pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device),
                gamma=1.5,
                label_smoothing=0.02,
                rank_weight=0.15,
                dice_weight=0.10,
            )
        lr = 1e-5 if MODEL_MODE == "checkpoint_finetune" else 3e-4
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=OPT_WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
        ema = ModelEMA(model, decay=0.995)

        best_val_score = -1.0
        best_val_mcc = 0.0
        best_val_auc_pr = 0.0
        best_threshold = 0.5
        best_state = None
        top_states = []
        epochs_without_improvement = 0

        # ---- 训练 + 验证（本折）----
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                epoch,
                fold_idx=fold,
                ema=ema,
            )
            print(f"[Fold {fold+1}] Epoch {epoch} Train Loss: {train_loss:.6f}")
            ema.store(model)
            ema.copy_to(model)
            val_metrics = evaluate(model, val_loader, device, split_name=f"Val-Fold{fold+1}")
            if PAIR_CONTACT_LOSS:
                evaluate_pair_contacts(model, val_loader, device, split_name=f"Val-Fold{fold+1}")
            val_score = validation_selection_score(val_metrics)
            scheduler.step(val_score)

            if val_score > best_val_score:
                best_val_score = val_score
                best_val_mcc = val_metrics["mcc"]
                best_val_auc_pr = val_metrics["auc_pr"]
                best_threshold = val_metrics["t_opt"]
                best_state = clone_state_dict_cpu(model)
                epochs_without_improvement = 0
                print(
                    f"==> [Fold {fold+1}] Update best model, "
                    f"Val score = {best_val_score:.4f}, "
                    f"MCC = {best_val_mcc:.4f}, AUPRC = {best_val_auc_pr:.4f}"
                )
            else:
                epochs_without_improvement += 1

            update_top_checkpoints(
                top_states,
                val_score,
                val_metrics["t_opt"],
                clone_state_dict_cpu(model),
                epoch,
            )

            if epochs_without_improvement >= patience:
                print(f"==> [Fold {fold+1}] Early stopping after {epoch} epochs")
                ema.restore(model)
                break
            ema.restore(model)

        # ---- 用本折最优模型在所有测试集上评估 ----
        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)

        if len(top_states) > 1:
            avg_state = average_state_dicts([item["state"] for item in top_states])
            model.load_state_dict(avg_state)
            model.to(device)
            avg_val_metrics = evaluate(
                model,
                val_loader,
                device,
                split_name=f"Val-Fold{fold+1}-Top{len(top_states)}Avg"
            )
            avg_val_score = validation_selection_score(avg_val_metrics)
            if avg_val_score >= best_val_score:
                best_state = clone_state_dict_cpu(model)
                best_threshold = avg_val_metrics["t_opt"]
                best_val_score = avg_val_score
                best_val_mcc = avg_val_metrics["mcc"]
                best_val_auc_pr = avg_val_metrics["auc_pr"]
                print(
                    f"==> [Fold {fold+1}] Use top-{len(top_states)} averaged model, "
                    f"Val score = {best_val_score:.4f}, "
                    f"MCC = {best_val_mcc:.4f}, AUPRC = {best_val_auc_pr:.4f}"
                )
            else:
                model.load_state_dict(best_state)
                model.to(device)

        val_labels, val_probs = predict_scores(model, val_loader, device)
        oof_labels.append(val_labels)
        oof_probs.append(val_probs)
        fold_artifacts.append({
            "state": copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}),
            "threshold": best_threshold,
            "val_mcc": best_val_mcc,
            "val_auc_pr": best_val_auc_pr,
            "val_selection_score": best_val_score,
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "atom_feat_mean": atom_feat_mean,
            "atom_feat_std": atom_feat_std,
            "surface_feat_mean": surface_feat_mean,
            "surface_feat_std": surface_feat_std,
            "sequence_feat_mean": sequence_feat_mean,
            "sequence_feat_std": sequence_feat_std,
            "plm_feat_mean": plm_feat_mean,
            "plm_feat_std": plm_feat_std,
            "plm_feature_dim": PLM_FEATURE_DIM,
            "use_plm_features": USE_PLM_FEATURES,
            "aux_plm_feat_mean": aux_plm_feat_mean,
            "aux_plm_feat_std": aux_plm_feat_std,
            "aux_plm_feature_dim": AUX_PLM_FEATURE_DIM,
            "use_aux_plm_features": USE_AUX_PLM_FEATURES,
            "view_logit_fusion": VIEW_LOGIT_FUSION,
            "model_dropout": MODEL_DROPOUT,
            "model_edge_dropout": MODEL_EDGE_DROPOUT,
            "weight_decay": OPT_WEIGHT_DECAY,
            "partner_conditioning": PARTNER_CONDITIONING,
            "partner_top_k": PARTNER_TOP_K,
            "partner_direct_fusion": PARTNER_DIRECT_FUSION,
            "partner_logit_mode": PARTNER_LOGIT_MODE,
            "partner_delta_scale": PARTNER_DELTA_SCALE,
            "partner_residue_encoder": PARTNER_RESIDUE_ENCODER,
            "partner_encoder_layers": PARTNER_ENCODER_LAYERS,
            "partner_target_fusion": PARTNER_TARGET_FUSION,
            "partner_contrast": PARTNER_CONTRAST,
            "partner_contrast_weight": PARTNER_CONTRAST_WEIGHT,
            "partner_contrast_margin": PARTNER_CONTRAST_MARGIN,
            "partner_contrast_warmup_epochs": PARTNER_CONTRAST_WARMUP_EPOCHS,
            "batch_size": batch_size,
            "pair_contact_loss": PAIR_CONTACT_LOSS,
            "pair_contact_loss_weight": PAIR_CONTACT_LOSS_WEIGHT,
            "pair_contact_pos_weight": PAIR_CONTACT_POS_WEIGHT,
            "pair_contact_head": PAIR_CONTACT_HEAD,
            "pair_contact_warmup_epochs": PAIR_CONTACT_WARMUP_EPOCHS,
            "pair_marginal_consistency": PAIR_MARGINAL_CONSISTENCY,
            "pair_marginal_consistency_weight": PAIR_MARGINAL_CONSISTENCY_WEIGHT,
            "pair_marginal_consistency_warmup_epochs": PAIR_MARGINAL_CONSISTENCY_WARMUP_EPOCHS,
            "pair_marginal_contrast": PAIR_MARGINAL_CONTRAST,
            "pair_marginal_contrast_weight": PAIR_MARGINAL_CONTRAST_WEIGHT,
            "pair_marginal_contrast_margin": PAIR_MARGINAL_CONTRAST_MARGIN,
            "pair_marginal_contrast_warmup_epochs": PAIR_MARGINAL_CONTRAST_WARMUP_EPOCHS,
            "pair_contact_contrast": PAIR_CONTACT_CONTRAST,
            "pair_contact_contrast_weight": PAIR_CONTACT_CONTRAST_WEIGHT,
            "pair_contact_contrast_margin": PAIR_CONTACT_CONTRAST_MARGIN,
            "pair_contact_contrast_hard_k": PAIR_CONTACT_CONTRAST_HARD_K,
            "pair_contact_contrast_warmup_epochs": PAIR_CONTACT_CONTRAST_WARMUP_EPOCHS,
            "pair_contact_contrast_mismatch_policy": "in_batch_nearest_length",
            "pair_contact_contrast_coverage": pair_contact_contrast_coverage,
            "max_folds": MAX_FOLDS,
            "skip_test_eval": SKIP_TEST_EVAL,
            "grouped_cv": GROUPED_CV,
            "cv_group_key": CV_GROUP_KEY,
            "test_sets": list(TEST_PKL_FILES),
        })
        fold_path = os.path.join(OUTPUT_DIR, f"fold{fold+1}_best.pt")
        torch.save({
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "threshold": best_threshold,
            "val_selection_score": best_val_score,
            "val_mcc": best_val_mcc,
            "val_auc_pr": best_val_auc_pr,
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "atom_feat_mean": atom_feat_mean,
            "atom_feat_std": atom_feat_std,
            "surface_feat_mean": surface_feat_mean,
            "surface_feat_std": surface_feat_std,
            "sequence_feat_mean": sequence_feat_mean,
            "sequence_feat_std": sequence_feat_std,
            "plm_feat_mean": plm_feat_mean,
            "plm_feat_std": plm_feat_std,
            "plm_feature_dim": PLM_FEATURE_DIM,
            "aux_plm_feat_mean": aux_plm_feat_mean,
            "aux_plm_feat_std": aux_plm_feat_std,
            "aux_plm_feature_dim": AUX_PLM_FEATURE_DIM,
            "model_mode": MODEL_MODE,
            "seed": SEED,
            "geometry_feature_mode": GEOMETRY_FEATURE_MODE,
            "surface_feature_mode": SURFACE_FEATURE_MODE,
            "sequence_feature_mode": SEQUENCE_FEATURE_MODE,
            "plm_feature_mode": PLM_FEATURE_MODE,
            "aux_plm_feature_mode": AUX_PLM_FEATURE_MODE,
            "use_plm_features": USE_PLM_FEATURES,
            "use_aux_plm_features": USE_AUX_PLM_FEATURES,
            "patch_label_distribution": PATCH_LABEL_DISTRIBUTION,
            "patch_label_steps": PATCH_LABEL_STEPS,
            "patch_label_alpha": PATCH_LABEL_ALPHA,
            "patch_label_max_unlabeled": PATCH_LABEL_MAX_UNLABELED,
            "partner_contact_aux": PARTNER_CONTACT_AUX,
            "partner_contact_aux_weight": PARTNER_CONTACT_AUX_WEIGHT,
            "partner_conditioning": PARTNER_CONDITIONING,
            "partner_top_k": PARTNER_TOP_K,
            "partner_direct_fusion": PARTNER_DIRECT_FUSION,
            "partner_logit_mode": PARTNER_LOGIT_MODE,
            "partner_delta_scale": PARTNER_DELTA_SCALE,
            "partner_residue_encoder": PARTNER_RESIDUE_ENCODER,
            "partner_encoder_layers": PARTNER_ENCODER_LAYERS,
            "partner_target_fusion": PARTNER_TARGET_FUSION,
            "partner_contrast": PARTNER_CONTRAST,
            "partner_contrast_weight": PARTNER_CONTRAST_WEIGHT,
            "partner_contrast_margin": PARTNER_CONTRAST_MARGIN,
            "partner_contrast_warmup_epochs": PARTNER_CONTRAST_WARMUP_EPOCHS,
            "batch_size": batch_size,
            "pair_contact_loss": PAIR_CONTACT_LOSS,
            "pair_contact_loss_weight": PAIR_CONTACT_LOSS_WEIGHT,
            "pair_contact_pos_weight": PAIR_CONTACT_POS_WEIGHT,
            "pair_contact_head": PAIR_CONTACT_HEAD,
            "pair_contact_warmup_epochs": PAIR_CONTACT_WARMUP_EPOCHS,
            "pair_marginal_consistency": PAIR_MARGINAL_CONSISTENCY,
            "pair_marginal_consistency_weight": PAIR_MARGINAL_CONSISTENCY_WEIGHT,
            "pair_marginal_consistency_warmup_epochs": PAIR_MARGINAL_CONSISTENCY_WARMUP_EPOCHS,
            "pair_marginal_contrast": PAIR_MARGINAL_CONTRAST,
            "pair_marginal_contrast_weight": PAIR_MARGINAL_CONTRAST_WEIGHT,
            "pair_marginal_contrast_margin": PAIR_MARGINAL_CONTRAST_MARGIN,
            "pair_marginal_contrast_warmup_epochs": PAIR_MARGINAL_CONTRAST_WARMUP_EPOCHS,
            "pair_contact_contrast": PAIR_CONTACT_CONTRAST,
            "pair_contact_contrast_weight": PAIR_CONTACT_CONTRAST_WEIGHT,
            "pair_contact_contrast_margin": PAIR_CONTACT_CONTRAST_MARGIN,
            "pair_contact_contrast_hard_k": PAIR_CONTACT_CONTRAST_HARD_K,
            "pair_contact_contrast_warmup_epochs": PAIR_CONTACT_CONTRAST_WARMUP_EPOCHS,
            "pair_contact_contrast_mismatch_policy": "in_batch_nearest_length",
            "pair_contact_contrast_coverage": pair_contact_contrast_coverage,
            "max_folds": MAX_FOLDS,
            "skip_test_eval": SKIP_TEST_EVAL,
            "two_head_binding": TWO_HEAD_BINDING,
            "two_head_rank_loss_weight": TWO_HEAD_RANK_LOSS_WEIGHT,
            "two_head_consistency_weight": TWO_HEAD_CONSISTENCY_WEIGHT,
            "two_head_rank_fusion": TWO_HEAD_RANK_FUSION,
            "two_head_rank_warmup_epochs": TWO_HEAD_RANK_WARMUP_EPOCHS,
            "view_logit_fusion": VIEW_LOGIT_FUSION,
            "model_dropout": MODEL_DROPOUT,
            "model_edge_dropout": MODEL_EDGE_DROPOUT,
            "weight_decay": OPT_WEIGHT_DECAY,
            "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
            "checkpoint_selection_aupr_weight": CHECKPOINT_SELECTION_AUPR_WEIGHT,
            "grouped_cv": GROUPED_CV,
            "cv_group_key": CV_GROUP_KEY,
            "test_sets": list(TEST_PKL_FILES),
        }, fold_path)
        print(f"Saved fold checkpoint: {fold_path}")

        for test_name, test_loader in test_loaders.items():
            test_metrics = evaluate(
                model,
                test_loader,
                device,
                split_name=f"Test-{test_name}-Fold{fold+1}",
                threshold=best_threshold
            )
            fold_metrics_by_test[test_name].append(test_metrics)
            csv_rows.append({
                "group": f"fold{fold+1}",
                "test_set": test_name,
                "acc": test_metrics["acc"],
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "f1": test_metrics["f1"],
                "mcc": test_metrics["mcc"],
                "auc_roc": test_metrics["auc_roc"],
                "auc_pr": test_metrics["auc_pr"],
                "threshold": test_metrics["t_opt"],
            })

    # ---- 5) 汇总 5 折 test 结果（逐测试集）----
    print("\n" + "=" * 30)
    if test_sets:
        print("====== CV on train, evaluated on independent test sets ======")
    else:
        print("====== CV on train, external test evaluation skipped ======")

    keys = ["auc_roc", "auc_pr", "acc", "precision", "recall", "f1", "mcc"]
    for test_name, metrics_list in fold_metrics_by_test.items():
        print("\n" + "-" * 12 + f" {test_name} " + "-" * 12)
        for k in keys:
            vals = np.array([m[k] for m in metrics_list])
            print(f"{k.upper():9s}: {vals.mean():.4f} ± {vals.std():.4f}")

    # ---- 6) Cross-fold ensemble: average probabilities from all fold models ----
    print("\n" + "=" * 30)
    print("====== CV probability ensemble with OOF-calibrated threshold ======")

    oof_y = np.concatenate(oof_labels, axis=0)
    oof_score = np.concatenate(oof_probs, axis=0)
    ensemble_threshold = best_mcc_threshold(oof_y, oof_score)
    metrics_from_scores(
        oof_y,
        oof_score,
        split_name="OOF-Validation-Calibration",
        threshold=ensemble_threshold,
    )
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "oof_predictions.npz"),
        labels=oof_y,
        probs=oof_score,
        threshold=np.array([ensemble_threshold], dtype=np.float32),
    )

    val_mcc_weights = normalized_positive_weights([artifact["val_mcc"] for artifact in fold_artifacts])
    val_aupr_weights = normalized_positive_weights([artifact["val_auc_pr"] for artifact in fold_artifacts])
    rank_threshold = best_mcc_threshold(oof_y, rank_normalize(oof_score))
    ensemble_strategies = {
        "mean": {"mode": "mean", "weights": None, "threshold": ensemble_threshold},
        "weighted_mcc": {"mode": "weighted", "weights": val_mcc_weights, "threshold": ensemble_threshold},
        "weighted_aupr": {"mode": "weighted", "weights": val_aupr_weights, "threshold": ensemble_threshold},
        "rank": {"mode": "rank", "weights": None, "threshold": rank_threshold},
    }

    print("Fold MCC weights:", np.round(val_mcc_weights, 4))
    print("Fold AUPRC weights:", np.round(val_aupr_weights, 4))

    ensemble_metrics_by_strategy = {name: {} for name in ensemble_strategies}
    for test_name, test_list in test_sets.items():
        fold_scores = []
        y_true = None

        for artifact in fold_artifacts:
            dataset = ProteinDataset(
                test_list,
                artifact["feat_mean"],
                artifact["feat_std"],
                artifact["atom_feat_mean"],
                artifact["atom_feat_std"],
                artifact["surface_feat_mean"],
                artifact["surface_feat_std"],
                artifact["sequence_feat_mean"],
                artifact["sequence_feat_std"],
                artifact["plm_feat_mean"],
                artifact["plm_feat_std"],
                artifact.get("plm_feature_dim", PLM_FEATURE_DIM),
                artifact.get("aux_plm_feat_mean"),
                artifact.get("aux_plm_feat_std"),
                artifact.get("aux_plm_feature_dim", AUX_PLM_FEATURE_DIM),
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_proteins,
                num_workers=0,
            )
            model = build_model(
                in_dim,
                atom_dim,
                device,
                plm_dim=artifact.get("plm_feature_dim", PLM_FEATURE_DIM),
                aux_plm_dim=artifact.get("aux_plm_feature_dim", AUX_PLM_FEATURE_DIM),
                partner_conditioning=bool(artifact.get("partner_conditioning", PARTNER_CONDITIONING)),
                partner_top_k=int(artifact.get("partner_top_k", PARTNER_TOP_K)),
                partner_direct_fusion=bool(artifact.get("partner_direct_fusion", PARTNER_DIRECT_FUSION)),
                partner_logit_mode=artifact.get("partner_logit_mode", PARTNER_LOGIT_MODE),
                partner_delta_scale=float(artifact.get("partner_delta_scale", PARTNER_DELTA_SCALE)),
                partner_residue_encoder=bool(artifact.get("partner_residue_encoder", False)),
                partner_encoder_layers=int(artifact.get("partner_encoder_layers", PARTNER_ENCODER_LAYERS)),
                partner_target_fusion=float(artifact.get("partner_target_fusion", PARTNER_TARGET_FUSION)),
                pair_contact_head=artifact.get("pair_contact_head", PAIR_CONTACT_HEAD),
            )
            if hasattr(model, "view_logit_fusion"):
                model.view_logit_fusion = float(artifact.get("view_logit_fusion", VIEW_LOGIT_FUSION))
            model.load_state_dict(artifact["state"])
            labels, probs = predict_scores(model, loader, device)
            y_true = labels if y_true is None else y_true
            fold_scores.append(probs)

        for strategy_name, strategy in ensemble_strategies.items():
            ensemble_score = combine_fold_scores(
                fold_scores,
                mode=strategy["mode"],
                weights=strategy["weights"],
            )
            metrics = metrics_from_scores(
                y_true,
                ensemble_score,
                split_name=f"Ensemble-{strategy_name}-{test_name}",
                threshold=strategy["threshold"],
            )
            ensemble_metrics_by_strategy[strategy_name][test_name] = metrics

            file_prefix = "ensemble" if strategy_name == "mean" else f"ensemble_{strategy_name}"
            np.savez_compressed(
                os.path.join(OUTPUT_DIR, f"{file_prefix}_{test_name}_predictions.npz"),
                labels=y_true,
                probs=ensemble_score,
                threshold=np.array([strategy["threshold"]], dtype=np.float32),
            )
            csv_rows.append({
                "group": f"ensemble_{strategy_name}",
                "test_set": test_name,
                "acc": metrics["acc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "mcc": metrics["mcc"],
                "auc_roc": metrics["auc_roc"],
                "auc_pr": metrics["auc_pr"],
                "threshold": metrics["t_opt"],
            })

    print("\n" + "-" * 12 + " Ensemble Summary " + "-" * 12)
    for strategy_name, metrics_by_test in ensemble_metrics_by_strategy.items():
        print(f"\n[{strategy_name}]")
        for test_name, metrics in metrics_by_test.items():
            print(
                f"{test_name:8s} "
                f"ACC={metrics['acc']:.4f} "
                f"Precision={metrics['precision']:.4f} "
                f"Recall={metrics['recall']:.4f} "
                f"F1={metrics['f1']:.4f} "
                f"MCC={metrics['mcc']:.4f} "
                f"AUROC={metrics['auc_roc']:.4f} "
                f"AUPRC={metrics['auc_pr']:.4f}"
            )

    write_metrics_csv(os.path.join(OUTPUT_DIR, "metrics_summary.csv"), csv_rows)
    save_json(os.path.join(OUTPUT_DIR, "ensemble_summary.json"), {
        "model_mode": MODEL_MODE,
        "seed": SEED,
        "geometry_feature_mode": GEOMETRY_FEATURE_MODE,
        "surface_feature_mode": SURFACE_FEATURE_MODE,
        "sequence_feature_mode": SEQUENCE_FEATURE_MODE,
        "plm_feature_mode": PLM_FEATURE_MODE,
        "plm_feature_dim": PLM_FEATURE_DIM,
        "aux_plm_feature_mode": AUX_PLM_FEATURE_MODE,
        "aux_plm_feature_dim": AUX_PLM_FEATURE_DIM,
        "patch_label_distribution": PATCH_LABEL_DISTRIBUTION,
        "patch_label_steps": PATCH_LABEL_STEPS,
        "patch_label_alpha": PATCH_LABEL_ALPHA,
        "patch_label_max_unlabeled": PATCH_LABEL_MAX_UNLABELED,
        "partner_contact_aux": PARTNER_CONTACT_AUX,
        "partner_contact_aux_weight": PARTNER_CONTACT_AUX_WEIGHT,
        "partner_conditioning": PARTNER_CONDITIONING,
        "partner_top_k": PARTNER_TOP_K,
        "partner_direct_fusion": PARTNER_DIRECT_FUSION,
        "partner_logit_mode": PARTNER_LOGIT_MODE,
        "partner_delta_scale": PARTNER_DELTA_SCALE,
        "partner_residue_encoder": PARTNER_RESIDUE_ENCODER,
        "partner_encoder_layers": PARTNER_ENCODER_LAYERS,
        "partner_target_fusion": PARTNER_TARGET_FUSION,
        "partner_contrast": PARTNER_CONTRAST,
        "partner_contrast_weight": PARTNER_CONTRAST_WEIGHT,
        "partner_contrast_margin": PARTNER_CONTRAST_MARGIN,
        "partner_contrast_warmup_epochs": PARTNER_CONTRAST_WARMUP_EPOCHS,
        "batch_size": batch_size,
        "pair_contact_loss": PAIR_CONTACT_LOSS,
        "pair_contact_loss_weight": PAIR_CONTACT_LOSS_WEIGHT,
        "pair_contact_pos_weight": PAIR_CONTACT_POS_WEIGHT,
        "pair_contact_head": PAIR_CONTACT_HEAD,
        "pair_contact_warmup_epochs": PAIR_CONTACT_WARMUP_EPOCHS,
        "pair_marginal_consistency": PAIR_MARGINAL_CONSISTENCY,
        "pair_marginal_consistency_weight": PAIR_MARGINAL_CONSISTENCY_WEIGHT,
        "pair_marginal_consistency_warmup_epochs": PAIR_MARGINAL_CONSISTENCY_WARMUP_EPOCHS,
        "pair_marginal_contrast": PAIR_MARGINAL_CONTRAST,
        "pair_marginal_contrast_weight": PAIR_MARGINAL_CONTRAST_WEIGHT,
        "pair_marginal_contrast_margin": PAIR_MARGINAL_CONTRAST_MARGIN,
        "pair_marginal_contrast_warmup_epochs": PAIR_MARGINAL_CONTRAST_WARMUP_EPOCHS,
        "pair_contact_contrast": PAIR_CONTACT_CONTRAST,
        "pair_contact_contrast_weight": PAIR_CONTACT_CONTRAST_WEIGHT,
        "pair_contact_contrast_margin": PAIR_CONTACT_CONTRAST_MARGIN,
        "pair_contact_contrast_hard_k": PAIR_CONTACT_CONTRAST_HARD_K,
        "pair_contact_contrast_warmup_epochs": PAIR_CONTACT_CONTRAST_WARMUP_EPOCHS,
        "pair_contact_contrast_mismatch_policy": "in_batch_nearest_length",
        "pair_contact_contrast_coverage": pair_contact_contrast_coverage,
        "max_folds": MAX_FOLDS,
        "skip_test_eval": SKIP_TEST_EVAL,
        "two_head_binding": TWO_HEAD_BINDING,
        "two_head_rank_loss_weight": TWO_HEAD_RANK_LOSS_WEIGHT,
        "two_head_consistency_weight": TWO_HEAD_CONSISTENCY_WEIGHT,
        "two_head_rank_fusion": TWO_HEAD_RANK_FUSION,
        "two_head_rank_warmup_epochs": TWO_HEAD_RANK_WARMUP_EPOCHS,
        "view_logit_fusion": VIEW_LOGIT_FUSION,
        "model_dropout": MODEL_DROPOUT,
        "model_edge_dropout": MODEL_EDGE_DROPOUT,
        "weight_decay": OPT_WEIGHT_DECAY,
        "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
        "checkpoint_selection_aupr_weight": CHECKPOINT_SELECTION_AUPR_WEIGHT,
        "grouped_cv": GROUPED_CV,
        "cv_group_key": CV_GROUP_KEY,
        "test_sets": list(TEST_PKL_FILES),
        "ensemble_threshold": ensemble_threshold,
        "rank_threshold": rank_threshold,
        "val_mcc_weights": val_mcc_weights,
        "val_aupr_weights": val_aupr_weights,
        "single_fold_summary": {
            test_name: {
                k: {
                    "mean": float(np.mean([m[k] for m in metrics_list])),
                    "std": float(np.std([m[k] for m in metrics_list])),
                }
                for k in keys
            }
            for test_name, metrics_list in fold_metrics_by_test.items()
        },
        "ensemble": ensemble_metrics_by_strategy,
    })
    print(f"\nSaved outputs to: {os.path.abspath(OUTPUT_DIR)}")
