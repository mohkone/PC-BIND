import itertools
import os
import pickle
import sys
import urllib.request

import numpy as np


DATA_DIR = "data"
PDB_CACHE = "pdb_cache"
OUT_DIR = os.path.join("data", "geo")
RESIDUE_CUTOFF = 10.0
ATOM_CUTOFF = 4.5
PARTNER_CONTACT_CUTOFF = 5.0
COMPUTE_ATOM_GEO_EDGES = False
SURFACE_RADII = (6.0, 8.0, 10.0, 12.0)
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "C", "PYL": "K",
}
AA_PROPERTIES = {
    # hydropathy, volume, polarity, pI
    "A": (1.8, 88.6, 8.1, 6.00), "C": (2.5, 108.5, 5.5, 5.07),
    "D": (-3.5, 111.1, 13.0, 2.77), "E": (-3.5, 138.4, 12.3, 3.22),
    "F": (2.8, 189.9, 5.2, 5.48), "G": (-0.4, 60.1, 9.0, 5.97),
    "H": (-3.2, 153.2, 10.4, 7.59), "I": (4.5, 166.7, 5.2, 6.02),
    "K": (-3.9, 168.6, 11.3, 9.74), "L": (3.8, 166.7, 4.9, 5.98),
    "M": (1.9, 162.9, 5.7, 5.74), "N": (-3.5, 114.1, 11.6, 5.41),
    "P": (-1.6, 112.7, 8.0, 6.30), "Q": (-3.5, 143.8, 10.5, 5.65),
    "R": (-4.5, 173.4, 10.5, 10.76), "S": (-0.8, 89.0, 9.2, 5.68),
    "T": (-0.7, 116.1, 8.6, 5.60), "V": (4.2, 140.0, 5.9, 5.96),
    "W": (-0.9, 227.8, 5.4, 5.89), "Y": (-1.3, 193.6, 6.2, 5.66),
}
RESIDUE_SEQUENCE_FEATURE_DIM = 39
PKL_FILES = [
    "Train335.pkl",
    "Test60.pkl",
    "Test287.pkl",
    "Test70.pkl",
    "TestB25.pkl",
    "TestUB25.pkl",
]


def find_existing_file(directory, filename):
    exact = os.path.join(directory, filename)
    if os.path.exists(exact):
        return exact
    target = filename.lower()
    if not os.path.isdir(directory):
        return exact
    for name in os.listdir(directory):
        if name.lower() == target:
            return os.path.join(directory, name)
    return exact


def parse_complex_code(code):
    code = str(code).strip().upper()
    parts = code.replace(":", "_").split("_")
    pdb_id = parts[0][:4]
    chain_hint = parts[1] if len(parts) > 1 else ""
    return pdb_id, chain_hint


def download_pdb(pdb_id):
    os.makedirs(PDB_CACHE, exist_ok=True)
    path = os.path.join(PDB_CACHE, f"{pdb_id}.pdb")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"Downloading {url}", flush=True)
    urllib.request.urlretrieve(url, path)
    return path


def parse_pdb_chains(path):
    chains = {}
    residue_order = {}
    atom_order = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            element = (line[76:78].strip() or atom_name[0]).upper()
            if element == "H":
                continue

            chain = line[21].strip() or "_"
            res_key = (
                chain,
                line[22:26].strip(),
                line[26].strip(),
                line[17:20].strip(),
            )
            xyz = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=np.float32,
            )

            chains.setdefault(chain, {"residues": [], "atoms": [], "backbone": {}})
            residue_order.setdefault(chain, set())
            atom_order.setdefault(chain, [])

            if res_key not in residue_order[chain]:
                residue_order[chain].add(res_key)
                chains[chain]["residues"].append(res_key)
            chains[chain]["atoms"].append((res_key, atom_name, xyz))
            if atom_name in {"N", "CA", "C"}:
                chains[chain]["backbone"].setdefault(res_key, {})[atom_name] = xyz

    return chains


def chain_coordinates(chain_data):
    residues = chain_data["residues"]
    atoms = chain_data["atoms"]
    backbone_lookup = chain_data["backbone"]

    atom_coords = np.stack([atom[2] for atom in atoms], axis=0).astype(np.float32)
    residue_coords = []
    residue_frames = []
    for res_key in residues:
        backbone = backbone_lookup.get(res_key, {})
        if "CA" in backbone:
            residue_coords.append(backbone["CA"])
        else:
            coords = [xyz for key, _, xyz in atoms if key == res_key]
            residue_coords.append(np.mean(coords, axis=0))
        residue_frames.append(residue_local_frame(backbone))
    residue_coords = np.stack(residue_coords, axis=0).astype(np.float32)
    residue_frames = np.stack(residue_frames, axis=0).astype(np.float32)
    return residue_coords, residue_frames, atom_coords


def residue_local_frame(backbone):
    """Right-handed local frame from N-CA-C; identity fallback if incomplete."""
    if not {"N", "CA", "C"}.issubset(backbone):
        return np.eye(3, dtype=np.float32)

    n = backbone["N"].astype(np.float32)
    ca = backbone["CA"].astype(np.float32)
    c = backbone["C"].astype(np.float32)

    e1 = c - ca
    e1_norm = np.linalg.norm(e1)
    if e1_norm < 1e-6:
        return np.eye(3, dtype=np.float32)
    e1 = e1 / e1_norm

    n_vec = n - ca
    n_vec = n_vec - np.dot(n_vec, e1) * e1
    e2_norm = np.linalg.norm(n_vec)
    if e2_norm < 1e-6:
        return np.eye(3, dtype=np.float32)
    e2 = n_vec / e2_norm

    e3 = np.cross(e1, e2)
    e3_norm = np.linalg.norm(e3)
    if e3_norm < 1e-6:
        return np.eye(3, dtype=np.float32)
    e3 = e3 / e3_norm
    return np.stack([e1, e2, e3], axis=1).astype(np.float32)


def radius_edges(coords, cutoff):
    if coords.shape[0] == 0:
        return np.empty((2, 0), dtype=np.int64)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    mask = (dist <= cutoff) & (dist > 0.0)
    src, dst = np.where(mask)
    return np.stack([src, dst], axis=0).astype(np.int64)


def partner_contact_labels(chain_data, chains, chain_id, cutoff=PARTNER_CONTACT_CUTOFF):
    """Training-time teacher labels from target-residue atoms to non-target-chain atoms."""
    residues = chain_data["residues"]
    target_atoms = chain_data["atoms"]
    partner_coords = [
        xyz
        for other_chain, other_data in chains.items()
        if other_chain != chain_id
        for _, _, xyz in other_data["atoms"]
    ]
    if not partner_coords:
        zeros = np.zeros((len(residues),), dtype=np.float32)
        return zeros, zeros, np.full((len(residues),), np.inf, dtype=np.float32)

    partner_coords = np.stack(partner_coords, axis=0).astype(np.float32)
    labels = []
    min_distances = []
    for res_key in residues:
        res_atoms = np.stack([xyz for key, _, xyz in target_atoms if key == res_key], axis=0).astype(np.float32)
        diff = res_atoms[:, None, :] - partner_coords[None, :, :]
        min_dist = float(np.linalg.norm(diff, axis=-1).min())
        min_distances.append(min_dist)
        labels.append(1.0 if min_dist <= cutoff else 0.0)

    labels = np.asarray(labels, dtype=np.float32)
    mask = np.ones_like(labels, dtype=np.float32)
    return labels, mask, np.asarray(min_distances, dtype=np.float32)


def residue_atom_arrays(chain_data, limit=None):
    """Residue-aligned atom coordinate arrays in the chain's residue order."""
    residues = chain_data["residues"] if limit is None else chain_data["residues"][:limit]
    atom_lookup = {res_key: [] for res_key in residues}
    for res_key, _, xyz in chain_data["atoms"]:
        if res_key in atom_lookup:
            atom_lookup[res_key].append(xyz)

    arrays = []
    for res_key in residues:
        coords = atom_lookup.get(res_key, [])
        if coords:
            arrays.append(np.stack(coords, axis=0).astype(np.float32))
        else:
            arrays.append(np.empty((0, 3), dtype=np.float32))
    return arrays


def sparse_partner_pair_contacts(chain_data, partner_atom_arrays, cutoff=PARTNER_CONTACT_CUTOFF):
    """Positive residue-pair contact labels. These are labels only, not input edges."""
    target_atom_arrays = residue_atom_arrays(chain_data)
    if not target_atom_arrays or not partner_atom_arrays:
        return np.empty((2, 0), dtype=np.int64)

    partner_coord_blocks = []
    partner_res_blocks = []
    for partner_idx, atoms in enumerate(partner_atom_arrays):
        if atoms.shape[0] == 0:
            continue
        partner_coord_blocks.append(atoms.astype(np.float32, copy=False))
        partner_res_blocks.append(np.full((atoms.shape[0],), partner_idx, dtype=np.int64))

    if not partner_coord_blocks:
        return np.empty((2, 0), dtype=np.int64)

    partner_coords = np.concatenate(partner_coord_blocks, axis=0)
    partner_atom_to_res = np.concatenate(partner_res_blocks, axis=0)
    positive_pairs = []

    for target_idx, target_atoms in enumerate(target_atom_arrays):
        if target_atoms.shape[0] == 0:
            continue
        diff = target_atoms[:, None, :] - partner_coords[None, :, :]
        min_dist_to_partner_atoms = np.linalg.norm(diff, axis=-1).min(axis=0)
        hit_atoms = np.flatnonzero(min_dist_to_partner_atoms <= cutoff)
        if hit_atoms.size == 0:
            continue
        for partner_idx in np.unique(partner_atom_to_res[hit_atoms]):
            positive_pairs.append((target_idx, int(partner_idx)))

    if not positive_pairs:
        return np.empty((2, 0), dtype=np.int64)
    return np.asarray(positive_pairs, dtype=np.int64).T


def partner_chain_descriptors(chains, chain_id):
    """Partner inputs in chain order; true cross-chain distances remain labels only."""
    surface_blocks = []
    sequence_blocks = []
    coord_blocks = []
    frame_blocks = []
    geo_edge_blocks = []
    sequence_strings = []
    accepted_chain_ids = []
    chain_slices = []
    chain_blocks = []
    partner_atom_arrays = []
    residue_offset = 0

    for other_chain, other_data in chains.items():
        if other_chain == chain_id:
            continue
        try:
            other_coords, other_frames, _ = chain_coordinates(other_data)
            surface = residue_surface_features(other_data, other_coords)
            sequence = residue_sequence_features(other_data)
            sequence_string = residue_sequence(other_data)
        except Exception:
            continue
        if surface.shape[0] == 0 or sequence.shape[0] == 0:
            continue
        width = min(
            surface.shape[0],
            sequence.shape[0],
            other_coords.shape[0],
            other_frames.shape[0],
            len(sequence_string),
        )
        if width == 0:
            continue
        surface_blocks.append(surface[:width].astype(np.float32, copy=False))
        sequence_blocks.append(sequence[:width].astype(np.float32, copy=False))
        coord_blocks.append(other_coords[:width].astype(np.float32, copy=False))
        frame_blocks.append(other_frames[:width].astype(np.float32, copy=False))
        geo_edges = radius_edges(other_coords[:width], RESIDUE_CUTOFF)
        if geo_edges.shape[1] > 0:
            geo_edge_blocks.append(geo_edges + residue_offset)
        sequence_strings.append(sequence_string[:width])
        accepted_chain_ids.append(other_chain)
        chain_slices.append((residue_offset, residue_offset + width))
        chain_blocks.extend([other_chain] * width)
        partner_atom_arrays.extend(residue_atom_arrays(other_data, limit=width))
        residue_offset += width

    if not surface_blocks:
        return {
            "surface": np.zeros((0, len(SURFACE_RADII) * 2 + 6), dtype=np.float32),
            "sequence_features": np.zeros((0, RESIDUE_SEQUENCE_FEATURE_DIM), dtype=np.float32),
            "coords": np.zeros((0, 3), dtype=np.float32),
            "frames": np.zeros((0, 3, 3), dtype=np.float32),
            "geo_edge": np.empty((2, 0), dtype=np.int64),
            "chains": [],
            "chain_ids": [],
            "chain_slices": np.empty((0, 2), dtype=np.int64),
            "sequences": [],
            "pair_contact_index": np.empty((2, 0), dtype=np.int64),
        }

    pair_contact_index = sparse_partner_pair_contacts(chains[chain_id], partner_atom_arrays)
    return {
        "surface": np.concatenate(surface_blocks, axis=0).astype(np.float32, copy=False),
        "sequence_features": np.concatenate(sequence_blocks, axis=0).astype(np.float32, copy=False),
        "coords": np.concatenate(coord_blocks, axis=0).astype(np.float32, copy=False),
        "frames": np.concatenate(frame_blocks, axis=0).astype(np.float32, copy=False),
        "geo_edge": (
            np.concatenate(geo_edge_blocks, axis=1).astype(np.int64, copy=False)
            if geo_edge_blocks
            else np.empty((2, 0), dtype=np.int64)
        ),
        "chains": chain_blocks,
        "chain_ids": accepted_chain_ids,
        "chain_slices": np.asarray(chain_slices, dtype=np.int64),
        "sequences": sequence_strings,
        "pair_contact_index": pair_contact_index,
    }


def residue_surface_features(chain_data, residue_coords):
    """Approximate solvent exposure and protrusion descriptors from target-chain geometry."""
    residues = chain_data["residues"]
    atoms = chain_data["atoms"]
    n_res = len(residues)
    if n_res == 0:
        return np.empty((0, 14), dtype=np.float32)

    atom_lookup = {res_key: [] for res_key in residues}
    for res_key, atom_name, xyz in atoms:
        if res_key in atom_lookup:
            atom_lookup[res_key].append((atom_name, xyz))

    diff = residue_coords[:, None, :] - residue_coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1).astype(np.float32)
    non_self = dist > 0.0
    safe_dist = np.where(non_self, dist, np.inf)

    counts = []
    densities = []
    for radius in SURFACE_RADII:
        mask = safe_dist <= radius
        counts.append(np.log1p(mask.sum(axis=1)).astype(np.float32))
        densities.append(np.exp(-np.square(np.where(non_self, dist / radius, np.inf))).sum(axis=1).astype(np.float32))

    centroid = residue_coords.mean(axis=0, keepdims=True)
    radial = np.linalg.norm(residue_coords - centroid, axis=1).astype(np.float32)
    radial_norm = radial / max(float(radial.max()), 1e-6)
    radial_z = (radial - radial.mean()) / max(float(radial.std()), 1e-6)

    nearest = np.min(safe_dist, axis=1)
    nearest = np.where(np.isfinite(nearest), nearest, 0.0).astype(np.float32)

    count10 = (safe_dist <= 10.0).sum(axis=1).astype(np.float32)
    exposure = (1.0 / (1.0 + count10)).astype(np.float32)
    local_mean_dist = np.zeros((n_res,), dtype=np.float32)
    local_aniso = np.zeros((n_res,), dtype=np.float32)
    local_flatness = np.zeros((n_res,), dtype=np.float32)
    for i in range(n_res):
        neigh = residue_coords[safe_dist[i] <= 10.0]
        if neigh.shape[0] > 0:
            local_mean_dist[i] = float(np.linalg.norm(neigh - residue_coords[i], axis=1).mean())
        if neigh.shape[0] >= 3:
            centered = neigh - neigh.mean(axis=0, keepdims=True)
            cov = (centered.T @ centered) / max(neigh.shape[0] - 1, 1)
            eigvals = np.linalg.eigvalsh(cov).clip(min=0.0)
            total = float(eigvals.sum())
            if total > 1e-6:
                local_aniso[i] = float(eigvals[-1] / total)
                local_flatness[i] = float(eigvals[0] / total)

    atom_counts = np.zeros((n_res,), dtype=np.float32)
    sidechain_extent = np.zeros((n_res,), dtype=np.float32)
    for i, res_key in enumerate(residues):
        res_atoms = atom_lookup.get(res_key, [])
        atom_counts[i] = np.log1p(len(res_atoms))
        sidechain_atoms = [xyz for atom_name, xyz in res_atoms if atom_name not in {"N", "CA", "C", "O"}]
        if sidechain_atoms:
            sc_centroid = np.stack(sidechain_atoms, axis=0).mean(axis=0)
            sidechain_extent[i] = float(np.linalg.norm(sc_centroid - residue_coords[i]))

    features = np.stack(
        [
            counts[0],
            counts[1],
            counts[2],
            counts[3],
            densities[1],
            densities[3],
            radial_norm,
            radial_z,
            nearest,
            exposure,
            local_mean_dist,
            local_aniso,
            local_flatness,
            atom_counts + 0.1 * sidechain_extent,
        ],
        axis=1,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def residue_sequence_features(chain_data):
    """Residue identity and biochemical descriptors from the matched target chain."""
    features = []
    aa_to_idx = {aa: i for i, aa in enumerate(AA_ORDER)}
    for res_key in chain_data["residues"]:
        res_name = res_key[3].upper()
        aa = AA3_TO_1.get(res_name, "X")
        one_hot = np.zeros((len(AA_ORDER),), dtype=np.float32)
        if aa in aa_to_idx:
            one_hot[aa_to_idx[aa]] = 1.0

        hydropathy, volume, polarity, iso_point = AA_PROPERTIES.get(aa, (0.0, 130.0, 8.5, 6.0))
        continuous = np.asarray(
            [
                hydropathy / 4.5,
                (volume - 130.0) / 60.0,
                (polarity - 8.5) / 4.5,
                (iso_point - 6.0) / 4.0,
            ],
            dtype=np.float32,
        )
        flags = np.asarray(
            [
                aa in {"D", "E"},
                aa in {"K", "R", "H"},
                aa in {"S", "T", "N", "Q", "C", "Y", "W", "H"},
                aa in {"A", "V", "I", "L", "M", "F", "W", "Y"},
                aa in {"F", "W", "Y", "H"},
                aa in {"G"},
                aa in {"P"},
                aa in {"C", "M"},
                aa in {"S", "T", "N", "Q", "Y", "W", "H", "K", "R", "D", "E"},
                aa in {"A", "V", "I", "L"},
                aa in {"N", "Q", "S", "T"},
                aa in {"D", "E", "K", "R", "H"},
                aa == "X",
                aa in {"C"},
                aa in {"W", "Y"},
            ],
            dtype=np.float32,
        )
        features.append(np.concatenate([one_hot, continuous, flags], axis=0))

    if not features:
        return np.empty((0, RESIDUE_SEQUENCE_FEATURE_DIM), dtype=np.float32)
    return np.stack(features, axis=0).astype(np.float32)


def residue_sequence(chain_data):
    return "".join(AA3_TO_1.get(res_key[3].upper(), "X") for res_key in chain_data["residues"])


def find_matching_chain(sample, chains, used_chains, preferred_chain=""):
    n_res = sample["residue_graph_node"].shape[0]
    n_atom = sample["atom_graph_node"].shape[0]
    preferred_candidates = []
    if preferred_chain:
        if preferred_chain in chains:
            preferred_candidates.append(preferred_chain)
        if len(preferred_chain) > 1:
            preferred_candidates.extend([ch for ch in preferred_chain if ch in chains])
    for chain_id in dict.fromkeys(preferred_candidates):
        chain_data = chains[chain_id]
        if len(chain_data["residues"]) == n_res:
            return chain_id, len(chain_data["atoms"]) == n_atom

    exact_candidates = []
    residue_candidates = []
    for chain_id, chain_data in chains.items():
        if len(chain_data["residues"]) == n_res:
            residue_candidates.append(chain_id)
            if len(chain_data["atoms"]) == n_atom:
                exact_candidates.append(chain_id)

    unused = [chain for chain in exact_candidates if chain not in used_chains]
    if unused:
        return unused[0], True
    if exact_candidates:
        return exact_candidates[0], True

    unused = [chain for chain in residue_candidates if chain not in used_chains]
    if unused:
        return unused[0], False
    if residue_candidates:
        return residue_candidates[0], False
    return None, False


def augment_file(filename):
    os.makedirs(OUT_DIR, exist_ok=True)
    geo_path = find_existing_file(OUT_DIR, filename)
    in_path = geo_path if os.path.exists(geo_path) else find_existing_file(DATA_DIR, filename)
    out_path = os.path.join(OUT_DIR, filename)

    with open(in_path, "rb") as f:
        samples = pickle.load(f)

    pdb_cache = {}
    used_by_pdb = {}
    matched = 0
    failed = []

    for i, sample in enumerate(samples):
        pdb_id, chain_hint = parse_complex_code(sample["complex_code"])
        try:
            if pdb_id not in pdb_cache:
                pdb_path = download_pdb(pdb_id)
                pdb_cache[pdb_id] = parse_pdb_chains(pdb_path)
            chains = pdb_cache[pdb_id]
            used = used_by_pdb.setdefault(pdb_id, set())
            chain_id, exact_atom_match = find_matching_chain(sample, chains, used, chain_hint)
            if chain_id is None:
                failed.append((i, pdb_id, sample["residue_graph_node"].shape[0], sample["atom_graph_node"].shape[0]))
                continue

            used.add(chain_id)
            residue_coords, residue_frames, atom_coords = chain_coordinates(chains[chain_id])
            sample["pdb_chain"] = chain_id
            sample["residue_coords"] = residue_coords
            sample["residue_frames"] = residue_frames
            sample["residue_sequence"] = residue_sequence(chains[chain_id])
            sample["residue_surface_features"] = residue_surface_features(chains[chain_id], residue_coords)
            sample["residue_sequence_features"] = residue_sequence_features(chains[chain_id])
            contact_label, contact_mask, contact_distance = partner_contact_labels(chains[chain_id], chains, chain_id)
            sample["partner_contact_label"] = contact_label
            sample["partner_contact_mask"] = contact_mask
            sample["partner_contact_distance"] = contact_distance
            partner_inputs = partner_chain_descriptors(chains, chain_id)
            sample["partner_residue_surface_features"] = partner_inputs["surface"]
            sample["partner_residue_sequence_features"] = partner_inputs["sequence_features"]
            sample["partner_residue_coords"] = partner_inputs["coords"]
            sample["partner_residue_frames"] = partner_inputs["frames"]
            sample["partner_residue_geo_edge"] = partner_inputs["geo_edge"]
            sample["partner_residue_chains"] = partner_inputs["chains"]
            sample["partner_chain_ids"] = partner_inputs["chain_ids"]
            sample["partner_chain_slices"] = partner_inputs["chain_slices"]
            sample["partner_residue_sequences"] = partner_inputs["sequences"]
            sample["partner_pair_contact_index"] = partner_inputs["pair_contact_index"]
            sample["partner_pair_contact_cutoff"] = PARTNER_CONTACT_CUTOFF
            if exact_atom_match:
                sample["atom_coords"] = atom_coords
            sample["residue_geo_edge"] = radius_edges(residue_coords, RESIDUE_CUTOFF)
            if exact_atom_match and COMPUTE_ATOM_GEO_EDGES:
                sample["atom_geo_edge"] = radius_edges(atom_coords, ATOM_CUTOFF)
            matched += 1
        except Exception as exc:
            failed.append((i, pdb_id, repr(exc)))

        if (i + 1) % 25 == 0 or i + 1 == len(samples):
            print(f"{filename}: {i + 1}/{len(samples)} processed, matched={matched}", flush=True)

    temp_path = f"{out_path}.tmp"
    with open(temp_path, "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp_path, out_path)

    print(f"Saved {out_path}")
    print(f"Matched {matched}/{len(samples)}")
    if failed:
        fail_path = os.path.join(OUT_DIR, f"{os.path.splitext(filename)[0]}_unmatched.txt")
        with open(fail_path, "w", encoding="utf-8") as f:
            for item in failed:
                f.write(repr(item) + "\n")
        print(f"Unmatched details: {fail_path}")


def main():
    filenames = sys.argv[1:] if len(sys.argv) > 1 else PKL_FILES
    for filename in filenames:
        augment_file(filename)


if __name__ == "__main__":
    main()
