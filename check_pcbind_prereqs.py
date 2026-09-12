import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


PKL_FILES = [
    "Train335.pkl",
    "Test60.pkl",
    "Test287.pkl",
    "Test70.pkl",
    "TestB25.pkl",
    "TestUB25.pkl",
]


def count_key(samples, key):
    return sum(key in sample for sample in samples)


def partner_residue_count(sample):
    surface = np.asarray(sample.get("partner_residue_surface_features", []))
    return int(surface.shape[0]) if surface.ndim == 2 else 0


def count_nonempty_partner_key(samples, key):
    return sum(key in sample and partner_residue_count(sample) > 0 for sample in samples)


def count_positive_pairs(samples):
    total = 0
    labelled = 0
    for sample in samples:
        if "partner_pair_contact_index" not in sample or partner_residue_count(sample) == 0:
            continue
        labelled += 1
        pair_index = np.asarray(sample["partner_pair_contact_index"])
        if pair_index.ndim == 2 and pair_index.shape[0] == 2:
            total += int(pair_index.shape[1])
    return labelled, total


def valid_partner_encoder_sample(sample):
    required = (
        "partner_residue_surface_features",
        "partner_residue_sequence_features",
        "partner_residue_coords",
        "partner_residue_frames",
        "partner_residue_geo_edge",
        "partner_residue_sequences",
        "partner_chain_slices",
        "partner_residue_plm_embedding",
        "partner_residue_plm_embedding_8m",
    )
    if any(key not in sample for key in required):
        return False

    surface = np.asarray(sample["partner_residue_surface_features"])
    sequence = np.asarray(sample["partner_residue_sequence_features"])
    coords = np.asarray(sample["partner_residue_coords"])
    frames = np.asarray(sample["partner_residue_frames"])
    edges = np.asarray(sample["partner_residue_geo_edge"])
    slices = np.asarray(sample["partner_chain_slices"])
    main_plm = np.asarray(sample["partner_residue_plm_embedding"])
    aux_plm = np.asarray(sample["partner_residue_plm_embedding_8m"])
    if surface.ndim != 2:
        return False
    n_res = int(surface.shape[0])
    if n_res <= 0:
        return False
    if sequence.ndim != 2 or sequence.shape[0] != n_res:
        return False
    if coords.shape != (n_res, 3) or frames.shape != (n_res, 3, 3):
        return False
    if edges.ndim != 2 or edges.shape[0] != 2:
        return False
    if edges.size and (edges.min() < 0 or edges.max() >= n_res):
        return False
    if slices.ndim != 2 or slices.shape[1] != 2 or slices.shape[0] == 0:
        return False
    if int(slices[0, 0]) != 0 or int(slices[-1, 1]) != n_res:
        return False
    if np.any(slices[:, 0] < 0) or np.any(slices[:, 1] <= slices[:, 0]):
        return False
    if slices.shape[0] > 1 and np.any(slices[1:, 0] != slices[:-1, 1]):
        return False
    if sum(len(str(seq)) for seq in sample["partner_residue_sequences"]) != n_res:
        return False
    if main_plm.ndim != 2 or main_plm.shape[0] != n_res or main_plm.shape[1] == 0:
        return False
    if aux_plm.ndim != 2 or aux_plm.shape[0] != n_res or aux_plm.shape[1] == 0:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Check PC-BIND input prerequisites.")
    parser.add_argument("--data-dir", default="data/geo", help="Directory containing geo pickle files.")
    parser.add_argument("--require-plm", action="store_true", help="Require both ESM-2 PLM feature fields.")
    parser.add_argument("--require-partner", action="store_true", help="Require partner descriptor fields.")
    parser.add_argument("--require-pair", action="store_true", help="Require sparse residue-pair contact labels.")
    parser.add_argument(
        "--require-partner-encoder",
        action="store_true",
        help="Require chain-aware partner geometry and both partner ESM embeddings.",
    )
    parser.add_argument("--min-partner-encoder-coverage", type=float, default=0.98)
    parser.add_argument("files", nargs="*", default=PKL_FILES)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    errors = []
    rows = []
    for filename in args.files:
        path = data_dir / filename
        if not path.exists():
            errors.append(f"{path}: missing")
            continue
        with path.open("rb") as handle:
            samples = pickle.load(handle)
        total = len(samples)
        main_plm = count_key(samples, "residue_plm_embedding")
        aux_plm = count_key(samples, "residue_plm_embedding_8m")
        partner_surface = count_nonempty_partner_key(
            samples, "partner_residue_surface_features"
        )
        partner_sequence = count_nonempty_partner_key(
            samples, "partner_residue_sequence_features"
        )
        pair_labelled, positive_pairs = count_positive_pairs(samples)
        partner_encoder = sum(valid_partner_encoder_sample(sample) for sample in samples)
        invalid_partner_samples = [
            f"{idx}:{sample.get('complex_code', '?')}"
            for idx, sample in enumerate(samples)
            if not valid_partner_encoder_sample(sample)
        ]
        rows.append((
            filename,
            total,
            main_plm,
            aux_plm,
            partner_surface,
            partner_sequence,
            pair_labelled,
            positive_pairs,
            partner_encoder,
        ))

        if args.require_plm and (main_plm == 0 or aux_plm == 0):
            errors.append(
                f"{path}: PLM coverage main={main_plm}/{total}, "
                f"aux={aux_plm}/{total}"
            )
        if args.require_partner and (partner_surface == 0 or partner_sequence == 0):
            errors.append(
                f"{path}: partner coverage surface={partner_surface}/{total}, "
                f"sequence={partner_sequence}/{total}"
            )
        if args.require_pair and pair_labelled == 0:
            errors.append(f"{path}: sparse pair-contact labels missing")
        if args.require_partner_encoder:
            coverage = partner_encoder / max(total, 1)
            if coverage < args.min_partner_encoder_coverage:
                errors.append(
                    f"{path}: partner encoder coverage={partner_encoder}/{total} "
                    f"({coverage:.1%}) below {args.min_partner_encoder_coverage:.1%}; "
                    f"invalid samples={', '.join(invalid_partner_samples[:20])}"
                )

    for (
        filename,
        total,
        main_plm,
        aux_plm,
        partner_surface,
        partner_sequence,
        pair_labelled,
        positive_pairs,
        partner_encoder,
    ) in rows:
        print(
            f"{filename}: main_plm={main_plm}/{total} aux_plm={aux_plm}/{total} "
            f"partner_surface={partner_surface}/{total} partner_sequence={partner_sequence}/{total} "
            f"pair_labels={pair_labelled}/{total} positive_pairs={positive_pairs}"
            f" partner_encoder={partner_encoder}/{total}"
        )

    if errors:
        print("\nPC-BIND prerequisites are not satisfied:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("\nPC-BIND prerequisites OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
