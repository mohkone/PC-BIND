import argparse
import json
import os
import pickle
from collections import defaultdict

import numpy as np


DEFAULT_FILES = (
    "Train335.pkl",
    "Test60.pkl",
    "Test287.pkl",
    "Test70.pkl",
    "TestB25.pkl",
    "TestUB25.pkl",
)


def complex_id(sample):
    return str(sample.get("complex_code", "")).strip().upper()


def chain_id(sample):
    return str(sample.get("pdb_chain", "")).strip()


def partner_chains(sample):
    values = np.asarray(sample.get("partner_residue_chains", []))
    return [str(value).strip() for value in values.tolist()]


def dataset_identifiers(path):
    with open(path, "rb") as handle:
        samples = pickle.load(handle)
    return {
        "pdb_ids": {complex_id(sample) for sample in samples},
        "pdb_chains": {(complex_id(sample), chain_id(sample)) for sample in samples},
        "sequences": {
            str(sample.get("residue_sequence", "")).strip().upper()
            for sample in samples
            if str(sample.get("residue_sequence", "")).strip()
        },
    }


def group_samples(samples):
    groups = defaultdict(list)
    for sample_idx, sample in enumerate(samples):
        groups[complex_id(sample)].append(sample_idx)
    return groups


def paired_records(samples, groups):
    records = []
    ambiguous = 0
    length_mismatch = 0

    for target_idx, target in enumerate(samples):
        target_chain = chain_id(target)
        target_partner_chains = partner_chains(target)
        candidates = []
        for candidate_idx in groups[complex_id(target)]:
            if candidate_idx == target_idx:
                continue
            candidate_chain = chain_id(samples[candidate_idx])
            if candidate_chain and candidate_chain in target_partner_chains:
                candidates.append(candidate_idx)

        if len(candidates) > 1:
            ambiguous += 1
        for candidate_idx in candidates:
            candidate = samples[candidate_idx]
            candidate_chain = chain_id(candidate)
            partner_positions = np.flatnonzero(
                np.asarray(target_partner_chains, dtype=object) == candidate_chain
            )
            candidate_length = int(candidate["residue_graph_node"].shape[0])
            aligned = int(partner_positions.size) == candidate_length
            length_mismatch += int(not aligned)
            records.append(
                {
                    "complex_code": complex_id(target),
                    "target_index": target_idx,
                    "target_chain": target_chain,
                    "partner_index": candidate_idx,
                    "partner_chain": candidate_chain,
                    "target_length": int(target["residue_graph_node"].shape[0]),
                    "partner_length": candidate_length,
                    "partner_offset_start": (
                        int(partner_positions[0]) if partner_positions.size else None
                    ),
                    "partner_offset_end": (
                        int(partner_positions[-1]) + 1 if partner_positions.size else None
                    ),
                    "partner_length_aligned": aligned,
                }
            )

    return records, ambiguous, length_mismatch


def reproduce_sample_folds(num_samples, seed, num_folds):
    rng = np.random.RandomState(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    fold_ids = np.arange(num_samples) % num_folds
    sample_folds = np.empty((num_samples,), dtype=np.int64)
    sample_folds[indices] = fold_ids
    return sample_folds


def split_leakage(groups, sample_folds):
    multi_groups = {key: value for key, value in groups.items() if len(value) > 1}
    leaking = {
        key: value
        for key, value in multi_groups.items()
        if len({int(sample_folds[idx]) for idx in value}) > 1
    }
    return multi_groups, leaking


def audit_file(path, seed, num_folds):
    with open(path, "rb") as handle:
        samples = pickle.load(handle)

    groups = group_samples(samples)
    records, ambiguous, length_mismatch = paired_records(samples, groups)
    paired_targets = {record["target_index"] for record in records}
    aligned_records = [record for record in records if record["partner_length_aligned"]]
    summary = {
        "file": os.path.abspath(path),
        "samples": len(samples),
        "unique_complexes": len(groups),
        "multi_sample_complexes": sum(len(indices) > 1 for indices in groups.values()),
        "samples_in_multi_sample_complexes": sum(
            len(indices) for indices in groups.values() if len(indices) > 1
        ),
        "paired_targets": len(paired_targets),
        "directed_pair_records": len(records),
        "aligned_pair_records": len(aligned_records),
        "ambiguous_targets": ambiguous,
        "length_mismatches": length_mismatch,
    }

    if os.path.basename(path).lower() == "train335.pkl":
        sample_folds = reproduce_sample_folds(len(samples), seed, num_folds)
        multi_groups, leaking = split_leakage(groups, sample_folds)
        summary.update(
            {
                "audit_seed": seed,
                "num_folds": num_folds,
                "multi_sample_complexes_crossing_folds": len(leaking),
                "cross_fold_rate": (
                    len(leaking) / len(multi_groups) if multi_groups else 0.0
                ),
                "samples_in_cross_fold_complexes": sum(
                    len(indices) for indices in leaking.values()
                ),
                "cross_fold_examples": [
                    {
                        "complex_code": key,
                        "samples": [
                            {
                                "index": idx,
                                "chain": chain_id(samples[idx]),
                                "fold": int(sample_folds[idx]),
                            }
                            for idx in indices
                        ],
                    }
                    for key, indices in list(leaking.items())[:10]
                ],
            }
        )

    return summary, aligned_records


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit full-sample partner pairing and sample-level CV leakage."
    )
    parser.add_argument("--data-dir", default=os.path.join("data", "geo"))
    parser.add_argument("--seed", type=int, default=2091)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", default="pcbind_pairing_audit.json")
    parser.add_argument(
        "files",
        nargs="*",
        default=list(DEFAULT_FILES),
        help="Dataset pickle filenames under --data-dir.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    report = {
        "purpose": (
            "Identify leakage-free complex groups and samples with a complete "
            "partner-side graph for Siamese PC-BIND."
        ),
        "datasets": {},
        "pair_manifest": {},
    }

    filenames = list(args.files)
    if "Train335.pkl" not in filenames:
        raise ValueError("The pairing audit requires Train335.pkl.")

    for filename in filenames:
        path = os.path.join(args.data_dir, filename)
        summary, records = audit_file(path, args.seed, args.folds)
        report["datasets"][filename] = summary
        report["pair_manifest"][filename] = records
        print(
            f"{filename}: samples={summary['samples']} "
            f"complexes={summary['unique_complexes']} "
            f"paired_targets={summary['paired_targets']} "
            f"aligned_pairs={summary['aligned_pair_records']}"
        )
        if "multi_sample_complexes_crossing_folds" in summary:
            print(
                "  Current sample-level CV: "
                f"{summary['multi_sample_complexes_crossing_folds']}/"
                f"{summary['multi_sample_complexes']} multi-sample complexes cross folds "
                f"({summary['cross_fold_rate']:.1%})."
            )

    identifiers = {
        filename: dataset_identifiers(os.path.join(args.data_dir, filename))
        for filename in filenames
    }
    train_ids = identifiers["Train335.pkl"]
    report["train_test_overlap"] = {}
    print("Train335 overlap audit:")
    for filename in filenames:
        if filename == "Train335.pkl":
            continue
        test_ids = identifiers[filename]
        overlap = {
            "shared_pdb_ids": len(train_ids["pdb_ids"] & test_ids["pdb_ids"]),
            "shared_pdb_chains": len(train_ids["pdb_chains"] & test_ids["pdb_chains"]),
            "shared_exact_sequences": len(train_ids["sequences"] & test_ids["sequences"]),
        }
        report["train_test_overlap"][filename] = overlap
        print(
            f"  {filename}: PDB={overlap['shared_pdb_ids']} "
            f"PDB+chain={overlap['shared_pdb_chains']} "
            f"exact_sequence={overlap['shared_exact_sequences']}"
        )

    output_path = os.path.abspath(args.output)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
