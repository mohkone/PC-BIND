import argparse
import os
import pickle

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


GEO_DIR = os.path.join("data", "geo")
DEFAULT_MODEL = "facebook/esm2_t12_35M_UR50D"
PLM_FEATURE_KEY = "residue_plm_embedding"
PLM_MODEL_KEY = "residue_plm_model"
MAX_RESIDUES = 1022
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


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path, obj):
    temp_path = f"{path}.tmp"
    with open(temp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp_path, path)


def fallback_sequence_from_biochem(sample):
    aa_order = "ACDEFGHIKLMNPQRSTVWY"
    feats = sample.get("residue_sequence_features")
    if feats is None:
        return None
    feats = np.asarray(feats)
    if feats.ndim != 2 or feats.shape[1] < len(aa_order):
        return None
    idx = feats[:, : len(aa_order)].argmax(axis=1)
    known = feats[:, : len(aa_order)].max(axis=1) > 0.5
    return "".join(aa_order[i] if ok else "X" for i, ok in zip(idx, known))


def normalized_sequence(sample):
    seq = sample.get("residue_sequence") or fallback_sequence_from_biochem(sample)
    if seq is None:
        return None
    seq = "".join(aa if aa in "ACDEFGHIKLMNPQRSTVWY" else "X" for aa in str(seq).upper())
    n_res = sample["residue_graph_node"].shape[0]
    if len(seq) != n_res:
        return None
    return seq


def normalized_partner_sequences(sample):
    sequences = sample.get("partner_residue_sequences")
    expected = int(np.asarray(sample.get("partner_residue_surface_features", [])).shape[0])
    if sequences is None:
        return [] if expected == 0 else None

    normalized = []
    for sequence in sequences:
        sequence = "".join(
            aa if aa in "ACDEFGHIKLMNPQRSTVWY" else "X"
            for aa in str(sequence).upper()
        )
        if sequence:
            normalized.append(sequence)
    if sum(map(len, normalized)) != expected:
        return None
    return normalized


def expected_embedding_rows(sample, partner=False):
    if partner:
        features = np.asarray(sample.get("partner_residue_surface_features", []))
        return int(features.shape[0]) if features.ndim == 2 else 0
    return int(sample["residue_graph_node"].shape[0])


def embed_sequence(seq, tokenizer, model, device, plm_dim, max_residues=MAX_RESIDUES):
    if not seq:
        return np.empty((0, plm_dim), dtype=np.float32)

    chunks = []
    for start in range(0, len(seq), max_residues):
        chunk = seq[start : start + max_residues]
        encoded = tokenizer(
            " ".join(chunk),
            return_tensors="pt",
            add_special_tokens=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)
        hidden = output.last_hidden_state[0, 1 : len(chunk) + 1].detach().cpu().float().numpy()
        chunks.append(hidden.astype(np.float32))
    return np.concatenate(chunks, axis=0).astype(np.float32)


def needs_embedding(
    sample,
    model_name,
    overwrite,
    feature_key=PLM_FEATURE_KEY,
    model_key=PLM_MODEL_KEY,
    partner=False,
):
    if overwrite:
        return True
    embedding = sample.get(feature_key)
    if embedding is None:
        return True
    if sample.get(model_key) != model_name:
        return True
    n_res = expected_embedding_rows(sample, partner=partner)
    embedding = np.asarray(embedding)
    return embedding.ndim != 2 or embedding.shape[0] != n_res


def extract_file(
    filename,
    tokenizer,
    model,
    device,
    model_name,
    plm_dim,
    overwrite=False,
    feature_key=PLM_FEATURE_KEY,
    model_key=PLM_MODEL_KEY,
    partner=False,
):
    path = find_existing_file(GEO_DIR, filename)
    samples = load_pickle(path)
    embedded = 0
    skipped = 0
    failed = []

    for i, sample in enumerate(samples):
        if not needs_embedding(
            sample,
            model_name,
            overwrite,
            feature_key=feature_key,
            model_key=model_key,
            partner=partner,
        ):
            skipped += 1
            continue

        sequences = normalized_partner_sequences(sample) if partner else [normalized_sequence(sample)]
        if sequences is None or any(sequence is None for sequence in sequences):
            n_res = expected_embedding_rows(sample, partner=partner)
            sample[feature_key] = np.zeros((n_res, plm_dim), dtype=np.float32)
            sample[model_key] = "missing_sequence"
            failed.append((i, str(sample.get("complex_code", "?")), n_res))
            continue

        embedding_blocks = [
            embed_sequence(sequence, tokenizer, model, device, plm_dim)
            for sequence in sequences
        ]
        embedding = (
            np.concatenate(embedding_blocks, axis=0).astype(np.float32, copy=False)
            if embedding_blocks
            else np.empty((0, plm_dim), dtype=np.float32)
        )
        expected_rows = expected_embedding_rows(sample, partner=partner)
        if embedding.shape[0] != expected_rows:
            raise RuntimeError(
                f"{filename}[{i}] embedding length mismatch: "
                f"{embedding.shape[0]} vs {expected_rows}"
            )
        sample[feature_key] = embedding
        sample[model_key] = model_name
        embedded += 1

        if (i + 1) % 25 == 0 or i + 1 == len(samples):
            print(
                f"{filename}: {i + 1}/{len(samples)} processed, embedded={embedded}, skipped={skipped}",
                flush=True,
            )

    save_pickle(path, samples)
    print(f"Saved {path}")
    print(f"Embedded {embedded}, skipped {skipped}, failed {len(failed)}")
    if failed:
        suffix = "partner_plm_failed" if partner else "plm_failed"
        fail_path = os.path.join(GEO_DIR, f"{os.path.splitext(filename)[0]}_{suffix}.txt")
        with open(fail_path, "w", encoding="utf-8") as f:
            for item in failed:
                f.write(repr(item) + "\n")
        print(f"PLM failure details: {fail_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract per-residue ESM-2 embeddings for geo pickle files.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--feature-key", default=PLM_FEATURE_KEY)
    parser.add_argument("--model-key", default=PLM_MODEL_KEY)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--partner",
        action="store_true",
        help="Embed chain-separated partner sequences instead of the target sequence.",
    )
    parser.add_argument("files", nargs="*", default=PKL_FILES)
    args = parser.parse_args()

    feature_key = args.feature_key
    model_key = args.model_key
    if args.partner:
        if not feature_key.startswith("partner_"):
            feature_key = f"partner_{feature_key}"
        if not model_key.startswith("partner_"):
            model_key = f"partner_{model_key}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PLM model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()
    plm_dim = int(getattr(model.config, "hidden_size"))
    print(f"PLM embedding dim: {plm_dim}")

    for filename in args.files:
        extract_file(
            filename,
            tokenizer,
            model,
            device,
            args.model,
            plm_dim,
            overwrite=args.overwrite,
            feature_key=feature_key,
            model_key=model_key,
            partner=args.partner,
        )


if __name__ == "__main__":
    main()
