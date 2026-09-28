# Read-only input integrity audit

Generated: 2026-09-27T09:11:40.556389+00:00. Input directory resolves to `C:\Users\Mohamed KONE\Desktop\article\data\geo`.

This audit reads the linked pickle files without altering them or downloading anything. It checks input coverage and exact target-sequence/PDB identity only. It is not a sequence homology, structural-independence, partner-independence, or label-alignment proof. Indices below are zero-based. File hashes and every affected sample are in `input_audit.json`.

| Dataset | Samples | Complex codes / PDB IDs | Valid target main / aux PLM | Valid partner encoder | Nonempty partner |
|---|---:|---:|---:|---:|---:|
| Train335.pkl | 335 | 209 / 209 | 334 / 334 | 334 | 334 |
| Test60.pkl | 60 | 58 / 58 | 60 / 60 | 60 | 60 |
| Test287.pkl | 287 | 217 / 217 | 285 / 285 | 285 | 285 |
| TestB25.pkl | 25 | 23 / 23 | 25 / 25 | 25 | 25 |
| TestUB25.pkl | 25 | 25 / 25 | 25 / 25 | 12 | 12 |
| Test70.pkl | 70 | 70 / 66 | 69 / 69 | 0 | 69 |

## Target input failures

- `Train335.pkl[106]`, `2J3R`: residue_plm_model: 'missing_sequence'; residue_plm_model_8m: 'missing_sequence'; missing or wrong-length target sequence.
- `Test287.pkl[55]`, `4M0W`: residue_plm_model: 'missing_sequence'; residue_plm_model_8m: 'missing_sequence'; missing or wrong-length target sequence.
- `Test287.pkl[161]`, `6MAV`: residue_plm_model: 'missing_sequence'; residue_plm_model_8m: 'missing_sequence'; missing or wrong-length target sequence.
- `Test70.pkl[63]`, `1K5D_AB`: residue_plm_model: 'missing_sequence'; residue_plm_model_8m: 'missing_sequence'; missing or wrong-length target sequence.

## Partner coverage and provenance

- `Train335.pkl`: 1 invalid or absent partner encoders. 1 × missing fields: partner_residue_surface_features, partner_residue_sequence_features, partner_residue_coords, partner_residue_frames, partner_residue_geo_edge, partner_residue_sequences, partner_chain_slices; 1 × zero partner residues (target-pathway fallback).
- `Test287.pkl`: 2 invalid or absent partner encoders. 2 × missing fields: partner_residue_surface_features, partner_residue_sequence_features, partner_residue_coords, partner_residue_frames, partner_residue_geo_edge, partner_residue_sequences, partner_chain_slices; 2 × zero partner residues (target-pathway fallback).
- `TestUB25.pkl`: 13 invalid or absent partner encoders. 13 × zero partner residues (target-pathway fallback).
- `Test70.pkl`: 70 invalid or absent partner encoders. 69 × missing fields: partner_residue_coords, partner_residue_frames, partner_residue_geo_edge, partner_residue_sequences, partner_chain_slices, partner_residue_plm_embedding, partner_residue_plm_embedding_8m; 1 × missing fields: partner_residue_surface_features, partner_residue_sequence_features, partner_residue_coords, partner_residue_frames, partner_residue_geo_edge, partner_residue_sequences, partner_chain_slices, partner_residue_plm_embedding, partner_residue_plm_embedding_8m; 1 × zero partner residues (target-pathway fallback).
- Existing mapping-verification metadata counts by dataset: Train335.pkl: 0/335; Test60.pkl: 0/60; Test287.pkl: 0/287; TestB25.pkl: 0/25; TestUB25.pkl: 0/25; Test70.pkl: 0/70. A successful shape check does not verify original residue/label or atom ordering.

## Exact overlap with Train335

| Test dataset | Shared full complex codes | Shared normalized PDB IDs | Test samples with identical target sequence |
|---|---:|---:|---:|
| Test60.pkl | 49 | 49 | 0 |
| Test287.pkl | 0 | 0 | 0 |
| TestB25.pkl | 18 | 18 | 0 |
| TestUB25.pkl | 0 | 0 | 0 |
| Test70.pkl | 0 | 58 | 53 |

Zero exact matches does not exclude close homologs. Missing sequences are not treated as identical. The sequence comparison uses SHA-256 fingerprints of normalized whole target strings; no local alignment or BLAST search was performed.

## Exact sequence overlap across input sides

Each cell reports affected test samples / distinct matched sequences. Partner comparisons use stored chain-separated partner sequences only; missing partner sequence fields are not reconstructed. These are input-identity overlaps, not evidence by themselves of label contamination.

| Test dataset | Test target → train partner | Test partner → train target | Test partner → train partner | Test samples with partner sequences |
|---|---:|---:|---:|---:|
| Test60.pkl | 49 / 49 | 49 / 50 | 15 / 36 | 60 |
| Test287.pkl | 0 / 0 | 0 / 0 | 0 / 0 | 285 |
| TestB25.pkl | 18 / 18 | 18 / 19 | 5 / 9 | 25 |
| TestUB25.pkl | 5 / 5 | 0 / 0 | 2 / 2 | 12 |
| Test70.pkl | 50 / 50 | 0 / 0 | 0 / 0 | 0 |

## Identifier formats

- `Train335.pkl`: 335 bare_pdb.
- `Test60.pkl`: 60 bare_pdb.
- `Test287.pkl`: 287 bare_pdb.
- `TestB25.pkl`: 25 bare_pdb.
- `TestUB25.pkl`: 25 bare_pdb.
- `Test70.pkl`: 70 chain_suffix.

Normalize PDB identity separately from full complex codes when auditing overlap: a chain suffix must not hide a shared source structure. Preserve the original complex grouping key in recorded experiments.
