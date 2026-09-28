# Read-only input integrity audit

Generated: 2026-09-28T02:43:21.985382+00:00. Input directory resolves to `C:\Users\Mohamed KONE\Desktop\PC-BIND-OT\data\geo_filtered_v1`.

This audit reads the linked pickle files without altering them or downloading anything. It checks input coverage and exact target-sequence/PDB identity only. It is not a sequence homology, structural-independence, partner-independence, or label-alignment proof. Indices below are zero-based. File hashes and every affected sample are in `input_audit.json`.

| Dataset | Samples | Complex codes / PDB IDs | Valid target main / aux PLM | Valid partner encoder | Nonempty partner |
|---|---:|---:|---:|---:|---:|
| Train335.pkl | 334 | 209 / 209 | 334 / 334 | 334 | 334 |
| Test287.pkl | 285 | 215 / 215 | 285 / 285 | 285 | 285 |

## Target input failures


## Partner coverage and provenance

- Existing mapping-verification metadata counts by dataset: Train335.pkl: 0/334; Test287.pkl: 0/285. A successful shape check does not verify original residue/label or atom ordering.

## Exact overlap with Train335

| Test dataset | Shared full complex codes | Shared normalized PDB IDs | Test samples with identical target sequence |
|---|---:|---:|---:|
| Test287.pkl | 0 | 0 | 0 |

Zero exact matches does not exclude close homologs. Missing sequences are not treated as identical. The sequence comparison uses SHA-256 fingerprints of normalized whole target strings; no local alignment or BLAST search was performed.

## Exact sequence overlap across input sides

Each cell reports affected test samples / distinct matched sequences. Partner comparisons use stored chain-separated partner sequences only; missing partner sequence fields are not reconstructed. These are input-identity overlaps, not evidence by themselves of label contamination.

| Test dataset | Test target → train partner | Test partner → train target | Test partner → train partner | Test samples with partner sequences |
|---|---:|---:|---:|---:|
| Test287.pkl | 0 / 0 | 0 / 0 | 0 / 0 | 285 |

## Identifier formats

- `Train335.pkl`: 334 bare_pdb.
- `Test287.pkl`: 285 bare_pdb.

Normalize PDB identity separately from full complex codes when auditing overlap: a chain suffix must not hide a shared source structure. Preserve the original complex grouping key in recorded experiments.
