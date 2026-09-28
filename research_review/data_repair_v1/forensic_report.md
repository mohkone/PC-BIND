# Data repair v1: read-only forensic report

No admissible exact repair is supported by the available metadata. Use an explicitly labelled filtered sensitivity cohort unless upstream target chain and original residue/atom identity order are recovered.

The three original records have bare PDB identifiers, no declared target chain, no target residue sequence or residue/atom identity list, no residue/atom coordinates, and no surface or biochemical sequence features. Their target PLM arrays are all-zero placeholders marked `missing_sequence`. The 1,024-column residue feature rows and internal atom-to-residue indices do not establish a PDB residue identity/order. Existing labels can be hashed and preserved, but cannot be aligned exactly to an undeclared chain.

| Original record (zero-based) | Target residues / atoms | Positive labels | Declared target chain | Exact repair |
|---|---:|---:|---|---|
| Train335.pkl[106] 2J3R | 158 / 1214 | 38 | Missing | Not admissible |
| Test287.pkl[55] 4M0W | 317 / 2493 | 32 | Missing | Not admissible |
| Test287.pkl[161] 6MAV | 161 / 1277 | 30 | Missing | Not admissible |

## Expected coordinate caches

The project-local `pdb_cache` directory is absent. The three pre-existing files were found in the resolved sibling article pipeline cache. These files were only read; no downloads or replacement occurred. Parsed chain counts below are descriptive and were not used to choose a target.

| Cached PDB | Parsed implicit-model chains: residues / atoms | Cache completeness |
|---|---|---|
| 2J3R | A: 159 / 1220; B: 157 / 1212 | Terminal END present |
| 4M0W | A: 228 / 1784 | Incomplete: final ANISOU line has 46 characters; no END/MASTER or final newline |
| 6MAV | A: 66 / 533 | Incomplete: final ANISOU line has 47 characters; no END/MASTER or final newline |

The incomplete 4M0W and 6MAV caches contain only part of chain A despite headers declaring chains A and B. Their parsed counts are not complete-structure counts. The 2J3R cache has a terminal END record, but neither parsed chain can establish the missing original residue mapping. A fresh download alone would not authorize a repair because the original target-chain declaration and identity mapping are absent.

## Provenance preserved

- `Train335.pkl` resolves to `C:\Users\Mohamed KONE\Desktop\article\data\geo\Train335.pkl`. SHA-256: `3b61ba47c66796ac9dfe326561fc917e78c9ec2ca6ef8c3f4ca5df9eb71e8a0a`. It matches the prior input audit: **True**.
- `Test287.pkl` resolves to `C:\Users\Mohamed KONE\Desktop\article\data\geo\Test287.pkl`. SHA-256: `6dec36a7e198a95fefe11b02999038919376c40cc01364acedd7d9236b7852de`. It matches the prior input audit: **True**.
- Cached `C:\Users\Mohamed KONE\Desktop\article\pdb_cache\2J3R.pdb`: SHA-256 `e94e01179f2d665b4e81cd39bb78a5992a81d8883d3535b1466b5e423a603512` (239598 bytes).
- Cached `C:\Users\Mohamed KONE\Desktop\article\pdb_cache\4M0W.pdb`: SHA-256 `211e69cfd43866fa7bd3d7b99ba62d2c5c2d92da70bffcd976b8e5caaf7731c7` (335872 bytes).
- Cached `C:\Users\Mohamed KONE\Desktop\article\pdb_cache\6MAV.pdb`: SHA-256 `9e65b31d1789045b5e69c232f0f7a4ea685fdf4c17d1993fcae7b5032b90886b` (155648 bytes).

The JSON companion records original array shape/dtype/content hashes, label counts, atom-to-residue index evidence, the exact missing fields, cache metadata, parser source hash, and source-file hashes. Original size and modification time remained unchanged during the reads.

## Safe next step

Use a versioned, explicitly filtered sensitivity cohort removing only Train335[106] (2J3R), Test287[55] (4M0W), and Test287[161] (6MAV): 334 training records and 285 Test287 records. Preserve an original-index map and copy remaining sample dictionaries without regenerating labels or features. This cohort is not a repaired full benchmark. Reconsider exact repair only with upstream target-chain declarations and residue/atom identity mappings that verify the original graph-feature and label order.

No original dataset, cached PDB, source script, label, or feature was modified.
