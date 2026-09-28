# Filtered-cohort protocol review

This is a read-only review of the proposed builder and audit gates. It does not certify that the commands below have already passed. The cohort is a filtered sensitivity analysis, not an exactly repaired full benchmark.

## Cohort identity

Use `data/geo_filtered_v1` for both arms. Retain the legacy filenames only for loader compatibility; report actual counts prominently.

| File | Original rows | Exclusions by original zero-based index and code | Filtered rows | Expected distinct complex codes |
|---|---:|---|---:|---:|
| Train335.pkl | 335 | 106: 2J3R | 334 | 209 |
| Test287.pkl | 287 | 55: 4M0W; 161: 6MAV | 285 | 215 |

The original training file has two `2J3R` records. Removing all records with that code would incorrectly remove a second, valid record. Exclude only the specified original index after asserting its code. Preserve relative order of every retained sample.

## Exact audit and gate commands

Run from `C:\Users\Mohamed KONE\Desktop\PC-BIND-OT`, after the builder completes and removes `INCOMPLETE.txt`:

```powershell
$ErrorActionPreference = 'Stop'
$FilteredDataDir = 'data/geo_filtered_v1'
if (Test-Path (Join-Path $FilteredDataDir 'INCOMPLETE.txt')) {
    throw 'Filtered cohort construction is incomplete.'
}

& .venv\Scripts\python.exe check_pcbind_prereqs.py `
    --data-dir $FilteredDataDir `
    --require-plm --require-partner --require-pair --require-partner-encoder `
    --min-partner-encoder-coverage 1.0 `
    Train335.pkl Test287.pkl
if ($LASTEXITCODE -ne 0) { throw 'Strict prerequisite gate failed.' }

& .venv\Scripts\python.exe audit_input_integrity.py `
    --data-dir $FilteredDataDir `
    --output-dir research_review/data_repair_v1/strict_audit `
    Train335.pkl Test287.pkl
if ($LASTEXITCODE -ne 0) { throw 'Input audit execution failed.' }
```

The coverage argument must be **1.0**. The default 0.98 permits incomplete partner inputs. The audit program is a report generator: its exit code alone does not reject coverage or overlap findings. Follow it with explicit result assertions:

```powershell
$Audit = Get-Content research_review/data_repair_v1/strict_audit/input_audit.json -Raw | ConvertFrom-Json
$Manifest = Get-Content data/geo_filtered_v1/cohort_manifest.json -Raw | ConvertFrom-Json
$Prior = Get-Content research_review/input_audit.json -Raw | ConvertFrom-Json
$ExpectedCounts = @{ 'Train335.pkl' = 334; 'Test287.pkl' = 285 }
$ExpectedGroups = @{ 'Train335.pkl' = 209; 'Test287.pkl' = 215 }

if ($Manifest.analysis_label -ne 'filtered-cohort analysis' -or
    $Manifest.status -ne 'content_verified' -or
    -not $Manifest.all_original_pickle_hashes_unchanged) {
    throw 'Cohort provenance or completion gate failed.'
}

foreach ($DatasetName in $ExpectedCounts.Keys) {
    $DatasetAudit = $Audit.datasets.PSObject.Properties[$DatasetName].Value
    $DatasetManifest = $Manifest.datasets.PSObject.Properties[$DatasetName].Value
    $Original = $Manifest.original_pickles.PSObject.Properties[$DatasetName].Value
    $PriorDataset = $Prior.datasets.PSObject.Properties[$DatasetName].Value
    if (-not $DatasetAudit -or -not $DatasetManifest -or -not $Original -or -not $PriorDataset) {
        throw "Missing provenance entry: $DatasetName"
    }
    foreach ($Field in @('samples', 'target_sequences_present',
                        'valid_target_main_plm', 'valid_target_aux_plm',
                        'valid_partner_encoder', 'nonempty_partner_samples',
                        'valid_pair_label_samples', 'samples_with_partner_chain_sequences')) {
        if ($DatasetAudit.PSObject.Properties[$Field].Value -ne $ExpectedCounts[$DatasetName]) {
            throw "Incomplete filtered coverage: $DatasetName $Field"
        }
    }
    if ($DatasetAudit.unique_complex_codes -ne $ExpectedGroups[$DatasetName] -or
        @($DatasetAudit.failed_or_fallback_samples).Count -ne 0 -or
        $DatasetAudit.sha256 -ne $DatasetManifest.sha256 -or
        $Original.sha256 -ne $PriorDataset.sha256 -or
        -not $DatasetManifest.retained_sample_content_unchanged -or
        $DatasetAudit.mapping_verified_true -ne 0) {
        throw "Filtered identity/content/mapping gate failed: $DatasetName"
    }
}

$Overlap = $Audit.train_test_exact_overlap.'Test287.pkl'
if (@($Overlap.shared_exact_complex_codes).Count -ne 0 -or
    @($Overlap.shared_normalized_pdb_ids).Count -ne 0 -or
    $Overlap.test_samples_with_exact_train_sequence -ne 0) {
    throw 'Exact target/PDB train-test overlap detected.'
}
foreach ($Comparison in $Audit.cross_side_exact_overlap.'Test287.pkl'.PSObject.Properties) {
    if ($Comparison.Value.affected_test_samples -ne 0) {
        throw "Exact cross-side overlap detected: $($Comparison.Name)"
    }
}
Write-Host 'Filtered-cohort coverage and exact-identity gates passed; biological mapping and homology remain unverified.'
```

These commands write audit reports only. They do not repair, download, augment, or train anything. Preserve their complete output and exit codes. The explicit original-hash comparison ties the new cohort to the previously audited originals, rather than merely to files that happen to have matching sample counts and exclusion codes.

## Builder review and assertions for identical arms

The reviewed `build_filtered_cohort.py` resolves paths before checking source/output separation, refuses existing destinations, writes derived pickles and reports exclusively, keeps an incomplete marker until successful completion, and compares every original pickle hash before and after derived writes. It filters by original index plus code, hashes all retained sample fields with dtype and array shape, reloads derived pickles, and verifies sample-content hashes in order. It preserves absent mapping-verification fields and records the number explicitly verified without promoting any mapping to verified status.

Both arms must consume the same derived file hashes, same retained sample index map, and same external row order. Preserve all sample dictionary fields, especially:

- Original labels, residue/atom graph nodes and edges, `a2r_map`, coordinates, frames, surface/sequence descriptors, and both target PLM arrays and model identifiers.
- Partner chains, chain slices, chain-separated sequences, partner graph/coordinates/frames, both partner PLM arrays and model identifiers, sparse pair-contact labels, and auxiliary contact labels/masks.
- Complex codes, chain metadata, original augmentation/mapping status, and the external manifest mapping filtered indices back to original indices. Do not insert invented residue identities or change unverified mapping metadata.

Before training, validate source and derivative hashes against the manifest and verify there is no incomplete marker. For every intended seed, require the training runtime's actual train/validation index lists and group IDs to equal the saved `grouped_folds_seed<seed>.json` lists exactly, with the same filtered training hash and `complex_code` grouping. Validate coverage of all 334 samples, nonempty folds, and disjoint complex groups. Reusing the same seed without checking the actual manifest is weaker than this identity check. The old 335-sample folds cannot be reused unchanged after filtering.

Require finite, binary labels of shape `(n_residues,)`, finite target node/geometry/descriptor matrices with consistent row counts, valid integer target/atom graph edge endpoints, and integer `a2r_map` of length `n_atoms` with values in `[0, n_residues)`. Existing prerequisite and input-audit functions do not check all of these target-side structural invariants. The builder presently checks sequence/label lengths, both target PLMs, partner-encoder structure, and rejects zero-filled target/partner embedding rows; the separate strict prerequisite gate checks pair-index validity. Retain explicit model identifiers: the prerequisite helper accepts a missing model identifier by defaulting it to the expected name, so that helper alone does not prove embedding provenance.

Only Test287 is the external dataset in this protocol. Test60, TestB25, Test70 and TestUB25 must not silently re-enter through default test-set lists or fallback data paths. Keep identical learning settings, model components other than the declared treatment, checkpoint-selection rules, and threshold-fitting procedure in the two arms. Same hashes and grouped folds are necessary but do not by themselves establish an architecture-matched ablation.

## What remains unproven

- No original record has `pdb_mapping_verified=True` in the source audit. Filtering three failed records does not verify the remaining target residue/atom ordering or feature-to-label correspondence. Dimensions, finite arrays, and content preservation are not biological alignment evidence.
- The source Test287 cohort has zero exact target, PDB, and cross-side sequence matches to Train335. Recheck this on the derivative, but do not call it proof of homology independence. Local alignments, homologous chains across both input sides, and structural similarity remain unaudited by `audit_input_integrity.py`.
- Grouping by complex code prevents records of one declared complex crossing folds. It does not establish sequence-cluster independence. `audit_pcbind_pairing.py` reports leakage for its legacy sample-level splitting algorithm; it does not validate the new saved grouped folds.
- Shape/model-name checks do not prove that PLMs were extracted from the declared sequence and chain order. The filtered builder preserves existing embedding content rather than re-establishing upstream extraction provenance.
- Nonempty, bounded sparse contact-index arrays do not prove the correctness or completeness of biological contacts or the provenance of auxiliary labels. The default primary training does not supervise pair contacts, but the auxiliary contact objective still depends on the retained labels.
- One seed with five folds supplies five overlapping training sets, not five independent seed replicates. Epoch/checkpoint selection and threshold fitting reuse development-fold labels; selected OOF performance is not an unbiased external estimate, and transferring that threshold to an ensemble remains a methodological assumption.

There is no identified blocker to constructing and auditing the explicitly labelled filtered sensitivity cohort. The missing chain/ordering evidence blocks a claim of an exactly repaired or biologically alignment-verified benchmark. No result from this filtered cohort should be presented as full Train335/Test287 benchmark reproduction.
