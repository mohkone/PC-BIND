# Filtered-cohort preparation result

Exact repair is not supported by the available metadata. The three failed records
have no declared target chain or original residue/atom identities with which to
align their labels. The existing 4M0W and 6MAV coordinate caches are truncated.
The [forensic report](forensic_report.md) records the evidence and cache hashes.

The authorized fallback was completed in **`data/geo_filtered_v1`**. This is a
**filtered-cohort analysis**, not a repaired full Train335/Test287 benchmark.
Legacy pickle filenames are retained for loader compatibility.

| File | Excluded original indices (zero-based) | Original valid PLM/partner coverage | Filtered valid main PLM / auxiliary PLM / partner encoder |
|---|---|---|---|
| Train335.pkl | 106: 2J3R | 334/335 | 334/334 for all three |
| Test287.pkl | 55: 4M0W; 161: 6MAV | 285/287 | 285/285 for all three |

The other valid 2J3R training record is retained. Both arms use the same filtered
files, sample order and labels. Each retained sample's full content, including
array dtype, shape and bytes, was hashed before serialization and checked again
after loading the new pickle. No labels, coordinates, PLMs or partner fields were
regenerated or zero-filled. The builder additionally rejects zero-filled rows in
both target and both partner PLMs.

SHA-256 records cover all six original pickles, including secondary test sets,
before and after construction; every original hash is unchanged. The source
junction still resolves outside this project, and no input was written there.
See [original hashes](original_pickle_hashes.json), the
[cohort manifest](cohort_manifest.json), and [builder](../../build_filtered_cohort.py).
The manifest records every excluded original sample, its failure evidence, all
retained original-to-filtered index mappings and the new file hashes.

## Evaluation population and folds

The shared [seed-2101 grouped split manifest](grouped_folds_seed2101.json)
was recomputed after exclusion. Its five validation folds partition all 334
training samples; all records of each of the 209 retained complex codes stay
in one fold. Both arms must use this manifest and the recorded training hash.
One seed is prepared; no independent seed replicates or accuracy results are
claimed.

Only the filtered Test287 population (285 samples, 215 complex codes) is the
external evaluation set. The new audit finds zero shared PDB identifiers,
exact target sequences, or exact cross-side whole-chain sequences with the
filtered training set. This does not establish homology independence.
Test60, TestB25 and Test70 remain overlap-contaminated relative to Train335;
TestUB25 also has known partner coverage and cross-side overlap limitations.
None of these four sets is included in this primary cohort directory.

## Verified prerequisites and remaining limits

The [strict audit](strict_audit/input_audit.md) reports no feature-coverage
failures in the filtered files. The [recorded gate commands and output](prerequisite_validation.json)
passed with both target PLMs required and partner-encoder coverage set to **1.0**.
Sparse pair-label indices are present and valid for all 334 and 285 samples.
The original full-cohort failure remains documented in the original input audit
and the new manifest; filtering does not turn 334/334 into 335/335.

**Verified biological PDB mappings remain 0/334 and 0/285.** Preserving arrays
and checking their shapes does not reconstruct original residue-to-label
alignment or establish PLM extraction provenance. This cohort supports an
explicitly labelled sensitivity analysis with those limitations. Full repair
requires upstream declared-chain and residue/atom mapping evidence.

The [retained-schema check](retained_schema_validation.json) also passes binary
label, target feature/geometry dimension, graph edge and atom-to-residue index
checks. Optional `atom_coords` remain absent from 97/334 training and 132/285
test records; all present arrays pass shape/finiteness checks. Filtering preserves
this pre-existing limitation rather than claiming complete atom-coordinate data.

The [protocol review](protocol_review.md) contains repeatable audit commands and
explicit JSON assertions, since the input audit's exit code alone does not gate
its findings. [Final validation](final_validation.json) records the local
verification: **68 regression tests**, launcher checks and all three OT/v5/v9
smoke checks passed in clean processes. The v5/v9 checks explicitly used the
filtered cohort. Current training code reproduced all saved fold indices and
group IDs exactly. No full model experiment was trained.

## Select the cohort explicitly

`run_seed.ps1 -DataDir` now forwards an absolute `PPI_DATA_DIR`. An explicitly
selected directory must contain every requested file: no fallback to original
data or silent skipping is allowed. Future run provenance and its embedded
checkpoint/final-summary copies record the selected cohort manifest hash and
the `filtered-cohort analysis` label.

The legacy OT/no-OT wrapper scripts do not accept `-DataDir`. Use the common
direct-run configuration below for this cohort. The
[saved dry runs](matched_dry_run.json) confirm that only transport activation and
output directory differ between the two arms. This is a transport-branch
comparison, not a parameter-count-matched capacity ablation.

```powershell
$common = @{
    DataDir = '.\data\geo_filtered_v1'; TestSets = 'Test287.pkl'; Seed = 2101
    GroupedCV = 1; CVGroupKey = 'complex_code'; MaxFolds = 5; BatchSize = 1
    TwoHead = 1; RankLossWeight = '0.55'; RankFusion = '0.35'; RankWarmupEpochs = 3
    ModelDropout = '0.25'; EdgeDropout = '0.06'; WeightDecay = '3e-4'
    PartnerConditioning = 1; PartnerTopK = 16; PartnerDirectFusion = 0
    PartnerLogitMode = 'standard'; PartnerResidueEncoder = 1; PartnerEncoderLayers = 1
    PartnerTargetFusion = '0.25'; PartnerContrast = 0; PairContactLoss = 0
    PairMarginalConsistency = 0; PairMarginalContrast = 0; PairContactContrast = 0
    UsePlm = 1; UseAuxPlm = 1; PartnerContactAux = 1; PatchLabels = 0
    SelectionMetric = 'mcc'; SelectionAuprWeight = '0.35'
}
& .\run_seed.ps1 @common -PartnerTransport 1 -OutputDir outputs_filtered_v1_ot_seed2101 -DryRun
& .\run_seed.ps1 @common -PartnerTransport 0 -OutputDir outputs_filtered_v1_no_ot_seed2101 -DryRun
```

These commands inspect settings without training. Before a future training run,
repeat the strict prerequisite and identity gates, compare the runtime fold
indices with the saved manifest, and remove `-DryRun` only for the intended run.
Keep threshold fitting, checkpoint selection and every other setting identical.
Use the same external population in both arms and report the filtered counts.
