# Shared training-seed development repeat

Registered on 2 October 2026 (client timezone Asia/Shanghai), after observing
the completed seed-2101 result. The user selected a shared multi-seed repeat.
This prospective record specifies **training seeds 2102, 2103 and 2104**, each
for OT and no-OT, with all five existing grouped folds held fixed. No new
training has been launched. [protocol.json](protocol.json) is the structured
record; this document explains its analysis and execution requirements.

## Question and reference

Does the direction and size of the OT-minus-no-OT development difference
persist across these three additional training seeds under the existing recipe?
The [completed seed-2101 comparison](../frozen_5fold_dev_v1/complete_oof/RESULT.md)
is a previously observed reference, excluded from the primary new-seed mean
and its intervals. Its PR-AUC difference was -0.0284 and MCC difference was
-0.0277. This registration does not make that completed result prospective.

Training randomness is one source of benchmark variation. Holding data, folds
and the recipe fixed while repeating training probes that source, following
the motivation in [Accounting for Variance in Machine Learning Benchmarks
(MLSys 2021)](https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html).
The choice of three additional seeds is a limited stability check, not a
power calculation or a comprehensive variance study.

## Fixed inputs and recipe

The filtered-cohort analysis retains 334 training samples, 209 complexes and
66,208 OOF residues. Use the exact `Train335.pkl`, cohort manifest and
`grouped_folds_seed2101.json` identified in the structured record. Consume
the stored fold membership and order verbatim: **training seed and fold seed
are separate**. Do not regenerate folds with seeds 2102, 2103 or 2104.
`Test287.pkl` retains its prerequisite hash/feature gate but receives no
prediction or metric evaluation in any arm.

The completed experiment's [freeze manifest](../frozen_5fold_dev_v1/freeze_manifest.json)
has SHA-256 `7f8f5f642cff676c8ef3438554dbf8b5ee8b6f5529b0e8deff58248277909265`.
Its [exact source archive](../frozen_5fold_dev_v1/recipe_source_archive.json)
preserves the recipe bytes, including original line endings. The structured
protocol binds these records, the completed comparison, both dataset hashes,
the cohort/fold hashes and the CPU runtime. Keep all model, feature, loss,
optimization and selection settings from this recipe. In particular:

- Both PLMs, partner conditioning and the shared partner encoder remain enabled.
- The OT arm uses `capacity_capped_dustbin_v2`; the control disables transport.
- All five folds run per arm, with batch size 1, maximum 30 epochs and
  early-stopping patience 8.
- MCC selects checkpoints; the existing top-three checkpoint-average candidate
  is accepted when its selection score is at least that of the best checkpoint.
- Each arm/seed selects its own threshold using the unchanged development
  procedure. Selected checkpoints and thresholds are outcomes, not new settings.
- The same recorded CPU runtime, dependency versions and thread settings apply
  to every arm. Any discrepancy fails preflight.

Only training randomness, run identities and output paths change relative to
the reference. Within each seed pair, transport enablement is the intended
model difference. Training seeds govern initialization, shuffle and stochastic
training operations through the existing seed-setting procedure.
Using the same numeric seed pairs the run conditions; unequal architectures
may consume random numbers differently, so their random histories are not
assumed to be synchronized.

## Execution and validation

Run six fresh processes serially in this order: OT 2102, no-OT 2102, OT 2103,
no-OT 2103, OT 2104, no-OT 2104. Output paths are declared in `protocol.json`.
Reject nonempty output directories and inherited configuration overrides.
Do not reuse pilot checkpoints, tune between runs, add seeds after inspecting
outcomes, or silently substitute a failed seed. Stop on failure and preserve
all evidence. A necessary change requires a new versioned protocol before
execution; preserve this record rather than rewriting it.

Before launching, implement a dedicated versioned execution path with explicit
training-seed and fold-seed fields. The current filtered-cohort wrappers and
verifier are locked to seed 2101 and cannot execute this plan unchanged.
Preserve those frozen sources. Review and freeze the new seed-routing code,
prove in dry runs and regression checks that only training seed varies and
that every arm consumes the original fold indices, and bind its exact source
hashes to new run provenance. This registration does not assert that this
execution path has already been implemented or validated.

For every completed arm, require all five folds, strict feature/hash gates,
matching runtime and recipe, skipped external evaluation, complete OOF
sample/residue/complex/fold identities, finite selected checkpoints and
reproducible saved metrics. Record the training seed, fixed fold seed,
protocol/source hashes, dataset/manifest hashes and actual split membership.
Require all three complete, validated seed pairs before computing an aggregate.
Report failures explicitly; a partial comparison cannot replace the planned
primary analysis. No result from this new experiment currently exists.

## Analysis fixed before the new runs

For each new seed, compute pooled OOF metrics over its five validation folds
and OT-minus-no-OT differences. **The primary estimate is the arithmetic mean
of the three seed-pair differences in trapezoidal PR-AUC.** PR-AUC here is
trapezoidal precision-recall area, not average precision. Secondary analysis
uses MCC at each arm/seed's saved pooled OOF threshold. AUROC is descriptive.
Use the reference metric definitions: `auc(recall, precision)` from the
precision-recall curve, `matthews_corrcoef` with prediction `probability >=
threshold`, and `roc_auc_score` for AUROC.
Do not concatenate repeated seed predictions as extra independent residues
or treat folds as independent experimental replicates.

Report each seed's two arm scores, thresholds and paired differences, followed
by the difference mean, sample standard deviation (`ddof=1`), range and counts
of positive, zero and negative differences. Report the observed seed-2101
reference separately. Do not select a favorable seed or perform a seed-level
hypothesis test from three new repetitions.

For conditional uncertainty, use 5,000 paired whole-complex bootstrap draws
with bootstrap seed 2101. In each draw, sample the 209 `complex_code` units
with replacement, keeping all samples and residues of each selected complex,
including its multiplicity. Apply the **same draw to both arms and all three
training seeds**, compute each seed's metric difference on that draw, then
average the three differences. Hold all six saved thresholds fixed. Report
95% percentile intervals for the mean PR-AUC and MCC differences, plus valid
and single-class-excluded draw counts. Missing identities or nonfinite values
are validation failures, not bootstrap exclusions. Do not rerun resampling
to obtain a favorable interval or report a bootstrap tail fraction as a
hypothesis-test p-value.

The 5,000 draws are attempts, without replenishing excluded single-class draws.
Use NumPy `Generator(PCG64(2101))`, sorted unique complex IDs after stripping
whitespace and uppercasing, and `integers(209, size=209)` for each draw. Compute
the endpoints with `quantile([0.025, 0.975], method="linear")` over valid draws.
Refuse an interval when no draw is valid. Record this RNG and metric behavior
in the new analysis source freeze before launching.

These intervals condition on all fitted predictions and selected thresholds.
They **do not include training-seed uncertainty**. Between-seed summaries
describe the three new observed repetitions; they do not precisely estimate
the full distribution of training randomness.

## Interpretation boundaries

Checkpoint and threshold selection uses the development labels, so reported
OOF scores remain selection-optimistic. Fold training populations overlap;
neither complex resampling nor training-seed repetition removes that fitting
dependence. This is a filtered sensitivity cohort with unverified biological
residue mappings and unestablished sequence-homology independence. Removing
transport changes parameter count, so the control is not capacity matched.
Fixed folds leave sampling, mapping, homology, architecture and hyperparameter
variation unprobed. Test287 remains unevaluated. Neither favorable nor negative
new results alone establish external generalization, biological partner
specificity or universal benefit/harm of transport.
