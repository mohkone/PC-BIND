# Transport projection-policy development ablation

Recorded on 7 October 2026 (client timezone Asia/Shanghai). Status:
**`prospective_protocol_recorded; execution_not_launched`**. This is an internal
prospective protocol, not an independently timestamped public registration.
[protocol.json](protocol.json) fixes the inputs, six fresh runs, acceptance gates
and analysis. No execution implementation or model training is authorized by
the existence of this document alone.

## Question and relation to the observed results

Within the same learned partner-transport architecture, what is the development
performance effect of the existing five-iteration projection/postprocessing
policy compared with the existing zero-iteration row-softmax policy?
Both arms retain the complete transport branch and all its trainable parameters.
The intended configuration difference is **`transport_sinkhorn_iters = 5`
versus `0`**. The arm names are `projected` and `rowsoftmax`; the reported
difference is always projected minus rowsoftmax.

The [completed three-seed OT-versus-no-OT result](../multiseed_dev_v1/complete_oof_v3/RESULT.md)
motivates this next question. Its primary PR-AUC difference averaged -0.030940
over seeds 2102, 2103 and 2104, and the mean MCC difference was -0.021680.
The [historical seed-2101 comparison](../frozen_5fold_dev_v1/complete_oof/RESULT.md)
is another observed reference. Those results have already been seen, as have
the original recipe and the chosen training seeds. This new contrast is
prospective relative to its own six new fits; it is not an independent
replication on unseen data, seeds or a newly selected recipe.

The ablation directly estimates a projection/postprocessing policy difference
**within transport**, rather than an uncapped-transport-versus-no-OT difference.
It cannot by itself establish whether uncapped transport outperforms no-OT.
Any contextual comparison with the observed old no-OT scores must be separately
labeled descriptive and excluded from this protocol's primary estimates and
bootstrap intervals. Reusing an old projected fit would fail the requirement
for six fresh paired fits.

## Precisely what changes

The accepted source is `CROSS5FOLD_multiseed_dev_v1.py`, SHA-256
`93de0890f172dc34a727e016c7e989fb344c9dc781826ce493f8068294c645f4`.
Its `sparse_sinkhorn` implementation at lines 1397-1436 already supports zero
iterations. With zero iterations it applies row softmax to the affinity scores
and learned dustbin logits. With five iterations it alternates row normalization
and upper column-cap projection, then assigns the remaining row deficit to the
dustbin without renormalizing the real partner columns. The upper capacity is
`max(1, n_target / max(n_partner, 1))`; it is not a required column marginal.

Therefore the contrast changes the complete existing projection/postprocessing
policy, including how real-pair and dustbin mass are formed and how gradients
flow through that policy. It does not isolate column capping alone, an individual
projection iteration, or a universal OT mechanism. Equal architecture and
parameter counts address the earlier parameter-count confound; they do not make
every downstream computation or gradient identical.

Keep `partner_transport = true` in both arms. Keep the dustbin, temperature
0.1, candidate top-k 32, entropy weight 0.01, sparsity weight 0.005 and logit
fusion 0.3 unchanged. Keep both PLMs, partner conditioning, the shared partner
encoder, all other features, auxiliary losses, fusion settings, optimizer,
learning-rate behavior, batch size, regularization and selection policy unchanged.
The structured record binds the complete recipe and its source archives; these
examples do not replace that full binding.

For each same-seed, same-fold pair, require exact equality of parameter names,
shapes, dtypes and trainability, total and trainable parameter counts, and
optimizer parameter-group membership and settings. Record canonical
initialization and RNG-state fingerprints before optimization for all five
folds in both arms. Require first-fold initial parameter equality within each
seed pair. Equality of the numeric seed or final parameter counts alone is
insufficient. The nontrainable iteration count may differ; no parameters may
be added, removed or frozen in either arm.

Preserve the existing seed-setting procedure: the accepted trainer seeds once
at run start, then processes folds without resetting that seed. Different
training trajectories and early-stopping durations can consume different RNG
histories, so later-fold initial weights need not match. Record any divergence
as part of the executed policy contrast rather than hiding it. Do not reseed
folds, force later-fold initialization equality or synchronize random draws;
those would change the accepted stochastic training procedure and require a
separate prospectively authorized amendment.

## Fixed population, folds and training policy

Use the exact filtered sensitivity cohort `geo_filtered_v1`: 334 training
samples, 209 complexes and 66,208 validation-residue observations over the
five fixed folds. Preserve the original `Train335.pkl`, `Test287.pkl`, cohort
manifest and `grouped_folds_seed2101.json` bytes and hashes. Each complete
complex stays together. Consume the stored train/validation indices and their
order verbatim, including validation sample counts 68, 67, 66, 66 and 67.
The fold seed remains **2101** for all six arms. Do not generate another split.

Training seeds are exactly **2102, 2103 and 2104**. They control the existing
initialization, shuffling and stochastic training operations; they do not control
fold assignment. All runs use the previously accepted CPU runtime, dependency
versions and 12 PyTorch threads. Runtime drift fails preflight.

Run all five folds for each arm, with the frozen maximum of 30 epochs,
early-stopping patience 8, MCC checkpoint selection and top-three checkpoint
average candidate. Accept that average only under the unchanged rule that its
selection score is at least the best single checkpoint's score. Each fitted
arm/seed retains its own saved pooled-OOF development-selected threshold.
Selected checkpoints, learned values and thresholds are outcomes, not tunable
inputs. Do not alter these procedures to improve a result.

`Test287.pkl` remains subject to its prerequisite hash/feature gate, with
**external evaluation skipped in every arm**. It receives no predictions or
metrics and is not consulted for model selection. The original and completed
experiment outputs remain untouched.

## Serial execution and failure policy

Create six fresh processes, serially, in this exact order:

1. `projected`, training seed 2102, five projection iterations.
2. `rowsoftmax`, training seed 2102, zero projection iterations.
3. `projected`, training seed 2103, five projection iterations.
4. `rowsoftmax`, training seed 2103, zero projection iterations.
5. `projected`, training seed 2104, five projection iterations.
6. `rowsoftmax`, training seed 2104, zero projection iterations.

Output directories are declared in `protocol.json`. Reject existing training
output directories, pre-existing aggregate destinations and concurrent workers.
Use an exclusive lock, clear inherited `PPI_*` settings, and verify the new
execution freeze, accepted ancestry, source, data, cohort, fixed folds and
runtime before every arm. Do not reuse previous checkpoints or predictions,
launch an isolated later arm, rerun a failed arm over its output, substitute a
seed, add seeds after inspection or tune between arms.

Any nonzero training return or failed post-run validation stops the queue
immediately. Preserve logs, process return codes, queue state and partial
artifacts. Do not launch later arms or a partial aggregate. A necessary
execution change requires a separately versioned amendment before a fresh
authorized launch, preserving this protocol and all failed-run evidence.

## Implementation and acceptance gates before any launch

The accepted multi-seed v2 verifier enforces its OT/no-OT recipe and cannot
serve as an execution authorization for the zero-iteration arm. Preserve all
accepted v2 and v3 source files, archives, manifests and acceptance evidence.
Implement a **new versioned execution path**, without editing those freezes or
the model's mathematical implementation. Only the iteration setting and the
new arm/provenance routing are permitted recipe differences for this ablation.

Before training, freeze the new runner, seed/arm routing, parameter/initialization
validator, artifact validator, analysis source and tests in an execution manifest.
Bind this protocol, the accepted v2/v3 ancestry, unchanged recipe/source hashes,
datasets, cohort, fixed folds, runtime, exact run order and destinations. A
separate acceptance record must bind all evidence by SHA-256 and prove:

- The complete relevant regression suite passes, with any pre-existing skip
  explained; no new failed or silently omitted required check is accepted.
- All six declared real-data dry runs pass strict feature and hash gates,
  skip external evaluation and preserve every fixed fold identity/order.
- The two arms differ only in iteration count and permitted run/outcome fields;
  training seed and fold seed route correctly and all model settings remain fixed.
- Exact parameter/schema/trainability/count and optimizer-group equality holds
  for each seed and all five folds; first-fold pre-optimization tensor equality
  holds within each pair, and all fold initialization/RNG fingerprints are saved.
- The worker refuses output collisions/concurrent workers and halts before any
  later arm or aggregate after forced process and validation failures.
- Validators reject wrong seeds or folds, runtime/source/data mismatches,
  malformed/partial/nonfinite OOF arrays, corrupted identities/checkpoints,
  unexpected external predictions/metrics, unequal parameter schemas/counts or
  optimizer settings, and unequal first-fold parameter initialization.
- Analysis tests verify the declared arm order, six-arm identity pairing,
  fixed thresholds, complex ordering, RNG, 5,000 attempted draws, exclusion
  policy and aggregate definitions.
- All six training output directories and the aggregate destination remain absent.

Dry runs and failure simulations must not fit research models or populate the
declared training outputs. No implementation, acceptance or training launch is
claimed by this prospective record. Launch requires a later explicit instruction
after the new concrete acceptance evidence exists.

For every fitted arm, require complete finite OOF arrays, exact labels and
sample/residue/complex/fold identities, all five finite selected checkpoints,
correct training/fold seeds, fixed source/data/cohort/fold/runtime hashes, strict
feature coverage, the declared transport/projection state, reproducible saved
metrics, required initialization/RNG evidence and no external artifacts. Require
all six complete validated arms before analysis.

## Analysis fixed before these fits

For each seed, compute one complete pooled-OOF score per arm over its five
validation folds. Trapezoidal PR-AUC is `auc(recall, precision)` from
`precision_recall_curve`, **not average precision**. MCC uses
`matthews_corrcoef(labels, probabilities >= saved_threshold)`. AUROC uses
`roc_auc_score(labels, probabilities)` and is descriptive.

Form projected-minus-rowsoftmax differences separately for seeds 2102, 2103
and 2104. The primary estimate is the arithmetic mean of these three PR-AUC
differences. The secondary estimate is the arithmetic mean MCC difference at
each arm/seed's own saved threshold. Report both arm scores and thresholds for
each seed, all individual differences, the mean, sample standard deviation
(`ddof=1`), range and positive/zero/negative counts. Report AUROC differences
with the same descriptive seed summaries, without a bootstrap interval or
seed-level hypothesis test. Do not treat folds as independent replicates,
pool seed predictions as extra independent residues, select a favorable seed
or add an adaptive analysis.

Use exactly **5,000 attempted paired whole-complex bootstrap draws**, with
NumPy `Generator(PCG64(2101))`. Normalize complex IDs by stripping whitespace
and uppercasing, then sort their unique values. On each attempt draw
`rng.integers(209, size=209)` and include every sample/residue row and each
drawn complex's multiplicity. Apply that **same draw to both arms and all three
seeds**, compute each seed's PR-AUC and MCC difference on the draw, then average
the three differences. Hold all six saved thresholds fixed throughout resampling.

Exclude and count only draws containing a single label class. Do not replenish
them: 5,000 is the number of attempts. Missing/corrupt identities, labels or
nonfinite probabilities are validation failures, never excluded draws. Report
attempted, valid and single-class-excluded counts. Refuse an interval if no draw
is valid. Use 95% percentile endpoints from
`numpy.quantile(values, [0.025, 0.975], method="linear")` for the mean PR-AUC and
MCC differences only. Do not rerun the bootstrap or overwrite a finished result.

The final JSON must contain the protocol, execution/acceptance freeze, source,
data, cohort, fold, runtime and complete artifact hashes, plus initialization
matching evidence. Historical seed 2101 and the observed three-seed OT/no-OT
results are separately labeled context; none of their predictions, checkpoints,
metrics or thresholds enter the prospective estimates or intervals.

## Interpretation limits

This is a development-only, filtered sensitivity analysis. Development labels
select checkpoints and thresholds, so OOF performance remains selection-optimistic.
The bootstrap conditions on six fitted predictions and saved thresholds; its
intervals do not measure training-seed uncertainty. Three selected seeds provide
limited descriptive stability information and were already used in the observed
OT/no-OT study. Fold training populations overlap, and neither this pairing nor
complex resampling removes fitting dependence. Fixed folds do not probe split
uncertainty. Biological residue/atom mappings remain unverified and sequence
homology independence remains unestablished. The recipe, cohort and follow-up
question were chosen with the prior negative findings known. `Test287` remains
unevaluated.

Parameter matching and first-fold initialization matching make this a narrower
architectural ablation. Later-fold initialization and stochastic histories may
diverge because earlier folds train or stop differently under once-per-run
seeding. This protocol compares fitted training policies, rather than holding
weights and random draws fixed for a pure causal intervention. It does not
establish a causal biological mechanism or isolate the cap from the
dustbin/postprocessing and gradient effects. Report the direction and size of
the projection-policy difference without a universal benefit/harm claim, an
external-generalization claim, a p-value, or a claim that row-softmax transport
beats no-OT on independent evidence.
