# Projection ablation infrastructure accepted; training unstarted

Acceptance was completed on 7 October 2026 for the preregistered projected
(five iterations) versus row-softmax (zero iterations) development ablation.
Both arms retain transport. There are no fitted ablation results.

- [Execution freeze revision 2](execution_manifest_v2.json):
  `fd31de86c6f959ed674f6912d1c3fac021ade3fc7ce9b54803fbed08376eb682`.
- [Acceptance record](execution_manifest_v2.acceptance.json):
  `fd3dcd976a361fadbdddff2b50726e33f81adee5b904a3740b7280a3bc41b661`.
- [Read-only acceptance verification](ACCEPTANCE_VERIFICATION_v2.json).
- [Independent metadata and initialization checks](implementation_validation_v2/independent_metadata_validation.json).

The accepted status is **`infrastructure_accepted_training_unstarted`**.
`training_started` and `launch_authorized` are both false. All six declared
training directories, the aggregate destination and the production process
directory remain absent. No launch-authorization file exists. A read-only call
to the launch gate confirmed that missing explicit authorization blocks execution;
the worker and trainer were not launched for this check.

## Acceptance evidence

The [full regression suite](implementation_validation_v2/full_tests.json) ran
213 tests: 212 passed, zero failed, zero errored, and one pre-existing test was
skipped because this Windows host cannot create symlinks. The [full test log](implementation_validation_v2/full_tests.log)
and all 18 test-source hashes are preserved. All 55 new infrastructure tests ran.

All [six real-data dry runs](implementation_validation_v2/all_six_dry_runs/validation.json)
passed without model fitting, forward passes, optimization or declared training
outputs. Their strict gates validated both PLM fields, partner/pair features
and partner-encoder coverage for 334/334 training records and 285/285
prerequisite-only Test287 records. Test287 evaluation remained skipped.
The actual trainer's run and prospective summary metadata independently match
the artifact validator's expected configuration derived from the accepted recipe.

Both arms have exactly **5,700,243 parameters, all trainable**, and one identical
AdamW parameter group. Every dry-run fold matches in parameter names, shapes,
dtypes, trainability, counts, buffers and optimizer settings. Within each seed
pair, first-fold parameter/state tensor digests and Python/NumPy/Torch CPU RNG
fingerprints are exactly equal. The three training seeds produce three distinct
first-fold parameter fingerprints. All five initialization observations are
saved in each dry-run record.

These dry runs contain initialization only. Future fitted runs retain the
original once-per-run seeding, save initialization tensors and RNG fingerprints
before every fold's optimization, and require the same structural/optimizer
proof for all folds. Later-fold tensor/RNG divergence is recorded and allowed;
no per-fold reseeding or later-fold equality assumption was introduced.

The [process-failure simulation](implementation_validation_v2/forced_process_failure/queue_state.json)
exited 7 at the first arm. The [validation-failure simulation](implementation_validation_v2/forced_validation_failure/queue_state.json)
rejected that arm after a zero exit. Both kept the other five arms queued and
aggregation waiting. Their children were harmless `python -c` exit commands.
Regression tests additionally exercise a failure at every arm position,
malformed or partial OOF identities, nonfinite/corrupt checkpoints, wrong
seeds/folds/source/runtime, external metrics, initialization mismatches,
output/lock collisions and unauthorized standalone trainer processes.

## Immutable scope

The [source archive](execution_source_archive_v2.json) binds ten new
implementation/test files and their exact archived copies. All 79 original
model, data, loss and training definitions, protected recipe assignments and
the optimization/checkpoint-selection loop pass AST preservation checks.
Accepted historical v2/v3 sources and archives, the registered protocol and
preregistration, cohort/data/fold hashes, CPU runtime, recipe and completed
research records remain unchanged.

The [preacceptance infrastructure amendment](INFRASTRUCTURE_AMENDMENT_v2.md)
adds a direct acceptance hash to the prospective final report and a serial
queue ownership gate. The first unaccepted freeze, source copies and passing
test/dry-run/failure evidence remain preserved. It was never authorized or
used to fit a model. No completed report or real-data bootstrap was recomputed.

## Launch remains a separate decision

[The preregistration](PREREGISTRATION.md) requires a later explicit instruction
after concrete acceptance evidence exists. Acceptance does not supply that
authorization. If subsequently authorized, exactly one locked serial worker
may run projected 2102, row-softmax 2102, projected 2103, row-softmax 2103,
projected 2104, row-softmax 2104, then the registered aggregate after every
arm and pair passes validation. Fold seed stays 2101, all five folds are fixed,
and external evaluation stays skipped. No failed arm is silently restarted.

The contrast tests the complete existing projection/postprocessing policy,
including dustbin and gradient effects. It does not isolate column capping
alone or establish row-softmax superiority over no-OT. Development selection,
conditional bootstrap, overlapping fold training sets, limited/pre-observed
seeds, filtered cohort, unverified biological mappings/homology, and possible
later-fold stochastic divergence remain limitations. Test287 stays untouched
for predictive evaluation.
