# Fixed-fold seed-routing implementation and validation

Completed on 2 October 2026, within the implementation-and-validation-only
scope. **No research model training or serial training queue has started.**
All six declared training output directories remain absent. Test287 received
only the strict input prerequisite checks, with no predictions or metrics.

The immutable [preregistration](PREREGISTRATION.md) and [protocol](protocol.json)
remain unchanged. The accepted source freeze is
[execution_manifest_v2.json](execution_manifest_v2.json), SHA-256
`ca69112d633de00ee80d23a7c7913a939d780cf2561f403393588f95a5eb0519`.
Its [acceptance record](execution_manifest_v2.acceptance.json) binds the passing
full suite, representative seed-2102 pair and six declared dry runs.

## Execution path

[run_multiseed_dev_v1.py](../../run_multiseed_dev_v1.py) supports source freezing
and serial dry runs. It removes inherited `PPI_*` settings and supplies the
exact reference environment, changing only training seed, transport arm and
output path. It invokes the unchanged strict prerequisite checker with fold
seed 2101, both declared pickles and all feature requirements.

[CROSS5FOLD_multiseed_dev_v1.py](../../CROSS5FOLD_multiseed_dev_v1.py) is a separate
versioned copy of the frozen trainer. Its explicit `--training-seed` governs
Python, NumPy and PyTorch RNGs; `--fold-seed` is fixed at 2101. The cohort and
fold verifiers receive the fold seed. Stored training/validation indices and
complex IDs are consumed in their original order; missing or changed manifests
fail rather than generating a replacement split.

[multiseed_protocol_v1.py](../../multiseed_protocol_v1.py) binds source, data,
cohort, folds, runtime, recipe and run identity. It rejects undeclared seeds,
mixed arm output paths, pre-existing output directories and recipe changes.
Training requires the accepted full-suite and six-dry-run record before output
creation. Run provenance and the checkpoint/summary serialization paths carry
`training_seed`, `fold_seed`, the original fold-manifest record and the complete
execution binding. Dry runs stop before feature-statistic fitting, model
construction, optimizer steps and checkpoint creation. No new checkpoints
were produced or evaluated in this stage.

The [narrow trainer diff](implementation_validation_v2/trainer_delta.patch)
contains routing and provenance changes. AST comparisons confirm all 77
protected reference definitions, existing module/main assignments and the
model/optimizer/loss/checkpoint-selection training loop are unchanged, after
removing only added provenance fields and verification calls. Effective epoch,
patience and checkpoint-average settings are also checked against the protocol.
The [exact source archive](execution_source_archive_v2.json) preserves the four
new execution/test source files. All fourteen original frozen sources and the
completed seed-2101 result artifacts retain their recorded hashes.

## Validation evidence

The [full-suite record](implementation_validation_v2/full_test_validation.json)
and [log](implementation_validation_v2/full_tests_final.txt) report **134 tests
run, 133 passed and one existing host-limited symlink test skipped**, with no
failures or errors. Twenty additional tests cover seed/fold separation,
unchanged fold order, rejected regeneration, changed fold seeds, unequal arm
seeds, output collisions, inherited environment overrides, strict gate hashes,
protected recipe changes and the pre-training acceptance gate. Lightweight RNG
checks verify repeatability within a seed and changed initialization, shuffle
and dropout across seeds; they do not train a research model.

The [representative pair](implementation_validation_v2/representative_seed2102/validation.json)
and [all-six audit](implementation_validation_v2/all_six/validation.json) passed
under the accepted source freeze:

| Training seed | OT dry run | no-OT dry run | Fold seed | External evaluation |
| --- | --- | --- | --- | --- |
| 2102 | Passed | Passed | 2101 | Skipped |
| 2103 | Passed | Passed | 2101 | Skipped |
| 2104 | Passed | Passed | 2101 | Skipped |

Every gate confirmed 334/334 training and 285/285 Test287 coverage for both PLMs
and partner encoders, plus required partner/pair fields. Every trainer dry run
validated the loaded training population and all five original split identities
against the frozen reference. Pair comparisons allow only arm/transport and
run identity/output differences. Across seeds, the actual model recipe and run
provenance differ only in the declared training seed and run identities/paths.
The [launch-gate check](implementation_validation_v2/launch_gate_check.json)
exercised the combined acceptance validator read-only and confirmed zero
training outputs. This verifies routing and prerequisites; it supplies no new
predictive-performance result or training-stability claim.

## Rechecking without training

Use a fresh evidence directory: validation refuses overwriting its logs.
From the repository root, this performs all six serial dry runs:

```powershell
.\.venv\Scripts\python.exe .\run_multiseed_dev_v1.py --dry-run `
    --execution-manifest .\research_review\multiseed_dev_v1\execution_manifest_v2.json `
    --evidence-dir .\research_review\multiseed_dev_v1\new_readonly_validation
```

The runner CLI performs validation only. A later training stage needs a serial
training worker and complete artifact validation appropriate to the new seed
fields; neither a queue nor training is started by these commands. The
preregistered analysis, fixed folds, no-adaptive-tuning policy and development
limitations continue to apply.

## Earlier implementation candidate

[execution_manifest_v1.json](execution_manifest_v1.json) and
`implementation_validation_v1/` preserve the initial 130-test suite and passing
representative pair. Review then strengthened protection for epoch/patience
assignments and empty output paths and added the full pre-training acceptance
gate. These were implementation checks, with no fitting or observed new model
outcomes. The initial candidate is superseded: its source hashes no longer
describe the current execution files and it cannot pass current source
verification. Use the accepted v2 source freeze. No frozen research source,
data artifact, preregistration or completed comparison was changed.
