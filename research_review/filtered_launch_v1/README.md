# Fixed-cohort OT/control launch protocol

The dedicated entry points bind this exact local `data/geo_filtered_v1` cohort:

- `run_pcbind_ot_filtered_v1_pilot.ps1`: transport enabled.
- `run_pcbind_noot_filtered_v1_control.ps1`: transport disabled.

Both delegate to `invoke_filtered_v1.ps1` and the same `run_seed.ps1` recipe.
They default to seed 2101, one fold and skipped external evaluation. They use
Train335.pkl (334 samples) for training and only Test287.pkl (285 samples) for
external evaluation when explicitly enabled. The seven requested facts appear
before any Python process is launched. Dry runs verify both pickle hashes and
the fold-manifest hash without creating a training output directory.

## Why the legacy double check was bypassed

`run_pcbind_ot_pilot.ps1` checks Train335, Test60, Test287 and TestB25 at its usual
partner-coverage threshold, then separately allows only 48% partner-encoder
coverage for TestUB25. The second call is a legacy coverage exception, not a
development/evaluation split. The dedicated flow uses exactly one prerequisite
invocation with **both filtered files**, both target PLMs required, partner and
pair checks enabled, and partner-encoder coverage **1.0**. It has no legacy
secondary-set call and no missing-input fallback.

## Explicit data and split contract

The wrappers validate the fixed manifest identity, safe declared file set,
resolved paths, sample counts, file sizes and SHA-256 hashes before displaying
the run. They reject linked cohort paths, incomplete builds, undeclared seeds,
changed pickles or folds, and nonempty output directories for training.

`run_seed.ps1` passes `--data-dir`, `--cohort-manifest` and `--fold-manifest` as
explicit Python arguments. The executable's data selection comes from these
arguments. `PPI_DATA_DIR` remains only as a compatibility field for imported
utilities and the settings display; it is not the executable's selection
channel.

The prerequisite checker and training executable independently hash both
declared pickles. They verify the fixed fold file against the cohort manifest,
and check saved indices, complex IDs and group IDs against the loaded training
population. Every validation sample appears once; each fold partitions the
population and keeps complex groups disjoint. Training consumes each saved
`train_indices` and `val_indices` list in its recorded order. It does not call
the random split generator for this cohort.

The real output directory contains:

- `launcher_provenance.json`: resolved directory, declared files, dataset and
  manifest hashes, arm, seed, scheduled fold count and evaluation scope.
- `prerequisite_provenance.json`: the same resolved selection, hashes, strict
  coverage checks and validated fixed-fold status.
- `run_provenance.json`: loaded-source identity, the complete verified cohort
  registry (including Test287 even when evaluation is skipped), fixed folds,
  runtime, source-code hashes, transport status and completed-fold status.

The data and split manifests are immutable inputs. Existing biological mapping
limitations remain those in the [preparation report](../data_repair_v1/RESULT.md).
This is a **filtered-cohort analysis**, not a repaired full benchmark or evidence
of an OT accuracy benefit. One development fold does not establish that benefit.

## Commands and recorded evidence

From the project root:

```powershell
.\run_pcbind_ot_filtered_v1_pilot.ps1 `
    -Seed 2101 -MaxFolds 1 -SkipTestEval 1 `
    -OutputDir .\outputs_pcbind_ot_filtered_seed2101_fold1 -DryRun
```

The [OT dry run](ot_dry_run.txt) prints exactly:

```text
Resolved data directory: C:\Users\Mohamed KONE\Desktop\PC-BIND-OT\data\geo_filtered_v1
Cohort manifest: C:\Users\Mohamed KONE\Desktop\PC-BIND-OT\data\geo_filtered_v1\cohort_manifest.json
Declared files: Train335.pkl, Test287.pkl
Training pickle: C:\Users\Mohamed KONE\Desktop\PC-BIND-OT\data\geo_filtered_v1\Train335.pkl
Grouped fold manifest: C:\Users\Mohamed KONE\Desktop\PC-BIND-OT\data\geo_filtered_v1\grouped_folds_seed2101.json
External evaluation: skipped
Partner transport: enabled
```

The [control dry run](noot_dry_run.txt) selects the identical data and folds,
with transport disabled and a distinct output directory.
[Machine-readable comparison](dry_run_validation.json) records the settings:
only transport activation and output paths differ.

Replace `-DryRun` with `-ValidateOnly` to execute the strict prerequisite checker
and training's actual loader/fixed-split validation, then exit before model
initialization. [Captured validation](training_validate_only.json) records all
saved folds and both verified dataset identities. The
[regression record](validation.json) contains commands, exit codes and source
hashes for Python tests, PowerShell launcher tests and OT/v5/v9 smoke checks.

The first real launch stopped in prerequisite-report serialization before any
model training. A NumPy coverage count needed conversion to a native integer.
The report now serializes completely before opening its output file, and a new
regression test covers this path. The
[follow-up verification](validation_after_report_fix.json) records **99 passing
tests and one host-limited symlink test skipped**, plus the real-cohort gate with
JSON provenance enabled. The failed attempt is preserved separately and recorded
in [its failure record](first_launch_failure.json).

After those checks, run the same command without either inspection switch to
start Fold 1. Use `-SkipTestEval 0` only when external evaluation is intended;
the sole external input remains the same filtered Test287 file. A later no-OT
run uses the control wrapper and its separate output directory.

## Completed Fold 1

After the dry-run and preflight checks, the requested OT Fold 1 was started in
`outputs_pcbind_ot_filtered_seed2101_fold1`. It uses 266 training and 68 validation
samples from saved fold index 0 and skips external evaluation. It completed
successfully with exit code 0 after 11 epochs (early stopping), taking about two
hours on CPU. The [completion audit](fold1_completion_audit.json) verifies final
provenance, the checkpoint, all 13,407 validation prediction identities and
labels, and exact saved-fold membership. The earlier
[launch snapshot](fold1_launch_snapshot.json) is retained as startup evidence.

The saved development predictions reproduce the checkpoint's MCC of **0.3837**
and trapezoidal PR-AUC of **0.4503**; AUROC is **0.7952**. These are results from
one selected development fold: its labels selected both the checkpoint and
threshold. Test287 was not evaluated, and the matched no-OT control has not yet
been trained. These scores establish neither independent generalization nor an
OT advantage.

The local background process saved its completion status and logs under
`outputs_pcbind_ot_filtered_seed2101_fold1_process_retry1/`. Read
`process_status.json` for process completion/failure, `stdout.log` for epochs and
validation output, and `stderr.log` for batch progress or errors. Final training
artifacts belong to the main output directory. The original failed preflight
output is preserved separately with the `_preflight_failed` suffix.
