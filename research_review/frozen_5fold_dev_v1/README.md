# Frozen five-fold development experiment

Both five-fold arms and the complete OOF comparison finished successfully.
The [final result](complete_oof/RESULT.md) and [JSON](complete_oof/comparison.json)
report lower OT scores under the frozen recipe: PR-AUC difference -0.0284
with paired 95% interval [-0.0454, -0.0124], and MCC difference -0.0277
with interval [-0.0431, -0.0125]. All 5,000 bootstrap draws were valid; none
were excluded. This is a negative development result, with the interpretation
limits recorded below. Test287 was skipped in both arms. The completed
experiment is preserved; no new experiment or external evaluation is launched
by this documentation update.

The [matched Fold 1 audit](../matched_fold1_v1/RESULT.md) verifies the existing
pair and records its descriptive negative result. The recipe is frozen without
tuning in response to that result. The [freeze manifest](freeze_manifest.json)
binds all training and prerequisite source hashes, both pickle hashes, cohort
and fold-manifest hashes, ordered split membership, runtime and model settings.
Its recipe source revision is `d8ff88dc155e0d775f64ef023e9e4ccbe82f07bf`.

Both arms use seed 2101, all five stored grouped folds, batch size 1 and skipped
external evaluation. Maximum epochs remain 30, early-stopping patience remains
8, and checkpoint selection remains MCC with the existing top-three-average
candidate. Model/data settings remain unchanged; the scheduled fold count and
output paths select these fresh runs. Windows updated from build 26200 to 26300
after the pilot pair. Both new arms are bound to the same current runtime;
Python, dependency versions, CPU device and 12-thread settings are unchanged.
Pilot predictions are not reused or pooled with the new pair. Training and
data preparation code is unchanged.

The authorized commands, from the project root, are:

```powershell
.\run_pcbind_ot_filtered_v1_pilot.ps1 -Seed 2101 -MaxFolds 5 -SkipTestEval 1 `
    -OutputDir .\outputs_filtered_v1_ot_seed2101_5fold_dev
.\run_pcbind_noot_filtered_v1_control.ps1 -Seed 2101 -MaxFolds 5 -SkipTestEval 1 `
    -OutputDir .\outputs_filtered_v1_noot_seed2101_5fold_dev
```

`run_frozen_development.py` runs those two jobs sequentially on the same CPU
runtime. These are fresh five-fold runs, including Fold 1, as requested by the
commands above; saved pilot checkpoints are not reused. Before each arm and
after it completes, the worker verifies frozen source, input and runtime
identity. Any failure stops the queue, retains evidence and prevents comparison.
Nonempty output directories are rejected; this worker does not resume or
overwrite earlier runs.

Local process evidence is saved in `outputs_frozen_5fold_dev_v1_process/`:
`queue_state.json`, `ot_stdout.log`, `ot_stderr.log`, `control_stdout.log` and
`control_stderr.log`. The control logs are created only when that arm starts.
The queue is a one-off background process, not a scheduled or recurring task;
the machine and worker must remain running. Dataset and checkpoint files remain
local and excluded from Git.

After both runs finish, `compare_oof_development.py` executes automatically. It
requires final complete status, all five scheduled/completed fold indices, the
same recipe and provenance, passed strict prerequisite gates and no external
predictions. It checks every OOF label, source sample, local residue index,
complex ID and fold ID against the stored validation membership in its recorded
order. Every sample/residue must appear exactly once. Selected checkpoints must
match their folds, run identities and settings, contain finite parameters and
reproduce their fold's saved development metrics.

The primary comparison is pooled OOF trapezoidal PR-AUC; MCC is secondary and
AUROC descriptive. The predeclared analysis uses 5,000 paired draws of complete
`complex_code` groups with bootstrap seed 2101 and percentile 95% intervals for
OT minus no-OT. Each draw is identical for both arms, retaining all target
samples and residues in each selected complex. Each arm's pooled OOF threshold
is held fixed. Single-class draws are excluded and counted. No bootstrap tail
fraction is presented as a hypothesis-test p-value.

Only a fully validated pair may produce `complete_oof/comparison.json` and
`complete_oof/RESULT.md`. Partial runs never produce a comparative result.
The live queue reached complete status for both jobs and their comparison.
Completion monitoring is paused; the frozen manifests and generated result
files remain unchanged.

The [completion verification](completion_verification.json) records the
successful read-only identity/checkpoint validation and twenty matching
artifact hashes. Its [log](completion_validation.txt),
[completed queue snapshot](queue_state_complete.json) and final provenance
copies preserve the evidence without including datasets or model binaries.
The [source archive](recipe_source_archive.json) identifies exact copies of
the fourteen frozen source files in `recipe_sources/`. Git preserves these
records' original bytes so their recorded SHA-256 hashes remain meaningful.

The next selected study is a [prospective shared training-seed repeat](../multiseed_dev_v1/PREREGISTRATION.md)
with training seeds 2102, 2103 and 2104 and these same seed-2101 folds. It has
been registered after observing this result and has not been launched. The
completed seed-2101 result remains a separate reference.

The [validation record](validation.json) contains the full local suite result:
114 tests ran, 113 passed and one host-limited symlink test was skipped. The
[dry-run audit](dry_run_validation.json) verifies that only transport and output
paths differ, including the prerequisite-report output path. The
[real partial-run rejection](partial_run_rejection.json) confirms that the
existing one-fold artifacts cannot produce a complete-five-fold comparison.

Development labels select checkpoints and thresholds, so these OOF scores are
selection-optimistic. Resampling holds fitted models and thresholds fixed and
does not quantify seed variability or dependence from overlapping fold training
sets. This is the filtered-cohort analysis, with unresolved biological mappings
and homology independence, and is not a parameter-count-matched ablation.
Test287 receives only the strict hash/feature prerequisite check, with no metric
evaluation or tuning. Independent external claims require a later locked
evaluation protocol.

The first freeze preflight rejected a comparison-layer record-shape mismatch:
training records `selection_mode=explicit`, while the shared cohort validator
and prerequisite records omit that training-specific field. The comparison
layer now separately requires explicit selection and compares the shared
fields exactly. A second preflight detected the Windows build update described
above. No training had started and no training source was changed. The
[resolution record](preflight_resolution.json) and both rejected candidate
manifests are retained.
