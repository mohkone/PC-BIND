# Matched Fold 1 development result

The saved OT and no-OT runs pass the paired provenance audit. Both use the
same hash-verified filtered cohort, fixed Fold 1, seed 2101, source files,
CPU runtime, model recipe and selection protocol. Test287 was skipped in both.
Only transport activation differs among the experimental settings. Run UUIDs,
creation times, output paths and selected thresholds are natural run differences.

The [machine-readable audit](pairing_audit.json) records dataset, manifest,
fold, checkpoint and prediction hashes. Raw copies of each arm's launcher,
prerequisite, run and ensemble provenance are preserved in this directory.
Checkpoint metadata, finite parameters, all 13,407 validation residue identities,
labels and selected metrics were checked against the saved 266/68 sample split.

| Selected Fold 1 development metric | OT | no-OT | OT minus no-OT |
| --- | ---: | ---: | ---: |
| MCC | 0.3837 | 0.3890 | -0.0054 |
| Trapezoidal PR-AUC | 0.4503 | 0.4836 | -0.0333 |
| AUROC | 0.7952 | 0.7974 | -0.0022 |
| Saved threshold | 0.539 | 0.569 | — |

These metrics were independently reproduced from the saved predictions. PR-AUC
means trapezoidal precision-recall area, not average precision. The checkpoint
and threshold were selected using these development labels. No statistical
significance, OT harm, external generalization or predictive advantage follows
from this one fold and seed. There is no preliminary Fold 1 indication of an
OT advantage under this recipe.

The user reports that the control selected epoch 3 and stopped after 11 epochs.
Its saved output does not contain a console log or selected-epoch metadata, so
those epoch details are user-reported rather than independently recovered. The
OT process log does document stopping after 11 epochs. The frozen source fixes
30 maximum epochs, patience 8, MCC selection and a top-three checkpoint-average
candidate, accepted only when its selection score is at least the best score.

The next stage is the [frozen five-fold development protocol](../frozen_5fold_dev_v1/README.md).
It runs both complete five-fold arms afresh and compares only complete OOF
predictions with whole-complex paired resampling. This remains a sensitivity
analysis of 334 retained training samples; original biological residue mappings
and sequence-homology independence remain unresolved. Transport removal also
changes parameter count, so this is not a capacity-matched ablation.
