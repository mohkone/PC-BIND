# PC-BIND

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22735579.svg)](https://doi.org/10.5281/zenodo.22735579)

PC-BIND is a sparse partner-conditioned framework for residue-level
protein-protein interface prediction. For each target residue, the v5 model
attends to the top 16 candidate residues in the supplied partner and fuses the
resulting cross-protein message with a target multi-view representation.

This workspace contains the PC-BIND v5 implementation, a target-only backbone
control, grouped cross-validation and evaluation code, experimental transport
matching, component ablations, and partner-intervention analyses.

## Research review and evidence status

Read the [research review](research_review/RESEARCH_REVIEW.md) before interpreting
the results. The [saved artifact audit](research_review/artifact_audit.md)
recomputes the retained metrics and records the actual run settings.

The three saved runs in this workspace each contain **one fold**, and all three
record `partner_transport=false`. In particular,
`outputs_ot_stable_seed2101` and `outputs_no_ot_matched_seed2101` differ in PLM
features and auxiliary supervision. Their difference cannot estimate an OT
effect. The documented five-fold v5/control result directories are absent.
No improved predictive accuracy is claimed by this code revision.

The target-only backbone control removes the shared target encoder introduced
with partner conditioning. It is a useful system comparison, but an additional
control retaining the same target-side architecture is needed to isolate the
contribution of partner information. Grouping by `complex_code` also does not
establish separation by sequence homology.

The input audit also finds PDB-level training overlap in 49/58 Test60 complexes
and 18/23 TestB25 complexes. Treat their scores as overlap-affected benchmark
results; they cannot establish generalization to unseen complexes. The full
overlap counts and failing records are in the [input audit](research_review/input_audit.md).

Cross-side checks find exact matches to training partner sequences for 49
Test60 targets, 18 TestB25 targets and 5 TestUB25 targets. Auditing only training
target sequences therefore misses input overlap in this partner-conditioned task.

## Repository Contents

- `CROSS5FOLD_multi_test.py`: model, training, grouped CV, threshold selection, and
  five-checkpoint inference.
- `run_pcbind_v5_pilot.ps1`: reference PC-BIND v5 training configuration.
- `run_grouped_baseline_pilot.ps1`: target-only backbone comparison.
- `prepare_pcbind_v5_data.ps1`: partner-coordinate and ESM-2 augmentation of
  authorized base benchmark files.
- `compare_pcbind_primary.py`: paired per-complex bootstrap comparison.
- `audit_saved_artifacts.py`: input-data-free metric and run-provenance audit.
- `audit_input_integrity.py`: read-only input coverage and exact-overlap audit.
- `audit_sequence_homology.py`: best-HSP BLASTP train-test overlap audit.
- `run_reviewer_ablations.ps1`: legacy no-PLM and no-auxiliary experiments;
  these are not matched frozen-v5 component studies.
- `run_pcbind_v6_*` through `run_pcbind_v9_*`: mechanistic development and
  stopping-gate analyses.

## Environment

The inspected environment uses Python 3.12.14. On Windows PowerShell:

```powershell
.\setup_env.ps1
```

This installs the pinned computational dependencies in `requirements.txt`.
PyTorch hardware builds vary by platform; install the appropriate official
PyTorch build first if CUDA acceleration is required.

## Data Layout

Place authorized base files at:

```text
data/geo/Train335.pkl
data/geo/Test60.pkl
data/geo/Test287.pkl
data/geo/TestB25.pkl
data/geo/TestUB25.pkl
```

Processed benchmark pickle files are intentionally not redistributed in this
repository because redistribution permission has not been established for every
upstream derivative. Users must obtain authorized base files from the upstream
benchmark providers under their applicable terms. The included preparation code
augments compatible base files; it does not reconstruct every benchmark from
raw PDB identifiers alone.

After obtaining the base files from authorized sources, augment and validate
them with:

```powershell
.\prepare_pcbind_v5_data.ps1
.\run_pcbind_v5_pilot.ps1 -Seed 2101 -SmokeOnly 1
```

The augmentation step retrieves PDB coordinates and ESM-2 weights when they are
not already cached, so network access is required on its first run.

Preparation updates the processed pickle files. In this workspace `data/geo`
is a junction to a sibling research directory, so updates would affect that
shared directory. The review used it read-only. See the
[input audit](research_review/input_audit.md) for resolved paths and coverage.
Current inputs fail the full target-PLM requirement for Train335 record 106
(`2J3R`) and Test287 records 55 (`4M0W`) and 161 (`6MAV`). Repair their verified
sequence mappings and embeddings before a full-feature experiment; do not
silently drop them or relabel the existing results.

The unbound TestUB25 data have partner inputs for 12/25 records; the other 13
use the target pathway. Aggregate results therefore mix these input regimes.
Report the coverage and both subgroups. Automatic PDB preparation uses other
parsed chains as candidate partners; a curated biological-partner/assembly
manifest is still required for a strong partner-specific interpretation.

## Reference five-fold experiment

After resolving the input failures, train the target-only backbone control and
PC-BIND v5 with the same grouped folds and seed:

```powershell
.\run_grouped_baseline_pilot.ps1 -Seed 2101 `
  -OutputDir outputs_grouped_full_seed2101

.\run_pcbind_v5_pilot.ps1 -Seed 2101 `
  -OutputDir outputs_pcbind_v5_shared_seed2101
```

Then reproduce the paired comparison:

```powershell
.\.venv\Scripts\python.exe .\compare_pcbind_primary.py `
  --control-dir outputs_grouped_full_seed2101 `
  --model-dir outputs_pcbind_v5_shared_seed2101 `
  --bootstrap 5000
```

Training uses five folds grouped by `complex_code`. The five checkpoints come
from one training seed and are not independent replicates. The external
threshold is selected from Train335 held-out-fold predictions, whose labels
were also used for checkpoint selection. These are development predictions,
not an unbiased nested-CV performance estimate or a probability calibration.
Transferring their threshold to an averaged ensemble requires validation on
training-only data. Test70 remains excluded from the reference primary set;
the supplied homology-audit script is a best-HSP diagnostic, not a certificate
that the remaining sets are homology independent.

The comparison reports pooled-residue trapezoidal AUPRC, AUROC and MCC. Its
complex-bootstrap intervals condition on fitted models and do not estimate
training-seed variability. Legacy `precision`, `recall` and `f1` columns are
macro averages over the two classes, not positive-interface-only metrics.
Legacy NPZ labels can be checked against a dataset, but alone cannot certify
residue identity or identical fold membership.

Future runs save `run_provenance.json` with data/code hashes, resolved input
paths, runtime information and the actual fold membership. New NPZ archives
include source sample/residue indices and dataset hashes; these indices do not
assert verified PDB residue numbering. Existing saved arrays remain unchanged.

## Transport experiments and validation

The optional transport branch is an **OT-inspired capacity-constrained matching
heuristic** with a no-match dustbin. It is not established as a converged solver
of a classical unbalanced OT objective. The corrected implementation records
`transport_implementation=capacity_capped_dustbin_v2`; previous transport
checkpoints must be re-evaluated or retrained before numerical comparison.

Inspect configurations without training or modifying saved outputs:

```powershell
.\run_pcbind_ot_pilot.ps1 -DryRun
.\run_pcbind_no_ot_control.ps1 -DryRun
.\run_pcbind_ot_ablation.ps1 -DryRun
```

The no-OT launcher disables the entire branch. The capacity ablation keeps
the branch, temperature, regularizers and fusion, and disables only capacity
projection. For a one-fold development run use `-MaxFolds 1 -SkipTestEval 1`;
external tests should remain untouched during model selection. If overriding
temperature or fusion, use the same values for the capacity ablation.

`run_seed.ps1` accepts explicit settings, clears inherited `PPI_*` variables
for execution, and restores the caller's environment afterwards. Specify
options as script arguments. `run_seed_onefold.ps1` now defaults to one fold.

Run the checks and reproduce the read-only audits:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\tests\test_launchers.ps1
.\.venv\Scripts\python.exe .\smoke_test_pcbind_ot.py
.\.venv\Scripts\python.exe .\audit_saved_artifacts.py
.\.venv\Scripts\python.exe .\audit_input_integrity.py
```

## Availability Status

The existing release metadata identifies `v1.0.0-pc-bind` under
[DOI 10.5281/zenodo.22735579](https://doi.org/10.5281/zenodo.22735579).
The version-independent concept DOI is
[10.5281/zenodo.22729835](https://doi.org/10.5281/zenodo.22729835).

That metadata refers to the prior code release. The local review corrections
have not been published as a new archive. The release is described as a code
bundle; local checkpoint and prediction directories are separate artifacts.

## License and Citation

Code is released under the [MIT License](LICENSE). Citation metadata is in
[`CITATION.cff`](CITATION.cff). Please cite the associated article once its
bibliographic details are available. The GraphPPIS-derived backbone attribution
is preserved in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
