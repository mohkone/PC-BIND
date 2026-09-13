# PC-BIND

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22735579.svg)](https://doi.org/10.5281/zenodo.22735579)

PC-BIND is a sparse partner-conditioned framework for residue-level
protein-protein interface prediction. For each target residue, the frozen model
attends to the top 16 candidate residues in the supplied partner and fuses the
resulting cross-protein message with a target multi-view representation.

This repository contains the primary PC-BIND v5 implementation, its matched
target-only control, grouped cross-validation and evaluation code, component
ablations, and partner-intervention analysis code.

## Repository Contents

- `CROSS5FOLD_multi_test.py`: model, training, grouped CV, calibration, and
  five-checkpoint inference.
- `run_pcbind_v5_pilot.ps1`: frozen primary PC-BIND v5 configuration.
- `run_grouped_baseline_pilot.ps1`: exact target-only comparison.
- `prepare_pcbind_v5_data.ps1`: partner-coordinate and ESM-2 augmentation of
  authorized base benchmark files.
- `compare_pcbind_primary.py`: paired per-complex bootstrap comparison.
- `audit_sequence_homology.py`: best-HSP BLASTP train-test overlap audit.
- `run_reviewer_ablations.ps1`: no-PLM and no-auxiliary experiments.
- `run_pcbind_v6_*` through `run_pcbind_v9_*`: mechanistic development and
  stopping-gate analyses.

## Environment

The frozen runs used Python 3.12.14. On Windows PowerShell:

```powershell
.\setup_env.ps1
```

This installs the tested computational dependencies in `requirements.txt`.
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

## Frozen Primary Experiment

Train the exact target-only control and PC-BIND v5 with the same grouped folds
and seed:

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
threshold is selected only from Train335 out-of-fold predictions. Test70 is not
part of the primary evaluation because the post hoc homology audit found
substantial Train335 overlap.

## Availability Status

The `v1.0.0-pc-bind` code snapshot at commit `3bfdfa2` is archived on Zenodo
under [DOI 10.5281/zenodo.22735579](https://doi.org/10.5281/zenodo.22735579).
The version-independent concept DOI is
[10.5281/zenodo.22729835](https://doi.org/10.5281/zenodo.22729835).

The archive is a code release rather than a data bundle. Trained checkpoints,
prediction arrays, and processed datasets are not included.

## License and Citation

Code is released under the [MIT License](LICENSE). Citation metadata is in
[`CITATION.cff`](CITATION.cff). Please cite the associated article once its
bibliographic details are available. The GraphPPIS-derived backbone attribution
is preserved in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
