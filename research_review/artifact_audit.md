# Saved artifact audit

Read-only audit of saved arrays and serialized metadata. Base datasets, chain identities, fold membership, and homology separation are not verified by this audit.

Only the saved probability mean and stored threshold are evaluated. No test-set strategy selection or threshold refitting is performed.

| Run | Seed | Folds present | Grouped CV | Transport | Main/aux PLM width | Contact auxiliary |
|---|---:|---:|---|---|---|---|
| outputs | 1234 | 1 | False | False | 480/320 | True |
| outputs_no_ot_matched_seed2101 | 2101 | 1 | True | False | 0/0 | False |
| outputs_ot_stable_seed2101 | 2101 | 1 | True | False | 480/320 | True |

AUPRC below is trapezoidal area; AP is non-interpolated average precision. Interface precision/recall/F1 refer only to the positive class, whereas the legacy CSV uses class-macro values.

## outputs

| Dataset | Residues | Prevalence | AUPRC | AP | MCC | Interface precision | Interface recall | Interface F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Test60 | 13144 | 0.1579 | 0.4688 | 0.4690 | 0.3521 | 0.4422 | 0.4742 | 0.4577 |
| Test287 | 60376 | 0.1419 | 0.4015 | 0.4016 | 0.3261 | 0.3784 | 0.5012 | 0.4312 |
| TestB25 | 5864 | 0.1260 | 0.4772 | 0.4778 | 0.3770 | 0.4450 | 0.4709 | 0.4576 |
| TestUB25 | 5917 | 0.1202 | 0.4253 | 0.4264 | 0.3617 | 0.4150 | 0.4740 | 0.4425 |

- Incomplete five-fold evidence; treat as a pilot.
- Run metadata does not establish complex-grouped cross-validation.
- Test60: residue identity absent; dataset ordering cannot be certified from NPZ alone.
- Test287: residue identity absent; dataset ordering cannot be certified from NPZ alone.
- TestB25: residue identity absent; dataset ordering cannot be certified from NPZ alone.
- TestUB25: residue identity absent; dataset ordering cannot be certified from NPZ alone.

## outputs_no_ot_matched_seed2101

| Dataset | Residues | Prevalence | AUPRC | AP | MCC | Interface precision | Interface recall | Interface F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Test60 | 13144 | 0.1579 | 0.4612 | 0.4616 | 0.3556 | 0.4500 | 0.4689 | 0.4593 |
| Test287 | 60376 | 0.1419 | 0.4100 | 0.4101 | 0.3383 | 0.3797 | 0.5290 | 0.4420 |
| TestB25 | 5864 | 0.1260 | 0.4614 | 0.4621 | 0.3886 | 0.4667 | 0.4641 | 0.4654 |
| TestUB25 | 5917 | 0.1202 | 0.4165 | 0.4177 | 0.3809 | 0.4433 | 0.4726 | 0.4575 |

- Incomplete five-fold evidence; treat as a pilot.
- Transport is disabled in serialized metadata; folder name is not treatment evidence.
- Test60: residue identity absent; dataset ordering cannot be certified from NPZ alone.
- Test287: residue identity absent; dataset ordering cannot be certified from NPZ alone.
- TestB25: residue identity absent; dataset ordering cannot be certified from NPZ alone.
- TestUB25: residue identity absent; dataset ordering cannot be certified from NPZ alone.

## outputs_ot_stable_seed2101

| Dataset | Residues | Prevalence | AUPRC | AP | MCC | Interface precision | Interface recall | Interface F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Test60 | 13144 | 0.1579 | 0.4613 | 0.4616 | 0.3473 | 0.4526 | 0.4467 | 0.4497 |
| Test287 | 60376 | 0.1419 | 0.4153 | 0.4155 | 0.3429 | 0.4026 | 0.4936 | 0.4435 |
| TestB25 | 5864 | 0.1260 | 0.4489 | 0.4495 | 0.3606 | 0.4548 | 0.4222 | 0.4379 |
| TestUB25 | 5917 | 0.1202 | 0.3951 | 0.3962 | 0.3597 | 0.4341 | 0.4402 | 0.4372 |

- Incomplete five-fold evidence; treat as a pilot.
- Transport is disabled in serialized metadata; folder name is not treatment evidence.
- Test60: residue identity absent; dataset ordering cannot be certified from NPZ alone.
- Test287: residue identity absent; dataset ordering cannot be certified from NPZ alone.
- TestB25: residue identity absent; dataset ordering cannot be certified from NPZ alone.
- TestUB25: residue identity absent; dataset ordering cannot be certified from NPZ alone.

These descriptive results do not establish an OT benefit, a matched causal contrast, homology-independent generalization, or variation across independent training seeds. The JSON records hashes and differences from the original metric summaries.
