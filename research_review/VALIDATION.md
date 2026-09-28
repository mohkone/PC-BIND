# Validation of the research corrections

Completed 28 September 2026 in the existing Python 3.12.14 environment,
NumPy 2.4.6, scikit-learn 1.8.0, PyTorch 2.12.0+cpu. CUDA is unavailable.
No dependencies were changed and no full experiment was trained.

The table below preserves the original correction-stage verification. The
[filtered-cohort preparation report](data_repair_v1/RESULT.md) and its
[final validation record](data_repair_v1/final_validation.json) document the
subsequent data-directory changes, larger regression suite and strict gate on
the new 334/285 sensitivity cohort. The original coverage failure is unchanged.

| Check | Result |
|---|---|
| `python -m unittest discover -s tests -p 'test_*.py' -v` | **53 tests passed**, latest verification run 7.765 seconds. |
| `tests/test_launchers.ps1` | Passed: OT/control/capacity-ablation differences, inherited-setting isolation, environment/location restoration, one-fold default and argument forwarding. |
| Python source compilation and PowerShell entry-point parsing | Passed. |
| `smoke_test_pcbind_ot.py` | Forward/backward, transport constraints, gradients and missing-partner fallback passed, including with deliberately conflicting inherited settings. |
| `smoke_test_pcbind_v5.py` | Linked-data forward/gradient/intervention checks and one training step passed. |
| `smoke_test_pcbind_v9.py` | Residual-path intervention and gradient smoke checks passed. |
| `audit_saved_artifacts.py` | All 12 retained run/dataset combinations reproduce recorded AUPRC, AUROC and MCC within `1e-6`. |
| Legacy comparison CLI on TestB25 | Passed with 23 complex groups and explicit unverified-provenance warnings; 20 resamples used only for a smoke check, with temporary outputs discarded. |
| Strict full-feature prerequisites | **Failed on existing data**, correctly detecting missing-sequence/embedding records in Train335 and Test287. See the input audit. |

The final regression log is `validation_tests.log`; `validation_tests.json`
records the command, timestamps and exit code. The fresh-process smoke log is
`validation_smoke_tests.log`. The new CI workflow is
configured to run data-free tests and launcher checks; a hosted CI run was not
performed here. Real-data smoke checks are local and do not depend on CI access
to private datasets.

The smoke tests establish executable code paths and selected invariants. They
do not establish accuracy improvements, unbiased uncertainty, dataset
independence, biological label correctness, or successful full retraining.
Original datasets, checkpoints and prediction arrays were preserved.

The [research review](RESEARCH_REVIEW.md) records methodological limitations
and the experiments needed next. The [input audit](input_audit.md) records
coverage/overlap, and the [artifact audit](artifact_audit.md) records actual
saved results and hashes.
