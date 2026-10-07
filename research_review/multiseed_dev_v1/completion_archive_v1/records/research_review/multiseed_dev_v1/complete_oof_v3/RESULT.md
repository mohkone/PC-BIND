# Three-seed fixed-fold development comparison

Primary analysis includes training seeds 2102, 2103 and 2104. Fixed fold seed: 2101. Test287 was not evaluated.

| Seed | Arm | Trapezoidal PR-AUC | MCC | AUROC | Saved threshold |
| --- | --- | ---: | ---: | ---: | ---: |
| 2102 | OT | 0.419784 | 0.346572 | 0.783893 | 0.559000 |
| 2102 | no-OT | 0.446950 | 0.366689 | 0.793741 | 0.575000 |
| 2103 | OT | 0.409008 | 0.340804 | 0.756171 | 0.609000 |
| 2103 | no-OT | 0.442522 | 0.365442 | 0.798666 | 0.564000 |
| 2104 | OT | 0.416721 | 0.347137 | 0.772162 | 0.594000 |
| 2104 | no-OT | 0.448862 | 0.367422 | 0.800469 | 0.565000 |

| Seed | OT − no-OT PR-AUC | OT − no-OT MCC | OT − no-OT AUROC |
| --- | ---: | ---: | ---: |
| 2102 | -0.027167 | -0.020117 | -0.009848 |
| 2103 | -0.033514 | -0.024638 | -0.042496 |
| 2104 | -0.032141 | -0.020285 | -0.028307 |

| Difference | Mean | Sample SD (ddof=1) | Range | Positive / zero / negative |
| --- | ---: | ---: | --- | --- |
| auc_pr_trapezoidal | -0.030940 | 0.003339 | [-0.033514, -0.027167] | 0 / 0 / 3 |
| mcc | -0.021680 | 0.002563 | [-0.024638, -0.020117] | 0 / 0 / 3 |
| auroc | -0.026884 | 0.016370 | [-0.042496, -0.009848] | 0 / 0 / 3 |

The 5,000 shared whole-complex bootstrap attempts produced 5,000 valid draws; 0 single-class draws were excluded without replacement attempts. Conditional 95% percentile intervals for the mean paired difference:

- auc_pr_trapezoidal: [-0.043168, -0.017972]
- mcc: [-0.032904, -0.010421]

Previously observed seed 2101 (excluded from the primary mean and intervals): PR-AUC difference -0.028387; MCC difference -0.027651.

These intervals condition on all fitted predictions and saved thresholds; they do not measure training-seed uncertainty. The three-seed SD and range describe only the observed repetitions.

- Checkpoint and threshold selection used development labels; the OOF scores remain selection-optimistic.
- Fold training populations overlap; complex resampling and seed repetition do not remove this fitting dependence.
- Only three new training seeds were observed; their variation is not a precise estimate of the full distribution of training randomness.
- The filtered cohort has unverified biological residue mappings and unestablished sequence-homology independence.
- Removing transport changes parameter count; the control is not capacity matched.
- Fixed folds leave sampling, mapping, homology, architecture and hyperparameter variation unprobed.
- Numeric seed pairing does not synchronize RNG histories across unequal architectures.
- Test287 remains unevaluated; these development results do not establish external generalization or biological partner specificity.
- No seed-level hypothesis test or bootstrap-tail p-value is reported.
