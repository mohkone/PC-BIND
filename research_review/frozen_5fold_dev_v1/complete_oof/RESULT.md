# Frozen five-fold development comparison

Both arms completed the same five fixed folds: 334 samples, 209 complexes and 66208 residues. Test287 was skipped. All source, runtime, dataset and split identities passed the frozen protocol gate.

| Complete OOF metric | OT | no-OT | OT minus no-OT | Paired 95% interval |
| --- | ---: | ---: | ---: | --- |
| auc_pr_trapezoidal | 0.4109 | 0.4393 | -0.0284 | [-0.0454, -0.0124] |
| mcc | 0.3369 | 0.3645 | -0.0277 | [-0.0431, -0.0125] |
| auroc | 0.7662 | 0.7916 | -0.0254 | Descriptive only |

PR-AUC is trapezoidal area under the precision-recall curve, not average precision. Primary PR-AUC and secondary MCC intervals use the same 5,000 whole-complex draws for both arms, with their saved pooled OOF thresholds held fixed.

- Checkpoint and threshold selection used these development labels; results are selection-optimistic.
- Bootstrap holds fitted predictions and each selected threshold fixed; it does not quantify seed or training variability.
- Training sets overlap across folds; complex resampling does not remove dependence caused by model fitting.
- This filtered sensitivity cohort has no verified original PDB residue mappings or established homology independence.
- The no-OT branch removes transport parameters; this is not a parameter-count-matched capacity ablation.
- No Test287 performance or external generalization claim is evaluated here.
