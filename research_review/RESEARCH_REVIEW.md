# PC-BIND research review and correction record

Review dates: 27–28 September 2026. Scope: the code, launchers, saved predictions,
serialized run settings, and dataset files reachable through this workspace.
No manuscript was supplied. Original checkpoints, predictions and datasets
were not rewritten. This is an implementation and evidence review, not a new
accuracy experiment.

Subsequent data preparation created an explicitly labelled
[filtered sensitivity cohort](data_repair_v1/RESULT.md), with 334 training and
285 Test287 samples, new hashes and shared grouped folds. Its strict coverage
gate passes; the original full-cohort failures and unverified biological
mappings remain recorded. No OT-versus-no-OT accuracy experiment has yet been
run on this cohort.

## Assessment

The central hypothesis is reasonable: a supplied interaction partner can help
distinguish a protein's relevant interface from its general binding propensity.
PC-BIND combines target residue, atom and substructure representations with
geometric, biochemical and ESM-2 features. The v5 path selects up to 16 partner
residues for each target residue and fuses a learned partner message into target
predictions. Classification/ranking heads and graph smoothing produce residue
scores. Later variants add pair-contact objectives, interventions, residual
heads, and an optional transport-style matching branch.

The input audit also finds that 49 of Test60's 58 distinct PDB entries and 18
of TestB25's 23 occur in Train335. These benchmarks are not disjoint from
training at the complex/PDB level, even though their target sequences have no
exact matches in the training target list. This is a major generalization
limitation requiring a complex- and partner-aware split redesign.

The implementation supports investigating this hypothesis, but the retained
experiments do not establish an OT benefit, partner specificity, superiority
over published methods, or homology-independent generalization. This conclusion
follows from the run metadata and experimental design, not from a presumption
that the architecture cannot work.

Partner-specific prediction already has precedents, including methods that
test the same target against alternative partners. Frame novelty around the
specific architecture, efficiency, or demonstrated behavior, not the existence
of partner conditioning itself. See the primary [BIPSPI paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6361243/).
Preserve attribution to the [GraphPPIS backbone](https://github.com/biomed-AI/GraphPPIS)
and distinguish its published task/inputs from any new comparison.

## What the existing results actually show

The [artifact audit](artifact_audit.md) recalculates the probability-mean
predictions at their saved thresholds. Its [JSON](artifact_audit.json) records
configuration fields, prediction hashes and errors against the original
summaries. Across the 12 retained run/dataset combinations, recomputed AUPRC,
AUROC and MCC agree with the recorded values within `1e-6`.

| Directory | Seed | Checkpoints | Grouped CV | Transport enabled | Primary/auxiliary PLM widths | Contact auxiliary |
|---|---:|---:|---|---|---|---|
| `outputs` | 1234 | 1 | No | No | 480 / 320 | Yes |
| `outputs_no_ot_matched_seed2101` | 2101 | 1 | Yes | No | 0 / 0 | No |
| `outputs_ot_stable_seed2101` | 2101 | 1 | Yes | No | 480 / 320 | Yes |

Both named comparison runs have partner conditioning enabled. Neither has
transport enabled in its saved summary. They change more than one component,
so their differences cannot identify the effect of transport or PLM features.
The reference five-fold v5 and backbone-control directories are absent.

For completeness, their *descriptive*, confounded comparison is:

| Dataset | `no_ot_matched` AUPRC | `ot_stable` AUPRC | Stable minus no-OT |
|---|---:|---:|---:|
| Test60 | 0.4612 | 0.4613 | +0.0001 |
| Test287 | 0.4100 | 0.4153 | +0.0054 |
| TestB25 | 0.4614 | 0.4489 | -0.0125 |
| TestUB25 | 0.4165 | 0.3951 | -0.0214 |

The unweighted mean difference is approximately -0.0071. This is a description
of these four saved results, not an estimate of an OT treatment effect. Do not
rename the directories and reinterpret them as a successful matched ablation.

## Corrections made

| Area | Defect | Correction |
|---|---|---|
| Experiment launch | The no-OT launcher enabled transport and shared the OT default output name. | It now forwards the same recipe with transport disabled and a distinct output path. |
| Ablation design | The purported capacity ablation also changed temperature and regularization. | It now changes only projection iterations; other settings match the OT launcher. |
| Configuration | `run_seed.ps1` exported undeclared variables and inherited settings from previous runs. | All exported options are declared; execution isolates and restores `PPI_*` settings and working directory. Dry-run output exposes the effective settings. |
| One-fold entry point | `run_seed_onefold.ps1` defaulted to five folds. | It delegates to the common launcher and defaults to one fold. |
| Smoke tests | Preflight tests could inherit another model's transport/logit configuration. | Each smoke process defines its own configuration. |
| Threshold selection | Search used `score > threshold`, while deployment used `>=`. | Both use `>=`, with a tied-score regression test. |
| Grouped splits | Missing group identifiers could silently fall back to individual samples. | Missing groups and insufficient distinct groups fail explicitly. |
| Transport numerics | Final row normalization could undo column capacity constraints; unsupported columns could produce undefined gradients. | Final excess mass goes to the dustbin; masked reductions have finite gradients. The revised implementation is versioned. |
| Auxiliary objectives | Attention contacts could count a positive twice; later mismatched-partner forwards could replace the plan used for regularization. | Positive pairs are deduplicated; regularization uses the saved true-partner forward plan. |
| PDB preparation | Multiple models/alternate conformers were mixed, and ambiguous chain sizes could silently choose a target. | Use the first model and a deterministic conformer; reject ambiguous or contradictory identity mappings; mark count-only mappings unverified. |
| Feature preparation | The script extracted partner PLMs but did not supply both required target PLMs. | Preparation includes target and partner extraction for both models. |
| Input validation | Existence of feature keys could be mistaken for valid coverage. | Check dimensions, numeric finiteness, embedding failure markers, graph/slice/contact bounds and complete target PLM coverage. |
| Statistical reporting | Comparison text asserted v5, seed 2101 and five folds without checking supplied runs. | Reports identify actual metadata, protocol differences and resolved data paths. |
| Pairing and uncertainty | Bootstrap grouping was not checked against dataset labels; some summaries substituted a fold-metric mean for an ensemble metric. | Validate label order and complex partitions; require the intended ensemble metric; disclose degenerate bootstrap draws and uncertainty scope. |
| Metric interpretation | Generic precision/recall/F1 labels obscured binary-class macro averaging. | Definitions are explicit; the artifact audit also reports positive-interface metrics and average precision. |
| Run provenance | Legacy arrays and summaries could not establish data version, fold membership or residue ordering. | Future runs save dataset/code hashes, resolved paths, runtime information, actual fold membership and row-aligned sample/residue identity. |

These changes improve correctness and reproducibility. They do not retroactively
correct the saved models. The transport implementation now records
`capacity_capped_dustbin_v2`; transport comparisons need new runs or an explicitly
labeled re-evaluation. Even with identical seeds, numerical or data changes
can change the training trajectory.

## Input findings and remaining data risks

`data/geo` is a filesystem junction resolving to the sibling
`Desktop/article/data/geo` directory. The `.venv` is also shared. The audit used
these links read-only; data preparation would update the shared processed
files. The [input audit](input_audit.md) and its [JSON](input_audit.json) record
the actual coverage and exact overlaps.

The exact overlap audit finds:

| Benchmark | Distinct PDB entries | PDB entries also in Train335 | Exact target-sequence matches to Train335 targets |
|---|---:|---:|---:|
| Test60 | 58 | 49 | 0 |
| Test287 | 217 | 0 | 0 |
| TestB25 | 23 | 18 | 0 |
| TestUB25 | 25 | 0 | 0 |
| Test70 | 66 | 58 | 53 of 70 records (69 have sequences) |

Shared structures may place different chains of the same complex on opposite
sides of the training/test boundary. Zero exact *target-to-target* matches do
not resolve that issue. These measurements establish overlap, not its exact
effect on predictive metrics. Train335 contains 335 target records from 209
distinct complex IDs. The primary files use bare PDB IDs; Test70 uses chain
suffixes, which must be normalized before comparing PDB overlap. No existing
sample records mapping-verification metadata.

The cross-side sequence check confirms that 49 Test60 targets and 18 TestB25
targets have exact sequence matches among **training partner chains**. Five
TestUB25 targets also exactly match training partner sequences despite no
shared PDB IDs. Thus even a target-sequence-only audit misses exposure through
the other model input. These are measured input overlaps; whether labels or
auxiliary supervision also cross the boundary needs an explicit residue/contact
mapping audit.

The strict full-feature check fails on existing target embeddings:

- Train335 record 106, `2J3R`, 158 residues: missing target sequence and failed embeddings.
- Test287 record 55, `4M0W`, 317 residues: missing target sequence and failed embeddings.
- Test287 record 161, `6MAV`, 161 residues: missing target sequence and failed embeddings.

Repair sequence-to-residue mappings against authorized source data before
regenerating embeddings. Blindly filling zeros, dropping records, or replacing
chains would change the benchmark or conceal failed inputs. Preserve the old
data version and publish the denominator of any justified exclusion.

TestUB25 has valid partner encoder inputs for 12/25 targets; the remaining 13
follow the target pathway. Report separate results for these strata and the
combined benchmark. Missing partners are a distinct condition from a deliberately
mismatched partner.

The preparation still aggregates other parsed PDB chains as candidate partners.
That does not verify the biological interaction partner, biological assembly,
or appropriate assembly transformations. Sequence agreement does not prove that
the atom-feature rows match PDB atom order. Legacy processed sequences may
themselves have come from count-based matching. A curated manifest must specify
PDB version, assembly, target/partner chains, residue IDs including insertion
codes, atom correspondence, label definition and coordinate source.

Complex-ID grouping prevents splits of a *verified shared identifier*. It is
not a homology split. The existing BLAST audit selects the best single HSP and
does not establish that no other qualifying hit exists. It also does not audit
partner sequences or every train/validation fold. Exact-sequence overlap counts
in the new input audit are lower-bound diagnostics, not a replacement for a
full homology audit.

## Methodological improvements required before stronger claims

1. **Define the estimand and controls.** Retain the backbone comparison as a
   system comparison. Add a target-only training control with the *same target
   encoder and target fusion capacity* as v5, and a partner-shuffled training
   control with the same branch capacity. A fixed trained model evaluated with
   null/mismatched partners is an intervention analysis, not a replacement for
   these training controls. For capacity matching, compare the corrected OT
   branch with its zero-projection ablation using identical features and losses.

2. **Establish dataset identity before training.** Curate the assembly/partner
   manifest, fix the three failed primary-set records, and hash the resulting
   datasets. Audit both target and partner sequence overlap. Construct connected
   homology groups across related chains/complexes under a prespecified identity
   and alignment-coverage rule; keep each component within one fold. Document
   any relationship between bound and unbound benchmarks.

3. **Separate selection from final evaluation.** Use grouped training-only
   development splits for architecture, hyperparameters and stopping gates.
   Use `SkipTestEval=1` during development. Current held-out-fold labels select
   epochs and averaged checkpoints, then select the decision threshold; the
   resulting OOF metrics are development estimates with selection optimism.
   If reporting internal generalization, add an outer held-out loop. Validate
   threshold selection for the *ensemble score* on training-only predictions;
   it need not share the distribution of individual-fold scores. Threshold
   selection is not probability calibration.

4. **Freeze and repeat matched experiments.** Use the same split manifest and
   seed list for all conditions, preferably at least three training seeds as an
   initial stability check. The legacy reviewer-ablation launcher uses unrelated
   seeds and defaults that differ from grouped v5; it is exploratory. Five CV
   checkpoints are overlapping training fits, not five independent experiments.
   Record hyperparameters, code version, data hashes, runtime and actual folds.

5. **Measure partner specificity directly.** For a fixed target with known
   alternative partners/interfaces, compare target residue predictions under
   each partner. Include multiple mismatches matched on sequence length and
   feature availability, donor identity checks, missing-partner baselines and
   paired uncertainty. A changed output or nonzero gradient proves sensitivity,
   not correct biological specificity. Attention weights and dustbin mass are
   model evidence, not experimentally validated contacts or calibrated binding
   probabilities.

6. **Report prespecified metrics and uncertainty.** Make one external scoring
   strategy primary. Report pooled-residue AUPRC with the exact definition,
   positive prevalence, AUROC, MCC, and positive-class precision/recall/F1.
   Complement it with an explicitly defined per-complex summary and missing-input
   strata. Resample whole independent complexes or homology components with both
   models paired; quantify training-seed variability separately. Do not bootstrap
   residues as independent observations. Correct multiplicity for a specified
   family when drawing multiple confirmatory conclusions.
   Legacy summary scripts that rank strategies on external test scores now label
   those rankings exploratory; selecting the best such score would remain
   optimistic even if its input predictions used training-selected thresholds.

7. **Benchmark biological utility and cost.** Use current, appropriate
   partner-specific comparators under equal input access and homology controls.
   Pair-contact evaluation should include retrieval coverage and missing true
   contacts, not only a candidate set augmented with known positives. Report
   runtime, peak memory, sequence lengths and candidate coverage. The scorer is
   sparse after selection, but it still constructs dense target-by-partner
   similarity/transport workspaces; do not claim fully linear memory scaling.

The implementation computes trapezoidal PR area, which differs from average
precision. Preserve this distinction when comparing publications; the
[scikit-learn definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
explains the difference. A confidence interval over four benchmark scores
assumes exchangeable datasets and does not include seed variability. With only
four nonzero paired differences, the smallest two-sided exact signed-rank
p-value is 0.125; those tests cannot support a 0.05 significance claim.

## Precise description of the transport branch

For target length `N` and partner length `M`, a dot-product retrieval step
chooses top-k candidates. A learned pair scorer assigns affinities on these
candidates, while a target-dependent dustbin logit represents no match. The
corrected finite projection routine keeps row mass at one and bounds each real
partner column by `max(1, N/M)`, placing final removed mass in the dustbin.
Zero projection iterations retain uncapped row-softmax matching.

This is a reproducible capacity-constrained, OT-inspired mechanism. It has no
demonstrated convergence guarantee to a specified unbalanced OT objective.
If the paper claims an OT solver, specify the cost, marginals, regularizers,
objective and convergence criterion, then verify the numerical implementation
against a reference. The foundational
[unbalanced transport scaling paper](https://arxiv.org/abs/1607.05816) is a
relevant mathematical reference, not evidence that this implementation solves
that objective.

## Suggested replacement research claims

**Aim:** "We investigate sparse partner conditioning for residue-level
protein-interface prediction using target and partner representations, with
an optional capacity-constrained matching module."

**Implementation:** "The framework provides grouped cross-validation,
training-derived decision thresholds, component comparisons and fixed-model
partner interventions. Its inputs and split groups require explicit provenance
and coverage audits."

**Current evidence:** "The retained artifacts are single-fold pilot runs.
They do not contain a matched transport-on/off experiment, and therefore do
not establish a transport advantage. Further matched, homology-controlled
experiments are required to assess partner specificity and generalization."

Avoid claims of state-of-the-art accuracy, statistical significance across
independent seeds, biological contact recovery, validated unbalanced-OT
optimization, or universal partner availability on the basis of these artifacts.

## Verification record

**53 regression tests pass**, along with the launcher checks and OT, v5 and v9
smoke checks. See the [validation record](VALIDATION.md) for commands, scope and
the existing input-prerequisite failure.

The [README](../README.md) gives repeatable commands. Regression checks cover
threshold ties, group leakage guards, sparse-plan mass/gradient constraints,
contact deduplication, the correct regularized forward pass, PDB parsing and
mapping, feature validation, complex pairing, bootstrap behavior and statistical
estimands. PowerShell checks verify matched configurations, caller environment
restoration and working-directory handling. Real-data coverage and saved-array
metric checks complement these synthetic tests.

Future runs write `run_provenance.json` and embed the manifest in checkpoints
and the final summary. Prediction archives include dataset SHA-256,
`sample_index`, `residue_index` and `complex_id`; OOF arrays also record the
fold index. Indices are zero-based source positions, not verified PDB residue
numbers. Provenance records trace an experiment; they do not themselves prove
biological label correctness or remove overlap.

The strict prerequisite failure is a finding, not a passing training run.
No full five-fold training, homology-filtered rerun, biological assembly audit,
or final test of a revised model was performed as part of this review.
