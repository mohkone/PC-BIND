# Infrastructure freeze revision 2, before acceptance or training

The first implementation freeze passed its full 211-test suite (210 passed;
one pre-existing Windows symlink capability skip), all six real-data dry runs,
and forced process/validation failure simulations. It was never accepted for
execution and no model was fitted. Its manifest, ten archived source copies,
test log, dry-run evidence and fault-injection evidence are preserved verbatim.

Final infrastructure review required two changes before acceptance:

- Bind the prospective final comparison JSON directly to the new acceptance
  record, and verify that record again after analysis.
- Require a non-dry trainer to prove ownership by the one authorized serial
  queue: matching lock/state/freeze/authorization/arm/command and the recorded
  trainer child PID. Account for the Windows virtual-environment launcher
  parent without allowing an unrelated standalone process.

Revision 2 changes only these execution/provenance gates, their regression
tests, and source-freeze/archive routing. The first source archive remains
immutable. The current ten new implementation/test files are archived again
under `execution_sources_v2`; `execution_manifest_v2.json` is the candidate
execution freeze. Repeat the full suite, all six real-data dry runs and both
forced-stop simulations against that freeze before writing its acceptance.

No accepted historical v2/v3 source or archive is changed. The preregistration,
protocol, cohort/data/fold hashes, runtime, recipe, seeds, output destinations,
once-per-run seeding, mathematical implementation and registered analysis are
unchanged. This amendment uses no fitted ablation outcome or external-test
evaluation. Training remains unstarted and a later explicit user launch
instruction remains required after concrete acceptance evidence exists.
