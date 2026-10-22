# Claim-Evidence Map

This map defines the manuscript's admissible claim boundary. It is a writing
aid, not an additional analysis.

## Confirmatory Claim

The fixed reserve-augmented candidate is superior to its fixed direct behavior
parent on generalized passenger seconds per departed passenger while satisfying
the preregistered completion-rate noninferiority condition.

- Evidence: `reproducibility/h2oplus_v20_formal_confirmation_results/formal_confirmation_analysis_v1.json`
- Independent analysis: `reproducibility/h2oplus_v20_formal_confirmation_results/formal_confirmation_analysis_verification_v1.json`
- Evidence closeout: `reproducibility/h2oplus_v20_formal_confirmation_results/evaluation_closeout_v1.json`
- Independent closeout: `reproducibility/h2oplus_v20_formal_confirmation_results/evaluation_closeout_verification_v1.json`
- Fixed candidate: `h2oplus-v19-completion-safety-quarter-load-sixty`
- Fixed direct parent: `h2oplus-v19-completion-safety-baseline`
- Formal units: 10 fresh environment blocks, 3 demand levels, 5 actor seeds,
  45 physical behaviors, and 1,350 endpoint episodes.
- Candidate-minus-parent means: -92.12 generalized passenger seconds per
  departed passenger, +0.00466 completion, and -77.22 unfinished passengers.
- Inference: 10/10 favorable primary blocks and 10/10 completion
  noninferiority blocks; one-sided exact sign-test value 0.0009766 for each
  component; the preregistered intersection-union test passed.

## Supporting Evidence

- The effect direction is favorable at all three demand levels and all five
  actor seeds.
- Descriptive lineage-control comparisons are supportive only. They do not
  enlarge the confirmatory claim because those policies are not the direct
  behavior parent.
- Artifact integrity and parent training stability passed their registered
  checks.

## Mechanistic Interpretation

The direct comparison isolates the final deterministic completion-safety
override because candidate and parent share actor tensors, checkpoints, state,
and inherited transforms. It does not isolate the full H2O+ training pipeline
from all alternatives.

The effect is not attributed to density-ratio correction. Held-out ratio
classification did not generalize, and the sealed actor training used unit
ratio weights.

## Required Limitations

- Both the target and online environments are simulators; no live-network
  effect is established.
- Candidate design used v14-v20 development outcomes before the fresh formal
  environment was opened.
- Training and formal evaluation used different SUMO versions.
- The replay has 3.28 million transitions but only 30 independent exogenous
  cells, sparse positive-action support, and a dominant exact-zero action.
- The final 60-second reserve reaches beyond dense behavior support in its
  guarded state region.
- Evidence is endpoint-only; no area-under-learning-curve claim is admissible.

## Manuscript Locations

- Main result and scope: abstract and Introduction.
- Exact candidate-parent intervention: Method.
- Design, estimand, tests, and results: Experiments.
- Interpretation and limitations: Discussion.
- Evidence package and diagnostics: Appendix.

