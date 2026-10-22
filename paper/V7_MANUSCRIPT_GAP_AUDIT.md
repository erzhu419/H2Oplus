# v7 Manuscript Gap Audit

Status: post-stable-rewrite audit. The frozen dataset, simulator contract,
reference audit, notation rewrite, and structural manuscript audit are now
closed. The semantic scanner retains exactly six findings, all caused by three
trainer-dependent decisions that still require fresh GPU gates: reward
conditioning and critic regression, hybrid-policy entropy targeting, and the
resulting full-dataset precision admission. This document is not a results
record, and no pending quantity below may be copied into the manuscript until
its named freeze or audit artifact exists.

## Authority order

When two sources disagree, use this order:

1. frozen v7 dataset, admission, training-plan, evaluation, and statistical
   artifacts, each identified by SHA-256;
2. executable v7 contracts and audited training code;
3. this gap audit and the v7 terminology ledger;
4. manuscript prose, tables, captions, and figures;
5. superseded v6 artifacts and historical notes.

Passing LaTeX compilation or `audit_manuscript.py` is not scientific evidence.
The current structural audit cannot detect a numerically well-formed but stale
claim. Run `audit_manuscript_v7.py`; its current machine-readable baseline is
`v7_manuscript_semantic_audit.json` and is expected to fail until the artifact-
backed rewrite is complete.

## Confirmed v7 contract

The following stable facts are fixed by executable code and hash-identified v7
artifacts. Trainer-dependent candidate values are listed explicitly as pending
and are not yet manuscript facts.

| Item | v7 fact | Authoritative implementation or artifact |
|---|---|---|
| Raw event observation | 16 physical features | `bus_h2o/mdp_contract_v5.py`, imported by `bus_h2o/mdp_contract_v7.py` |
| Local event observation | 13 features after removing time and two system-wide passenger counts | `LOCAL_EVENT_STATE_NAMES` in `bus_h2o/mdp_contract_v7.py` |
| Raw packed state | 245 dimensions | `RAW_STATE_DIM` in `bus_h2o/mdp_contract_v7.py` |
| Canonical state | 229 dimensions | `CANONICAL_STATE_DIM` in `bus_h2o/mdp_contract_v7.py` |
| Canonical batch prefix | 5 values | `CANONICAL_PREFIX_NAMES` in `bus_h2o/mdp_contract_v7.py` |
| Event slots | 16 slots, 14 values per slot | `MAX_SIMULTANEOUS_DECISIONS` and `CANONICAL_SLOT_WIDTH` |
| Dataset reward | negative generalized passenger time in physical passenger-seconds; data reward unit is 1 | `TransitMDPConfigV7.reward_unit_passenger_seconds` |
| Training reward scale | pending fresh gate; `1e-4` is the registered candidate | reward-conditioning gate on the frozen pilot split, followed by full-dataset admission |
| Critic regression | pending fresh gate; Huber loss with delta 1 is the registered candidate | reward-conditioning gate on the frozen pilot split, followed by full-dataset admission |
| Hybrid-policy entropy target | pending fresh gate; 0 is the registered candidate | hybrid-entropy gate bound cryptographically to the selected reward/loss artifact |
| Episode horizon | 24,000 physical seconds | `TransitMDPConfigV7.horizon_seconds` |
| Maximum hold | 60 physical seconds | `TransitMDPConfigV7.max_hold_seconds` |
| Offline policy set | zero hold, two random hurdle policies, four headway-gain policies, and one myopic-welfare policy | `training_environment_design_v7_formal_v1.json` plus the two collection matrices |
| Frozen complete episodes | 240: 30 zero-hold and 210 nonzero-behavior episodes | audited formal-v2 HDF5 |
| Frozen replay rows | 3,284,718 | audited formal-v2 HDF5 |
| Frozen HDF5 identity | 589,504,302 bytes; SHA-256 `6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899` | `bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5` |
| Audited edge cases | 2,145 exact upper-endpoint actions; 1,025,206 zero-duration rows; 80.2246% exact-zero actions | formal-v2 dataset audit |
| Intervention fidelity | 240 matched SUMO and 240 simulator episodes; all 21 nonzero policy--demand strata pass point and bootstrap-interval thresholds for eight metrics | `bus_h2o/calibrated_env/intervention_fidelity_audit_v7_formal_v3.json` |
| Resource boundary | Every canonical CUDA job runs only on registered GPU servers; H2O+ and non-H2O+ jobs retain separate server execution classes | user-fixed execution boundary and v7 training-plan builder |
| Server payload | compact, hash-locked v7 HDF5 plus code/config only; no legacy snapshot corpus | `deployment_bundle_v7.py` |

The cache data type, admitted process concurrency, formal training run count,
checkpoint digests, effect estimates, confidence intervals, and significance
tests are deliberately not asserted here.

## Runtime and placement controls

- The existing `offline-sumo` WSRL and RLPD runs establish only that the two
  server lanes contain a usable environment and implementation. Their 16- or
  17-dimensional legacy datasets are ineligible as v7 evidence. Every formal
  server job must consume the same admitted 229-dimensional v7 HDF5 as the
  server H2O+ jobs.
- Direct inspection of the retained `revision_hold_v3_20260718` artifacts
  closes the reuse question: their HDF5 has 3,374,285 rows with 16-dimensional
  observations (SHA-256 `f0d9613798b82f868f56c68981fe201a2ce6d9e6d2308ae01ee0c6db88566c49`),
  and the RLPD/WSRL actor input layers implement that legacy contract. Those
  checkpoints cannot initialize, resume, or substitute for a 229-dimensional
  v7 run.
- The 200 GB legacy snapshot corpus belongs only to the historical H2O+
  construction workflow. The approved cleanup deleted 30 superseded files
  occupying 129,782,956,032 bytes while re-verifying the canonical formal-v7
  HDF5. The deletion is recorded in
  `reproducibility/cleanup_manifests/legacy_snapshot_cleanup_20260810_receipt.json`.
  No deleted file may be restored, described as a formal-v7 input, or copied to
  a GPU server for H2O+, WSRL, RLPD, ratio-SAC, SAC, CQL, IQL, or BC.
- Historical v6 performance gates show that simulator rollout was not the
  dominant H2O+ gate cost: 64 decision events took about one second. The first
  compiled learner block took 13--23 seconds, whereas later 16-update blocks
  took about 0.6--0.9 seconds. These are diagnostic observations, not formal v7
  timing results.
- A formal epoch contains 500 decision events and 500 H2O+ updates, internally
  interleaved in at most 16-event chunks. This update timing is part of the
  learning protocol and cannot be enlarged merely to improve utilization
  without a preregistered algorithmic justification.
- Server concurrency is not fixed by an OOM guess. On the frozen full HDF5,
  compare one-process and concurrent end-to-end throughput for the admitted
  cache precision. Admit a method-specific concurrency above one only when all
  workers complete, the configured free-memory reserve remains, and aggregate
  update throughput improves by the preregistered margin.
- The one dispatch lane assigned two simultaneous RLPD seeds requires an exact
  host/GPU two-process capacity gate. Other server lanes run one RLPD seed, and
  all other server methods remain one process per GPU unless an equally
  explicit method-specific capacity artifact passes.
- GPU-utilization samples are valid only when their process identities are
  attributable to the audited workers. Historical local samples collected
  while unrelated GPU jobs were active cannot support utilization claims.
- The portable v4 trainer-gate bundle locks Python 3.8.20 and PyTorch 2.4.1 with
  CUDA 12.4 together with Python, Torch-extension, CUDA-library, compiler,
  package, and compatibility-header hashes. The current formal gate is running
  on `jtl110gpu2`; no other server lane is admitted merely because it has a
  similarly named environment. Admission, matrix launch, and deployment
  verification must fail closed on any runtime drift. Server launch also pins
  the system `libcurl.so.4` preload required by h5py on the older glibc host.
- The retained server throughput probes are diagnostic only. Five measured
  source files differ from the current tree, and the WSRL difference includes
  the now-correct target subset sampling with replacement, two-head CQL-penalty
  subsampling, offline pre-update actor/critic semantics, and next-state
  temperature batches. Current-source throughput and the RLPD two-process
  capacity decision must be remeasured after the trainer admission freezes.
- The official WSRL batch of 1,024 belongs only to online high-UTD fine-tuning.
  Its preceding 250,000-step CQL checkpoint is trained with batch size 256 via
  the separate locomotion CQL launcher. Any runtime or memory projection that
  applies 1,024 to the offline stage is invalid.

## Rewrite disposition

| Status | Location | Previous mismatch | Current disposition |
|---|---|---|---|
| Closed | Abstract, introduction, problem, experiments, appendix | 2,831,150 rows, 210 episodes, seven policies | Replaced by the audited 3,284,718 rows, 240 episodes, and eight behavior policies. |
| Closed | Problem, experiments, appendix, terminology ledger | 275-dimensional canonical state | Replaced by 229 dimensions with one 5-value prefix and 16 slots of 14 values. |
| Closed | Problem and terminology ledger | each local vector is 16-dimensional | Raw 16-feature and deduplicated 13-feature vectors are now distinguished. |
| Partial: gate pending | Problem, method, appendix | physical reward unit conflated with a `1e6` divisor or unscaled training reward | Stored reward is now physical passenger-seconds. The trainer multiplier remains withheld until the fresh reward/loss gate passes. |
| Pending | Method and appendix | target entropy is -1 | The old value remains one of the six semantic findings. Replace it only with the fresh entropy-gate selection. |
| Pending | Method | critic objective is a squared residual | The old objective remains one of the six semantic findings. Replace it only with the fresh reward/loss-gate selection. |
| Closed | Method and appendix | three-value AEC prefix | Replaced by the five canonical prefix values. |
| Pending | Experiments and appendix | cache data type stated before admission | The data type is withheld until full-HDF5 precision and capacity gates pass. |
| Closed | All sections | `v6` equation/algorithm labels and formal-v6 status prose | Stable labels and protocol prose use v7 consistently. |
| Pending | Experiments | formal training implied complete | Results and run-specific quantities remain explicitly withheld until admitted formal runs finish. |
| Closed | Related work and experiment baselines | incomplete or semantically misplaced literature | All 23 active records and their placements are source-verified in `REFERENCE_AUDIT_V7.md`. |
| Closed | Abstract, introduction, and simulator acceptance audit | only zero-policy fidelity was visible although intervention-response evidence had passed | Added the 240-pair intervention audit and a worst-stratum table; `audit_manuscript_v7.py` now binds every displayed value to the frozen artifact. |
| Closed | Discussion | seven behavior policies and unaudited support | Updated to eight policies and frozen action-support statistics. |
| Pending | Result figures and tables | generated results may encode v6 or incomplete runs | Regenerate result assets from frozen v7 aggregates; never relabel stale pixels. |

## Rewrite controls

### Citations

- Every bibliography record must be verified against a publisher, proceedings,
  DOI registry, or the paper itself.
- Every citation site must be checked for entailment: the source must support
  the exact adjacent clause, not merely the broad topic.
- Related work, method provenance, and baseline implementation citations are
  separate obligations. One citation cannot silently serve all three roles.
- Unused records and records relevant only to a rejected method must not enter
  the final bibliography.
- The implemented 23-record bibliography and every registered placement are
  audited in `REFERENCE_AUDIT_V7.md`. Future records return to quarantine until
  their exact target sentence and primary-source support are both audited.

### Symbols

- One ledger row must contain exactly one canonical symbol and one meaning.
- Tuple components, parameter pairs, clipping bounds, and operators each need
  separate rows; grouped rows in `TERMINOLOGY_LEDGER_V6.md` are not acceptable.
- A symbol is introduced in prose immediately before or after first use. A
  definition only in the appendix does not license an earlier occurrence.
- A physical quantity, normalized feature, learned parameter, estimator, loss,
  and hyperparameter must never share a symbol.
- The raw data reward, scaled training reward, physical hold duration, target
  update coefficient, replay mixture, and bootstrap mixture require distinct
  symbols.

### Acronyms

- Audit document order after TeX expansion: abstract, body, captions, tables,
  algorithms, appendix, and declarations.
- The first prose occurrence must be `full name (ACRONYM)`; subsequent prose
  uses the acronym only.
- Title and keyword metadata do not satisfy the first-prose-use requirement.
- Official capitalization and expansions must be preserved, especially SUMO,
  RLPD, WSRL, HDF5, and SHA-256.

### Numbered objects

- Every equation, figure, table, and algorithm label must be unique and cited
  in the body.
- Every body reference must resolve to the intended object after final ordering.
- Figure and table captions must be understandable without relying on a stale
  v6 section number or filename.
- Numeric claims in captions and tables must be generated from the same frozen
  aggregate used by the prose.

## Release sequence

1. **Complete:** freeze both v7 collection matrices.
2. **Complete:** assemble and independently audit the full v7 HDF5 and both
   zero-action and intervention-fidelity suites.
3. **In progress:** serialized-credit, zero-policy fidelity, and intervention
   fidelity are complete. The portable v4 reward/loss, hybrid-entropy, and
   device-precision chain is running; the method-protocol freeze and RLPD
   capacity gate follow only after that chain passes.
4. **In progress:** the server-H2O+/server-non-H2O+ boundary and deployment
   validators are implemented; freeze final plans and manifests only after the
   remaining trainer and capacity gates pass.
5. Complete formal training and evaluation; freeze checkpoints and aggregates.
6. **Stable portion complete:** rewrite the paper against the frozen data,
   simulator, reference, and notation artifacts. Trainer selections and formal
   results remain withheld.
7. **Baseline complete:** structural, bibliography, acronym, symbol,
   LaTeX-warning, and rendered-PDF audits have a clean pre-gate baseline. Re-run
   all of them after the six trainer-dependent semantic findings and final
   result assets are resolved.
8. **Complete for the approved legacy-data scope:** the dry-run manifest was
   approved, 30 superseded files were deleted, and the canonical formal-v7 HDF5
   was re-verified. Any future cleanup requires a new manifest and approval.
