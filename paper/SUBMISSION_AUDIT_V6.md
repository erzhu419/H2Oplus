# Submission Audit (v6)

Last updated: 2026-08-09

## Evidence lock

- Canonical offline data: `transit_smdp_v6_formal_v1.h5`
- Rows / episodes: 2,831,150 / 210
- Data SHA-256: `da6b5ebd5dfd4ce582907d5bb009c6741b385e9d7ac75bfcbe310754abced709`
- Simulator runtime SHA-256: `c1f7bc434a1fd2c0aaf32b7efe0db47e3d9292f8a3b9cccaa9d0e9934664675a`
- Simulator aggregate SHA-256: `793910a1327a3b178a9d426f673936983547bdc4fee38ab8b03fd1739973304e`
- Journey artifact SHA-256: `1c98250932a41fb854a23736dd6e2f3be21a476751f052b3ba8e522824e4eb52`
- Formal fidelity audit: `fidelity_audit_v6_formal_v6.json`
- Fidelity-audit SHA-256: `ba278d2d3d5f353317d6acebc78399e98526bda09b63f03627dcc6e32d0bbc93`
- Frozen runtime package: `v6_gate_runtime_v19` (423 manifest entries)
- Runtime aggregate SHA-256: `cec314293e0f1289f467a6e6ab22b8d4fdb7554a92780cf5e338cea30855594d`
- All-service control audit: `control_scope_audit_v6_formal_v1.json`
- Control-scope-audit SHA-256: `eb89db516fde23dd654066009f14b3bd11d5a59151b2bac303330a5460501518`
- Current pre-result PDF: 29 pages; SHA-256 `65bae8e5c7a85148f5171c7080d664ab338e4a36f74aa2ba6e8afbf0f37042db`

The control-scope audit scanned every action-bearing row rather than a sample.
All 2,831,150 rows have an active current slot and matching raw/canonical line
encoding. The per-service transition counts are: 102S 277,201; 102X 234,290;
122S 230,923; 122X 229,672; 311S 236,955; 311X 226,512; 406S 219,893;
406X 226,205; 705S 237,441; 705X 203,648; 7S 260,943; and 7X 247,467.
The 389 scheduled trip identifiers are globally unique. Thus Line 7X is only a
visual reference in the topology figure; the shared policy controls eligible
events on all 12 directional services.

## Formal-plan lock

| Partition | Plan SHA-256 |
|---|---|
| Local H2O+ core | `950ba8c812f98d0b289b1b1a9ac974701f7d562be4b363796fe8026f1694fcf5` |
| Server core baselines | `434aad14471a72a196c90c890f94d98452f86cd438aea4f944a07fc638f4e396` |
| Local H2O+ ablations | `9e4e749c6715bff7cdf8441578ff99b75e464c8cdcb38edc6b135e3e2481b857` |
| Local H2O+ coefficient sensitivity | `92565d535167f943c7338eaded9ea227b99446475622a58e703d165d3655cf8b` |
| Server ratio-SAC sensitivity | `9077e76e3bf0523306232e80e0deadeb4d7af761fd034ce4eb26acaa04031cd7` |
| Server core dispatch | `6470d35ca57a45e5626dfba27daa06e6c122330d04584a48d8af25dd4cb22b8b` |

The signed core dispatch contains exactly 35 server jobs. H2O+ and its direct
ablations/sensitivity profiles are assigned to the local GPU. RLPD, WSRL, SAC,
offline controls, and ratio-SAC are assigned to GPU servers. No server baseline
reads a legacy snapshot corpus; no pre-v6 checkpoint can satisfy a v19 plan.

## Experiment readiness

| Gate | State | Evidence / blocker |
|---|---|---|
| Current test suite | PASS | 208 tests passed; three import-time SWIG deprecation warnings only. |
| Runtime verification | PASS | Exact v19 environment verified locally and on `jtl110gpu` and `jtl110gpu2`. |
| Local H2O+ GPU gate | PASS | 8/8 runs; independent audit has no errors or warnings. SHA-256 `e1a43c3b78e6294c57a2c67a5cb9677de2d6a23977e26eaf9f43537940f3a2d7`. |
| Server baseline GPU gate | PASS | 8/8 runs; independent audit has no errors or warnings. SHA-256 `8d35de4b63cb46e997123b27aedd10ba0b3435e411593b36ea0b0bc321ca5d80`. |
| Formal local training | RUNNING | `h2oplus-v19-formal-local.service`; H2O+ core, ablation, and coefficient plans run serially on the local GPU. |
| Formal server core | RUNNING | `h2oplus-v19-formal-core.service` on both GPU servers; RLPD alone permits two jobs per GPU. |
| Formal ratio-SAC sensitivity | QUEUED AFTER CORE | Server-side plan is hash-locked and must not overlap the core dispatch. |
| Formal SUMO evaluation | BLOCKED BY TRAINING | Must use all disjoint registered cells, common random numbers, final checkpoints, and invariant rejection. Before launch, freeze and verify an evaluation-only runtime refresh that removes `sim_obj/signal.py` standard-library shadowing; the running v19 training package remains immutable. |
| Legacy cleanup | NOT AUTHORIZED YET | Produce a dry-run deletion manifest only after final v6 results replace old artifacts and the user confirms it. |

The displayed `offline update .../10000` denominator in run logs is the current
checkpoint chunk, not the total stage budget. The durable
`offline_updates_completed` counter advances across chunks until the signed
50,000- or 250,000-update budget is reached.

## Manuscript state

| Component | State | Evidence / remaining work |
|---|---|---|
| Title, abstract, introduction | V6 PROTOCOL DRAFT | Contains no pre-v6 outcome claim; final effect estimates remain withheld. |
| Related work and bibliography | VERIFIED | 14/14 cited records checked against primary proceedings, publisher records, DOI/Crossref, and source papers; see `REFERENCE_AUDIT_V6.md`. |
| Problem and method | V6 REBUILT | Chronological SMDP, exact welfare, hurdle action, IQL stage, conditional duration ratio, and executable schedule are stated. |
| Experimental design | V6 REBUILT | Data split, fidelity thresholds, baselines, event cap, statistics, placement, and provenance are fixed. |
| Results, discussion, conclusion | OUTCOMES PENDING | No earlier checkpoint or table is admissible; update only after complete formal evaluation. |
| Symbol and acronym audit | PASS PRE-RESULT | The canonical ledger has no duplicate symbol or duplicate meaning; all manuscript math atoms are registered, and all 23 tracked abbreviations expand exactly once at first use. |
| Figures and PDF | PASS PRE-RESULT | 14 citations, 49 labels, and 26 referenced numbered objects pass the structural audit. Clean compilation has no undefined reference, overflow, or underflow warning; all 29 rendered pages were inspected, all fonts are embedded, and no Type 3 font is present. The two BibTeX warnings are the verified page-less ICLR 2021/2022 records. |

## Explicitly excluded evidence

- Any 16-dimensional, hold-v3, pre-v6, or `offline-sumo` checkpoint/output.
- Any result using a horizon other than 24,000 s or an objective other than the
  exact non-overlapping generalized passenger-time ledger.
- Any result using a two-dimensional action, snapshot-reset online simulator,
  headway-deviation reward, or legacy H2O+ components.
- Any service-invalid or accounting-invalid evaluation, even when the process
  exits successfully.
- Any post hoc selection of a behavior gain, training seed, checkpoint, demand
  condition, or evaluation cell using target outcomes.

## Final release gates

1. Require every signed training job to have a valid summary, final checkpoint
   hash, exact event/update counts, and accepted resume history.
2. Evaluate only the declared final checkpoint on all 30 held-out SUMO cells.
3. Reject outputs failing chronology, passenger-ledger, terminal, vehicle-trip,
   or provenance invariants before aggregation.
4. Produce hierarchical intervals, paired block tests, Holm corrections, ratio
   diagnostics, phase timings, utilization, and peak-memory summaries.
5. Insert results, limitations, tables, and figures only from the accepted
   formal matrix; then repeat citation-placement and claim-support review.
6. Compile from a clean auxiliary state and inspect all warnings, references,
   equations, tables, figures, captions, glyphs, and every rendered PDF page.
7. Archive final data, source, runtime, plan, result, checkpoint, and PDF hashes.
8. Only then generate a legacy-deletion dry-run manifest and seek confirmation.
