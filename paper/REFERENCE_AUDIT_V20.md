# Reference and Manuscript Audit

Rechecked 2026-10-28, following the 2026-09-30 audit. The manuscript contains 25 references: 14 arXiv
records and 11 published records. arXiv metadata and original abstracts or
methods were used for the preprints; publisher metadata, publisher-deposited
Crossref records, and author/institutional copies were used for published
papers. No verified same-paper arXiv record was found for the 11 published
entries, so their journal or conference metadata is retained.

## Reference-to-Claim Mapping

| Reference key | Verified record | Placement and supported statement |
|---|---|---|
| niu2022h2o | [arXiv:2206.13464](https://arxiv.org/abs/2206.13464) | Introduction, Related Work, formulation: target replay plus imperfect-simulator interaction and dynamics-aware correction |
| niu2025h2oplus | [arXiv:2309.12716](https://arxiv.org/abs/2309.12716) | Introduction, Related Work, formulation, mixed-value bootstrap, diagnostic: improved H2O framework across base learners |
| eysenbach2021off | [arXiv:2006.13916](https://arxiv.org/abs/2006.13916) | Related Work and diagnostic: classifier-based conditional dynamics ratio; the source method corrects rewards |
| haarnoja2018soft | [arXiv:1801.01290](https://arxiv.org/abs/1801.01290) | Related Work and training: entropy-regularized off-policy actor-critic |
| haarnoja2018applications | [arXiv:1812.05905](https://arxiv.org/abs/1812.05905) | Related Work and training: automatic entropy-temperature adjustment |
| kumar2020cql | [arXiv:2006.04779](https://arxiv.org/abs/2006.04779) | Related Work: conservative values for offline action-support mismatch |
| kostrikov2022iql | [arXiv:2110.06169](https://arxiv.org/abs/2110.06169) | Related Work and initialization: expectile value and advantage-weighted actor extraction |
| lee2022balanced | [arXiv:2107.00591](https://arxiv.org/abs/2107.00591) | Related Work: prioritized balanced replay and pessimistic ensemble, not an attribution of the present fixed 50/50 split |
| nakamoto2023calql | [arXiv:2303.05479](https://arxiv.org/abs/2303.05479) | Related Work: policy-averaged conservative estimates lower-bounded by a reference-policy value |
| zhang2023pex | [arXiv:2302.00935](https://arxiv.org/abs/2302.00935) | Related Work: retained offline policy combined adaptively with a new policy |
| ball2023efficient | [arXiv:2302.02948](https://arxiv.org/abs/2302.02948) | Related Work: online learning with prior replay from random initialization |
| zhou2025wsrl | [arXiv:2412.07762](https://arxiv.org/abs/2412.07762) | Related Work: frozen-policy warmup followed by online replay without retaining offline data |
| hu2025rgcql | [arXiv:2501.14199](https://arxiv.org/abs/2501.14199) | Related Work: offline-to-online coordination of ride-pooling and public transit, a distinct application |
| wang2021asynchronous | [arXiv:2105.00376](https://arxiv.org/abs/2105.00376) | Related Work: asynchronous multi-agent bus holding, delimiting novelty |
| sutton1999between | [Publisher DOI](https://doi.org/10.1016/S0004-3702(99)00052-1) | Formulation: semi-Markov temporal abstraction, not proof that this compressed observation is Markov |
| rezazada2024review | [Publisher DOI](https://doi.org/10.1080/01441647.2024.2313969) | Introduction and Related Work: demand/supply/control causes of bus bunching |
| daganzo2009headway | [Publisher DOI](https://doi.org/10.1016/j.trb.2009.04.002) | Introduction and Related Work: headway-based holding and stability |
| xuan2011dynamic | [Publisher DOI](https://doi.org/10.1016/j.trb.2011.07.009) | Introduction and Related Work: virtual-schedule arrival deviations and schedule reliability, distinguished from pure headway control |
| delgado2012holding | [Publisher DOI](https://doi.org/10.1016/j.trb.2012.04.005) | Introduction and Related Work: holding/boarding-limit tradeoffs under passenger-time objectives |
| sanchezmartinez2016realtime | [Publisher DOI](https://doi.org/10.1016/j.trb.2015.11.013) | Related Work: holding optimization with dynamic running times and demand |
| andres2017predictive | [Publisher DOI](https://doi.org/10.1016/j.trb.2017.06.013) | Related Work: data-driven headway prediction combined with dynamic holding |
| alesiani2018reinforcement | [Publisher DOI](https://doi.org/10.1109/ITSC.2018.8569473) | Introduction and Related Work: high-frequency bus holding with RL |
| wang2020real | [Publisher DOI](https://doi.org/10.1016/j.trc.2020.102661) | Introduction and Related Work: multi-agent dynamic holding |
| wang2023multiobjective | [Publisher DOI](https://doi.org/10.1016/j.trc.2023.104309) | Introduction and Related Work: multi-objective multiline shared-corridor holding |
| lopez2018microscopic | [DLR original paper](https://elib.dlr.de/127994/1/08569938.pdf) | Formulation: SUMO microsimulation, not support for this study's training or performance claim |

## Corrections

Corrected author names/order in Off-Dynamics RL, Cal-QL, and PEX; completed
the SAC follow-up author list and preserved accented surnames. The SUMO
author list follows the original paper rather than inconsistent deposited
metadata. Ceder's parenthetical nickname is omitted to avoid a spurious second
initial in the formatted bibliography. Preprint years now follow first arXiv submission; internal citation
keys are identifiers and may retain an earlier publication-year suffix.
No unrelated reference was retained as support for the final effect.

The method now reports the positive branch's median, the actual linear
terminal taper, preceding freeze/ramp, fixed actor interpolation, reward
conditioning, mixed entropy bootstrap, and critic target. Statistical symbols
distinguish the completion margin, point mass, entropy temperature, and test
level. The logged-runtime kernel is distinguished from the formal-evaluation
kernel, and the conditional ratio is assigned to the former. Tables use
explicitly defined paired differences. Appendix A supplies
two elementary proofs called from the formulation and actor sections;
Appendix B's implementation, diagnostics, support, and seed effects are linked
from the corresponding main sections. Limitations are collected in Discussion.
The final build has no missing citation, duplicate label, or unreferenced
numbered item: 25 references, 26 equations, 7 tables, 1 figure, and 2 propositions.

## 2026-10-28 Recheck

All 25 entries match the primary records for identity, title, authors, year,
and identifier. No bibliography entry required replacement. The SUMO author
order is retained from the original paper and the
[official SUMO citation guidance](https://sumo.dlr.de/userdoc/Publications.html),
which differ from Crossref's deposited order. First-submission years for arXiv
entries remain intentional; internal key suffixes do not determine citation years.

The recheck clarified Cal-QL's reference-value lower bound and placed the H2O+
attribution directly beside the mixed-value bootstrap. The action definition
now states additional holding with a fixed speed ratio. The actor description
distinguishes the ideal continuous distribution from the inward endpoint
projection used for logged 60-s actions; the added likelihood-input and machine-
epsilon symbols are defined and entered in the notation ledger. The primary
table now states that differences are computed before rounding its displayed means.
The PDF check also corrected repeated appendix prefixes and the appendix plural
in cross-references; subsection numbers now appear as A.1/B.1, with the appendix
name supplied once by the reference formatter.

The effective bibliography contains all 25 cited entries (42 citation occurrences).
All 26 numbered equations, 7 tables, 1 figure, and 2 propositions are referenced;
the two appendix propositions are explicitly cited from the main text. No missing
citation, unused entry, duplicate label, missing cross-reference, or unresolved
PDF placeholder was found. The 35-page PDF rebuild has no undefined-reference
or overfull-box warning. Scientific source files and result records were not changed.

## Remaining Scope

The evidence supports the fixed candidate-parent comparison in one simulator
scenario family. It does not establish Markov sufficiency of the observation,
convergence, a learned density-ratio benefit, or field effectiveness. Author,
funding, competing-interest, and data-sharing declarations still require
author confirmation, as recorded in SUBMISSION_READINESS_V20.md.
