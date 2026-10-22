# Verified Reference Candidates for the v7 Manuscript

Status: implemented source-verified insertion plan. All nine records are now in
the active bibliography at the bounded claim locations below. The complete
23-entry result and release checks are recorded in `REFERENCE_AUDIT_V7.md`.

Metadata was checked against the publisher or conference record and Crossref
on 2026-08-09. A candidate is retained here only when it fills a specific
literature gap and has a bounded semantic role in the manuscript.

## Recommended additions

| Proposed key | Verified metadata | Intended manuscript location and supported claim | Semantic boundary |
|---|---|---|---|
| `lee2022balanced` | Seunghyun Lee, Younggyo Seo, Kimin Lee, Pieter Abbeel, and Jinwoo Shin. "Offline-to-Online Reinforcement Learning via Balanced Replay and Pessimistic Q-Ensemble." *Proceedings of the 5th Conference on Robot Learning*, PMLR 164:1702--1712, 2022. [PMLR](https://proceedings.mlr.press/v164/lee22d.html) | Related Work, before RLPD/WSRL: offline-to-online fine-tuning can destroy a useful offline policy through state-action shift and bootstrap error; balanced replay and pessimistic ensembles are one stabilization route. | Does not study a dynamics gap, transit control, or the present chronological SMDP contract. It must not be cited for those claims. |
| `nakamoto2023calql` | Mitsuhiko Nakamoto, Simon Zhai, Anikait Singh, Max Sobol Mark, Yi Ma, Chelsea Finn, Aviral Kumar, and Sergey Levine. "Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning." *Advances in Neural Information Processing Systems* 36:62244--62269, 2023. DOI [10.52202/075280-2719](https://doi.org/10.52202/075280-2719). [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/c44a04289beaf0a7d968a94066a1d696-Abstract-Conference.html) | Related Work: calibrated conservative values provide another route to stable online fine-tuning. Experiments: cite when disclosing that the WSRL paper's main study uses Cal-QL, whereas the registered transit variant uses its paper-supported CQL initialization. | Must not be presented as the initializer actually used by the registered CQL-initialized WSRL jobs. |
| `zhang2023pex` | Haichao Zhang, Wei Xu, and Haonan Yu. "Policy Expansion for Bridging Offline-to-Online Reinforcement Learning." *International Conference on Learning Representations*, 2023. [OpenReview](https://openreview.net/forum?id=-Y34L45JR6z). | Related Work: retaining the offline policy as one candidate while learning an expansion policy is a distinct way to avoid destructive early fine-tuning. | PEX is contextual literature, not an experimental baseline here; no DOI or page range should be invented. |
| `sutton1999between` | Richard S. Sutton, Doina Precup, and Satinder Singh. "Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning." *Artificial Intelligence* 112(1--2):181--211, 1999. DOI [10.1016/S0004-3702(99)00052-1](https://doi.org/10.1016/S0004-3702(99)00052-1). | Problem Formulation, first SMDP definition: supports the semi-Markov foundation for actions or decisions with non-unit, variable duration. | Does not support the manuscript's serialized simultaneous-event construction, passenger ledger, or v7 discount convention; those are study-specific definitions. |
| `rezazada2024review` | Mustafa Rezazada, Neema Nassir, Egemen Tanin, and Avishai (Avi) Ceder. "Bus bunching: a comprehensive review from demand, supply, and decision-making perspectives." *Transport Reviews* 44(4):766--790, 2024. DOI [10.1080/01441647.2024.2313969](https://doi.org/10.1080/01441647.2024.2313969). | Introduction and Bus Holding Related Work: supports the broad bunching mechanism, control taxonomy, data sources, and the need to balance passenger and operator measures. | A review is not evidence that the present simulator is faithful or that the proposed method is effective. |
| `delgado2012holding` | Felipe Delgado, Juan Carlos Munoz, and Ricardo Giesen. "How much can holding and/or limiting boarding improve transit performance?" *Transportation Research Part B: Methodological* 46(9):1202--1217, 2012. DOI [10.1016/j.trb.2012.04.005](https://doi.org/10.1016/j.trb.2012.04.005). | Introduction and Bus Holding Related Work: supports the passenger-performance trade-off created by holding and the comparison between holding alone and holding with boarding limits. | The present action is holding only. The citation must not imply that boarding limits are implemented. |
| `sanchezmartinez2016realtime` | G. E. Sanchez-Martinez, H. N. Koutsopoulos, and N. H. M. Wilson. "Real-time holding control for high-frequency transit with dynamics." *Transportation Research Part B: Methodological* 83:1--19, 2016. DOI [10.1016/j.trb.2015.11.013](https://doi.org/10.1016/j.trb.2015.11.013). | Bus Holding Related Work: supports real-time holding with time-varying running times and demand, including the waiting-time versus onboard-time trade-off. | It is an optimization study, not offline-to-online RL and not evidence for the present learned transition ratio. |
| `andres2017predictive` | Matthias Andres and Rahul Nair. "A predictive-control framework to address bus bunching." *Transportation Research Part B: Methodological* 104:123--148, 2017. DOI [10.1016/j.trb.2017.06.013](https://doi.org/10.1016/j.trb.2017.06.013). | Bus Holding Related Work: supports data-driven headway prediction coupled to dynamic holding and helps distinguish predictive control from the present model-free policy. | The present method does not predict future headways and must not be described as predictive control. |
| `hu2025rgcql` | Yulong Hu, Tingting Dong, and Sen Li. "Coordinating ride-pooling with public transit using Reward-Guided Conservative Q-Learning: An offline training and online fine-tuning reinforcement learning framework." *Transportation Research Part C: Emerging Technologies* 174:105051, 2025. DOI [10.1016/j.trc.2025.105051](https://doi.org/10.1016/j.trc.2025.105051). | Related Work, offline-to-online RL: establishes that offline pretraining and online fine-tuning have been applied in a public-transport-adjacent ride-pooling coordination problem. It should narrow any broad claim that offline-to-online RL is unresolved for all transit applications. | The task is multimodal ride-pooling coordination, not fixed-route bus holding; it does not study a target/online-simulator dynamics gap or an SMDP holding action. |

## Planned insertion logic

1. Open the offline-to-online subsection with the generic failure modes and the
   balanced-replay, calibrated-value, and policy-expansion solution families.
   Then introduce RLPD and WSRL as the two registered source-aligned controls.
2. Cite the transportation RG-CQL study before narrowing the open problem to
   cross-fidelity, asynchronous fixed-route bus holding. Do not claim that
   offline-to-online RL is new to transportation or public transit generally.
3. Cite Sutton et al. at the first formal SMDP definition, not beside the
   study-specific serialization or reward equations.
4. Use the bus-bunching review to frame the field, then assign Delgado et al.,
   Sanchez-Martinez et al., and Andres and Nair to their specific operational
   mechanisms. Keep the existing analytical and RL holding citations.
5. Cite Cal-QL wherever the manuscript contrasts the WSRL paper's main
   initializer with the registered CQL-initialized variant.

## Deliberate exclusions

- AWAC is not needed once the directly relevant fine-tuning routes above are
  cited; adding an arXiv-only record without an experimental or argumentative
  role would widen the bibliography without supporting a new claim.
- Action-free offline-to-online RL, autonomous-driving fine-tuning, and generic
  sim-to-sim robotics papers address different data or deployment contracts and
  are not suitable support for the bus-holding formulation.
- The 2026 multi-line bus scheduling work concerns offline planning followed by
  online scheduling decisions, not offline-to-online RL fine-tuning of a
  holding policy; it should not be used to support the present O2O claim.
- No 2026 graph-aware or multi-line holding paper will be added until its final
  publisher metadata and exact semantic contribution are checked against a
  claim that survives the formal v7 results.

## Release gate

After insertion, rerun cited-key equality, duplicate-work, BibTeX warning,
first-use acronym, symbol-ledger, and citation-placement audits. Every added
entry must be cited, and no adjacent citation may be reused to support a claim
outside the boundary stated above.
