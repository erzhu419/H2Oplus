# Reference and Citation Audit (v7)

Audit scope: all 23 records in `references.bib`, every citation command in the
expanded manuscript, bibliographic metadata, citation placement, and the exact
claim supported at each placement. Metadata was rechecked against publisher or
conference records and DOI/Crossref metadata on 2026-08-10.

## Release result

- Bibliography entries: 23.
- Distinct cited keys: 23.
- Verified entries: 23.
- Uncited entries, undefined keys, duplicate keys, and duplicate works: 0.
- Missing registered citation placements: 0.
- Structurally misplaced citations: 0 after the v7 rewrite.
- Biber datamodel validation: passed for all 23 entries.
- DOI/Crossref comparison: 12 exact records and three publisher-resolved
  metadata exceptions among the 15 DOI-bearing entries.
- LaTeX build result: successful, with no undefined citations or references.
- Authoritative bibliography SHA-256:
  87461f3391a11cf1e5cc40e6787a9462ec91699dc11725907c73b0fa98456e82.
- Reviewed citation-paragraph SHA-256:
  8f1775e56d7c0b1c4c21a0169b83ff92921fc0ff05f3a43f72846ae25afe333d.

The three Crossref exceptions are not bibliography errors. Crossref folds the
NeurIPS volume into the container title for niu2022h2o and
nakamoto2023calql, whereas the official NeurIPS records identify volumes 35
and 36 explicitly. For lopez2018microscopic, Crossref returns a different
author order; the SUMO-maintained recommended citation and the conference paper
itself support the ten-author order retained here. These exceptions are
resolved against the primary records rather than by deleting correct fields to
match an incomplete aggregator record.

The `elsarticle-harv` BibTeX style reports `empty pages` for the accepted ICLR
records `eysenbach2021off`, `kostrikov2022iql`, and `zhang2023pex`. Their
official OpenReview/proceedings records provide no page range, so no pages were
invented to silence a style warning. The ICLR 2025 proceedings record for WSRL
does provide volume 2025 and pages 32343--32368, which are retained.

## Entry-by-entry verification

| Key | Authoritative record | Supported placement and semantic boundary |
|---|---|---|
| `niu2022h2o` | [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/ed3cd2520148b577039adfade82a5566-Abstract-Conference.html), DOI `10.52202/068431-2652` | Supports the original H2O problem: immutable target replay, simulator interaction, and dynamics-aware correction. It is not cited for H2O+ generality or transit-specific serialization. Seven authors, volume 35, pages 36599--36612, and DOI verified. |
| `niu2025h2oplus` | [IEEE DOI](https://doi.org/10.1109/ICRA55743.2025.11127503), [DBLP](https://dblp.org/rec/conf/icra/NiuJLZZZHZHZ25) | Supports H2O+ coupling of offline and online learners under dynamics gaps and the source-schedule control. It does not support the new hurdle action, duration conditioning, or passenger ledger. Ten authors and order, ICRA 2025, pages 1421--1428 verified. |
| `eysenbach2021off` | [ICLR 2021 OpenReview](https://openreview.net/forum?id=eqBwg3AcIAK) | Supports the two-domain-classifier conditional dynamics-ratio factorization. It is not evidence for transit fidelity or effectiveness. Conference author order is Eysenbach, Chaudhari, Asawa, Levine, Salakhutdinov; no official pages. |
| `haarnoja2018soft` | [PMLR 80](https://proceedings.mlr.press/v80/haarnoja18b.html) | Supports the entropy-regularized off-policy SAC backbone and bootstrap only. Four authors and pages 1861--1870 verified. |
| `kumar2020cql` | [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html) | Supports conservative offline value learning, the CQL control, and the declared CQL initializer. WSRL-specific behavior is attributed separately to WSRL. Four authors, volume 33, pages 1179--1191 verified. |
| `kostrikov2022iql` | [ICLR 2022 OpenReview](https://openreview.net/forum?id=68n2s9ZJWF8) | Supports expectile value fitting, in-sample targets, and advantage-weighted extraction used in the warm start and IQL control. Three authors verified; no official pages. |
| `lee2022balanced` | [PMLR 164](https://proceedings.mlr.press/v164/lee22d.html) | Supports destructive early fine-tuning as a generic offline-to-online risk and balanced replay plus a pessimistic ensemble as one remedy. It is not cited for dynamics gaps or transit. Five authors and pages 1702--1712 verified. |
| `nakamoto2023calql` | [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/c44a04289beaf0a7d968a94066a1d696-Abstract-Conference.html), DOI `10.52202/075280-2719` | Supports calibrated conservative values as a fine-tuning route and discloses the main initializer context in the WSRL paper. It is not the initializer used by the registered transit WSRL jobs. Eight authors, volume 36, pages 62244--62269 verified. |
| `zhang2023pex` | [ICLR 2023 OpenReview](https://openreview.net/forum?id=-Y34L45JR6z) | Supports retaining the offline policy while learning an expansion policy as a distinct stabilization route. PEX is contextual literature, not an experiment here. Three authors verified; no DOI or official pages. |
| `ball2023efficient` | [PMLR 202](https://proceedings.mlr.press/v202/ball23a.html) | Supports RLPD prior/new replay, critic normalization, ensemble design, high UTD, and random initialization without offline pretraining. The manuscript calls the implementation source-aligned, not byte-identical. Four authors and pages 1577--1594 verified. |
| `zhou2025wsrl` | [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/504491292cb71e7681eedfe0e602b72f-Abstract-Conference.html) | Supports offline initialization, frozen-policy warmup, and online-only fine-tuning without retaining offline data. The text distinguishes the paper's main Cal-QL setup from its paper-supported CQL variant used here. Five authors, volume 2025, pages 32343--32368 verified. |
| `hu2025rgcql` | [Transportation Research Part C](https://www.sciencedirect.com/science/article/pii/S0968090X25000555), DOI `10.1016/j.trc.2025.105051` | Establishes offline training plus online fine-tuning in public-transit-adjacent ride-pooling coordination, preventing an overbroad transportation novelty claim. It does not study fixed-route holding, cross-fidelity dynamics, or this SMDP. Three authors, volume 174, article 105051 verified. |
| `sutton1999between` | [Artificial Intelligence](https://www.sciencedirect.com/science/article/pii/S0004370299000521), DOI `10.1016/S0004-3702(99)00052-1` | Supports only the semi-Markov foundation for variable-duration decisions at the first SMDP definition. Serialization and the passenger ledger remain study-specific. Three authors, volume 112(1--2), pages 181--211 verified. |
| `rezazada2024review` | [Transport Reviews](https://www.tandfonline.com/doi/abs/10.1080/01441647.2024.2313969), DOI `10.1080/01441647.2024.2313969` | Supports the broad bunching mechanism, control taxonomy, and passenger/operator trade-offs in the Introduction and bus-control review. It is not fidelity or effectiveness evidence. Four authors, volume 44(4), pages 766--790 verified. |
| `daganzo2009headway` | [Transportation Research Part B](https://www.sciencedirect.com/science/article/abs/pii/S0191261509000484), DOI `10.1016/j.trb.2009.04.002` | Supports analytical headway-feedback holding under structured assumptions. Sole author, volume 43(10), pages 913--921 verified. |
| `xuan2011dynamic` | [Transportation Research Part B](https://www.sciencedirect.com/science/article/pii/S0191261511001093), DOI `10.1016/j.trb.2011.07.009` | Supports dynamic holding for schedule reliability using optimal linear control. Three authors, volume 45(10), pages 1831--1845 verified. |
| `delgado2012holding` | [Transportation Research Part B](https://www.sciencedirect.com/science/article/pii/S0191261512000653), DOI `10.1016/j.trb.2012.04.005` | Supports the passenger-performance trade-off of holding and boarding limits. The manuscript explicitly states that its own action is holding only. Three authors, volume 46(9), pages 1202--1217 verified. |
| `sanchezmartinez2016realtime` | [Transportation Research Part B](https://www.sciencedirect.com/science/article/pii/S019126151500257X), DOI `10.1016/j.trb.2015.11.013` | Supports real-time holding with time-varying running times and demand. It is an optimization study, not offline-to-online RL or ratio evidence. Three authors, volume 83, pages 1--19 verified. |
| `andres2017predictive` | [Transportation Research Part B](https://www.sciencedirect.com/science/article/abs/pii/S019126151630892X), DOI `10.1016/j.trb.2017.06.013` | Supports predictive headway control; the manuscript explicitly distinguishes its non-predictive policy. Two authors, volume 104, pages 123--148 verified. |
| `alesiani2018reinforcement` | [University of Twente record](https://research.utwente.nl/en/publications/reinforcement-learning-based-bus-holding-for-high-frequency-servi), DOI `10.1109/ITSC.2018.8569473` | Supports RL-based holding for high-frequency service, not cross-fidelity transfer. Two authors and pages 3162--3168 verified. |
| `wang2020real` | [Transportation Research Part C](https://www.sciencedirect.com/science/article/pii/S0968090X20305763), DOI `10.1016/j.trc.2020.102661` | Supports multi-agent dynamic holding. The stable citation key contains `real`, but the text does not claim a real-system deployment. Two authors, volume 116, article 102661 verified. |
| `wang2023multiobjective` | [Transportation Research Part C](https://www.sciencedirect.com/science/article/pii/S0968090X2300298X), DOI `10.1016/j.trc.2023.104309` | Supports multi-objective multi-agent holding for multiline services on a shared corridor. Two authors, volume 155, article 104309 verified. |
| `lopez2018microscopic` | [SUMO recommended citation](https://sumo.dlr.de/about/), DOI `10.1109/ITSC.2018.8569938` | Supports identifying SUMO as the microscopic traffic simulator, not fidelity of this benchmark. The SUMO-maintained ten-author order, pages 2575--2582, and DOI are retained. |

## Placement conclusions

1. H2O, H2O+, off-dynamics RL, SAC, CQL, IQL, RLPD, and WSRL are each cited
   only for their own algorithmic contribution; transit-specific state, action,
   reward, duration, and deployment choices are not attributed to them.
2. Balanced replay, Cal-QL, and PEX are contextual fine-tuning routes. They
   motivate failure modes and design alternatives but are not silently promoted
   to experimental baselines.
3. RG-CQL narrows the novelty statement to cross-fidelity asynchronous
   fixed-route holding rather than offline-to-online transportation broadly.
4. The bus-control references are assigned to review, analytical feedback,
   boarding-limit comparison, real-time optimization, predictive control, or RL
   holding according to the primary source. None supports simulator fidelity.
5. SUMO identifies the software; only the preregistered matched-cell audits
   support this study's simulator-acceptance claims.

## Change-control gate

After any citation edit, rerun cited-key/bibliography-key equality, duplicate
work checks, registered placement checks, BibTeX, acronym first-use checks, and
the full manuscript build. An adjacent citation may not be reused for a new
claim unless its primary source directly supports that claim.
