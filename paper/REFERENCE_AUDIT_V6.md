# Reference and Citation Audit (v6)

Audit scope: every entry in `references.bib`, every citation command in the
submission source, bibliographic metadata, and the claim made at each citation
location. Metadata was checked against publisher or conference records; DOI
records were also cross-checked through Crossref on 2026-08-09.

## Result

- Bibliography entries: 14.
- Distinct cited keys: 14.
- Verified entries: 14.
- Uncited bibliography entries: 0.
- Undefined citation keys: 0.
- Duplicate works: 0.
- Semantically misplaced citations: 0 after the corrections recorded below.

The manuscript uses `elsarticle-harv`, so author-year rendering and title case
are controlled by the journal style. The accepted ICLR 2021 and 2022 records
for `eysenbach2021off` and `kostrikov2022iql` have no official page ranges; no
page numbers were invented to suppress BibTeX warnings. In contrast, the ICLR
2025 proceedings BibTeX explicitly assigns volume 2025 and pages 32343--32368
to `zhou2025wsrl`, and those fields are retained.

## Entry-by-entry verification

| Key | Primary metadata record | Claim supported in this manuscript | Status and correction history |
|---|---|---|---|
| `niu2022h2o` | [NeurIPS 2022 proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/ed3cd2520148b577039adfade82a5566-Abstract-Conference.html), DOI `10.52202/068431-2652` | Defines H2O as fixed target-domain offline data plus online interaction in an imperfect simulator with dynamics-aware correction; cited in the Introduction, Related Work, and problem objective. | Verified title, seven authors and order, volume 35, pages 36599--36612, year, and DOI. DOI added. |
| `niu2025h2oplus` | [IEEE DOI record](https://doi.org/10.1109/ICRA55743.2025.11127503); [DBLP final record](https://dblp.org/rec/conf/icra/NiuJLZZZHZHZ25) | Supports H2O+ as a flexible coupling of offline and online base learners under a dynamics gap; cited for the adaptation, mixed method, and source control. | Verified final ICRA 2025 record, ten authors and order, pages 1421--1428. The preprint-only record was replaced by the final conference record. |
| `eysenbach2021off` | [ICLR 2021 OpenReview record](https://openreview.net/forum?id=eqBwg3AcIAK) | Supports estimating off-dynamics correction with transition-domain classifiers without an explicit dynamics model; cited in Related Work and the ratio factorization. | Verified. Author order corrected to Eysenbach, Chaudhari, Asawa, Levine, Salakhutdinov. No official pages. |
| `haarnoja2018soft` | [PMLR volume 80](https://proceedings.mlr.press/v80/haarnoja18b.html) | Supports the entropy-regularized off-policy actor-critic backbone and SAC bootstrap. | Verified four authors, ICML 2018, volume 80, pages 1861--1870. |
| `kumar2020cql` | [NeurIPS 2020 proceedings](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html) | Supports conservative value learning for offline data; used as an offline control and as the declared WSRL initializer. | Verified four authors, volume 33, pages 1179--1191. The text does not attribute WSRL-specific behavior to CQL. |
| `kostrikov2022iql` | [ICLR 2022 OpenReview record](https://openreview.net/forum?id=68n2s9ZJWF8) | Supports expectile value fitting, in-sample critic targets, and advantage-weighted policy extraction; used for the H2O+ warm start and offline control. | Verified three authors and ICLR 2022. No official pages. |
| `ball2023efficient` | [PMLR volume 202](https://proceedings.mlr.press/v202/ball23a.html) | Supports RLPD's prior/new replay mixing, critic normalization, ensemble design, high UTD, and absence of offline pretraining. | Verified four authors, ICML 2023, pages 1577--1594. The transit implementation is explicitly described as source-aligned, not byte-identical. |
| `zhou2025wsrl` | [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/504491292cb71e7681eedfe0e602b72f-Abstract-Conference.html) | Supports offline initialization, frozen-policy warmup, then online-only fine-tuning without retaining offline replay. | Verified five authors, volume 2025, and pages 32343--32368 against the proceedings BibTeX. The main experiments use CalQL initialization, while Appendix H/Figure 22 reports that CQL and IQL initializations also work similarly. The manuscript therefore identifies its CQL initialization as one paper-supported variant, not the sole WSRL default. |
| `daganzo2009headway` | DOI `10.1016/j.trb.2009.04.002` | Supports analytical headway-feedback holding for suppressing bus bunching. | Verified sole author, journal, volume 43(10), pages 913--921. |
| `xuan2011dynamic` | DOI `10.1016/j.trb.2011.07.009` | Supports dynamic holding for schedule reliability using optimal linear control. | Verified three authors and order, volume 45(10), pages 1831--1845. |
| `alesiani2018reinforcement` | DOI `10.1109/ITSC.2018.8569473` | Supports RL-based holding for high-frequency bus service. | Verified two authors, ITSC 2018, pages 3162--3168. |
| `wang2020real` | DOI `10.1016/j.trc.2020.102661` | Supports multi-agent dynamic bus holding to avoid bunching. | Verified two authors, Transportation Research Part C, volume 116, article 102661. The citation key is retained for stability; the manuscript does not call the study a real-system deployment. |
| `wang2023multiobjective` | DOI `10.1016/j.trc.2023.104309` | Supports multi-objective, multi-agent holding for multiline services on a shared corridor. | Verified two authors, Transportation Research Part C, volume **155**, article 104309. An earlier volume-156 value was corrected. |
| `lopez2018microscopic` | [SUMO project's recommended citation](https://sumo.dlr.de/userdoc/Publications.html), DOI `10.1109/ITSC.2018.8569938` | Supports identifying SUMO as the microscopic target traffic simulator. | Verified title, pages 2575--2582, and DOI. The bibliography follows the SUMO project's recommended ten-author order; Crossref currently places Evamarie Wiessner second, so the project-maintained primary citation is treated as authoritative for author order. |

## Citation-placement conclusions

1. The Introduction now cites H2O only for the original hybrid setting and H2O+
   separately for generalization across base learners. This removes an earlier
   combined citation whose sentence blurred the two contributions.
2. The conditional-ratio discussion cites off-dynamics RL for the domain-
   classifier construction and H2O+ for its use in hybrid offline/online
   learning. Neither source is used to support transit-specific SMDP duration
   handling; that adaptation is claimed as this implementation's design.
3. RLPD and WSRL citations support their defining data flows and update
   principles. Exact transit encoder, hurdle actor, critic count, and update
   ordering are disclosed as registered adaptations and are not misrepresented
   as untouched official implementations.
4. Bus-control references are confined to the operational formulation they
   actually study. They are not used as evidence for cross-fidelity transfer,
   causal journey generation, or the present evaluation protocol.
5. The SUMO citation supports the simulator software, not fidelity of this
   benchmark. Fidelity is supported only by the manuscript's preregistered
   matched-cell audit.

## Final release gate

Repeat the cited-key/bibliography-key equality check and BibTeX warning scan
after any reference edit. A new empirical or methodological claim may not reuse
an adjacent citation unless the primary source directly supports that claim.
