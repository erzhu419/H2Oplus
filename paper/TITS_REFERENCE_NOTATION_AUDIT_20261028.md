# TITS Submission: Reference and Notation Check

Checked 2026-10-28. Target: `TITS_Bus_Holding_Manuscript.tex`, not `main.tex`.

## References

The manuscript has 26 cited works across 32 citation commands (40 individual
key occurrences). The shared bibliography's existing 25 entries retain their
previously verified metadata; citation placement was checked against this
manuscript's own wording. The missing `wang2023robust` entry was added from
[arXiv:2111.01946](https://arxiv.org/abs/2111.01946): Wang and Sun,
*Robust Dynamic Bus Control: A Distributional Multi-Agent Reinforcement
Learning Approach*, 2021. It directly supports the distributional bus-control
statement. Its internal key retains the publication-year suffix; the cited
arXiv year is 2021. No unrelated substitute was introduced.

The generated IEEE bibliography has 26 entries, in first-citation order, with
no missing, duplicate, or unused key. References [1]--[7] and [10]--[13] support
the bus-control discussion; [8], [9], and [14]--[24] support the learning
discussion; [25] supports temporal abstraction, and [26] identifies SUMO.
The 15 arXiv and 11 published records are consistently formatted by IEEEtran.

## Mathematics and Corrections

Equations (1)--(18) were checked analytically: action scaling, accrued passenger
time, the logistic-normal Jacobian and median, expectile weighting, mixed
bootstrap, holding transforms, paired aggregation, and the binomial sign tail
are consistent. The 60-s candidate formula correctly subsumes the inherited
20-s rule because the latter's activation window is a subset of the former's.

Definitions were added for states/actions, actor and value parameters, normal
parameters, replay sampling distributions, losses, expectation, indicator,
clipping, and physical time. Completion differences now consistently use
`R_cmp`; guard bounds no longer reuse physical-time `u`, and the binomial
summation no longer reuses decision index `k`. Policy expectations explicitly
condition actions on their states. All 18 equations, 4 tables, and 3 figures
now have valid textual cross-references. This version contains no appendix or
unconnected appendix proof. Algorithm code and experimental outcomes were unchanged.

The statement that the value network continues updating on offline replay is
correct: `SimpleSAC/transit_v3/trainer.py` updates it whenever an offline batch
is present, including simulator training. The release variants retain that path.

## Outstanding Files

`figures/controller_overview.pdf` and `figures/block_effects.pdf` are missing.
Normal compilation stops at the first missing figure. A separate diagnostic
build in `/tmp/h2oplus_tits_audit_20261028` uses graphicx demo boxes only to
check mathematics, bibliography, and references; it has no undefined-reference,
undefined-citation, or overfull-box warning. It is not a submission PDF.
The actual figures, their consistency with the text, and final pagination must
be checked once those two files are supplied.

## Complete Package Applied

The revisions were subsequently applied to
`TITS_Bus_Holding/TITS_Bus_Holding_Manuscript.tex` and its local bibliography.
That complete package already contained the Wang reference and all three figures;
available arXiv versions were used without duplicating entries. Its native build
produces a 9-page PDF with all figures present, 26 resolved bibliography entries,
and no undefined citations/references or overfull boxes. The rendered figure
pages were checked against their captions. The missing-file limitation above
applies only to the earlier standalone source, not to this complete package.
