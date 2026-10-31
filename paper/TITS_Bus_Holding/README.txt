T-ITS BUS HOLDING MANUSCRIPT
===========================
Prepared: 22 October 2026
Reference/notation revision: 28 October 2026

The current TeX, bibliography, and PDF include that revision. PROVENANCE.json,
VALIDATION.json, and SHA256SUMS.txt are retained as receipts of the original
22 October package, not as validation of subsequently edited files.

Title
Completion-Aware Bus Holding Using Hybrid Offline and Simulation-Based
Reinforcement Learning

Main files
----------
TITS_Bus_Holding_Manuscript.pdf   Complete 9-page manuscript
TITS_Bus_Holding_Manuscript.tex   Editable main source
references.bib                  26 references
IEEEtran.cls / IEEEtran.bst     Unmodified IEEE LaTeX class and bibliography style
figures/                       Vector figures and previews
scripts/                       Figure preparation and result verification
data/                          Compact evidence used for tables and the block plot
EDITORIAL_NOTES_zh.txt          Chinese explanation of positioning and revisions
PROVENANCE.json                Source commits, policy URLs, and evidence paths

Compilation
-----------
Run from this directory:

    latexmk -pdf -interaction=nonstopmode -halt-on-error TITS_Bus_Holding_Manuscript.tex

Alternatively:

    pdflatex TITS_Bus_Holding_Manuscript.tex
    bibtex TITS_Bus_Holding_Manuscript
    pdflatex TITS_Bus_Holding_Manuscript.tex
    pdflatex TITS_Bus_Holding_Manuscript.tex

The supplied template's journal layout is retained: IEEEtran, letter paper,
10-point type, and two columns. Margins, body type size, and line spacing have
not been reduced to meet the page limit. IEEEtran's standard reference-column
break balances the final page; reconsider its reference number after edits
that materially change the document length.

The PDF includes the abstract, all eight main sections, four tables, three
figures, acknowledgment, and 26 references. It preserves the author names,
affiliations, and email addresses supplied in the existing project manuscript.
It does not invent a corresponding author, funding details, or author biographies.

Regenerating figures
--------------------
Python packages: matplotlib, numpy, and PyMuPDF.

    python3 scripts/plot_controller_overview.py
    python3 scripts/plot_block_effects.py
    python3 scripts/prepare_network_figure.py

The network geometry is reused unchanged from the project's original vector
map. The preparation script enlarges its legend for a single-column figure.
The original is retained as figures/network_topology_original.pdf.
The block plot reads the existing experiment results. It adds no experiments,
bootstrap intervals, smoothing, or new hypotheses.

Result data and units
---------------------
data/paired_episode_endpoints.csv contains the 300 evaluation records used for
the primary comparison. Each pair shares training seed, traffic block, demand
level, and trained policy parameters. The fields are:

primary       Generalized seconds per passenger who started a journey: 2W + V
waiting       Waiting seconds per passenger who started a journey
in_vehicle    In-vehicle seconds per passenger who started a journey
completion    Fraction of departed passengers completing by 24,000 seconds
unfinished    Count of departed passengers not completing by that horizon
departed      Number of passengers who started a journey
actor         Trained policy parameter hash
ledger_error  Difference between reward and independent time accounting
*_total       Undivided accrued passenger-time quantities
endpoint      Common horizon in seconds
path          Relative path to the original project result record

Reported means give equal weight to each scenario-policy pair. They are not
pooled passenger averages. Completion differences in the manuscript and figure
are percentage points; stored completion fractions must be multiplied by 100.

Legacy controller identifiers are preserved in the evidence for traceability:

h2oplus-v19-completion-safety-quarter-load-sixty
    Proposed controller: the 60-second rule in [0.15, 0.75).
h2oplus-v19-completion-safety-baseline
    Matched reference: a holding floor of 20 seconds in [0.50, 0.75).
h2oplus-v14-terminal-quarter
    Learned policy alone, before hold calibration and completion rules.
h2oplus
    Transit-adapted H2O+ control with active dynamics weights.
sac-online
    Online-only SAC.
h2oplus-v10-delayed-actor
    Actor updates from simulator data.
h2oplus-v11-actor-hold
    Delayed actor updates.
h2oplus-v12-actor-ramp
    Gradual increase in actor learning rate.
h2oplus-v13-proximal-strong
    Actor updates with parameter anchoring.

In data/verified_results.json, the inherited keys "with" and "without" mean
"proposed" and "matched reference", respectively. BOTH controllers include a
service rule. "Without" must never be interpreted as an entirely rule-free
controller. The manuscript and figures use the explicit comparison names.

Recompute the primary results with:

    python3 scripts/verify_results.py

The full comparison matrix has 1,350 episodes. The primary comparison has
150 matched pairs and ten independent environment blocks. Each block averages
five trained policies and three demand levels. The exact sign tests use the
ten block differences, not 150 independent pairs or millions of transitions.

Scope
-----
This package prepares a manuscript from the supplied research. It does not
contain large training replay files, checkpoints, original road/stop XML, or
generated scenario inputs. Their absence does not prevent compilation or
recalculation of the included endpoint summaries, but full experiment reruns
require the original research inputs. No new simulation experiments were run
for this rewrite.

Current journal guidance
------------------------
Regular papers: 10 pages before mandatory overlength charges.
Overlength: USD 175 per page above 10 after publication formatting.
Traditional publication and optional open access have different fee treatment.
The 9-page draft leaves room for final author information and revisions, but
the final publisher proof must also stay within 10 pages.

Official requirements and fee sources are listed in PROVENANCE.json.
The special-issue deadline is transcribed from the user-supplied call for papers,
not independently verified online.
