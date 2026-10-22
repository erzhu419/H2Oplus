Create a clean, publication-ready academic workflow illustration for a transportation reinforcement-learning paper.

Canvas and style:
- Landscape 3:2 canvas, white background, vector-diagram appearance, crisp sans-serif typography.
- Restrained academic palette: muted blue for target data/environment, muted green for H2O+ learning/correction modules, light gray for baselines, orange for evaluation, muted red for claim boundary.
- Thick dark-gray arrows with clear arrowheads. No crossed arrows.
- No icons, no photos, no 3D, no glow, no heavy shadows, no decorative background.
- Use only the exact short English labels below. Avoid paragraphs and avoid tiny text.

Overall layout:
- Two stacked horizontal lanes with generous spacing.
- Top lane title: "H2O / H2O+ template"
- Bottom lane title: "Bus-holding instantiation"
- Two thin vertical adaptation arrows from the top lane to the bottom lane.

Top lane, left to right:
1. Blue box: "Target data D_real"
2. Blue box: "Imperfect simulator M_sim"
3. Green box: "Offline warm start"
4. Green box: "Dynamics-gap estimator"
5. Green box: "IS-weighted hybrid update"
6. Orange box: "Target-domain policy"

Top-lane arrows:
- "D_real" flows to "Offline warm start".
- "M_sim" flows to "IS-weighted hybrid update".
- "Dynamics-gap estimator" supplies "importance weights" to "IS-weighted hybrid update".
- "Offline warm start" flows to "IS-weighted hybrid update".
- "IS-weighted hybrid update" flows to "Target-domain policy".

Bottom lane, left to right:
1. Blue box: "SUMO target M_real"
2. Blue box: "D_off: 675K + ep39"
3. Green box: "sim-core online rollouts"
4. Green box: "H2O+ variants"
5. Gray box: "WSRL / RLPD / Cal-QL"
6. Orange box: "Paired SUMO stress grid"
7. Red box: "Claim boundary"

Bottom-lane arrows:
- "SUMO target M_real" and "D_off: 675K + ep39" flow into "H2O+ variants".
- "sim-core online rollouts" flow into "H2O+ variants".
- "H2O+ variants" and "WSRL / RLPD / Cal-QL" both flow into "Paired SUMO stress grid".
- "Paired SUMO stress grid" flows into "Claim boundary".

Visual hierarchy:
- The two lane titles should be larger than module labels.
- The bottom lane should be slightly more prominent than the top lane, because it is this paper's contribution.
- Keep labels large and readable at two-column paper width.
- Leave enough whitespace around every box so the diagram does not feel crowded.
