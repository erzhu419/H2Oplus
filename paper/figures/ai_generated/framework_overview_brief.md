Title / label: Study-level framework adapted from H2O/H2O+ for bus holding.

One-line claim: The paper instantiates the H2O/H2O+ offline-data-plus-imperfect-simulator template in bus holding, then evaluates whether dynamics correction adds value beyond simpler offline-to-online baselines.

Key visual elements:
- Two horizontal lanes.
- Top lane: generic H2O/H2O+ abstraction.
- Bottom lane: this paper's SUMO bus-holding instantiation.
- Use a restrained academic palette: blue for target data/environment, green for learning/correction, gray for baselines, orange for evaluation, red for claim boundary.
- Show thick left-to-right arrows and two vertical adaptation arrows from the generic template to this paper's implementation.

Data grounding:
- Use only large, short labels inside the image.
- Include: D_real, M_sim, offline warm start, dynamics-gap estimator / IS weights, hybrid policy update.
- Include: SUMO target M_real, D_off 675K + ep39, sim-core online rollouts, H2O+ variants, baselines, paired SUMO evaluation, claim boundary.
- Do not include exact p-values, long captions, or dense tables in the image.

Where it goes:
- Figure 1 in the Introduction, before the Setting paragraph.
- Caption carries the detailed interpretation.
