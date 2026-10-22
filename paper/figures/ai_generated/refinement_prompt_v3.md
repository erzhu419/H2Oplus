Final refinement for the bottom comparison branch.

Use a clean two-lane academic diagram. Keep all labels readable.

Top lane:
- Same as before: Target data D_real and Imperfect simulator M_sim feed Offline warm start, Dynamics-gap estimator, IS-weighted hybrid update, and Target-domain policy.

Bottom lane layout must be exactly this:
- Left column: three stacked input boxes:
  1. "SUMO target M_real"
  2. "D_off: 675K + ep39"
  3. "sim-core online rollouts"
- Middle upper box: "H2O+ variants"
- Middle lower gray box: "WSRL / RLPD / Cal-QL"
- Right orange box: "Paired SUMO stress grid"
- Far-right red box: "Claim boundary"

Bottom lane arrows:
- The three left input boxes point only to "H2O+ variants".
- "H2O+ variants" points directly to "Paired SUMO stress grid".
- "WSRL / RLPD / Cal-QL" points directly to "Paired SUMO stress grid".
- There must be no arrow from H2O+ variants to WSRL / RLPD / Cal-QL.
- There must be no arrow from WSRL / RLPD / Cal-QL to H2O+ variants.
- "Paired SUMO stress grid" points to "Claim boundary".

The baseline box is a parallel comparison branch, not a processing stage.
Use thick dark arrows. Avoid any shared bus line in the bottom lane.
