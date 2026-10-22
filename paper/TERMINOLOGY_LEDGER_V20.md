# Terminology Ledger

Use these terms consistently in the manuscript and submission materials.

| Preferred term | Meaning |
|---|---|
| target SUMO environment | The SUMO 1.25.0 microsimulation used for endpoint evaluation |
| logged SUMO environment | The SUMO 1.27.1 counterpart that generated offline replay |
| online event-driven simulator | The lower-cost causal simulator used for continuation training |
| cross-fidelity simulator transfer | Transfer between the two simulators; not sim-to-real transfer |
| frozen actor | A preregistered actor checkpoint with no evaluation-time learning |
| reserve-augmented candidate | The fixed actor plus the final guarded 60-second reserve |
| direct behavior parent | The identical policy with the final reserve disabled |
| lineage control | A fixed earlier algorithm label used for supportive comparison |
| formal environment block | One independent fresh jitter/seed block crossed with demand levels |
| generalized passenger seconds per departed passenger | The primary endpoint |
| completion rate | Completed passenger journeys divided by departed passenger journeys |
| endpoint-only confirmation | Inference based on fixed terminal checkpoints, not learning curves |
| formal confirmation | The preregistered candidate-parent comparison on fresh environment blocks |

The result is a paired endpoint effect on the registered simulator family.
Lineage comparisons are supportive. The term "completion-safety" in registered
method identities denotes the variant, not a universal safety guarantee.

## Mathematical Notation

Every entry below is defined in the manuscript where it first enters an
argument. Subscripts identify components or indices, not new meanings.

| Symbol | Unique meaning |
|---|---|
| $\mathcal{L},\ell$ | Set of directional services and one service index |
| $m,j,n_m,e_{m,j}$ | Physical batch index, slot index, batch size, and slot event |
| $k,K$ | Chronological AEC token index and total episode token count |
| $t,t_m,t_0,T_{\mathrm{hor}},u$ | Physical time, batch time, first controllable time, horizon, and time integration variable |
| $x^{\mathrm{raw}}_{m,j},x_{m,j}$ | Raw event observation and deduplicated local event vector |
| $s^{\mathrm{raw}}_k,s_k,s'$ | Packed raw state, canonical state, and successor state |
| $a,a_k,a^+$ | Normalized action, token action, and conditional positive-branch action |
| $\tau_k,\tau_{\max},\Delta t_k$ | Executed hold, hold cap, and transition elapsed time |
| $z_k,\Gamma_k$ | Terminal indicator and continuation factor |
| $N_{\mathrm{w}},N_{\mathrm{v}}$ | System waiting and in-vehicle passenger counts |
| $\omega_{\mathrm{w}},\omega_{\mathrm{v}}$ | Fixed waiting and in-vehicle passenger-time weights |
| $C_k,C_{\mathrm{pre}},r_k$ | Token cost, pre-control cost, and negative token cost reward |
| $N_{\mathrm{dep}},N_{\mathrm{cmp}},Y,R_{\mathrm{cmp}},U$ | Departed and completed passenger counts, primary endpoint, completion rate, and unfinished count |
| $\mathcal{M}_{\mathrm{tgt}},\mathcal{M}_{\mathrm{log}},\mathcal{M}_{\mathrm{sim}},P_{\mathrm{tgt}},P_{\mathrm{log}},P_{\mathrm{sim}}$ | Evaluation, logged, and training environments and their respective transition-duration kernels |
| $\mathcal{D}_{\mathrm{off}},\mathcal{D}_{\mathrm{sim}},\iota,N_{\mathrm{off}}$ | Offline and simulator replay, offline row index, and offline row count |
| $J_{\mathrm{tgt}},\mathbb{E}$ | Expected target return and expectation operator |
| $\pi,\pi_\phi,\phi$ | Admissible policy, learned hurdle policy, and actor parameters |
| $p^+_\phi,g_\phi,\mu_\phi,\sigma_\phi,\zeta$ | Positive-branch probability, gate logit, latent normal mean, standard deviation, and latent coordinate |
| $\delta_{-1},f_\phi,\varpi_\phi,\mathfrak{n}_\phi,\mathbb{I}$ | No-hold point mass, conditional action density, mixed-measure density, latent normal density, and indicator |
| $\theta_i,\bar\theta_i,i,\bar Q_{\min},V_\nu,\nu,A,w_{\mathrm{IQL}}$ | Online and target critic parameters, critic-head index, target critic minimum, value function, value parameters, advantage, and actor regression weight |
| $\kappa_r,\eta_{\mathrm{ent}},B_{\mathrm{SAC}},B_{\mathrm{mix}},y_k$ | Trainer reward scale, entropy temperature, SAC bootstrap, mixed bootstrap, and critic training target |
| $F_{\psi_J},G_{\psi_M},\psi_J,\psi_M$ | Joint and marginal target-class logits and their parameters |
| $\rho,\rho_{\mathrm{lo}},\rho_{\mathrm{hi}}$ | Horizon fraction and guard-window endpoints |
| $H_f,H_b,H_f^*,H_b^*,I_f,I_b$ | Forward/backward headways, target headways, and vehicle-presence indicators |
| $h_\phi,h_{\mathrm{hb}},h_{16},h_{\mathrm{par}},h_{\mathrm{safe}}$ | Actor proposal, headway reserve, inherited calibrated proposal, direct-parent hold, and candidate hold |
| $n_{\mathrm{on}},c,\lambda_{\mathrm{arr}},\mathcal{G}$ | Local onboard count, capacity, eligible arrival rate, and binary completion guard |
| $\beta,\xi,d,\mathcal{D}_{\mathrm{dem}}$ | Formal block index, training-seed index, demand multiplier, and multiplier set |
| $\Delta Y,\Delta R_{\mathrm{cmp}},\Delta U$ | Mean candidate-minus-parent endpoint differences over the specified subset |
| $\varepsilon_{\mathrm{NI}},\alpha_{\mathrm{sig}},p_{\mathrm{sign}},w,\upsilon$ | Completion margin, significance level, sign-test probability, observed win count, and possible-win summation index |

## Abbreviations

The abstract first expands RL, H2O, SMDP, SUMO, and IQL; H2O+ is introduced
as the improved H2O framework. Related Work first expands SAC, CQL, Cal-QL,
PEX, RLPD, WSRL, and RG-CQL. Problem Formulation expands AEC. Experiments
expands AULC, AUC, ESS, and JSON. The appendix expands Adam. Declarations
expand AI and CRediT. Later prose uses these abbreviations. Bibliography titles
and official method identities retain the source spelling.

