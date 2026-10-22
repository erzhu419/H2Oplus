# Terminology and Notation Ledger (v7)

Status: normative draft for the v7 rewrite. Dataset and simulator anchors are
frozen; trainer-dependent reward/loss, entropy, precision, capacity, and runtime
anchors remain release blockers until their fresh audits pass.

## Naming rules

1. `target environment` means the high-fidelity SUMO environment;
   `online simulator` means the calibrated reduced-order event simulator.
2. Neither environment is `real`. The evidence is simulator-to-simulator on a
   real network topology, not sim-to-real deployment.
3. A `decision event` is one bus-stop control opportunity. A `decision batch`
   contains all simultaneous events at one physical time. A `decision token`
   is one serialized action inside that batch.
4. A `target transition` comes from immutable SUMO offline data. A `simulator
   transition` is collected online in the reduced-order simulator.
5. `Generalized passenger time` is the scientific objective. `Training reward
   scale` is optimizer conditioning and must never be described as a physical
   reward unit.
6. `Online interaction budget` counts simulator decision events. Offline
   updates, online updates, target evaluations, and wall time are separate
   quantities.
7. `Raw event observation`, `deduplicated local event vector`, `raw packed
   state`, and `canonical state` are distinct objects and cannot be shortened
   collectively to `state vector` when dimensions are reported.
8. Every canonical CUDA job runs on a registered GPU server. H2O+ family jobs
   use the `server-h2oplus` execution class; WSRL, RLPD, ratio-SAC, SAC
   controls, and offline controls use `server-non-h2oplus`. Placement is
   provenance, not a methodological difference.
9. The legacy snapshot corpus belongs to a historical H2O+ workflow and is not
   an input to any canonical v7 method. Do not use `snapshot-based` to describe
   the formal v7 offline data, simulator interaction, or training protocol.

## Acronym ledger

The first prose occurrence in expanded document order must use the exact full
form followed by the acronym in parentheses. Later prose uses only the acronym.
Title and keyword metadata do not satisfy first prose use.

| Acronym | Required first use | Notes |
|---|---|---|
| RL | reinforcement learning (RL) | Define in the abstract. |
| H2O | Hybrid Offline-and-Online (H2O) | Problem-family name. |
| H2O+ | improved framework for Hybrid Offline-and-Online RL with dynamics gaps (H2O+) | Method name used by the cited source. |
| SUMO | Simulation of Urban MObility (SUMO) | Preserve official capitalization. |
| SMDP | semi-Markov decision process (SMDP) | Variable-duration decision process. |
| AEC | agent-environment cycle (AEC) | Serialization interface. |
| IQL | implicit Q-learning (IQL) | H2O+ warm start. |
| SAC | soft actor-critic (SAC) | Online actor-critic backbone. |
| IS | importance sampling (IS) | Conditional kernel-ratio weighting only. |
| CQL | conservative Q-learning (CQL) | Registered WSRL initializer and offline control. |
| Cal-QL | calibrated Q-learning (Cal-QL) | Context for the WSRL source paper, not the registered initializer. |
| BC | behavior cloning (BC) | Offline control. |
| PEX | policy expansion (PEX) | Contextual literature, not an experimental control. |
| RG-CQL | Reward-Guided Conservative Q-Learning (RG-CQL) | Transportation literature only. |
| RLPD | Reinforcement Learning with Prior Data (RLPD) | Preserve official title capitalization. |
| WSRL | Warm-Start Reinforcement Learning (WSRL) | Preserve repository-title hyphenation. |
| UTD | update-to-data (UTD) ratio | Critic updates per new event-equivalent. |
| GPU | graphics processing unit (GPU) | Compute placement. |
| HDF5 | Hierarchical Data Format version 5 (HDF5) | Canonical replay format. |
| SHA-256 | Secure Hash Algorithm 256-bit (SHA-256) | Provenance digest. |
| CUDA | Compute Unified Device Architecture (CUDA) | NVIDIA runtime. |
| AVL | Automatic Vehicle Location (AVL) | Future operational validation only. |
| APC | Automatic Passenger Counting (APC) | Future operational validation only. |
| AFC | Automated Fare Collection (AFC) | Future operational validation only. |
| AI | Artificial Intelligence (AI) | Declaration only. |
| CRediT | Contributor Roles Taxonomy (CRediT) | Authorship declaration only. |

## Canonical symbols

Each row contains one symbol family and one unique meaning. Subscripts identify
an instance of that family; they do not license a second meaning. The `First
definition` column is a release obligation, not merely a suggested location.

| Symbol | Unique meaning | First definition |
|---|---|---|
| $\mathcal{B}$ | set of buses controlled by the shared policy | Problem, controlled system |
| $\mathcal{L}$ | set of controlled directional services | Problem, controlled system |
| $\mathcal{K}_{\ell}$ | ordered stop sequence belonging to directional service $\ell$ | Problem, controlled system |
| $\ell$ | directional-service index | Problem, controlled system |
| $m$ | physical decision-batch index | Problem, chronology |
| $j$ | within-batch decision-token position | Problem, chronology |
| $k$ | global chronological decision-token index | Problem, chronology |
| $\iota$ | immutable offline-replay row index | Problem, dataset |
| $i$ | critic index | Method, critic definition |
| $t$ | physical time; $t_m$ is the time of batch $m$ | Problem, chronology |
| $u$ | passenger-time integration variable | Problem, reward |
| $n_m$ | number of controllable events in decision batch $m$ | Problem, chronology |
| $e_{m,j}$ | physical control event in batch position $(m,j)$ | Problem, chronology |
| $x^{\mathrm{raw}}$ | 16-feature raw SUMO event observation | Problem, state contract |
| $x$ | 13-feature deduplicated local event vector | Problem, state contract |
| $s^{\mathrm{raw}}$ | 245-dimensional raw packed batch state | Problem, state contract |
| $s$ | 229-dimensional canonical centralized state; a prime denotes its successor | Problem, state contract |
| $\operatorname{pack}$ | deterministic raw-batch packing operation | Problem, state contract |
| $\operatorname{normalize}$ | deterministic raw-to-canonical state transformation | Problem, state contract |
| $a$ | normalized scalar policy action in the closed interval $[-1,1]$ | Problem, action |
| $\tau$ | physical hold duration induced by the policy action | Problem, action |
| $\tau_{\max}$ | maximum permitted physical hold duration of 60 seconds | Problem, action |
| $\Delta t$ | physical elapsed time carried by one transition | Problem, duration |
| $z$ | terminal-transition indicator | Problem, continuation |
| $\Gamma$ | stored continuation multiplier equal to one minus the terminal indicator | Problem, continuation |
| $N_{\mathrm{w}}$ | system-wide number of waiting passengers as a function of physical time | Problem, reward |
| $N_{\mathrm{v}}$ | system-wide number of in-vehicle passengers as a function of physical time | Problem, reward |
| $C$ | generalized passenger-time cost accrued by one transition | Problem, reward |
| $\omega_{\mathrm{w}}$ | waiting-time value coefficient fixed at 2 | Problem, reward |
| $\omega_{\mathrm{v}}$ | in-vehicle-time value coefficient fixed at 1 | Problem, reward |
| $r$ | dataset reward in physical generalized passenger-seconds | Problem, reward |
| $r^{\mathrm{train}}$ | numerically scaled reward supplied to a trainer update | Method, conditioning |
| $\kappa_{\mathrm{r}}$ | training reward multiplier selected by the fresh reward/loss gate | Method, conditioning; pending gate |
| $T_{\mathrm{hor}}$ | physical episode horizon fixed at 24,000 seconds | Problem, horizon |
| $\mathcal{M}_{\mathrm{tgt}}$ | target high-fidelity SUMO SMDP | Problem, environments |
| $\mathcal{M}_{\mathrm{sim}}$ | calibrated online-simulator SMDP | Problem, environments |
| $P_{\mathrm{tgt}}$ | target transition-duration kernel | Problem, environments |
| $P_{\mathrm{sim}}$ | simulator transition-duration kernel | Problem, environments |
| $\mathcal{D}_{\mathrm{off}}$ | immutable target-environment offline replay | Problem, dataset |
| $\mathcal{D}_{\mathrm{sim}}$ | chronological replay collected in the online simulator | Problem, dataset |
| $\mathcal{D}$ | one explicitly subscripted empirical replay distribution | Method preamble |
| $N_{\mathrm{off}}$ | audited number of rows in the frozen v7 offline replay, 3,284,718 | Problem, dataset; frozen HDF5 audit |
| $\mathbb{E}$ | expectation under the distribution named by its subscript | Problem, objective |
| $\pi$ | admissible hurdle-policy family | Problem, objective |
| $\pi_{\phi}$ | learned hurdle policy parameterized by $\phi$ | Problem, objective |
| $\phi$ | actor parameter vector | Problem, objective |
| $J_{\mathrm{tgt}}$ | expected target-environment return functional | Problem, objective |
| $K$ | number of chronological tokens before the episode horizon | Problem, objective |
| $\operatorname{sigmoid}$ | logistic sigmoid function | Method, hurdle actor |
| $g_{\phi}$ | actor gate logit function | Method, hurdle actor |
| $p^{+}_{\phi}$ | positive-hold branch probability | Method, hurdle actor |
| $p^{0}_{\phi}$ | exact-zero-hold branch probability | Method, hurdle actor |
| $\zeta$ | latent normal variate on the positive-hold branch | Method, hurdle actor |
| $\mu_{\phi}$ | actor output for latent-normal mean | Method, hurdle actor |
| $\sigma_{\phi}$ | actor output for latent-normal standard deviation | Method, hurdle actor |
| $\mathcal{N}$ | normal probability distribution | Method, hurdle actor |
| $a^{+}$ | transformed action on the positive-hold branch | Method, hurdle actor |
| $\delta_{-1}$ | point-mass probability measure at action minus one | Method, hurdle actor |
| $f_{\phi}$ | transformed-normal density on the positive-hold branch | Method, hurdle actor |
| $\mathbb{I}$ | indicator function | Method, hurdle actor |
| $\epsilon_{a}$ | machine precision of the stored action data type | Method, action boundary |
| $Q_{\theta_i}$ | online critic with index $i$ | Method, critic definition |
| $Q_{\bar\theta_i}$ | target critic with index $i$ | Method, critic definition |
| $\theta_i$ | parameter vector of online critic $i$ | Method, critic definition |
| $\bar\theta_i$ | parameter vector of target critic $i$ | Method, critic definition |
| $\theta$ | ordered collection of the two online-critic parameter vectors | Method, critic definition |
| $\bar\theta$ | ordered collection of the two target-critic parameter vectors | Method, critic definition |
| $\kappa_{\mathrm{tgt}}$ | Polyak target-network update coefficient | Method, critic definition |
| $V_{\nu}$ | IQL state-value function | Method, IQL |
| $\nu$ | IQL value-function parameter vector | Method, IQL |
| $q_{\min}$ | minimum estimate over the two target critics | Method, IQL |
| $A_{\nu,\bar\theta}$ | target-critic advantage relative to the IQL value function | Method, IQL |
| $\eta_{\mathrm{exp}}$ | IQL expectile level fixed at 0.7 | Method, IQL |
| $\beta_{\mathrm{aw}}$ | IQL advantage-exponent coefficient fixed at 3 | Method, IQL |
| $y_{\mathrm{IQL}}$ | IQL critic regression target | Method, IQL |
| $\mathcal{L}_{V}$ | IQL value-function loss | Method, IQL |
| $\mathcal{L}^{\mathrm{IQL}}_{\pi}$ | advantage-weighted IQL actor loss | Method, IQL |
| $F_{\psi_J}$ | joint target-versus-simulator classifier logit | Method, ratio estimator |
| $G_{\psi_M}$ | state-action marginal target-versus-simulator classifier logit | Method, ratio estimator |
| $\psi_J$ | joint domain-classifier parameter vector | Method, ratio estimator |
| $\psi_M$ | marginal domain-classifier parameter vector | Method, ratio estimator |
| $\psi$ | ordered collection of the joint and marginal classifier parameter vectors | Method, ratio estimator |
| $\widetilde{w}_{\psi}$ | unclipped conditional target-to-simulator kernel-ratio estimate | Method, ratio estimator |
| $w_{\psi}$ | clipped conditional target-to-simulator kernel-ratio weight | Method, ratio estimator |
| $w_{\min}$ | fixed lower clipping bound of $10^{-5}$ | Method, ratio estimator |
| $w_{\max}$ | fixed upper clipping bound of 1 | Method, ratio estimator |
| $\operatorname{clip}$ | truncation operator to a stated closed interval | Method, ratio estimator |
| $\mathcal{L}_{\mathrm{ratio}}$ | sum of the two balanced domain-classification losses | Method, ratio estimator |
| $a'$ | next-state action sampled from the learned policy | Method, online backup |
| $B_{\mathrm{SAC}}$ | entropy-regularized target-critic bootstrap | Method, online backup |
| $\alpha_{\mathrm{ent}}$ | learned SAC entropy-temperature coefficient | Method, online backup |
| $\mathcal{H}_{\mathrm{tar}}$ | hybrid-policy entropy target selected by the fresh entropy gate | Method, online backup; pending gate |
| $B_{\mathrm{mix}}$ | convex IQL-value and SAC bootstrap used by H2O+ | Method, online backup |
| $\lambda_{\mathrm{mix}}$ | H2O+ bootstrap-mixture coefficient fixed at 0.1 | Method, online backup |
| $y_{\mathrm{mix}}$ | H2O+ Bellman regression target | Method, online backup |
| $\varepsilon_{Q,i}$ | temporal-difference residual of critic $i$ | Method, critic loss |
| $h_{\delta_H}$ | scalar Huber penalty when Huber regression is selected | Method, critic loss; pending gate |
| $\delta_H$ | Huber transition point when Huber regression is selected | Method, critic loss; pending gate |
| $\chi$ | scalar argument of the Huber penalty | Method, critic loss |
| $\rho_{\mathrm{sim}}$ | simulator fraction in a mixed critic minibatch fixed at 0.5 | Method, critic loss |
| $\mathcal{L}_{Q}$ | H2O+ critic loss | Method, critic loss |
| $\mathcal{L}_{\pi}$ | online SAC actor loss | Method, actor loss |
| $\mathrm{d}$ | differential or integration-measure operator | Method, hurdle actor |
| $\in$ | set-membership relation | First mathematical use |
| $\sim$ | sampling or probability-distribution relation | First mathematical use |
| $\mid$ | probabilistic-conditioning relation | First mathematical use |
| $\approx$ | stated approximation relation | Method, ratio estimator |
| $=$ | equality relation | First mathematical use |
| $<$ | strict less-than relation | First mathematical use |
| $\leq$ | less-than-or-equal relation | First mathematical use |
| $\geq$ | greater-than-or-equal relation | First mathematical use |
| $+$ | arithmetic addition operator | First mathematical use |
| $-$ | arithmetic subtraction or unary negation operator | First mathematical use |
| $\times$ | scalar multiplication operator | First mathematical use |
| $/$ | arithmetic division operator | First mathematical use |
| $|\cdot|$ | absolute-value operator | Method, IQL |
| $\sum$ | finite summation operator | Problem, objective |
| $\int$ | definite integration operator | Problem, reward |
| $\min$ | minimum operator | Method, IQL |
| $\max$ | maximum operator | Method, stated bounds |
| $\exp$ | exponential function | Method, IQL |
| $\log$ | natural logarithm | Method, hurdle actor |
| $\cdot$ | placeholder argument of a function or probability measure | Method, hurdle actor |
| $\mathbb{R}$ | set of real numbers | Problem, state contract |

## Prohibited aliases and overloads

- Do not use $c_{\mathrm{r}}$: v7 has no manuscript-level data reward divisor.
- Do not use an unqualified $\epsilon$ for both action precision and numerical
  stabilization; action precision is $\epsilon_a$.
- Do not use $\tau$ for target-network updates; $\tau$ is physical holding and
  the target update coefficient is $\kappa_{\mathrm{tgt}}$.
- Do not use an unqualified $\lambda$ for replay composition, bootstrap mixing,
  regularization, or ratio weighting.
- Do not use $E$ for expectation or ensemble size; expectation is $\mathbb E$
  and the main critic count is stated directly as two.
- Do not use $\delta$ for terminal status, point-mass measure, Huber threshold,
  and temporal-difference residual interchangeably. The ledger assignments are
  $z$, $\delta_{-1}$, $\delta_H$, and $\varepsilon_{Q,i}$.
- Do not call both the 16-feature raw event observation and the 13-feature local
  event vector $x$ without the raw superscript.
- Do not use a second symbol for stored reward, scaled training reward, target
  entropy, simulator replay fraction, or mixed-bootstrap coefficient.

## Final gates

- Compare every mathematical atom in expanded TeX against this table.
- Verify the first-definition anchor in document order, including captions,
  algorithms, tables, and appendix material.
- Reject grouped symbol rows, duplicate normalized symbols, and duplicate
  normalized meanings.
- Inspect each equation manually for dimensional consistency and agreement with
  executable trainer code.
- Replace every `pending freeze` anchor with a hash-identified artifact before
  submission.
