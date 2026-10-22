# Terminology and Notation Ledger (v6)

Status: normative for the v6 submission manuscript. The final source and PDF
audits must agree with this ledger.

## Naming rules

1. Use `target environment` for the high-fidelity SUMO environment and
   `online simulator` for the calibrated reduced-order event simulator.
2. Do not call either environment `real`. The study is simulator-to-simulator
   transfer on a real network topology, not sim-to-real deployment.
3. Use `decision event` for one bus-stop control opportunity and `decision
   batch` for simultaneous events at one physical time.
4. Use `decision token` for one serialized action in an agent-environment-cycle
   batch. Only the final token of a simultaneous batch advances physical time.
5. Use `target transition` for a row from the immutable SUMO offline data and
   `simulator transition` for a row collected online in the reduced-order
   simulator.
6. Use `generalized passenger time` for the objective. Do not call the v6
   objective a headway reward or holding penalty.
7. Use `online interaction budget` only for simulator decision events. Offline
   gradient updates are a separate budget and must be reported separately.

## Acronym ledger

Each row gives the only permitted first-use expansion. After first use in
document order, use the acronym alone. Full terms in the title and keywords are
metadata and do not replace the first prose expansion in the abstract.

| Acronym | Required first use | Notes |
|---|---|---|
| RL | reinforcement learning (RL) | Define in the abstract. |
| H2O | Hybrid Offline-and-Online (H2O) | Use for the problem family, followed by RL where needed. |
| H2O+ | improved framework for Hybrid Offline-and-Online RL with dynamics gaps (H2O+) | Define in the abstract; thereafter use H2O+ only. |
| SUMO | Simulation of Urban MObility (SUMO) | Preserve official capitalization. |
| SMDP | semi-Markov decision process (SMDP) | v6 has event-dependent durations. |
| AEC | agent-environment cycle (AEC) | Serialization of simultaneous decisions. |
| IQL | implicit Q-learning (IQL) | Main H2O+ warm start. |
| SAC | soft actor-critic (SAC) | Online actor-critic backbone. |
| IS | importance sampling (IS) | Only for density-ratio weighting. |
| CQL | conservative Q-learning (CQL) | Offline baseline/initializer. |
| BC | behavior cloning (BC) | Offline baseline. |
| RLPD | Reinforcement Learning with Prior Data (RLPD) | Expansion verified against the official repository and PMLR paper page. |
| WSRL | Warm-Start Reinforcement Learning (WSRL) | Hyphenation follows the official repository title; the paper body uses `Warm Start`. |
| UTD | update-to-data (UTD) ratio | Number of critic updates per new event-equivalent. |
| GPU | graphics processing unit (GPU) | Needed only in reproducibility text. |
| HDF5 | Hierarchical Data Format version 5 (HDF5) | Canonical replay file format. |
| SHA-256 | Secure Hash Algorithm 256-bit (SHA-256) | Provenance digest. |
| CUDA | Compute Unified Device Architecture (CUDA) | NVIDIA runtime; define in the appendix. |
| AVL | Automatic Vehicle Location (AVL) | Future operational validation only. |
| APC | Automatic Passenger Counting (APC) | Future operational validation only. |
| AFC | Automated Fare Collection (AFC) | Future operational validation only. |
| AI | Artificial Intelligence (AI) | Declaration only. |
| CRediT | Contributor Roles Taxonomy (CRediT) | Authorship declaration heading only. |

## Canonical symbols

No symbol below may be reused with another meaning. No listed meaning may be
represented by a second symbol in final prose.

| Symbol | Unique meaning |
|---|---|
| $\mathcal{B}$ | set of controlled buses |
| $\mathcal{L}$ | set of directional bus services |
| $\mathcal{K}_{\ell}$ | ordered stop sequence of service $\ell$ |
| $\ell$ | directional-service index |
| $m$ | physical decision-batch index |
| $j$ | within-batch event/token position |
| $k$ | global chronological decision-token index |
| $\iota$ | offline-dataset row index |
| $i$ | critic index, with $i\in\{1,2\}$ in the main H2O+ profile |
| $t_m$ | physical time of decision batch $m$ |
| $t$ | generic physical-time argument of a passenger-count function |
| $u$ | integration variable for passenger-time cost |
| $n_m$ | number of controllable events in batch $m$ |
| $e_{m,j}$ | event occupying position $j$ in batch $m$ |
| $x_{m,j}$ | 16-dimensional raw local feature vector of $e_{m,j}$ |
| $\R$ | set of real numbers; $\R^d$ is the $d$-dimensional real vector space |
| $s_k$ | 275-dimensional canonical centralized state at token $k$ |
| $\operatorname{pack}$ | deterministic canonical-state packing operation |
| $a_k$ | one-dimensional policy action in $[-1,1]$ |
| $\tau_k$ | physical hold duration induced by $a_k$ |
| $\tau_{\max}$ | maximum hold duration, fixed at 60 s |
| $\Delta t_k$ | physical duration from token $k$ to token $k+1$ |
| $z_k$ | terminal indicator for transition $k$ |
| $\Gamma_k$ | stored continuation discount, $1-z_k$ in v6 |
| $N_{\mathrm{w}}(t)$ | number of waiting passengers at physical time $t$ |
| $N_{\mathrm{v}}(t)$ | number of in-vehicle passengers at physical time $t$ |
| $C_k$ | generalized passenger-time cost accrued on transition $k$ |
| $c_{\mathrm{r}}$ | reward unit, fixed at $10^6$ passenger-seconds |
| $r_k$ | training reward, $-C_k/c_{\mathrm{r}}$ |
| $T_{\mathrm{hor}}$ | episode horizon, fixed at 24,000 s |
| $\omega_{\mathrm{w}}$ | waiting-time value, fixed at 2 |
| $\omega_{\mathrm{v}}$ | in-vehicle-time value, fixed at 1 |
| $\mathcal{M}_{\mathrm{tgt}}$ | target high-fidelity SUMO SMDP |
| $\mathcal{M}_{\mathrm{sim}}$ | calibrated online-simulator SMDP |
| $P_{\mathrm{tgt}}$ | target transition-duration kernel |
| $P_{\mathrm{sim}}$ | simulator transition-duration kernel |
| $s,a,r,s',\Gamma,\Delta t,z$ | generic state, action, reward, successor state, continuation, duration, and terminal flag of one transition |
| $\mathcal{D}_{\mathrm{off}}$ | immutable target-environment offline data |
| $\mathcal{D}_{\mathrm{sim}}$ | replay collected online in the simulator |
| $\mathcal{D}$ | generic replay named in an empirical expectation |
| $N_{\mathrm{off}}$ | number of offline transitions, 2,831,150 |
| $\E$ | expectation; its subscript identifies the environment, policy, or empirical replay distribution |
| $\pi$ | generic admissible hurdle policy |
| $\pi_{\phi}$ | learned hurdle policy with parameters $\phi$ |
| $\phi$ | actor/policy parameter vector |
| $J_{\mathrm{tgt}}(\pi)$ | expected target-environment return of policy $\pi$ |
| $K$ | number of chronological decision tokens before the horizon |
| $\sigmoid$ | logistic sigmoid function |
| $\mathcal{N}(\mu,\sigma^2)$ | normal distribution with mean $\mu$ and variance $\sigma^2$ |
| $g_{\phi}(s)$ | actor gate logit |
| $p^+_{\phi}(s)$ | probability of the positive-hold branch |
| $p^0_{\phi}(s)$ | probability of the exact-zero-hold branch |
| $\zeta$ | latent normal variable for the positive action branch |
| $\mu_{\phi}(s)$ | latent-normal mean output by the actor |
| $\sigma_{\phi}(s)$ | latent-normal standard deviation output by the actor |
| $a^+$ | transformed action drawn from the positive-hold branch |
| $\delta_{-1}$ | point-mass measure at action $-1$ |
| $f_{\phi}(a\mid s)$ | transformed-normal density on the positive branch |
| $\mathbb{I}$ | indicator function |
| $\mathrm{d}a$ | continuous-action integration element |
| $\epsilon$ | machine precision of the replay-action data type |
| $Q_{\theta_i}$ | critic $i$; $i\in\{1,2\}$ for main H2O+ |
| $Q_{\bar\theta_i}$ | target copy of critic $i$ |
| $\theta=(\theta_1,\theta_2)$ | online-critic parameter pair |
| $\bar\theta=(\bar\theta_1,\bar\theta_2)$ | target-critic parameter pair |
| $V_{\nu}$ | IQL state-value function |
| $\nu$ | value-function parameter vector |
| $q_{\min}(s,a)$ | minimum of the two target-critic estimates |
| $A_{\nu,\bar\theta}(s,a)$ | target-critic advantage over $V_{\nu}$ |
| $\alpha_{\mathrm{ent}}$ | learned SAC entropy temperature |
| $\eta_{\mathrm{exp}}$ | IQL expectile, fixed at 0.7 |
| $\beta_{\mathrm{aw}}$ | IQL advantage exponent coefficient, fixed at 3 |
| $y_{\mathrm{IQL}}$ | IQL critic regression target |
| $\mathcal{L}^{\mathrm{IQL}}_{\pi}$ | advantage-weighted IQL actor loss |
| $\lambda_{\mathrm{mix}}$ | H2O+ value/SAC bootstrap mixture, fixed at 0.1 in the main profile |
| $\rho_{\mathrm{sim}}$ | simulator fraction in a mixed critic minibatch, fixed at 0.5 |
| $F_{\psi_J}$ | joint target-vs-simulator domain-classifier logit |
| $G_{\psi_M}$ | state-action marginal domain-classifier logit |
| $\psi_J$ | joint-classifier parameter vector |
| $\psi_M$ | marginal-classifier parameter vector |
| $\psi=(\psi_J,\psi_M)$ | combined ratio-estimator parameter vector |
| $\widetilde{w}_{\psi}$ | unclipped conditional target-to-simulator kernel-ratio estimate |
| $w_{\psi}$ | clipped conditional target-to-simulator kernel ratio |
| $w_{\min},w_{\max}$ | ratio clipping bounds, fixed at $10^{-5}$ and 1 |
| $\clip$ | truncation of its first argument to bounds given by its second and third arguments |
| $a'$ | next-state action sampled from $\pi_{\phi}(\cdot\mid s')$ |
| $B_{\mathrm{SAC}}(s')$ | entropy-regularized target-critic bootstrap |
| $B_{\mathrm{mix}}(s')$ | convex value/SAC bootstrap used by H2O+ |
| $y_{\mathrm{mix}}$ | H2O+ Bellman target |
| $\mathcal{L}_Q$ | critic loss |
| $\mathcal{L}_V$ | IQL value loss |
| $\mathcal{L}_{\pi}$ | online SAC policy loss |
| $\mathcal{L}_{\mathrm{ratio}}$ | sum of the two balanced domain-classification losses |
| $\in,\sim,\mid,\approx$ | standard set membership, sampling/distribution, conditioning, and approximation relations |
| $\cdot$ | placeholder for the action argument in a policy distribution |
| $\min,\max,\exp,\log,|\cdot|,\sum,\int$ | standard minimum, maximum, exponential, natural logarithm, absolute value, summation, and integration operators |
| $=,<,\leq,\geq,+,-,\times,/$ | standard equality, ordering, and arithmetic operators |

## Prohibited legacy terms and symbols

The following belong to the pre-v6 manuscript and must not appear in the final
submission except in an explicitly historical audit note:

- `real environment`, $\mathcal{M}_{\mathrm{real}}$, and $P_{\mathrm{real}}$;
- 675,000 transitions, four behavior policies, ep39, or an 18,000 s horizon;
- a two-dimensional action, separate hold gate action, or Gaussian-only policy;
- LSTM sim-core, snapshot reset, snapshot injection, or a 200 GB snapshot corpus;
- headway-deviation reward, reward-scale ratio 0.2, Q-floor, LCB target,
  InfoNCE ratio, action-invariance assumption, or five-critic H2O+;
- using $E$ for both expectation and ensemble size;
- using $\tau$ for both physical holding and target-network update rate;
- using an unqualified $\lambda$ for both replay weighting and mixed backup.

## Final audit gates

- Every acronym occurrence must be scanned in document order, including title,
  abstract, keywords, captions, tables, algorithms, appendix, and declarations.
- Every math token in displayed and inline mathematics must map to this ledger
  or be defined locally before or immediately after first use.
- Temporary TeX compatibility aliases for legacy sections must be removed after
  those sections are rewritten.
- The final reference audit must preserve the primary-source-verified RLPD and
  WSRL expansions above.
