# v7 Equation Rewrite Specification

Status: normative rewrite plan derived from the executable v7 state, reward,
policy, and trainer contracts. Equations shown here are design specifications,
not manuscript-ready prose. Dataset counts and result values remain pending.

## Label policy

Use semantic labels without protocol-version suffixes. Proposed labels are:

| Object | Final label |
|---|---|
| Raw and canonical state contract | `eq:state_contract` |
| Action-to-hold map | `eq:hold_action` |
| Continuation multiplier | `eq:continuation` |
| Generalized passenger-time cost | `eq:passenger_cost` |
| Physical dataset reward | `eq:data_reward` |
| Training reward scaling | `eq:training_reward` |
| Target and simulator kernels | `eq:transition_kernels` |
| Immutable offline replay | `eq:offline_dataset` |
| Target-environment objective | `eq:target_objective` |
| Hurdle policy measure | `eq:hurdle_policy` |
| IQL advantage | `eq:iql_advantage` |
| IQL value loss | `eq:iql_value` |
| IQL critic target | `eq:iql_target` |
| IQL actor loss | `eq:iql_actor` |
| Conditional kernel ratio | `eq:conditional_ratio_target` |
| Conditional log-ratio estimator | `eq:conditional_log_ratio` |
| Clipped ratio weight | `eq:ratio_weight` |
| SAC bootstrap | `eq:sac_bootstrap` |
| H2O+ mixed bootstrap | `eq:h2oplus_target` |
| Huber penalty | `eq:huber_penalty` |
| H2O+ critic loss | `eq:h2oplus_critic` |
| Online actor loss | `eq:h2oplus_actor` |
| Training algorithm | `alg:h2oplus_training` |

No final label may contain `v6`, `v7`, a run revision, or a result-table index.
Each numbered object must have at least one body reference after final ordering.

## State contract

The manuscript must distinguish four objects. The raw SUMO event observation
has 16 physical features, while the deduplicated local event vector removes
physical time and the two system-wide passenger counts:

```tex
x^{\mathrm{raw}}_{m,j}\in\mathbb{R}^{16},
\qquad
x_{m,j}\in\mathbb{R}^{13}.
```

The raw packed batch state stores five global values once and then 16 slots.
Each raw slot contains one active flag, 13 local features, and one prior action:

```tex
s^{\mathrm{raw}}_k
=\operatorname{pack}\!\left(
t_m,n_m,j,N_{\mathrm{w}}(t_m),N_{\mathrm{v}}(t_m),
\{x_{m,q},a_{m,q}^{\mathrm{prior}}\}_{q=1}^{16}
\right)
\in\mathbb{R}^{245}.
```

The canonical transform has five normalized prefix values and 16 slots of
width 14:

```tex
s_k=\operatorname{normalize}(s^{\mathrm{raw}}_k)
\in\mathbb{R}^{5+16\times14}
=\mathbb{R}^{229}.
\label{eq:state_contract}
```

Do not introduce $q$ or $a^{\mathrm{prior}}$ in the final equation unless both
are added to the terminology ledger and defined at first use. An acceptable
alternative is to describe the slot sequence in prose and keep only the two
dimensional statements in the numbered equation.

The five canonical prefix fields must be listed in implementation order:
horizon progress, scaled simultaneous-decision count, scaled current-decision
index, scaled system waiting count, and scaled system in-vehicle count. The 14
slot fields must be tied to `CANONICAL_SLOT_NAMES` rather than reconstructed
from memory.

## Action and duration

The normalized action has one meaning and one physical map:

```tex
\tau_k=\frac{a_k+1}{2}\tau_{\max},
\qquad
a_k\in[-1,1],
\qquad
\tau_{\max}=60\ \mathrm{s}.
\label{eq:hold_action}
```

Action $a_k=-1$ means exact zero hold; $a_k\in(-1,1]$ means positive hold.
The policy has no separate gate action coordinate. The physical transition
duration is $\Delta t_k$ and the terminal flag is $z_k$:

```tex
\Gamma_k=1-z_k.
\label{eq:continuation}
```

The finite-horizon v7 objective is undiscounted. Intermediate serialized tokens
inside one simultaneous batch have $\Delta t_k=0$; only the final token advances
physical time and receives the elapsed passenger-time ledger increment.

## Physical reward and numerical conditioning

The physical generalized passenger-time cost remains:

```tex
C_k=
\omega_{\mathrm{w}}
\int_{t_m}^{t_m+\Delta t_k}N_{\mathrm{w}}(u)\,\mathrm{d}u
+\omega_{\mathrm{v}}
\int_{t_m}^{t_m+\Delta t_k}N_{\mathrm{v}}(u)\,\mathrm{d}u,
\qquad
\omega_{\mathrm{w}}=2,
\quad
\omega_{\mathrm{v}}=1.
\label{eq:passenger_cost}
```

The immutable dataset stores a physical reward with no million-second divisor:

```tex
r_k=-C_k
\quad\text{generalized passenger-seconds}.
\label{eq:data_reward}
```

The trainer applies a distinct numerical multiplier:

```tex
r^{\mathrm{train}}_k=\kappa_{\mathrm{r}}r_k,
\qquad
\kappa_{\mathrm{r}}=10^{-4}.
\label{eq:training_reward}
```

The final value of $\kappa_{\mathrm{r}}$ must be bound to the reward/loss pilot
and training admission. The method equations use $r^{\mathrm{train}}$, while
the scientific objective, reported welfare, and dataset description use $r$.
Because $\kappa_{\mathrm{r}}>0$, this numerical conditioning does not change
policy ordering.

## Environments and data

The target and simulator are joint successor-state/duration kernels:

```tex
P_{\mathrm{tgt}}(s',\Delta t\mid s,a),
\qquad
P_{\mathrm{sim}}(s',\Delta t\mid s,a).
\label{eq:transition_kernels}
```

The offline replay equation uses the audited row count only after the HDF5
freeze:

```tex
\mathcal{D}_{\mathrm{off}}=
\left\{
(s_{\iota},a_{\iota},r_{\iota},s'_{\iota},
\Gamma_{\iota},\Delta t_{\iota},z_{\iota})
\right\}_{\iota=1}^{N_{\mathrm{off}}}.
\label{eq:offline_dataset}
```

Do not place a provisional value for $N_{\mathrm{off}}$ in TeX. The final
objective remains in physical reward units:

```tex
J_{\mathrm{tgt}}(\pi)=
\mathbb{E}_{\mathcal{M}_{\mathrm{tgt}},\pi}
\left[\sum_{k=0}^{K-1}r_k\right],
\qquad
\max_{\pi}J_{\mathrm{tgt}}(\pi).
\label{eq:target_objective}
```

## Hurdle policy

Retain the mixed counting-plus-Lebesgue policy measure, but use the ledger's
$\epsilon_a$ for action precision. The positive branch is open at both action
endpoints; the environment action set remains closed. The exact count of stored
$a=1$ rows must come from the frozen full-data audit.

The final equation may retain the current form:

```tex
\pi_{\phi}(\mathrm{d}a\mid s)=
p^0_{\phi}(s)\delta_{-1}(\mathrm{d}a)
+p^+_{\phi}(s)f_{\phi}(a\mid s)
\mathbb{I}\{a\in(-1,1)\}\,\mathrm{d}a.
\label{eq:hurdle_policy}
```

Immediately define the gate logit, branch probabilities, latent normal,
transformed action, point mass, density, indicator, and base measure. Do not use
`Gaussian policy` as shorthand for the complete hurdle distribution.

## Offline targets

The IQL structure is unchanged, but every critic target must use scaled training
reward:

```tex
y_{\mathrm{IQL}}
=r^{\mathrm{train}}+\Gamma V_{\nu}(s').
\label{eq:iql_target}
```

Keep the expectile value and advantage-weighted actor equations only if their
symbols match `TERMINOLOGY_LEDGER_V7.md`. The actor-weight cap of 100 is a fixed
implementation constant and must be stated once.

## Conditional ratio

The scientific ratio remains

```tex
\frac{P_{\mathrm{tgt}}(s',\Delta t\mid s,a)}
     {P_{\mathrm{sim}}(s',\Delta t\mid s,a)}.
\label{eq:conditional_ratio_target}
```

The joint classifier receives normalized duration
$\Delta t/T_{\mathrm{hor}}$. Balanced target/simulator class priors are part of
the estimator contract. Keep separate symbols for joint and marginal
classifier parameters and separate lower and upper clipping bounds.

## H2O+ target and Huber critic

The SAC and mixed bootstraps retain their current structure. The Bellman target
must use the scaled reward:

```tex
B_{\mathrm{mix}}(s')=
\lambda_{\mathrm{mix}}V_{\nu}(s')
+(1-\lambda_{\mathrm{mix}})B_{\mathrm{SAC}}(s'),
\qquad
y_{\mathrm{mix}}=r^{\mathrm{train}}+\Gamma B_{\mathrm{mix}}(s').
\label{eq:h2oplus_target}
```

Define the critic residual and admitted Huber penalty explicitly:

```tex
\varepsilon_{Q,i}(s,a)=Q_{\theta_i}(s,a)-y_{\mathrm{mix}},
```

```tex
h_{\delta_H}(\chi)=
\begin{cases}
\tfrac{1}{2}\chi^2, & |\chi|\leq\delta_H,\\
\delta_H\left(|\chi|-\tfrac{1}{2}\delta_H\right), & |\chi|>\delta_H,
\end{cases}
\qquad
\delta_H=1.
\label{eq:huber_penalty}
```

Then replace every squared TD residual in the current manuscript:

```tex
\begin{aligned}
\mathcal{L}_Q={}&(1-\rho_{\mathrm{sim}})
\mathbb{E}_{\mathcal{D}_{\mathrm{off}}}
\left[\frac{1}{2}\sum_{i=1}^{2}
h_{\delta_H}\!\left(\varepsilon_{Q,i}(s,a)\right)\right]\\
&+\rho_{\mathrm{sim}}
\mathbb{E}_{\mathcal{D}_{\mathrm{sim}}}
\left[\frac{w_{\psi}(s,a,s',\Delta t)}{2}
\sum_{i=1}^{2}
h_{\delta_H}\!\left(\varepsilon_{Q,i}(s,a)\right)\right],
\qquad
\rho_{\mathrm{sim}}=0.5.
\end{aligned}
\label{eq:h2oplus_critic}
```

The factor $1/2$ averages the two critics. The ratio weight multiplies only the
simulator Huber contribution. It does not change stored or scaled rewards,
targets, actor gradients, or the IQL value loss.

## Hybrid entropy

State that the learned temperature is optimized against
$\mathcal{H}_{\mathrm{tar}}=0$. This is a calibrated target for the complete
mixed hurdle policy, not the target entropy of a one-dimensional Gaussian.
Retain the actor loss only after checking its branch expectation against the
trainer. If a temperature-loss equation is added, introduce a distinct loss
symbol and the log-temperature parameter in the ledger before first use.

## Dimensional and implementation checks

- $r$, $r^{\mathrm{train}}$, $C$, and both Bellman targets are scalar.
- $a$ and $\tau$ are scalar; $s$ has width 229.
- $P_{\mathrm{tgt}}$ and $P_{\mathrm{sim}}$ condition on identical state/action
  definitions and return the same successor/duration variables.
- $w_{\psi}$ is scalar, nonnegative, and clipped to $[10^{-5},1]$.
- $h_{\delta_H}$ is scalar and nonnegative.
- No use of MSE or `squared residual` may remain outside a rejected-candidate
  pilot discussion.
- Equation order, labels, and references must be re-audited after insertion;
  the numbers printed in this specification are not fixed manuscript numbers.
