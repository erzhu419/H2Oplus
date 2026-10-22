# V6 method protocol audit

Audit date: 2026-08-08

This file freezes the source evidence used to define the formal v6 method
profiles. It separates published algorithms from transit-specific adaptations;
the latter must not be reported as an unmodified implementation of a baseline.

## Primary sources

| Method | Paper | Reference implementation | Frozen revision |
|---|---|---|---|
| H2O+ | https://arxiv.org/abs/2309.12716v2 and https://doi.org/10.1109/ICRA55743.2025.11127503 | https://github.com/t6-thu/H2Oplus | `02171768283d307244700c69003423704db83b05` |
| RLPD | https://proceedings.mlr.press/v202/ball23a.html | https://github.com/ikostrikov/rlpd | `c90fd4baf28c9c9ef40a81460a2e395092844f88` |
| WSRL | https://proceedings.iclr.cc/paper_files/paper/2025/hash/504491292cb71e7681eedfe0e602b72f-Abstract-Conference.html | https://github.com/zhouzypaul/wsrl | `ad4dc1248a138bc15d6e053f2d1dba1b8cfbaca2` |

## H2O+ findings

The published v2 paper defines the mixed target as

```text
B_mix = r + lambda * gamma * V(s')
          + (1 - lambda) * gamma * E_pi[Q_target(s', a') - beta log pi(a'|s')].
```

It applies the target-to-simulator conditional transition ratio only to the
simulator Bellman residual. The state value is learned only from target-domain
offline data, while the SAC actor is optimized on the union of target and
simulator samples.

The paper explicitly states `lambda = 0.1` for all comparative experiments and
reports a lambda ablation over 0, 0.1, 0.2, 0.5, and 1. The public code's
`get_default_config()` instead contains `exploit_coeff = 0.5`. The formal
protocol therefore follows the published experimental value, not the
repository default. `lambda = 0.5` remains a named sensitivity profile.

The public code uses expectile/quantile `0.5`, no separate offline pretraining
by default (`warmup_steps = 0`), one simulator rollout before each block of
updates, balanced target/simulator batches, two critics, batch size 256, and a
ratio-classifier update on every gradient step. It clips the simulator TD
weight to `[1e-5, 1]` and implements weighted MSE by multiplying prediction and
target by the square root of that weight.

## Registered H2O+ profiles

| Profile | Role | Deliberate deviations |
|---|---|---|
| `h2oplus-source` | Paper/source control | Uses paper `lambda=0.1`, source expectile 0.5, no separate IQL or ratio pretraining, 1,000-event initial rollout, and per-update ratio training. The transit hurdle actor and chronological SMDP representation are task-required adaptations. |
| `h2oplus` | Proposed transit adaptation | Adds 50,000 IQL updates, a 5,000-event frozen-policy warmup, 1,000 ratio-pretraining updates, expectile 0.7, and ratio updates every four critic steps. |
| `h2oplus-no-ratio` | Component ablation | Removes conditional dynamics-ratio weighting from the transit adaptation. |
| `h2oplus-lambda0` | Component ablation | Removes the offline value anchor from the mixed target while retaining ratio correction and the warm start. |
| `h2oplus-lambda0p2` | Sensitivity | Uses the published ablation value 0.2. |
| `h2oplus-lambda0p5` | Sensitivity | Uses the public-code default and published ablation value 0.5. |
| `h2oplus-lambda1` | Sensitivity | Uses the published ablation endpoint 1.0, which removes the SAC term from the mixed bootstrap. |
| `h2oplus-no-warmstart` | Component ablation | Removes separate IQL pretraining but retains the transit warmup and ratio pretraining. It is not the source control. |

## RLPD mapping

The state-based official protocol uses random initialization, a 50/50
offline/online replay mixture, ten Q-functions, two randomly selected target
Q-functions, critic layer normalization, batch size 256 per critic update, UTD
20, and one actor/temperature update after the 20 critic updates. Its actor loss
uses the mean of all ten online Q-functions. Its scalar-action target entropy is
`-action_dim/2 = -0.5`, and it collects 5,000 online steps before training. The
`rlpd` profile preserves this replay and update protocol. The transit
implementation evaluates only the two sampled target Q heads when constructing
each Bellman target; it does not evaluate all ten and discard eight afterward.

## WSRL mapping

WSRL is a fine-tuning framework initialized by a pretrained offline agent; it is
not tied to one offline algorithm. The paper's main locomotion experiments use
Cal-QL initialization, and its initializer study also evaluates CQL and IQL.
The registered transit profile deliberately uses the supported CQL variant and
must therefore be reported as **WSRL (CQL initialization)**, not as an
unmodified reproduction of the paper's main Cal-QL-initialized result.

The official workflow first creates a CQL checkpoint with `--use_redq`, then
resumes it through the WSRL launcher. The registered profile therefore performs
250,000 CQL initialization updates with batch size 256, ten Q-functions, and
actor/critic layer normalization. Every TD loss trains all ten heads, while the
CQL penalty uses a separately sampled two-head subset with replacement. The
offline actor, critic, and temperature losses use the same pre-update parameters;
temperature samples ten actions per next state.

WSRL then collects 5,000 transitions with the frozen pretrained policy,
discards offline data during fine-tuning, and uses critic UTD 4. Each target
uses the minimum of two target Q-functions sampled with replacement. The online
batch of 1,024 is split into four critic minibatches of 256, followed by one
actor/temperature update on the aggregate batch; that actor loss uses the
minimum of all ten online Q-functions and temperature uses the aggregate next
states. It uses actor/temperature learning rate `1e-4`, critic learning rate
`3e-4`, no entropy term in the Bellman backup, and CQL alpha 5 during
initialization. These choices preserve the WSRL fine-tuning structure while
making the offline initializer explicit.

## Claim boundary

The PyTorch implementations are equation- and update-protocol reproductions,
not line-for-line ports of the JAX or original PyTorch repositories. All methods
share the same v6 chronological SMDP state, hurdle action distribution, reward,
online interaction cap, and target-domain evaluation grid. The RLPD and WSRL
profiles use ten independently parameterized Q heads after one common transit
structured encoder; this differs from ten fully independent source Q networks.
Their target reductions and actor reductions remain source-aligned as described
above. No general actor or critic gradient clipping is applied in the registered
profiles. The ratio estimator alone retains its separate gradient-norm cap of
5. Architecture or task-required changes shared across methods must be
described as controlled adaptations. Only `h2oplus-source` may be used as the
original-H2O+ control; `h2oplus` must be named as the transit-adapted method in
tables and prose. The `wsrl` profile must be labeled WSRL (CQL initialization).

## Controlled SAC comparator

The `sac-mixed` profile is a task-specific control, not a reproduction of RLPD.
It uses SAC updates on balanced target/simulator replay while matching the
transit H2O+ actor, two-critic ensemble, batch size, activation, normalization,
temperature initialization, and UTD ratio. Its purpose is to remove the IQL
warm start, mixed value anchor, and dynamics-ratio correction together without
also changing network capacity.
