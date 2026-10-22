# V7 method protocol audit

Audit date: 2026-08-08

This audit maps each registered method to a pinned paper and implementation.
It does not treat the common transit state, action, reward, or entropy design as
part of the original algorithms. Machine-verifiable commit, file-hash, and
profile checks are recorded in `method_protocol_audit_v7.json`.

## Frozen primary sources

| Method | Paper | Official implementation | Commit |
|---|---|---|---|
| H2O+ | arXiv:2309.12716v2; ICRA 2025, DOI 10.1109/ICRA55743.2025.11127503 | https://github.com/t6-thu/H2Oplus | `02171768283d307244700c69003423704db83b05` |
| RLPD | PMLR 202:1577--1594 (2023) | https://github.com/ikostrikov/rlpd | `c90fd4baf28c9c9ef40a81460a2e395092844f88` |
| WSRL | ICLR 2025, OpenReview HN0CYZbAPw | https://github.com/zhouzypaul/wsrl | `ad4dc1248a138bc15d6e053f2d1dba1b8cfbaca2` |

The WSRL commit was fetched directly from the official repository. The audit
locks its README, training loop, REDQ/WSRL/CQL/SAC configs, locomotion CQL and
WSRL launch scripts, and SAC/CQL agents by SHA-256; the retained server copy
matches those eleven file hashes.

## H2O+

The paper's mixed target combines the offline state-value bootstrap and the
maximum-entropy target-Q bootstrap with coefficient `lambda`. It applies the
target-to-simulator conditional transition ratio only to the simulator TD
residual, learns the state value from target-domain offline data, and trains the
actor on the mixed target/simulator state batch. The paper fixes `lambda=0.1`
in comparative experiments and reports the grid `{0, 0.1, 0.2, 0.5, 1}`. The
public class default is `0.5`; the registered source control follows the paper's
experimental value, while `0.5` remains a named sensitivity condition.

The official code computes actor, value, and critic losses from pre-update
networks and applies the actor optimizer before the critic optimizer. V7 now
registers this as `actor-before-critic` for the complete H2O+ family. The
`h2oplus-source` control uses two Q-functions, batch 256, 50/50 offline/simulator
replay, expectile 0.5, no separate offline or ratio pretraining, one initial
1,000-event simulator rollout, and per-update ratio training. The proposed
`h2oplus` profile deliberately adds 50,000 IQL updates, a 5,000-event frozen
policy warm-up, 1,000 ratio updates before online learning, expectile 0.7, and
ratio updates every four critic steps. It must be labeled a transit adaptation,
not an unmodified H2O+ reproduction.

## RLPD

The independent official state-based RLPD implementation uses random
initialization, 5,000 uniformly random environment actions before updates,
50/50 offline/online replay, batch size 256 per critic step, ten Q-functions,
two randomly sampled target Q-functions, critic layer normalization, and UTD
20. It performs 20 critic updates and then one actor/temperature update on the
last critic minibatch; the actor objective uses the mean of all ten Q values.

V7 therefore registers `uniform-box` warm-up actions and
`critic-before-actor`, rather than sampling the randomly initialized hurdle
actor. The WSRL repository also contains an RLPD-named comparison with UTD 4;
that script is not the authoritative RLPD protocol and is not used here.

## WSRL

WSRL is an offline-initialization plus warm-up and no-retention fine-tuning
framework, not one fixed offline algorithm. Most paper experiments use Cal-QL,
but Appendix B explicitly uses CQL for all nine dense-reward MuJoCo locomotion
tasks because their datasets lack terminal return-to-go values required by the
Cal-QL regularizer. The transit comparison is therefore labeled **WSRL (CQL
initialization)** and follows this published locomotion variant.

The official README specifies a two-stage workflow: first run the locomotion CQL
launcher with `--use_redq`, then resume its checkpoint with the WSRL launcher.
The CQL stage performs 250,000 updates with its default batch size 256. REDQ
configuration supplies ten Q-functions, two-head subsampling with replacement,
and actor/critic layer normalization. The TD loss trains all ten heads, while
each CQL penalty uses a separately sampled two-head subset. Actor, critic, and
temperature gradients in an offline step are computed from the same pre-update
parameters. Temperature entropy is sampled from next states with ten actions
per state, matching the CQL max-target proposal count.

After checkpoint loading, WSRL collects 5,000 transitions with the frozen
policy, retains no offline data, and uses critic UTD 4. Its online batch of
1,024 is split into four critic minibatches of 256, followed by one
actor/temperature update on the aggregate batch. The actor takes the minimum of
all ten Q heads; temperature uses one independently sampled action per aggregate
next state. Actor and temperature learning rates are `1e-4`, critic learning
rate is `3e-4`, CQL alpha is 5 during initialization, max-target backup is used
during initialization, and entropy is excluded from the Bellman backup.

## Shared adaptation boundary

All methods use the same v7 chronological transit SMDP, structured 229-D state,
scalar hold action, generalized passenger-time reward, continuous-time
discount, and evaluation cells. The hurdle actor uses a point mass at zero plus
a continuous positive-hold density. Its entropy is defined against a
counting-plus-Lebesgue base measure, so continuous-control target-entropy
defaults from the source repositories are not transferable. A common explicit
v7 entropy target and reward scale must be frozen after registered diagnostics
and before any formal training. The hybrid entropy is bounded above by
`log(3)` but has no finite lower bound; negative candidate targets are therefore
mathematically valid and must not be rejected by an artificial zero lower bound.

The PyTorch implementations are equation- and update-protocol reproductions,
not line-for-line ports. The ten Q heads share the same transit state encoder
but retain independent post-encoder parameters. Papers and tables must disclose
this architectural adaptation, call the main method `transit-adapted H2O+`, use
`h2oplus-source` only for the source control, and label WSRL with its CQL
initializer.
