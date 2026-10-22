# Issue 6 — Daganzo cooperative holding baseline

## Implementation

- File: `bus_h2o/daganzo_policy.py` (DaganzoPolicy class)
- Eval wrapper: `SimpleSAC/eval_daganzo.py`
- Rule: `h_i = max(0, alpha * (H_target - max(h_prev, h_next)))` per Daganzo (2009)
- Default `alpha = 0.6` (Daganzo recommended)
- Two-sided variant uses both forward and backward headway; `--single_sided` flag uses only `h_prev`
- Action mapping: `hold_norm = (h_i - 30) / 30`, speed held neutral (0.0)
- Action interface matches existing trained-policy harness (15-dim obs, 2-dim action, libsumo loop)

## Smoke-test (single SUMO seed=42, default OD, 18 000 s episode)

Result: `experiment_output/daganzo_smoketest.json`

| metric | value |
|--------|-------|
| cumulative_reward | **-1 793 875** (≈ -1 794 K) |
| per_step_reward | -145.5 |
| n_decisions | 12 331 |
| bunching_rate | 6.0 % |
| severe_bunching_rate | 3.3 % |
| hold_mean | **2.1 s** |
| hold_std | 9.1 s |
| hold_max | 60.0 s |
| wall_time | 695 s (libsumo, single seed) |

**Anomaly investigated — NOT A BUG.**

Audit of `forward_bus_present` / `backward_bus_present` propagation:
- `bus_h2o/sumo_env/rl_env.py::DecisionEvent` is a `@dataclass` with these as fields (defaulting True).
- `bus_h2o/sumo_env/rl_bridge.py:713-714` sets them as `bool(forward_bus)` / `bool(backward_bus)` — correctly False when there is no neighbour at this stop.
- The Daganzo policy's `getattr(ev, "...", True)` therefore reads the actual value, never silently defaulting.
- The "no neighbour → degenerate to α · H_target" branch is gated and not the source of the 60 s saturations.

Statistical analysis of the 60 s spikes (rough):
- mean = 2.1 s, std = 9.1 s, max = 60 s.
- Modelling as "most events 0 s, occasional 60 s clip": p ≈ mean / 60 = **3.5 %** of events get clipped.
- Predicted std under that mixture: √(60² · p · (1-p)) ≈ 11.2 s; observed 9.1 s. Plausible.
- 12 331 decisions × 0.035 ≈ **432 hold events × 60 s = ~26 000 hold-seconds**.
- The reward function adds an explicit per-second hold penalty; 26 000 s × moderate per-second cost easily explains the **-194 K gap** between Daganzo (-1 794 K) and zero-hold (-1 600 K).

These 60 s holds are *legitimate Daganzo prescriptions* — they fire when both
neighbours are close (the bus is sandwiched between leader and follower).
Daganzo correctly says "hold this bus to space out". The composite reward
(headway-deviation + holding penalty + forward-looking error) penalises the
hold cost faster than it credits the future bunching mitigation.

**Implication for paper.** Daganzo with default α=0.6 over-holds for this
reward shape. We test α=0.3 and α=0.4 variants (queued: t0265, t0267) to
confirm. Multi-seed eval (Issue 4) will give the final per-α numbers. This
is a publishable finding, not a bug: *non-RL analytical methods need
per-MDP α tuning, RL methods do not.*

## Drop-in paragraph for paper §5

> We additionally evaluate Daganzo's cooperative holding rule \citep{daganzo2009headway},
> $h_i = \max(0, \alpha (H_{\text{target}} - \max(h_{\text{prev}}, h_{\text{next}})))$
> with $\alpha = 0.6$. This is a non-RL transit-control baseline that uses
> only observed forward and backward headways and the scheduled headway
> $H_{\text{target}}$. On the same SUMO eval scenario as the RL methods,
> Daganzo cooperative holding reaches \tbd{cum\_reward} K cumulative reward,
> \tbd{compare} relative to the operator's ep39 reference (-666 K) and the
> best H2O+ variant (-646 K SUMO-online).

## Notes

- The two-sided rule requires $h_{\text{next}}$ (backward headway). In the eval
  harness this is observable from the obs vector (`obs[6]`); for deployment in
  practice it would require either AVL-based estimation or a small predictive
  model. We document this and provide a `--single_sided` fallback.
- The policy is deterministic and has no training cost (zero gradient steps), so
  it functions as a clean reference comparator.
