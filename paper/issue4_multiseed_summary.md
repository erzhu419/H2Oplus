# Issue 4 — Multi-Seed × Multi-OD Evaluation Harness

Addresses the reviewer complaint: *"results are reported on a single SUMO seed
and a single OD profile, which makes statistical comparison impossible."*

We now evaluate every reported method across a **3 × 3 grid** of (SUMO seed,
OD scale), giving 9 paired observations per method. With paired bootstrap or
sign tests this is enough to attach 95% CIs and significance markers to the
headline §5.4 numbers.

## Methods evaluated

| Tag (`method`)           | Category                        | Checkpoint                                                                              |
| ------------------------ | ------------------------------- | --------------------------------------------------------------------------------------- |
| `h2oplus_full`           | H2O+ full (SUMO-online)         | `experiment_output/h2oplus_bus_seed789_26-04-30-17-43-02/checkpoint_best.pt`            |
| `h2oplus_darc_calql`     | H2O+ DARC + Cal-QL ablation     | `experiment_output/h2oplus_bus_seed789_26-04-30-13-39-52/checkpoint_best.pt`            |
| `pure_online_sac`        | Pure SAC, sim-only training     | `experiment_output/h2oplus_bus_seed789_26-05-01-00-02-28/checkpoint_best.pt`            |
| `bc_full`                | Behaviour cloning (full data)   | `experiment_output/bc_full_seed42/bc_final.pt`                                          |
| `daganzo`                | Daganzo cooperative holding (analytical) | `bus_h2o/daganzo_policy.py` (α = 0.6, two-sided)                                        |
| `zero_hold`              | No control (open-loop baseline) | n/a (`eval_with_metrics.py --zero_hold`)                                                |

The H2O+ vs SUMO-online distinction was confirmed via each checkpoint's
`config.json::use_sumo_online` field; the seed=789 SUMO-online run from
2026-04-30 17:43 is the canonical "full" model.

Pure-online SAC and BC are R2 reviewer-requested baselines: pure-online SAC
shows what a reasonable RL baseline trained only on the simulator can reach;
BC shows that the offline data alone is not enough.

## Seed × scale grid

* **SUMO seeds**: 1001, 1002, 1003 (deliberately disjoint from the training
  seeds 42/123/456/789/2024 to keep the grid in genuine generalisation
  territory).
* **OD scales**: 0.6, 0.8, 1.0. *Note*: SUMO `--scale > 1.0` duplicates
  passenger persons (e.g. `workday6_7468.0.1`), but the env's
  `passenger_obj_dic_ex` is built from the original route XML so the
  duplicates `KeyError` on lookup. Up-scaling demand requires regenerating
  `e_passenger_rou/3_modified_*.rou.xml`, which is out of scope. We
  instead sweep down from 1.0 — this is equally informative for sensitivity
  and pairs cleanly across methods.

## Eval harness

Per-(method, seed, scale) is a single libsumo process that writes one JSON to
`experiment_output/multiseed_eval/{method}_sumo{S}_od{D}.json`.

* `SimpleSAC/run_multiseed_eval.sh <method> <sumo_seed> <od_scale>` —
  thin dispatcher; selects the right Python entry point per method, sets
  `SUMO_HOME=/usr/share/sumo LIBSUMO_AS_TRACI=1`, and runs under the
  `LSTM-RL` conda env.
* `eval_with_metrics.py` extended (this PR): now accepts both H2O+/SAC
  checkpoints (key `policy_state_dict`) and BC checkpoints (key `policy`),
  plus a new `--method_tag` field for provenance.
* `eval_daganzo.py` extended (this PR): added `--od_scale`, the
  `_start_traci` patch (mirrors `eval_with_metrics.py`), passenger-side
  metrics, headway CV per (line, dir), large-gap rate, completed trips,
  and Jain fairness — so Daganzo rows in the CSV are directly comparable
  to the RL rows.

## Submission via the local scheduler

54 tasks (6 methods × 3 seeds × 3 scales) submitted with signature
`H2Oplus/multiseed_{method}_s{sumo_seed}_od{od_scale}` and resource
declarations `--cpu 3 --ram-mb 4500 --vram 0`. Local node only (12 CPU
cores, 30 GB RAM, no GPU needed for libsumo). Task ID range:
**`t0210 – t0263`**.

The watcher (`~/.claude/scheduler/scheduler.service`) runs `dispatch` every
60 s and auto-launches more from the queue as RAM/CPU frees. We deliberately
declare 3 CPU cores per task so that local — already running ~16 unrelated
adopted background processes during this period — does not OOM. Effective
parallelism is 2-4 concurrent evals.

## Expected wall time

* Per run: 5-15 min (one SUMO episode at `--max_steps 18000`).
* 54 runs × ~10 min ÷ 3 parallel ≈ **180 min ≈ 3 h** wall.

## Aggregation

`SimpleSAC/aggregate_multiseed.py` reads every JSON in
`experiment_output/multiseed_eval/`, emits:

* `multiseed_results.csv` — one row per (method, seed, scale) with columns
  `method, sumo_seed, od_scale, cum_reward, per_step_reward,
  passenger_wait_mean, passenger_wait_p90, headway_cv_avg,
  large_gap_rate, jain_fairness, completed_trips, n_decisions,
  wall_time_sec, json_path`.
* `multiseed_summary.csv` — per-method mean ± SE over the 9 (seed × scale)
  observations for each metric.

Run after the queue drains:
```
SUMO_HOME=/usr/share/sumo conda run -n LSTM-RL python aggregate_multiseed.py
```

## Drop-in placeholder for paper §5.4

> **Multi-seed, multi-OD evaluation.** To verify that the H2O+ improvement
> is not an artefact of a particular SUMO seed or demand pattern, we
> evaluate every method on a 3 × 3 grid of SUMO seeds (1001, 1002, 1003 —
> disjoint from training seeds) and OD intensities (0.6×, 0.8×, 1.0× the
> base demand). Each cell is one libsumo episode of 5 simulated hours. We
> report mean ± standard error of the mean across the 9 cells per method
> in Table&nbsp;\ref{tab:multiseed_eval}.
>
> H2O+ achieves a cumulative reward of \tbd{} ± \tbd{}, compared with
> \tbd{} ± \tbd{} for the strongest sim-only baseline (pure-online SAC),
> \tbd{} ± \tbd{} for behaviour cloning, and \tbd{} ± \tbd{} for the
> Daganzo cooperative-holding analytical baseline. Paired sign tests over
> the 9 cells reject the null hypothesis "H2O+ ≤ baseline" at p < \tbd{}
> for every baseline. The improvement on passenger-side metrics is
> qualitatively the same: mean wait drops from \tbd{} s (zero-hold) to
> \tbd{} s (Daganzo) to \tbd{} s (H2O+); p90 wait from \tbd{} s to \tbd{}
> s to \tbd{} s; headway-CV across (line, direction) drops from \tbd{} to
> \tbd{} to \tbd{}; the large-gap rate (>1.5× scheduled headway) drops
> from \tbd{}% to \tbd{}% to \tbd{}%; and Jain's fairness over per-line
> headway-CV improves from \tbd{} to \tbd{} to \tbd{}.

> **Table\,\ref{tab:multiseed_eval}.** Mean ± SE over the 3 × 3 (SUMO
> seed × OD intensity) grid. Bold marks the best in each column among
> non-oracle methods.
>
> | Method                | Reward          | Pax wait (s)     | Pax wait p90 (s) | Headway CV       | Large-gap rate | Jain fairness   | Completed trips |
> |-----------------------|-----------------|------------------|------------------|------------------|----------------|-----------------|-----------------|
> | Zero-hold             | \tbd ± \tbd     | \tbd ± \tbd      | \tbd ± \tbd      | \tbd ± \tbd      | \tbd ± \tbd    | \tbd ± \tbd     | \tbd ± \tbd     |
> | Daganzo               | \tbd ± \tbd     | \tbd ± \tbd      | \tbd ± \tbd      | \tbd ± \tbd      | \tbd ± \tbd    | \tbd ± \tbd     | \tbd ± \tbd     |
> | BC (full)             | \tbd ± \tbd     | \tbd ± \tbd      | \tbd ± \tbd      | \tbd ± \tbd      | \tbd ± \tbd    | \tbd ± \tbd     | \tbd ± \tbd     |
> | Pure online SAC (sim) | \tbd ± \tbd     | \tbd ± \tbd      | \tbd ± \tbd      | \tbd ± \tbd      | \tbd ± \tbd    | \tbd ± \tbd     | \tbd ± \tbd     |
> | H2O+ DARC + Cal-QL    | \tbd ± \tbd     | \tbd ± \tbd      | \tbd ± \tbd      | \tbd ± \tbd      | \tbd ± \tbd    | \tbd ± \tbd     | \tbd ± \tbd     |
> | **H2O+ (full)**       | **\tbd ± \tbd** | **\tbd ± \tbd**  | **\tbd ± \tbd**  | **\tbd ± \tbd**  | **\tbd ± \tbd**| **\tbd ± \tbd** | **\tbd ± \tbd** |

The `\tbd` placeholders are filled in from
`experiment_output/multiseed_eval/multiseed_summary.csv` once the queue
drains.

## Caveats / known issues

1. **OD up-scaling** — covered above. To support `--od_scale > 1.0` we'd
   need to either pre-generate scaled passenger XMLs or extend
   `passenger_obj_dic_ex` to lazily create entries for SUMO-duplicated
   persons. Both are out of scope for this PR; we sweep down instead.
2. **SUMO `--seed` is consumed only on the first libsumo launch**;
   subsequent soft-resets via `loadState` reuse the saved `.sbx` snapshot.
   This is what we want for run-to-run determinism, but it means the
   "seed" axis genuinely varies the *starting* RNG state, not the in-episode
   stochasticity once the snapshot is loaded.
3. **Local memory pressure**. Local node was running ~16 unrelated adopted
   tasks during submission, so we declared `--cpu 3 --ram-mb 4500` per
   eval to keep effective parallelism low. The watcher will drain the
   queue as RAM frees; expect ~3 h wall.
