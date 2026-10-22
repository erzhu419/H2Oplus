# Issue 5 — Operational Evaluation Extension

Extends `SimpleSAC/eval_with_metrics.py` with passenger-side and
operational metrics requested by the reviewer. The script is now the
single entry point for all evaluation reporting in the paper.

## New CLI flags

| Flag             | Default | Behaviour                                                                                                                                   |
| ---------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `--sumo_seed`    | 42      | Propagated as `--seed N` to libsumo when starting the SUMO process. Patches `SumoRLBridge._start_traci` so we don't fork the upstream file. |
| `--od_scale`     | 1.0     | Propagated as SUMO `--scale F`. SUMO uniformly duplicates/discards both vehicles **and** persons, which is the cleanest in-binary OD knob.  |
| `--large_gap_factor` | 1.5 | Threshold (in units of scheduled headway) for `large_gap_rate`.                                                                             |

## New JSON fields written by the script

```
passenger_wait_mean         scalar (s)
passenger_wait_p90          scalar (s)
passenger_wait_stats        {n, mean, std, min, p25, p50, p75, p90, max}
in_vehicle_delay_stats      same shape
total_travel_time_stats     same shape
excess_waiting_time_stats   same shape
expected_wait_reference_sec H_ref / 2 used to compute excess wait
headway_cv_per_line         dict keyed by "<line>_<dir_letter>_dir<0|1>"
large_gap_rate              scalar in [0,1]
hold_time_dist              {n, mean, std, min, p25, p50, p75, p90, max}
completed_trips             int
jain_fairness               J = (Σ x_i)^2 / (n · Σ x_i^2) over per-line CV
sumo_seed                   echoed run config
od_scale                    echoed run config
```

The original fields (`cumulative_reward`, `headway.*`, `per_line`,
`hw_std_by_hour`, `action_by_gap`, …) are retained unchanged.

## Data sources

* **Passenger waits / travel times.** SUMO does simulate individual
  persons in this configuration (`e_passenger_rou/3_modified_*.rou.xml`).
  `SumoRLBridge.passenger_obj_dic` holds a `Passenger` object per person;
  each completed leg appends `[arrive, board, alight, wait, travel,
  bus_id]` to `Passenger.travel_data_l` (see
  `SUMO_ruiguang/online_control/sim_obj/passenger.py`). We aggregate over
  all completed legs across all persons. We snapshot the dict before
  `bridge.close()` to avoid losing data if upstream cleanup ever drops
  the dict.
* **Excess wait.** Computed as `max(realised − H_ref/2, 0)` where
  `H_ref` is the median of `bridge.line_headways` (per-line scheduled
  headways). The classic "wait = H/2 under random arrivals + perfect
  service" is used as the reference.
* **Headway CV / large-gap rate.** Computed from the per-stop arrival
  event stream that the script already collects (`fwd_hw` vs.
  `target_hw`). Keys include direction (S/X is encoded in the line ID
  suffix; `direction` 0 ≡ X, 1 ≡ S — kept in the key for clarity).
* **Hold-time distribution.** Min / p25 / p50 / p75 / p90 / max of the
  `hold` value applied per decision (already tracked).
* **Completed trips.** Buses with at least one entry in
  `bus_obj.trajectory_dict` at the end of the run. This counts buses
  that served at least one stop. A stricter "served the last scheduled
  stop" version is straightforward but would miss buses that legitimately
  finish early due to off-route termination, so we report the looser
  count and document it.
* **Jain fairness.** Standard Jain index over per-line headway-CV
  (`J = (Σ x)² / (n · Σ x²)`, range `(0, 1]`). Lines with zero CV are
  excluded so a degenerate single-line case returns 1.0.

## Caveats

1. **`--od_scale` granularity.** SUMO `--scale` rescales total demand by
   probabilistic duplication/dropping. It is the only built-in knob that
   is uniform across the loaded passenger route files. We did **not**
   re-write the route XMLs (would require regenerating
   `e_passenger_rou/3_modified_*.rou.xml`). For paper-grade
   intensity-sweeps this is the canonical SUMO approach.
2. **Excess-wait reference.** We use `H_ref = median(line_headways)`
   across all 12 lines, not per-line. Per-line excess can be derived
   from the per-leg `bus_id → line` mapping if needed; reviewers asked
   for one global number, which is what we report.
3. **`completed_trips` is a lower bound** on full-cycle completions in
   short evaluation windows because some buses are still mid-route at
   `--max_steps`. For paper numbers, run with the default
   `--max_steps 18000` (~5h sim).
4. **Seed reproducibility.** The patched `_start_traci` injects
   `--seed N` only on **first** SUMO launch. Subsequent soft-resets via
   `traci.simulation.loadState` reuse the same RNG state from the saved
   `.sbx` snapshot, which is what we want for run-to-run determinism.

## Smoke test

Run on `experiment_output/h2oplus_bus_seed42_26-04-29-10-09-18/checkpoint_best.pt`
with `--max_steps 6000 --sumo_seed 42 --od_scale 1.0`.

```
Pax wait mean: 182.5s   p90: 380.0s   n=73 completed legs
Pax total tt:  931.2s   p90: 1440.0s
In-veh delay:  748.8s   p90: 1294.0s
Excess wait:    38.5s   p90:  140.0s
Completed trips: 123
Jain fairness: 0.797
Large-gap rate: 0.0%  (>=1.5x sched)
Headway CV per line ranges 0.07–0.42 across 12 (line,dir) groups.
```

JSON output: `experiment_output/eval_metrics_smoketest.json`.

## File deltas

* `SimpleSAC/eval_with_metrics.py`: 328 → 534 lines (+206).
* No upstream files (`bus_h2o/sumo_env/rl_bridge.py`,
  `SUMO_ruiguang/online_control/*`) modified — seed/scale wiring is done
  by monkey-patching `_start_traci` from inside the eval script.
