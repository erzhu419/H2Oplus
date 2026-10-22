# H2O+ Paper — Round 2 Revision Plan

Target: address GPT review issues **3, 4, 5, 6, 8** (require new experiments / data work, not just text edits).

Status legend: `[exists]` script ready · `[partial]` data partially in repo · `[new]` need to write or collect.

---

## Issue 3 — Real-corridor calibration evidence

**Reviewer ask:** route-level travel-time MAPE/RMSE; stop-level headway dist; dwell-time fit; passenger boarding/alighting fit; bunching-rate / headway-CV agreement; train/calibration/test day separation.

**What we have:**
- `bus_h2o/calibrated_env/data/{102S,102X,122S,122X,311S,311X,406S,406X,705S,705X,7S,7X}/` — 12 per-line directories with `passenger_OD.xlsx`, `route_news.xlsx`, `stop_news.xlsx`, `time_table.xlsx`. **[partial]** This is the calibration *input*, not validation output.
- `bus_h2o/verify_results/v3_travel_times.csv` — segment-level `mean_sec, std_sec, min_sec, max_sec` per direction (X01→X02 etc.). **[exists]** This is SUMO segment travel-time, but unclear if validated against real AVL.
- `bus_h2o/compare_results/sumo_episode_metrics.csv`, `sim_episode_metrics.csv` — episode-level boardings/alightings/wall-time per env. **[exists]** sim vs SUMO comparison done; not vs real AVL.
- `bus_h2o/calibrate_speeds.py` — calibration script. **[exists]** Need to read what data source it uses.

**What's missing:**
- Real AVL/APC/IC-card data files. Did you receive any? If yes, where?
- Train/calibration/test day separation logic. The current single OD profile doesn't split.

**Action items:**
1. **Ask user** (highest priority): is real AVL/APC data available, and where? If no, the only honest path is to reframe §6.2 calibration disclaimer harder — admit "synthetic OD on real topology" is the actual contribution, and rule out a calibration-claim addition for this revision.
2. If data exists: write `bus_h2o/calibration_report.py` that produces:
   - Route-level travel-time MAPE table (real vs SUMO per line × direction)
   - Stop-level headway-CV table (real vs SUMO)
   - Dwell-time KS-test or Wasserstein distance per stop
   - Bunching-rate comparison
   - Train/calibration/test day list
3. Append `\subsection{Calibration validation results}` to §6.2 with the produced numbers and a single calibration figure (4-panel: travel-time scatter, headway-CV scatter, dwell-time density overlay, bunching-rate bar).

**Effort estimate:** if real data exists and is in repo-readable format → 1 day to write report script + half day to write text. If data does NOT exist → only text-side reframing (~2 hours).

---

## Issue 4 — Multi-seed / multi-scenario evaluation

**Reviewer ask:** 5–10 evaluation seeds per training seed; varied passenger demand days; traffic-intensity perturbations; incident scenarios; paired evaluation under common random numbers; heuristic baseline on the same seed set.

**What we have:**
- `eval_with_metrics.py` — supports `--max_steps` and reads checkpoint. **[exists]** No `--seed` flag yet visible from header; need to add.
- ep39 reference policy as code in heuristic_policy module.
- Multiple H2O+ checkpoints in `experiment_output/h2oplus_bus_seed{42,123,456,789,2024}_*`. **[exists]**

**What's missing:**
- Multi-seed SUMO eval harness (currently each checkpoint → 1 SUMO eval).
- Demand perturbation scenarios (scale OD by ×0.8, ×1.0, ×1.2; or shift peak).
- Incident scenarios (random link blockage; lane closure).
- Paired CRN evaluation (same SUMO seed for all methods on each evaluation point).

**Action items:**
1. Extend `eval_with_metrics.py` with `--sumo_seed N --od_scale F --incident_config PATH` flags. **Critical:** SUMO single-libsumo-session constraint means each (checkpoint, sumo_seed) is one process; budget accordingly.
2. Write `run_multiseed_eval.sh` that loops:
   ```
   for method in {h2oplus_sim_is, h2oplus_sumo_transdisc, calql_offline, ep39, zero_hold}:
     for sumo_seed in {1001..1010}:
       for od_scale in {0.8, 1.0, 1.2}:
         eval --checkpoint $method --sumo_seed $sumo_seed --od_scale $od_scale
   ```
3. Aggregate to a CSV with `method, training_seed, sumo_seed, od_scale, cum_reward, ...`.
4. New table: **Tab. multi-seed**, mean ± SE across (training_seed × sumo_seed × od_scale), with paired Wilcoxon p-values vs ep39 on the same scenario set.
5. Replace §6.3 "Single-scenario evaluation" disclaimer with concrete results.

**Effort estimate:** SUMO eval is ≈3 min per (checkpoint, scenario). Budget: 5 methods × 5 training seeds × 10 sumo seeds × 3 OD scales = 750 runs × 3 min = 37 hours sequential. If `H2O+ SUMO-online` is included that's another 4 × 10 × 3 = 120 runs. Realistic on overnight + weekend with `libsumo` serial. **2 days wall time + 1 day to write/aggregate.**

---

## Issue 5 — Operational (passenger-side) metrics

**Reviewer ask:** mean & 90th-pct passenger waiting time, in-vehicle delay, total passenger travel time, excess waiting time, headway CV by route/direction, large-gap rate, holding-time distribution, completed trips, line-level fairness.

**What we have:**
- `eval_with_metrics.py` already reports: cumulative reward, headway std, bunching rate, action distribution, per-line breakdown. **[exists]**
- `bus_h2o/compare_results/sumo_stop_metrics.csv` — stop-level metrics. **[exists]** Need to inspect what's in it.

**What's missing in `eval_with_metrics.py`:**
- Passenger waiting time (need to track per-passenger arrival time and boarding time → likely already in SUMO trip output)
- In-vehicle delay vs scheduled (need scheduled travel time + actual)
- Excess waiting time = mean(actual wait) − scheduled wait based on headway
- Headway CV per route+direction (currently just std)
- Large-gap rate (already partially via bunching threshold)
- Holding time distribution (script logs actions, need histogram)
- Completed trips count (in SUMO output)
- Line-level fairness (Jain's index over per-line headway-CV)

**Action items:**
1. Extend `eval_with_metrics.py`:
   - Hook into SUMO trip output (`tripinfo-output`) for passenger times.
   - Add `passenger_wait_mean, wait_p90, in_vehicle_delay, excess_wait, headway_cv_per_line, large_gap_rate, hold_time_dist, completed_trips, jain_fairness` to JSON output.
2. Re-run eval on the 5 main checkpoints (TransDisc-sumo, Contrastive-sim, CalQL-offline, SAC-online, ep39) under the multi-seed protocol from Issue 4.
3. New table: **Tab. operational**, with passenger-side metrics by method.
4. Move existing §6.4 disclaimer text to actually-reported subsection.

**Effort estimate:** 1 day to extend script + bundle with Issue 4 multi-seed runs (no extra SUMO time, just more output fields per run). **+1 day for table + figure.**

---

## Issue 6 — Missing baselines

**Reviewer ask:** BC on full data, BC on ep39-only data, IQL/AWAC/RLPD, Daganzo/Xuan analytical holding, longer pure-online SAC.

**What we have:**
- `SimpleSAC/sim2real_td3bc.py` + `sim2real_td3bc_main.py` — TD3+BC implementation. **[exists]**
- `SimpleSAC/train_offline_only.py` — offline-only training entry. **[exists]** Likely supports CalQL; check if it can do BC by setting `cql_alpha=0`.
- ep39 heuristic — used as reference.
- No Daganzo / Xuan analytical holding implementation.
- No IQL or AWAC implementation.

**Action items:**
1. **BC**: re-use `train_offline_only.py` with policy-only loss (no Q-loss); train on (a) full $\Doff$ and (b) ep39-only subset. → 2 new BC checkpoints.
2. **TD3+BC**: re-train using `sim2real_td3bc_main.py` on full $\Doff$ → 1 checkpoint per seed × 5 seeds.
3. **IQL**: write `SimpleSAC/train_iql.py` (~200 lines, standard implementation). Train 5 seeds.
4. **Daganzo cooperative holding**: write `bus_h2o/daganzo_policy.py` — implements forward-headway-tracking holding rule from Daganzo (2009). No training needed; just an evaluable policy.
5. **Pure online SAC longer budget**: re-run pure-online SAC for 1000 epochs (vs current 200) on 3 seeds to test whether the `-1654K` failure is budget-bound. → cheap (sim-core only).
6. New rows in main results table; expanded baseline section.

**Effort estimate:** BC (1 day code + 1 day train), TD3+BC (already coded, 1 day train), IQL (2 days code + 1 day train), Daganzo (1 day code + 0 train), longer SAC (0 code, 1 day train sim-core). **Total: ≈8 days**. If we drop IQL → 6 days.

**Recommendation:** prioritize **BC + Daganzo + longer-SAC** (5 days) for biggest reviewer-perceived value. IQL/AWAC/RLPD can be acknowledged as "future comparison" if time-bound.

---

## Issue 8 — Action-invariance data-level verification

**Reviewer ask:** report data-level histogram check (not just learned-discriminator permutation sensitivity).

**What we have:**
- `SimpleSAC/verify_assumption1.py` (224 lines) — **already implements** PCA + binning + log-ratio check + heatmap output. **[exists, just needs running.]**

**What's missing:**
- A SUMO-side h5 transition file: `--real_h5 ../bus_h2o/datasets_v2/merged_all_v2.h5` (referenced in script docstring). Need to verify this file exists.
- A sim-core h5 from the same initial state distribution: `../experiment_output/sim_rollouts_for_verification.h5`. Script header references `rollout_sim_for_verification.py` to generate this.

**Action items:**
1. Verify `merged_all_v2.h5` exists (the offline buffer). If not, build it from the 4 behaviour-policy h5s.
2. Find or write `rollout_sim_for_verification.py` to generate matching sim-core rollouts from snapshots in $\Doff$ (re-using snapshot injection from training).
3. Run `verify_assumption1.py --real_h5 ... --sim_h5 ... --out_dir ...`.
4. Report `action_dep_ratio.txt` value in §5 Analysis subsection. Embed `heatmap_per_action.pdf` as a new figure (or supplement).
5. Replace the current honest-note paragraph with concrete numbers.

**Effort estimate:** 1 hour to verify h5s exist + 0.5 day to write rollout-for-verification script (if missing) + 1 hour to run + 1 hour to write text. **Total: ≈1 day.**

---

## Suggested execution order (by impact / effort ratio)

| Order | Issue | Effort | Why first |
|-------|-------|--------|-----------|
| 1 | Issue 8 (action invariance) | 1 day | Cheapest; closes a specific reviewer-pointed loose end; script exists |
| 2 | Issue 5 (operational metrics) | +2 days | Reuses existing eval script; bundles with Issue 4 runs at no extra SUMO cost |
| 3 | Issue 4 (multi-seed eval) | 3 days wall | Requires SUMO time but easily parallelisable across days; new tables + fixes statistical significance hole |
| 4 | Issue 6 (baselines) | 5–8 days | Mostly new code; can run BC/Daganzo first (cheap), TD3+BC retrains second, longer-SAC overnight |
| 5 | Issue 3 (calibration) | depends on data availability | **Blocked by user input on real AVL data** |

**Critical path / blocker:** Issue 3 requires answering: *do we have real AVL/APC data, and where?* If no, the honest path is to leave §6.2 as a limitation and not over-promise calibration in this revision.

**Total effort (optimistic):** ≈12 working days + SUMO wall time. Realistic: **3 weeks** for a complete round-2 revision.

---

## Per-section paper changes after experiments land

- **§5.1 Main table:** add BC, BC-ep39, TD3+BC, Daganzo rows; add multi-seed mean ± SE (replacing single-seed numbers); add paired Wilcoxon p-values.
- **§5.2 Operational metrics table (NEW):** passenger wait, headway CV, large-gap rate, hold-time stats per method.
- **§5.3 Action-invariance subsection:** replace text-only honest note with `action_dep_ratio` value and heatmap figure.
- **§5.x Multi-scenario robustness (NEW):** OD-perturbation results in a small table.
- **§6.2 Calibration:** *if data available*, replace disclaimer with results subsection. *If not*, strengthen the limitation language.
- **§6.3 Single-scenario eval:** delete (replaced by §5.x).
- **§6.4 Operational metrics:** delete (replaced by §5.2).
- **§7 Conclusion:** update numbers and remove "deferred to revision" language for everything we actually did.

---

## What blocks each item

- **Issue 3:** real-data availability (user input needed).
- **Issue 4:** SUMO wall-clock time only.
- **Issue 5:** Issue 4 runs + script extension.
- **Issue 6:** code time + GPU/CPU compute.
- **Issue 8:** likely just needs h5 verification + small wrapper script.
