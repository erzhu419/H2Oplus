"""
collect_sim_log.py  (v3 — libsumo bridge + SimpleSim dual logging + timing)
============================================================================
Records every decision and execution event from BOTH SUMO and SimpleSim into
the same CSV schema, tagged  env_type='sumo' / 'sim'.

SUMO side  — uses the SAME libsumo bridge as sac_v2_bus_SUMO.py
             (SUMO_ruiguang.online_control.rl_bridge:build_bridge)
             Passes through SumoBusHoldingEnv; DecisionEvents become 'sumo' rows.

SimpleSim  — MultiLineSimEnv stepped at 1-sec ticks, locked to SUMO episode.
             With prob p_reset the sim is fully reset.

Timing     — wall-clock time is measured separately for SUMO and SimpleSim
             steps each episode.  If  sim_time / sumo_time > 0.40  for the
             majority of episodes, cProfile is dumped automatically.

Run modes
---------
  # Without SUMO (sim-only, for testing / CI):
  cd .../H2Oplus/bus_h2o
  python collect_sim_log.py --n_episodes 10 --p_reset 0.01

  # With SUMO (libsumo, same as LSTM-RL training):
  cd .../H2Oplus/bus_h2o
  python collect_sim_log.py --n_episodes 10 --p_reset 0.01 --with_sumo

Output files  (in --log_dir, default sim_logs/)
-----------------------------------------------
  decisions_{ts}.csv      — one row per decision event (both envs)
  executions_{ts}.csv     — one row per bus-at-station (sim only)
  episode_summary_{ts}.csv
  timing_{ts}.csv
  analysis_{ts}.txt
  [profile_{ts}.pstat]    — only when sim is too slow
"""

import argparse, csv, os, sys, random, datetime, time, io, math
import cProfile, pstats
import importlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# ── path setup ────────────────────────────────────────────────────────────────
_HERE  = os.path.dirname(os.path.abspath(__file__))
_ROOT  = os.path.abspath(os.path.join(_HERE, "../.."))   # sumo-rl/
_SUMO_OC = os.path.join(_ROOT, "SUMO_ruiguang", "online_control")
for p in [_HERE, _ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from envs.bus_sim_env import MultiLineSimEnv
from common.data_utils import extract_structured_context, set_route_length

CALIB_PATH = os.path.join(_HERE, "calibrated_env")
SUMO_ROOT  = _SUMO_OC
SUMO_SCHEDULE = "initialize_obj/save_obj_bus.add.xml"
SUMO_BRIDGE   = "SUMO_ruiguang.online_control.rl_bridge:build_bridge"

SLOW_RATIO_THRESHOLD = 0.40   # sim/sumo wall-time threshold
SLOW_EPISODE_FRACTION = 0.5   # trigger cProfile if >50% episodes are slow
# Buses first depart at ~1200s, arrive at first station ~2000-2500s.
# Don't allow resets until after that warmup.
MIN_STEPS_BEFORE_RESET = 2500

# ── shared column schema ──────────────────────────────────────────────────────
OBS_FIELDS = [
    "obs_line_idx", "obs_bus_idx", "obs_station_id", "obs_time_period",
    "obs_direction", "obs_forward_hw", "obs_backward_hw", "obs_waiting_pax",
    "obs_target_hw", "obs_base_dwell", "obs_sim_time", "obs_gap",
    "obs_co_fwd_hw", "obs_co_bwd_hw", "obs_seg_speed",
]
Z_FIELDS  = [f"z_{i}" for i in range(30)]
DEC_COLS  = (
    ["env_type", "episode", "sumo_step", "sim_time",
     "line_id", "bus_id", "station_id", "station_name", "direction"]
    + OBS_FIELDS + ["action", "reward"] + Z_FIELDS
)
EXC_COLS  = [
    "env_type", "episode", "sumo_step", "sim_time",
    "line_id", "bus_id", "station_id", "station_name", "direction",
    "action_executed", "passengers_onboard", "waiting_at_station",
    "forward_headway", "backward_headway",
]
EP_COLS   = [
    "episode", "sumo_steps",
    "sim_decisions", "sumo_decisions",
    "sim_executions", "sim_resets",
    "sim_reward", "sumo_reward",
    "sumo_wall_sec", "sim_wall_sec", "sim_sumo_ratio",
]
TIMING_COLS = ["episode", "sumo_step", "sumo_step_ms", "sim_step_ms"]


# ── helpers ───────────────────────────────────────────────────────────────────

def ms() -> float:
    return time.perf_counter() * 1000.0

def zero_actions_sim(env: MultiLineSimEnv) -> dict:
    return {lid: {k: 0.0 for k in range(le.max_agent_num)}
            for lid, le in env.line_map.items()}

def any_obs(state: dict) -> bool:
    return any(bool(v) for bd in state.values() for v in bd.values())

def make_snapshot(env: MultiLineSimEnv) -> dict:
    buses, stations = [], []
    for lid, le in env.line_map.items():
        cum, rc = 0.0, [0.0]
        for r in le.routes:
            cum += r.distance; rc.append(cum)
        for bus in le.bus_all:
            if getattr(bus, "on_route", False):
                buses.append({
                    "pos":   bus.absolute_distance,
                    "speed": le.route_state[0] if le.route_state else 5.0,
                    "load":  len(getattr(bus, "passengers", [])),
                })
        for i, st in enumerate(le.stations):
            wp = getattr(st, "waiting_passengers", [])
            stations.append({
                "pos":           rc[min(i, len(rc) - 1)],
                "waiting_count": int(len(wp)) if hasattr(wp, "__len__") else 0,
            })
    return {"sim_time": env.current_time, "all_buses": buses, "all_stations": stations}

def compute_z(env: MultiLineSimEnv, route_len: float) -> np.ndarray:
    set_route_length(route_len)
    return extract_structured_context(make_snapshot(env))

def null_z() -> list:
    return [0.0] * 30

def policy(obs: list) -> float:
    """Zero-hold. Replace with trained policy for real experiments."""
    return 0.0

def sta_name(env: MultiLineSimEnv, lid: str, sid: int) -> str:
    le = env.line_map.get(lid)
    if le:
        for st in le.stations:
            if st.station_id == sid:
                return st.station_name
    return f"id_{sid}"


# ── SUMO bridge loader (mirrors sac_v2_bus_SUMO.py) ──────────────────────────

def load_sumo_env(sumo_root: str, update_freq: int):
    """
    Loads SumoBusHoldingEnv via the LSTM-RL bridge (libsumo).
    Returns (sumo_env, close_callback) or raises on failure.
    """
    # Same import chain as sac_v2_bus_SUMO.py
    mod_name, _, fn_name = SUMO_BRIDGE.partition(":")
    bridge_module = importlib.import_module(mod_name)
    factory = getattr(bridge_module, fn_name or "build_bridge")
    bridge = factory(root_dir=sumo_root, gui=False, update_freq=update_freq)

    if isinstance(bridge, tuple):
        decision_provider = bridge[0]
        action_executor   = bridge[1]
        reset_cb          = bridge[2] if len(bridge) > 2 else None
        close_cb          = bridge[3] if len(bridge) > 3 else None
    elif isinstance(bridge, dict):
        decision_provider = bridge["decision_provider"]
        action_executor   = bridge["action_executor"]
        reset_cb          = bridge.get("reset_callback")
        close_cb          = bridge.get("close_callback")
    else:
        raise ValueError("Bridge must return tuple or dict")

    from SUMO_ruiguang.online_control.rl_env import SumoBusHoldingEnv
    sumo_env = SumoBusHoldingEnv(
        root_dir=sumo_root,
        schedule_file=SUMO_SCHEDULE,
        decision_provider=decision_provider,
        action_executor=action_executor,
        reset_callback=reset_cb,
        close_callback=close_cb,
    )
    return sumo_env, close_cb


# ── row builders ──────────────────────────────────────────────────────────────

def make_sumo_dec_row(
    episode: int, sumo_step: int,
    line_id: str, bus_id, obs15: list,
    action: float, reward: float,
) -> Dict[str, Any]:
    """Build a 'sumo' decision row from a 15-dim obs and DecisionEvent fields."""
    # obs15 layout mirrors _register_event in rl_env.py:
    # [line_idx, bus_idx, station_idx, time_period, direction,
    #  fwd_hw, bwd_hw, waiting, target_hw, base_dwell, sim_time, gap,
    #  co_fwd_hw, co_bwd_hw, seg_speed]
    row: Dict[str, Any] = {
        "env_type":    "sumo",
        "episode":     episode,
        "sumo_step":   sumo_step,
        "sim_time":    obs15[10] if len(obs15) > 10 else 0.0,
        "line_id":     line_id,
        "bus_id":      bus_id,
        "station_id":  int(obs15[2]) if len(obs15) > 2 else -1,
        "station_name":str(line_id) + "_sta" + str(int(obs15[2])) if len(obs15) > 2 else "?",
        "direction":   int(obs15[4]) if len(obs15) > 4 else 1,
        "action":      round(float(action), 3),
        "reward":      round(float(reward), 4),
    }
    for i, fn in enumerate(OBS_FIELDS):
        row[fn] = round(float(obs15[i]), 4) if i < len(obs15) else 0.0
    # z is 0 unless we have a real SUMO snapshot extractor
    for zf in Z_FIELDS:
        row[zf] = 0.0
    return row


def make_sim_dec_rows(
    state: dict, reward: dict, sim_env: MultiLineSimEnv,
    z: np.ndarray, episode: int, sumo_step: int,
) -> List[Dict]:
    rows = []
    for lid, bd in state.items():
        for bus_id, v in bd.items():
            if not v:
                continue
            obs = v[-1]
            act = policy(obs)
            rew = float((reward.get(lid) or {}).get(bus_id) or 0.0)
            sid = int(obs[2]) if len(obs) > 2 else -1
            le  = sim_env.line_map.get(lid)
            row: Dict[str, Any] = {
                "env_type":    "sim",
                "episode":     episode,
                "sumo_step":   sumo_step,
                "sim_time":    le.current_time if le else 0.0,
                "line_id":     lid,
                "bus_id":      bus_id,
                "station_id":  sid,
                "station_name":sta_name(sim_env, lid, sid),
                "direction":   int(obs[4]) if len(obs) > 4 else 1,
                "action":      round(act, 3),
                "reward":      round(rew, 4),
            }
            for i, fn in enumerate(OBS_FIELDS):
                row[fn] = round(float(obs[i]), 4) if i < len(obs) else 0.0
            for j, zf in enumerate(Z_FIELDS):
                row[zf] = round(float(z[j]), 5) if j < len(z) else 0.0
            rows.append(row)
    return rows


def collect_sim_executions(sim_env, episode, sumo_step):
    rows = []
    for lid, le in sim_env.line_map.items():
        for bus in le.bus_all:
            if not getattr(bus, "on_route", False): continue
            if not getattr(bus, "in_station",  False): continue
            st = bus.last_station
            wp = getattr(st, "waiting_passengers", [])
            fw = getattr(bus, "forward_headway",  None)
            bw = getattr(bus, "backward_headway", None)
            rows.append({
                "env_type":           "sim",
                "episode":            episode,
                "sumo_step":          sumo_step,
                "sim_time":           le.current_time,
                "line_id":            lid,
                "bus_id":             bus.bus_id,
                "station_id":         st.station_id,
                "station_name":       st.station_name,
                "direction":          int(getattr(bus, "direction", 1)),
                "action_executed":    round(getattr(bus, "holding_time", 0.0), 3),
                "passengers_onboard": len(getattr(bus, "passengers", [])),
                "waiting_at_station": int(len(wp)) if hasattr(wp, "__len__") else 0,
                "forward_headway":    round(float(fw), 3) if fw is not None else -1.0,
                "backward_headway":   round(float(bw), 3) if bw is not None else -1.0,
            })
    return rows


# ── main ──────────────────────────────────────────────────────────────────────

def run_episodes(args, log_dir: str, ts: str,
                 dec_writer, exc_writer, ep_writer, timing_writer,
                 profiler=None) -> List[float]:
    """Core loop. Returns list of sim/sumo timing ratios per episode."""

    # ── init route_len for z ──────────────────────────────────────────────
    _tmp = MultiLineSimEnv(CALIB_PATH)
    route_len = max(
        sum(r.distance for r in le.routes) if le.routes else 14000.0
        for le in _tmp.line_map.values()
    )
    del _tmp
    set_route_length(route_len)

    # ── SUMO env (optional) ───────────────────────────────────────────────
    sumo_env = None
    sumo_close = None
    sumo_step_count = 0   # counts SumoBusHoldingEnv.step() calls
    if args.with_sumo:
        print("  Loading SUMO (libsumo) bridge ...")
        try:
            sumo_env, sumo_close = load_sumo_env(SUMO_ROOT, args.passenger_update_freq)
            print("  SUMO bridge OK.")
        except Exception as e:
            print(f"  SUMO bridge FAILED: {e}")
            print("  Falling back to sim-only mode.")
            sumo_env = None

    ratios: List[float] = []

    for ep in range(1, args.n_episodes + 1):
        print(f"\n{'='*60}")
        print(f"  Episode {ep}/{args.n_episodes}  [{datetime.datetime.now().strftime('%H:%M:%S')}]")
        print(f"{'='*60}")

        # ── Init SimpleSim ────────────────────────────────────────────────
        sim_env = MultiLineSimEnv(CALIB_PATH)
        sim_env.reset()
        # Fast-forward past warmup (buses depart at ~1200s, reach stations ~2000s)
        sim_env.initialize_state()
        sim_actions = zero_actions_sim(sim_env)
        sim_done        = False
        steps_since_reset = MIN_STEPS_BEFORE_RESET  # already warmed up, allow reset early if needed

        # ── Init SUMO episode ─────────────────────────────────────────────
        sumo_done        = False
        sumo_obs         = {}
        sumo_rew         = {}
        sumo_action_dict = {}
        station_feature_idx = 2   # obs[2] == station_id in rl_env scheme

        if sumo_env is not None:
            sumo_env.reset()
            sumo_obs, sumo_rew, sumo_done = sumo_env.reset()
            sumo_action_dict = {
                lid: {bid: None for bid in buses}
                for lid, buses in sumo_obs.items()
            }

        ep_sim_dec   = 0
        ep_sumo_dec  = 0
        ep_sim_exc   = 0
        ep_resets    = 0
        ep_sim_rew   = 0.0
        ep_sumo_rew  = 0.0
        sumo_wall    = 0.0   # total ms in sumo_env.step()
        sim_wall     = 0.0   # total ms in sim_env.step()

        sumo_step = 0

        while not sumo_done:
            sumo_step += 1
            sumo_step_ms = 0.0

            # ── SUMO step ─────────────────────────────────────────────────
            if sumo_env is not None:
                # Build action dict using zero-hold policy on current obs
                for lid, buses in sumo_obs.items():
                    if lid not in sumo_action_dict:
                        sumo_action_dict[lid] = {}
                    for bus_id, hist in buses.items():
                        if hist:
                            obs15 = hist[-1]
                            sumo_action_dict[lid][bus_id] = policy(obs15)

                t0 = ms()
                try:
                    sumo_obs, sumo_rew, sumo_done, _ = sumo_env.step(sumo_action_dict)
                except Exception as ex:
                    print(f"  [ep{ep} step{sumo_step}] SUMO step error: {ex}")
                    sumo_done = True
                    break
                sumo_step_ms = ms() - t0
                sumo_wall += sumo_step_ms

                # Log SUMO decision events
                for lid, buses in sumo_obs.items():
                    for bus_id, hist in buses.items():
                        if not hist:
                            continue
                        obs15  = hist[-1]
                        act    = policy(obs15)
                        rew    = float((sumo_rew.get(lid) or {}).get(bus_id) or 0.0)
                        ep_sumo_rew += rew
                        row = make_sumo_dec_row(ep, sumo_step, lid, bus_id,
                                                obs15, act, rew)
                        dec_writer.writerow(row)
                        ep_sumo_dec += 1

            else:
                # No SUMO: fake episode ends after sumo_steps
                if sumo_step >= args.sumo_steps:
                    sumo_done = True
                    break

            # ── 0.01-prob SimpleSim reset ─────────────────────────────
            # Guard: only allow reset after MIN_STEPS_BEFORE_RESET steps
            # since last reset — otherwise buses never reach stations.
            steps_since_reset += 1
            if (random.random() < args.p_reset
                    and not sim_done
                    and steps_since_reset >= MIN_STEPS_BEFORE_RESET):
                sim_env = MultiLineSimEnv(CALIB_PATH)
                sim_env.reset()
                sim_actions = zero_actions_sim(sim_env)
                sim_done = False
                ep_resets += 1
                steps_since_reset = 0
                print(f"  [ep{ep} step{sumo_step}] *** SimpleSim RESET #{ep_resets} ***")

            # ── SimpleSim step ────────────────────────────────────────────
            if not sim_done:
                t0 = ms()
                try:
                    state, reward, sim_done = sim_env.step(sim_actions)
                except Exception as ex:
                    print(f"  [ep{ep} step{sumo_step}] sim error: {ex}")
                    state, reward, sim_done = {}, {}, False
                sim_step_ms = ms() - t0
                sim_wall += sim_step_ms
            else:
                sim_step_ms = 0.0

            timing_writer.writerow({
                "episode":      ep,
                "sumo_step":    sumo_step,
                "sumo_step_ms": round(sumo_step_ms, 3),
                "sim_step_ms":  round(sim_step_ms, 3),
            })

            # ── Execution log ─────────────────────────────────────────────
            if not sim_done:
                exc_rows = collect_sim_executions(sim_env, ep, sumo_step)
                for r in exc_rows:
                    exc_writer.writerow(r)
                ep_sim_exc += len(exc_rows)

                # Decision log
                if any_obs(state):
                    z = compute_z(sim_env, route_len)
                    dec_rows = make_sim_dec_rows(
                        state, reward, sim_env, z, ep, sumo_step
                    )
                    for r in dec_rows:
                        dec_writer.writerow(r)
                    ep_sim_dec += len(dec_rows)
                    ep_sim_rew += sum(r["reward"] for r in dec_rows)

        # ── Episode timing ────────────────────────────────────────────────
        ratio = (sim_wall / sumo_wall) if sumo_wall > 0 else float("nan")
        ratios.append(ratio)
        slow_flag = "⚠️ SLOW" if (not math.isnan(ratio) and ratio > SLOW_RATIO_THRESHOLD) else "OK"
        print(f"  Episode {ep}: sumo_steps={sumo_step}, "
              f"sim_dec={ep_sim_dec}, sumo_dec={ep_sumo_dec}, "
              f"resets={ep_resets}, "
              f"sim_wall={sim_wall/1000:.2f}s, sumo_wall={sumo_wall/1000:.2f}s, "
              f"ratio={ratio:.3f} {slow_flag}")

        ep_writer.writerow({
            "episode":       ep,
            "sumo_steps":    sumo_step,
            "sim_decisions": ep_sim_dec,
            "sumo_decisions":ep_sumo_dec,
            "sim_executions":ep_sim_exc,
            "sim_resets":    ep_resets,
            "sim_reward":    round(ep_sim_rew, 3),
            "sumo_reward":   round(ep_sumo_rew, 3),
            "sumo_wall_sec": round(sumo_wall / 1000, 3),
            "sim_wall_sec":  round(sim_wall / 1000, 3),
            "sim_sumo_ratio":round(ratio, 4) if not math.isnan(ratio) else "nan",
        })

    if sumo_close is not None:
        try:
            sumo_close()
        except Exception:
            pass

    return ratios


def maybe_run_profile(log_dir: str, ts: str, ratios: List[float]):
    """If too many episodes are slow, re-run sim for 1 episode under cProfile."""
    valid = [r for r in ratios if not math.isnan(r)]
    if not valid:
        return
    n_slow = sum(1 for r in valid if r > SLOW_RATIO_THRESHOLD)
    frac   = n_slow / len(valid)
    print(f"\n  Timing: {n_slow}/{len(valid)} episodes slow (frac={frac:.2f})")
    if frac <= SLOW_EPISODE_FRACTION:
        print("  SimpleSim speed OK — no cProfile needed.")
        return

    print(f"  sim/sumo ratio > {SLOW_RATIO_THRESHOLD:.0%} in majority of episodes")
    print("  Running cProfile on 1 SimpleSim episode ...")
    prof_path = os.path.join(log_dir, f"profile_{ts}.pstat")

    def _profile_run():
        env = MultiLineSimEnv(CALIB_PATH)
        env.reset()
        actions = zero_actions_sim(env)
        for _ in range(5000):
            try:
                _, _, done = env.step(actions)
            except Exception:
                break
            if done:
                break

    pr = cProfile.Profile()
    pr.enable()
    _profile_run()
    pr.disable()
    pr.dump_stats(prof_path)

    # Also print top-20 by cumtime
    buf = io.StringIO()
    ps  = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(20)
    report = buf.getvalue()
    print(report)

    txt_path = prof_path.replace(".pstat", ".txt")
    with open(txt_path, "w") as f:
        f.write(report)
    print(f"  cProfile saved: {prof_path}")
    print(f"  Top-20 report : {txt_path}")


def main(args):
    log_dir = os.path.join(_HERE, args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    dec_path     = os.path.join(log_dir, f"decisions_{ts}.csv")
    exc_path     = os.path.join(log_dir, f"executions_{ts}.csv")
    ep_path      = os.path.join(log_dir, f"episode_summary_{ts}.csv")
    timing_path  = os.path.join(log_dir, f"timing_{ts}.csv")
    ana_path     = os.path.join(log_dir, f"analysis_{ts}.txt")

    print(f"\n{'='*65}")
    print(f"collect_sim_log v3  |  "
          f"n_episodes={args.n_episodes}  p_reset={args.p_reset}  "
          f"sumo={'ON' if args.with_sumo else 'OFF (sim-only)'}")
    print(f"logs → {log_dir}")
    print(f"{'='*65}\n")

    all_decisions:  List[Dict] = []
    all_executions: List[Dict] = []
    ep_summaries:   List[Dict] = []

    with (
        open(dec_path,    "w", newline="") as df,
        open(exc_path,    "w", newline="") as ef,
        open(ep_path,     "w", newline="") as pf,
        open(timing_path, "w", newline="") as tf,
    ):
        dec_w     = csv.DictWriter(df, fieldnames=DEC_COLS,  extrasaction="ignore")
        exc_w     = csv.DictWriter(ef, fieldnames=EXC_COLS,  extrasaction="ignore")
        ep_w      = csv.DictWriter(pf, fieldnames=EP_COLS,   extrasaction="ignore")
        timing_w  = csv.DictWriter(tf, fieldnames=TIMING_COLS, extrasaction="ignore")
        for w in [dec_w, exc_w, ep_w, timing_w]:
            w.writeheader()

        # ── collect decisions/executions row-by-row (streaming) ───────────
        # We'll also mirror into memory for the analysis pass.
        class TeeWriter:
            def __init__(self, real_writer, store):
                self._w = real_writer
                self._s = store
            def writerow(self, row):
                self._w.writerow(row)
                self._s.append(dict(row))

        dec_tee = TeeWriter(dec_w, all_decisions)
        exc_tee = TeeWriter(exc_w, all_executions)

        ratios = run_episodes(args, log_dir, ts, dec_tee, exc_tee, ep_w, timing_w)

    print(f"\n  decisions  → {os.path.basename(dec_path)}  ({len(all_decisions):,} rows)")
    print(f"  executions → {os.path.basename(exc_path)}  ({len(all_executions):,} rows)")

    # ── analysis ──────────────────────────────────────────────────────────
    analyze(all_decisions, all_executions, ratios, ana_path)

    # ── cProfile if needed ────────────────────────────────────────────────
    maybe_run_profile(log_dir, ts, ratios)

    print(f"\nAll done. Logs: {log_dir}")


# ── analysis ──────────────────────────────────────────────────────────────────

def analyze(decisions, executions, ratios, out_path):
    from collections import defaultdict, Counter
    lines_out = []
    W = lines_out.append

    W("=" * 72)
    W("SimpleSim + SUMO  Log Analysis")
    W("=" * 72)

    sim_dec  = [d for d in decisions if d["env_type"] == "sim"]
    sumo_dec = [d for d in decisions if d["env_type"] == "sumo"]
    W(f"Total decision rows: sim={len(sim_dec)}  sumo={len(sumo_dec)}")
    W(f"Total execution rows: sim={len(executions)}")

    # Timing summary
    valid_r = [r for r in ratios if not math.isnan(r)]
    if valid_r:
        W(f"\n── Timing (sim / sumo wall-time ratio per episode) ──")
        for ep_i, r in enumerate(ratios, 1):
            flag = " ⚠️ SLOW" if r > SLOW_RATIO_THRESHOLD else ""
            W(f"  ep{ep_i:2d}: {r:.4f}{flag}")
        W(f"  Mean ratio: {sum(valid_r)/len(valid_r):.4f}  "
          f"threshold={SLOW_RATIO_THRESHOLD:.0%}")

    # Per-line counts
    for tag, rows in [("sim", sim_dec), ("sumo", sumo_dec)]:
        if not rows:
            continue
        W(f"\n── [{tag}] Per-line decisions ──")
        by_line: Counter = Counter(d["line_id"] for d in rows)
        for lid in sorted(by_line):
            W(f"  {lid:6s}: {by_line[lid]}")

    # Episode 1 decision table
    ep1 = [d for d in sim_dec if d["episode"] == 1]
    if ep1:
        W(f"\n── [sim] Episode 1 — first 15 decisions ──")
        W(f"  {'ep':>3} {'step':>6} {'sim_t':>7} {'line':>5} {'bus':>5} "
          f"{'station':>20} {'dir':>3} {'fwd_hw':>8} {'bwd_hw':>8} "
          f"{'act':>5} {'rew':>8}")
        for d in ep1[:15]:
            W(f"  {d['episode']:3d} {d['sumo_step']:6d} {d['sim_time']:7.0f} "
              f"{d['line_id']:>5} {str(d['bus_id']):>5} "
              f"{d['station_name']:>20} {d['direction']:3d} "
              f"{d['obs_forward_hw']:8.1f} {d['obs_backward_hw']:8.1f} "
              f"{d['action']:5.1f} {d['reward']:8.3f}")

    # Episode 1 execution table
    ep1_exc = [e for e in executions if e["episode"] == 1]
    if ep1_exc:
        W(f"\n── [sim] Episode 1 — first 10 executions ──")
        W(f"  {'ep':>3} {'step':>6} {'sim_t':>7} {'line':>5} {'bus':>5} "
          f"{'station':>20} {'dir':>3} {'hold':>5} {'onboard':>7} {'wait':>7}")
        for e in ep1_exc[:10]:
            W(f"  {e['episode']:3d} {e['sumo_step']:6d} {e['sim_time']:7.0f} "
              f"{e['line_id']:>5} {str(e['bus_id']):>5} "
              f"{e['station_name']:>20} {e['direction']:3d} "
              f"{e['action_executed']:5.1f} "
              f"{e['passengers_onboard']:7d} {e['waiting_at_station']:7d}")

    # Obs range
    W("\n── Obs Range (sim) ──")
    for fn in OBS_FIELDS:
        vals = [d.get(fn, 0.0) for d in sim_dec]
        if vals:
            W(f"  {fn:22s}: min={min(vals):10.3f}  max={max(vals):10.3f}  "
              f"mean={sum(vals)/len(vals):10.3f}")

    # Bug checks
    W("\n── Bug Checks ──")
    n_zero = sum(1 for d in sim_dec
                 if all(d.get(f, 0.0) == 0.0 for f in OBS_FIELDS))
    W(f"  All-zero obs rows: {n_zero}")

    n_nan = sum(1 for d in sim_dec
                for f in OBS_FIELDS
                if not math.isfinite(d.get(f, 0.0)))
    W(f"  NaN/Inf obs values: {n_nan}")

    by_lid_idx = defaultdict(set)
    for d in sim_dec:
        by_lid_idx[d["line_id"]].add(d["obs_line_idx"])
    W("  obs_line_idx consistency per line:")
    for lid in sorted(by_lid_idx):
        ids = by_lid_idx[lid]
        W(f"    {lid:6s}: {ids}  {'✓' if len(ids)==1 else '✗ MISMATCH'}")

    # SUMO vs Sim comparison
    if sumo_dec:
        W("\n── SUMO vs Sim — Obs Feature Means ──")
        W(f"  {'feature':22s}  {'sumo_mean':>10}  {'sim_mean':>10}")
        for fn in OBS_FIELDS:
            sv = [d[fn] for d in sumo_dec]
            rv = [d[fn] for d in sim_dec]
            if sv and rv:
                W(f"  {fn:22s}  {sum(sv)/len(sv):10.3f}  {sum(rv)/len(rv):10.3f}")
    else:
        W("\n  No SUMO rows (run with --with_sumo for comparison)")

    W("\n" + "=" * 72)
    W("End of Analysis")
    W("=" * 72)

    report = "\n".join(lines_out)
    print("\n" + report)
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\n  Analysis → {os.path.basename(out_path)}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SUMO+SimpleSim dual episode logger v3")
    parser.add_argument("--n_episodes",            type=int,   default=10)
    parser.add_argument("--p_reset",               type=float, default=0.01)
    parser.add_argument("--sumo_steps",            type=int,   default=18000,
                        help="Max fake-SUMO steps per episode (sim-only mode)")
    parser.add_argument("--log_dir",               type=str,   default="sim_logs")
    parser.add_argument("--with_sumo",             action="store_true",
                        help="Enable real SUMO via libsumo bridge")
    parser.add_argument("--passenger_update_freq", type=int,   default=10)
    args = parser.parse_args()
    main(args)
