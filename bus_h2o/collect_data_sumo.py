"""
collect_data_sumo.py
====================
Phase 1.2 — 从 SUMO 采集 H2O+ 所需的 offline 数据。

数据格式 (每条 transition):
    obs       (15,)   — 车辆到站决策时刻的 15 维观测
    action    (1,)    — 执行的驻站时间 (zero-hold 下全为 0)
    reward    (1,)    — headway-based 奖励
    next_obs  (15,)   — 同一车辆到达下一站时的 15 维观测
    z_t       (30,)   — 出发站时刻的路网宏观特征
    z_t1      (30,)   — 到站时刻的路网宏观特征
    terminal  (1,)    — 是否为最后一站 (bool)
    sim_time  ()      — 出发站时刻的仿真时间

采集模式:
    使用 Pending-Cache 模式 (参考 sac_v2_bus.py 的 state_dict pattern):
    1. 车辆 A 到站 k   → 提取 obs, z_t, action → 存入 pending[A]
    2. 车辆 A 到站 k+1 → 提取 next_obs, z_t1, reward → 结算为完整 transition

⚠️ 不导入 SumoBusHoldingEnv (会触发 libsumo 冲突)!

Usage
-----
    cd /home/erzhu419/mine_code/sumo-rl/H2Oplus/bus_h2o
    python collect_data_sumo.py [--max_steps 18000] [--out datasets/sumo_offline.h5]
"""

import argparse
import os
import sys
import time
import numpy as np

# ── path setup ─────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

SUMO_DIR = os.path.normpath(os.path.join(
    _HERE, os.pardir, os.pardir, "SUMO_ruiguang", "online_control"))
sys.path.insert(0, SUMO_DIR)
sys.path.insert(0, os.path.join(SUMO_DIR, "sim_obj"))
_CASE_DIR = os.path.join(_HERE, "sumo_env", "case")
if os.path.isdir(_CASE_DIR):
    sys.path.insert(0, _CASE_DIR)

# ⚠️ 只导入 SumoRLBridge 和快照工具，绝不导入 SumoBusHoldingEnv / rl_env
from sumo_env.rl_bridge import SumoRLBridge                     # noqa: E402
from sumo_env.sumo_snapshot import bridge_to_snapshot            # noqa: E402
from common.data_utils import (build_edge_linear_map,            # noqa: E402
                               extract_structured_context,
                               set_route_length)

EDGE_XML = os.path.join(_HERE, "network_data", "a_sorted_busline_edge.xml")
LINE_ID  = "7X"
OBS_DIM  = 15
Z_DIM    = 30


# ── obs 构建 ──────────────────────────────────────────────────────────
# 与 rl_env.py _register_event() 完全一致的 15 维 obs

# 索引登记簿 (运行过程中自增)
_line_index    = {}
_fleet_index   = {}
_station_index = {}
_time_period_index = {}
_line_headway  = {}    # line_id → median headway (s)


def _encode_line(line_id: str) -> int:
    if line_id not in _line_index:
        _line_index[line_id] = len(_line_index)
    return _line_index[line_id]

def _encode_fleet(fleet_idx_raw) -> int:
    key = str(fleet_idx_raw)
    if key not in _fleet_index:
        _fleet_index[key] = len(_fleet_index)
    return _fleet_index[key]

def _encode_station(line_id: str, stop_id: str, stop_idx: int) -> int:
    key = (line_id, stop_id)
    if key not in _station_index:
        _station_index[key] = stop_idx if (stop_idx is not None and stop_idx >= 0) else len(_station_index)
    return _station_index[key]

def _encode_time_period(sim_time: float, span: int = 3600) -> int:
    period = int(sim_time // span)
    if period not in _time_period_index:
        _time_period_index[period] = len(_time_period_index)
    return _time_period_index[period]


def event_to_obs(ev, headway_fallback: float = 360.0) -> np.ndarray:
    """从 DecisionEvent 提取 15 维 obs — 与 rl_env.py _register_event() 一致."""
    line_idx    = _encode_line(ev.line_id)
    fleet_idx   = _encode_fleet(ev.bus_id)
    station_idx = _encode_station(ev.line_id, ev.stop_id, ev.stop_idx)
    tp_idx      = _encode_time_period(ev.sim_time)
    direction   = int(ev.direction)

    target_headway = _line_headway.get(ev.line_id, headway_fallback)
    gap = (target_headway - ev.forward_headway) if getattr(ev, 'forward_bus_present', True) else 0.0

    obs = np.array([
        float(line_idx),                    # 0: line_id (cat)
        float(fleet_idx),                   # 1: fleet_bus_id (cat)
        float(station_idx),                 # 2: station_id (cat)
        float(tp_idx),                      # 3: time_period (cat)
        float(direction),                   # 4: direction (cat)
        float(ev.forward_headway),          # 5
        float(ev.backward_headway),         # 6
        float(ev.waiting_passengers),       # 7
        float(target_headway),              # 8
        float(ev.base_stop_duration),       # 9
        float(ev.sim_time),                 # 10
        float(gap),                         # 11
        float(ev.co_line_forward_headway),  # 12
        float(ev.co_line_backward_headway), # 13
        float(ev.segment_mean_speed),       # 14
    ], dtype=np.float32)
    return obs


def compute_reward(ev, headway_fallback: float = 360.0) -> float:
    """Linear-penalty reward — 与 sac_ensemble_SUMO_linear_penalty.py 一致."""
    def headway_reward(hw, target):
        return -abs(hw - target)

    target_fwd = getattr(ev, 'target_forward_headway', headway_fallback)
    target_bwd = getattr(ev, 'target_backward_headway', headway_fallback)
    fwd_present = getattr(ev, 'forward_bus_present', True)
    bwd_present = getattr(ev, 'backward_bus_present', True)

    r_fwd = headway_reward(ev.forward_headway, target_fwd) if fwd_present else None
    r_bwd = headway_reward(ev.backward_headway, target_bwd) if bwd_present else None

    if r_fwd is not None and r_bwd is not None:
        fwd_dev = abs(ev.forward_headway - target_fwd)
        bwd_dev = abs(ev.backward_headway - target_bwd)
        w = fwd_dev / (fwd_dev + bwd_dev + 1e-6)
        R = target_fwd / max(target_bwd, 1e-6)
        sim_bonus = -abs(ev.forward_headway - R * ev.backward_headway) * 0.5 / ((1 + R) / 2)
        return r_fwd * w + r_bwd * (1 - w) + sim_bonus
    elif r_fwd is not None:
        return r_fwd
    elif r_bwd is not None:
        return r_bwd
    else:
        return -50.0


# ── 主流程 ─────────────────────────────────────────────────────────────

def main(args):
    try:
        import h5py
    except ImportError:
        print("ERROR: 需要安装 h5py。pip install h5py"); sys.exit(1)

    # 1. Edge map + route length
    print(f"[1] 构建 edge map: {EDGE_XML} ...")
    edge_map = build_edge_linear_map(EDGE_XML, LINE_ID) if os.path.exists(EDGE_XML) else {}
    route_len = max(edge_map.values()) if edge_map else 13119.0
    set_route_length(route_len)
    print(f"    路线长: {route_len:.0f} m, edges: {len(edge_map)}")

    # 2. 初始化 Bridge (不导入 SumoBusHoldingEnv!)
    print(f"[2] 初始化 SumoRLBridge (root_dir={SUMO_DIR}) ...")
    bridge = SumoRLBridge(
        root_dir=SUMO_DIR, gui=args.gui, max_steps=args.max_steps)
    bridge.reset()

    # 计算 line headways (bridge 加载后有此数据)
    _line_headway.update(bridge.line_headways)

    # 3. 采集循环 — Pending Cache 模式
    print(f"\n[3] 开始采集 (最多 {args.max_steps} sim steps) ...")
    t0 = time.time()

    # pending_cache: bus_id → {obs, action, z_t, sim_time, station_idx}
    pending = {}
    transitions = []
    event_count = 0

    for step_i in range(args.max_steps * 5):   # safety limit
        events, done, departed = bridge.fetch_events()

        # 已到终点的车 — 清理 pending (防止残留)
        for bus_id in departed:
            pending.pop(bus_id, None)

        if done:
            break
        if not events:
            continue

        # 本批次的快照和 z (所有到站车共享同一时刻)
        snap = bridge_to_snapshot(bridge, edge_map)
        z_now = extract_structured_context(snap)

        for ev in events:
            bus_id = ev.bus_id
            obs = event_to_obs(ev)
            station_idx = int(obs[2])  # obs[2] == station_id
            reward = compute_reward(ev)

            # ── 结算: 该车有挂起的上一站记录 ──
            if bus_id in pending:
                prev = pending.pop(bus_id)
                # 确保 station 变了 (bus 实际走了一站)
                if station_idx != prev["station_idx"]:
                    transitions.append({
                        "obs":      prev["obs"],                           # (15,)
                        "action":   np.array([prev["action"]], np.float32),# (1,)
                        "reward":   np.array([reward], np.float32),        # (1,)
                        "next_obs": obs,                                   # (15,)
                        "z_t":      prev["z_t"],                           # (30,)
                        "z_t1":     z_now.copy(),                          # (30,)
                        "terminal": np.array([0.0], np.float32),           # (1,)
                        "sim_time": prev["sim_time"],
                    })

            # ── 新开单: 存入 pending ──
            action = 0.0   # zero-hold policy
            bridge.apply_action(ev, action)

            pending[bus_id] = {
                "obs":         obs.copy(),
                "action":      action,
                "z_t":         z_now.copy(),
                "sim_time":    ev.sim_time,
                "station_idx": station_idx,
            }
            event_count += 1

        if event_count % 200 == 0 and event_count > 0:
            print(f"    event={event_count:5d}  transitions={len(transitions):5d}  "
                  f"t={bridge.current_time:.0f}s  elapsed={time.time()-t0:.1f}s", flush=True)

    bridge.close()
    elapsed = time.time() - t0
    print(f"\n    采集完成: {event_count} 事件 → {len(transitions)} 条 transition "
          f"({elapsed:.1f}s)")

    if not transitions:
        print("\n⚠️ 没有采集到任何 transition — 检查 SUMO 配置。")
        return

    # 4. 验证
    print(f"\n[4] 数据验证 ...")
    _validate_transitions(transitions)

    # 5. 保存 HDF5
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"\n[5] 保存 {len(transitions)} 条 transition → {args.out}")

    with h5py.File(args.out, "w") as f:
        f.create_dataset("obs",      data=np.stack([t["obs"]      for t in transitions]), compression="gzip")
        f.create_dataset("action",   data=np.stack([t["action"]   for t in transitions]), compression="gzip")
        f.create_dataset("reward",   data=np.stack([t["reward"]   for t in transitions]), compression="gzip")
        f.create_dataset("next_obs", data=np.stack([t["next_obs"] for t in transitions]), compression="gzip")
        f.create_dataset("z_t",      data=np.stack([t["z_t"]      for t in transitions]), compression="gzip")
        f.create_dataset("z_t1",     data=np.stack([t["z_t1"]     for t in transitions]), compression="gzip")
        f.create_dataset("terminal", data=np.stack([t["terminal"] for t in transitions]), compression="gzip")
        f.create_dataset("sim_time", data=np.array([t["sim_time"] for t in transitions]), compression="gzip")
        f.attrs["n_transitions"] = len(transitions)
        f.attrs["policy"]        = "zero"
        f.attrs["route_len"]     = route_len
        f.attrs["source"]        = "real"   # SUMO 数据标记为 "real"

    print(f"    已保存。各 key 的 shape:")
    with h5py.File(args.out, "r") as f:
        for k in f.keys():
            print(f"      {k:12s}: {f[k].shape}")

    print("\n✅ collect_data_sumo.py 完成。")


def _validate_transitions(transitions: list):
    """打印验证摘要，检查数值合理性。"""
    N = len(transitions)
    obs_arr = np.stack([t["obs"] for t in transitions])
    nobs_arr = np.stack([t["next_obs"] for t in transitions])
    rew_arr = np.stack([t["reward"] for t in transitions])
    z_t_arr = np.stack([t["z_t"] for t in transitions])
    z_t1_arr = np.stack([t["z_t1"] for t in transitions])

    print(f"    transitions: {N}")
    print(f"    obs  shape: {obs_arr.shape}, range: [{obs_arr.min():.2f}, {obs_arr.max():.2f}]")
    print(f"    nobs shape: {nobs_arr.shape}")
    print(f"    reward range: [{rew_arr.min():.2f}, {rew_arr.max():.2f}], mean: {rew_arr.mean():.2f}")
    print(f"    z_t  norm range: [{np.linalg.norm(z_t_arr, axis=1).min():.2f}, "
          f"{np.linalg.norm(z_t_arr, axis=1).max():.2f}]")
    print(f"    z_t1 norm range: [{np.linalg.norm(z_t1_arr, axis=1).min():.2f}, "
          f"{np.linalg.norm(z_t1_arr, axis=1).max():.2f}]")

    # 检查 station_id 前进 (obs[2] → next_obs[2])
    station_changed = 0
    station_went_forward = 0
    for i in range(N):
        s0, s1 = obs_arr[i, 2], nobs_arr[i, 2]
        if s0 != s1:
            station_changed += 1
        if s1 > s0:
            station_went_forward += 1
    print(f"    station_id 变化: {station_changed}/{N}")
    print(f"    station_id 前进 (s1>s0): {station_went_forward}/{N}")

    # z 非零检查
    z_all_zero = np.sum(np.linalg.norm(z_t_arr, axis=1) < 1e-6)
    print(f"    z_t 全零: {z_all_zero}/{N}" +
          ("  ⚠️ set_route_length 可能未调用!" if z_all_zero > N // 2 else "  ✅"))

    # 打印 5 条样本
    print(f"\n    ── 前 5 条 transition 样本 ──")
    for i in range(min(5, N)):
        t = transitions[i]
        print(f"    [{i}] t={t['sim_time']:.0f}s  line={t['obs'][0]:.0f}  "
              f"fleet={t['obs'][1]:.0f}  stn={t['obs'][2]:.0f}→{t['next_obs'][2]:.0f}  "
              f"fwd_hw={t['obs'][5]:.1f}  bwd_hw={t['obs'][6]:.1f}  "
              f"reward={t['reward'][0]:.2f}  "
              f"||z_t||={np.linalg.norm(t['z_t']):.2f}  ||z_t1||={np.linalg.norm(t['z_t1']):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 1.2: 从 SUMO 采集 offline 数据 (H2O+)")
    parser.add_argument("--max_steps", type=int, default=18000,
                        help="最大仿真步数 (default: 18000)")
    parser.add_argument("--out", type=str, default="datasets/sumo_offline.h5",
                        help="输出 HDF5 路径 (default: datasets/sumo_offline.h5)")
    parser.add_argument("--gui", action="store_true",
                        help="启动 SUMO GUI (慢, 仅调试用)")
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        print(f"\n!!! FATAL: {e}")
        import traceback; traceback.print_exc()
