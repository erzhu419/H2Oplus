"""
sumo_gym_env.py
===============
Wraps SumoRLBridge to provide the same interface as MultiLineSimEnv / BusSimEnv,
so it can be used as a drop-in replacement in bus_sampler.py for H2O+ online rollout.

This allows testing H2O+ with SUMO as both offline AND online environment
(zero dynamics gap), isolating algorithm issues from SIM alignment issues.

Usage in h2o+_bus_main.py:
    from envs.sumo_gym_env import SumoGymEnv
    sim_env = SumoGymEnv(sumo_dir=..., edge_xml=..., max_steps=18000)
"""

import os
import sys
import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUS_H2O = os.path.dirname(_HERE)
sys.path.insert(0, _BUS_H2O)

from common.data_utils import (
    build_edge_linear_map, extract_structured_context, set_route_length
)


class SumoGymEnv:
    """SUMO environment wrapper matching MultiLineSimEnv interface for bus_sampler.py.

    Required interface:
        - reset()
        - step_to_event(action_dict) -> (state, reward, done)
        - step_fast(action_dict) -> (state, reward, done)  [alias for step_to_event]
        - state: dict {bus_id: [obs_list]}
        - reward: dict {bus_id: float}
        - capture_full_system_snapshot() -> dict for z-feature extraction
        - max_agent_num: int
        - done: bool
    """

    def __init__(self, sumo_dir, edge_xml, schedule_xml=None, max_steps=18000,
                 gui=False, line_id="7X"):
        # Lazy import to avoid libsumo conflicts
        self._sumo_dir = sumo_dir
        self._gui = gui
        self._max_steps = max_steps
        self._line_id = line_id
        self._bridge = None

        # Build edge maps and SUMO indices
        self._edge_xml = edge_xml
        self._all_edge_maps = {}
        self._line_route_lengths = {}
        if os.path.exists(edge_xml):
            tree = ET.parse(edge_xml)
            root = tree.getroot()
            for bl in root.findall("busline"):
                lid = bl.get("id")
                self._all_edge_maps[lid] = build_edge_linear_map(edge_xml, lid)
                total_len = sum(float(e.get("length", 0)) for e in bl.findall("element"))
                self._line_route_lengths[lid] = total_len
            route_len = self._line_route_lengths.get(line_id, 13119.0)
            set_route_length(route_len)

        # SUMO indices for obs construction
        self._schedule_xml = schedule_xml or os.path.join(
            sumo_dir, "initialize_obj", "save_obj_bus.add.xml"
        )
        self._line_idx_map, self._bus_idx_map = {}, {}
        if os.path.exists(self._schedule_xml):
            self._line_idx_map, self._bus_idx_map = self._build_sumo_indices()

        # State tracking
        self.state = {}       # {bus_id: [[obs_vec]]}
        self.reward = {}      # {bus_id: float}
        self.done = False
        self.max_agent_num = 25  # per-line max; will be updated after first reset

        # Internal
        self._station_index = {}
        self._time_period_index = {}
        self._line_headway = {}
        self._current_events = []
        self._event_edge_maps = {}

        # Snapshot pool for buffer reset
        self._snapshot_pool = []       # list of file paths
        self._snapshot_priorities = []  # priority scores (for JTT)
        self._snapshot_pool_max = 50   # max saved states
        self._snapshot_save_interval = 100  # save every N events
        self._event_counter = 0
        self._last_saved_state = None

    def _build_sumo_indices(self):
        tree = ET.parse(self._schedule_xml)
        root = tree.getroot()
        line_deps = defaultdict(list)
        for elem in root.findall(".//bus_obj"):
            lid = elem.get("belong_line_id_s")
            bid = elem.get("bus_id_s")
            st = float(elem.get("start_time_n", "0"))
            if lid and bid:
                line_deps[lid].append((st, bid))
        for entries in line_deps.values():
            entries.sort(key=lambda p: p[0])
        line_idx = {lid: i for i, lid in enumerate(sorted(line_deps.keys()))}
        bus_idx = {}
        counter = 0
        for lid, deps in line_deps.items():
            for _, bid in deps:
                if bid not in bus_idx:
                    bus_idx[bid] = counter
                    counter += 1
        self.max_agent_num = max(counter, 25)
        return line_idx, bus_idx

    def _ensure_bridge(self):
        if self._bridge is None:
            # Add SUMO paths
            sys.path.insert(0, self._sumo_dir)
            sys.path.insert(0, os.path.join(self._sumo_dir, "sim_obj"))
            from sumo_env.rl_bridge import SumoRLBridge
            self._bridge = SumoRLBridge(
                root_dir=self._sumo_dir, gui=self._gui, max_steps=self._max_steps
            )

    def save_sumo_state(self, path=None):
        """Save SUMO simulation state for later restore.

        Returns the path to the saved state file.
        Uses traci.simulation.saveState() for full fidelity.
        """
        if path is None:
            import tempfile
            path = os.path.join(tempfile.gettempdir(), "sumo_snapshot.xml")
        import traci
        traci.simulation.saveState(path)
        self._last_saved_state = path
        return path

    def reset(self, snapshot=None):
        """Reset SUMO.

        Args:
            snapshot: If string path → load SUMO state from file (traci.simulation.loadState)
                      If dict → ignored (custom SnapshotDict not supported for SUMO)
                      If None → standard reset from t=0
        """
        self._ensure_bridge()

        if isinstance(snapshot, str) and os.path.exists(snapshot):
            # Restore from saved SUMO state file, then rebuild bridge objects
            import traci
            traci.simulation.loadState(snapshot)
            # Rebuild bridge's Python objects to match the restored SUMO state
            self._bridge._load_objects()
            self._bridge.done = False
            self._bridge.current_time = traci.simulation.getTime()
            self._bridge.steps = 0
            self._bridge.decision_queue.clear()
            self._bridge.pending_events.clear()
            self._bridge.active_events.clear()
            self._bridge.arrival_history.clear()
            self._bridge.depart_history.clear()
            self._bridge.active_bus_ids = set()
            self._bridge.just_departed_buses = []
            self._line_headway.update(self._bridge.line_headways)
        else:
            # Standard reset
            self._bridge.reset()
            self._line_headway.update(self._bridge.line_headways)

        self._station_index.clear()
        self._time_period_index.clear()
        self.state = {}
        self.reward = {}
        self.done = False
        self._current_events = []
        self._event_counter = 0

    def _event_to_obs(self, ev, headway_fallback=360.0):
        """Convert SUMO event to 15-dim obs vector (same format as SIM)."""
        line_idx = self._line_idx_map.get(ev.line_id, 0)
        bus_idx = self._bus_idx_map.get(str(ev.bus_id), 0)

        sk = (ev.line_id, ev.stop_id)
        if sk not in self._station_index:
            self._station_index[sk] = (
                ev.stop_idx if ev.stop_idx is not None and ev.stop_idx >= 0
                else len(self._station_index)
            )
        station_idx = self._station_index[sk]

        tp = int(ev.sim_time // 3600)
        if tp not in self._time_period_index:
            self._time_period_index[tp] = len(self._time_period_index)
        tp_idx = self._time_period_index[tp]

        target_hw = self._line_headway.get(ev.line_id, headway_fallback)
        dyn_target = getattr(ev, 'target_forward_headway', target_hw)
        fp = getattr(ev, 'forward_bus_present', True)
        gap = (dyn_target - ev.forward_headway) if fp else 0.0

        return np.array([
            float(line_idx), float(bus_idx), float(station_idx),
            float(tp_idx), float(int(ev.direction)),
            float(ev.forward_headway), float(ev.backward_headway),
            float(ev.waiting_passengers), float(target_hw),
            float(ev.base_stop_duration), float(ev.sim_time), float(gap),
            float(ev.co_line_forward_headway), float(ev.co_line_backward_headway),
            float(ev.segment_mean_speed),
        ], dtype=np.float32)

    def _compute_reward(self, ev, headway_fallback=360.0):
        """Compute reward from SUMO event (matches rl_env._compute_reward_linear)."""
        def hr(hw, t):
            return -abs(hw - t)
        t_f = getattr(ev, 'target_forward_headway', headway_fallback)
        t_b = getattr(ev, 'target_backward_headway', headway_fallback)
        fp = getattr(ev, 'forward_bus_present', True)
        bp = getattr(ev, 'backward_bus_present', True)
        rf = hr(ev.forward_headway, t_f) if fp else None
        rb = hr(ev.backward_headway, t_b) if bp else None
        if rf is not None and rb is not None:
            fd = abs(ev.forward_headway - t_f)
            bd = abs(ev.backward_headway - t_b)
            w = fd / (fd + bd + 1e-6)
            R = t_f / max(t_b, 1e-6)
            sb = -abs(ev.forward_headway - R * ev.backward_headway) * 0.5 / ((1 + R) / 2)
            reward = rf * w + rb * (1 - w) + sb
        elif rf is not None:
            reward = rf
        elif rb is not None:
            reward = rb
        else:
            return -50.0
        # Tanh large-deviation penalty
        f_pen = (20.0 * np.tanh((abs(ev.forward_headway - t_f) - 0.5 * t_f) / 30.0)
                 if fp and t_f > 0 else 0.0)
        b_pen = (20.0 * np.tanh((abs(ev.backward_headway - t_b) - 0.5 * t_b) / 30.0)
                 if bp and t_b > 0 else 0.0)
        reward -= max(0.0, f_pen + b_pen)
        return reward

    def step_to_event(self, action_dict):
        """Step SUMO until next decision event(s).

        Args:
            action_dict: {bus_id: [hold, speed] or None}
                         bus_id is int (matched to ev.bus_id from previous events)

        Returns:
            (state, reward, done) matching MultiLineSimEnv interface
        """
        # Apply pending actions from previous events
        for ev in self._current_events:
            bid = ev.bus_id
            action = action_dict.get(bid)
            if action is not None:
                if isinstance(action, (list, np.ndarray)) and len(action) >= 2:
                    self._bridge.apply_action(ev, [float(action[0]), float(action[1])])
                elif isinstance(action, (int, float)):
                    self._bridge.apply_action(ev, [float(action), 1.0])
                else:
                    self._bridge.apply_action(ev, [0.0, 1.0])
            else:
                self._bridge.apply_action(ev, [0.0, 1.0])

        # Fetch next events
        self.state = {}
        self.reward = {}

        for _ in range(100000):
            events, done, departed = self._bridge.fetch_events()
            if done:
                self.done = True
                return self.state, self.reward, True
            if not events:
                continue

            self._current_events = events
            for ev in events:
                bid = ev.bus_id
                obs = self._event_to_obs(ev)
                rew = self._compute_reward(ev)
                self.state[bid] = [[obs.tolist()]]
                self.reward[bid] = rew

            # Auto-save SUMO state for snapshot pool
            self._event_counter += len(events)
            if (self._event_counter >= self._snapshot_save_interval
                    and len(self._snapshot_pool) < self._snapshot_pool_max):
                self._event_counter = 0  # reset counter after each save
                import tempfile
                snap_path = os.path.join(
                    tempfile.gettempdir(),
                    f"sumo_snap_{len(self._snapshot_pool)}.xml"
                )
                try:
                    import traci
                    traci.simulation.saveState(snap_path)
                    self._snapshot_pool.append(snap_path)
                    self._snapshot_priorities.append(1.0)  # initial uniform priority
                except Exception:
                    pass

            return self.state, self.reward, False

        self.done = True
        return self.state, self.reward, True

    def step_fast(self, action_dict):
        """Alias for step_to_event (used during init)."""
        return self.step_to_event(action_dict)

    def step(self, action_dict):
        """Alias for step_to_event."""
        return self.step_to_event(action_dict)

    def capture_full_system_snapshot(self):
        """Capture current SUMO state as a snapshot for z-feature extraction.

        Returns a dict compatible with extract_structured_context().
        """
        if self._bridge is None:
            return {"all_buses": [], "all_stations": []}

        from sumo_env.sumo_snapshot import bridge_to_snapshot
        # Use the 7X edge map for z extraction
        edge_map = self._all_edge_maps.get(self._line_id, {})
        try:
            snap = bridge_to_snapshot(self._bridge, edge_map)
            return snap
        except Exception:
            return {"all_buses": [], "all_stations": []}

    def get_random_snapshot(self):
        """Return a random snapshot path from the pool, or None if empty."""
        if self._snapshot_pool:
            import random
            return random.choice(self._snapshot_pool)
        return None

    def get_prioritized_snapshot(self, temperature=1.0):
        """Return a snapshot weighted by priority (for JTT). Falls back to random."""
        if not self._snapshot_pool:
            return None
        if not self._snapshot_priorities or len(self._snapshot_priorities) != len(self._snapshot_pool):
            return self.get_random_snapshot()
        import random
        p = np.array(self._snapshot_priorities, dtype=np.float64)
        p = np.clip(p, 0, None)
        if p.sum() < 1e-12 or temperature > 50.0:
            return random.choice(self._snapshot_pool)
        p = p ** (1.0 / max(temperature, 1e-4))
        p = p / (p.sum() + 1e-8)
        idx = np.random.choice(len(self._snapshot_pool), p=p)
        return self._snapshot_pool[idx]

    def update_snapshot_priority(self, snap_path, priority):
        """Update priority score for a specific snapshot (called by JTT)."""
        if snap_path in self._snapshot_pool:
            idx = self._snapshot_pool.index(snap_path)
            if idx < len(self._snapshot_priorities):
                ema = 0.1
                self._snapshot_priorities[idx] = (1 - ema) * self._snapshot_priorities[idx] + ema * priority

    def close(self):
        if self._bridge is not None:
            self._bridge.close()
            self._bridge = None
        # Clean up snapshot files
        for p in self._snapshot_pool:
            try:
                os.remove(p)
            except OSError:
                pass
        self._snapshot_pool.clear()

    # ── Properties for compatibility ──
    @property
    def stations(self):
        """Stub for compatibility."""
        return []

    @property
    def line_map(self):
        """Stub — SumoGymEnv is not a MultiLineEnv."""
        return {}
