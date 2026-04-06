"""
envs/bus_sim_env.py
====================
Phase 2: BusSimEnv — gym-compatible wrapper around the local sim_core env_bus.

Extends env_bus (H2Oplus/bus_h2o/sim_core/sim.py) with:
    1. `capture_full_system_snapshot()` — serialize entire sim state → SnapshotDict
    2. `restore_full_system_snapshot(snapshot)` — god-mode reset to any past state
    3. `reset(snapshot=None)` — standard reset (no snapshot) or buffer-seed reset
    4. `step(action_dict)` — unchanged behaviour + returns snapshot in `info`

SnapshotDict schema (all buses and stations at a given sim time):
    {
        "sim_time": float,
        "current_time": float,     # same as sim_time
        "all_buses": [
            {
                "bus_id":          int,
                "trip_id":         int,
                "direction":       int (1=up, 0=down),
                "absolute_distance": float,   # metres from route origin
                "current_speed":   float,
                "load":            int,        # passengers on board
                "holding_time":    float,
                "forward_headway": float,
                "backward_headway":float,
                "last_station_id": int,
                "next_station_id": int,
                "state":           str,        # BusState enum name
                "on_route":        bool,
                "trip_id_list":    list[int],
                "next_station_dis":float,
                "last_station_dis":float,
            },
            ...
        ],
        "all_stations": [
            {
                "station_id":    int,
                "station_name":  str,
                "direction":     bool,
                "waiting_count": int,
                "pos":           float,  # cumulative abs distance from route origin (m)
            },
            ...
        ],
        "launched_trips":   list[int],   # trip indices already launched
        "timetable_state":  list[bool],  # timetable[i].launched
    }

Usage:
    from envs.bus_sim_env import BusSimEnv

    env = BusSimEnv(path="/path/to/LSTM-RL-legacy/env")
    obs = env.reset()

    # Standard rollout
    for t in range(max_steps):
        actions = {bus_id: agent.act(obs[bus_id]) for bus_id in obs}
        obs, rewards, done, info = env.step(actions)
        snap_T2 = info["snapshot"]   # SnapshotDict at this time step

    # Snapshot-seeded reset (Phase 3 buffer reset)
    obs = env.reset(snapshot=snap_T2)
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Import from local sim_core package (self-contained copy of LSTM-RL-legacy/env)
# ---------------------------------------------------------------------------
_BUS_H2O_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BUS_H2O_DIR not in sys.path:
    sys.path.insert(0, _BUS_H2O_DIR)

from sim_core.sim import env_bus        # noqa: E402
from sim_core.bus import BusState       # noqa: E402
from sim_core.co_line_scheduler import VirtualCoLineScheduler  # noqa: E402


class BusSimEnv(env_bus):
    """
    Drop-in extension of env_bus that supports snapshot-based reset.

    All existing env_bus behaviour is preserved.  New capabilities:
        - env.reset()          → behaves exactly as before (standard reset)
        - env.reset(snapshot)  → injects a past system state (god-mode)
        - env.step(actions)    → returns (obs, rewards, done, info)
                                 where info["snapshot"] is a SnapshotDict

    Coordinate convention
    ---------------------
    `pos` in each station entry equals the station's cumulative linear distance
    from the route origin (m).  Computed once during __init__ from route distances.
    Consistent with sumo_pos_to_linear() in common/data_utils.py.
    """

    def __init__(self, path: str, debug: bool = False, render: bool = False) -> None:
        # sim_core/sim.py no longer has the CWD-based sys.path hack,
        # so we can call super().__init__ directly.
        super().__init__(path, debug=debug, render=render)
        # Pre-compute station linear positions once (used for snapshot `pos` field)
        self._station_linear_pos: dict[str, float] = self._compute_station_positions()

        # SUMO alignment: inject line-level constants into the env so
        # launch_bus() propagates them to each Bus instance.
        # 7X is alphabetically last among 12 SUMO lines → index 11.
        # Median headway for 7X = 360.0s (from save_obj_bus.add.xml schedule).
        self.line_idx = 11       # _SUMO_LINE_INDEX['7X']
        self.line_headway = 360.0  # matches rl_env._line_headway['7X']

        # Virtual co-line scheduler: analytically tracks 102X/705X bus positions
        # so co-line headways (obs[12-13]) match SUMO rl_bridge logic.
        self._co_scheduler = VirtualCoLineScheduler(
            x7_station_positions=self._station_linear_pos
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, snapshot: Optional[dict] = None) -> dict:
        """
        Reset the environment.

        Args:
            snapshot: If None, performs the standard env_bus reset.
                      If a SnapshotDict, performs a god-mode state injection
                      (buffer reset for Phase 3 H2O+ training).

        Returns:
            obs : dict[bus_id → state_vector]  (same format as env_bus)
        """
        if snapshot is None:
            # Standard reset — delegate to parent
            super().reset()
            # Do NOT call initialize_state() here — it runs the sim
            # with None actions to completion. Let the sampler loop
            # step forward to the first decision event.
            return self.state
        else:
            # God-mode buffer reset
            super().reset()                      # reinitialise objects
            self.restore_full_system_snapshot(snapshot)
            # Return current obs (may be empty if no bus is at a decision point yet)
            return self.state

    def initialize_state(self, render: bool = False):
        """Override to handle the 3-tuple returned by BusSimEnv.step_fast()."""
        def count_non_empty(lst):
            return sum(1 for v in lst if v)

        while count_non_empty(list(self.state.values())) == 0:
            state, reward, done = self.step_fast(self.action_dict, render=render)
            if done:
                break

        return self.state, self.reward, self.done

    def step(self, action_dict: dict, **kwargs) -> tuple[dict, dict, bool, dict]:
        """
        Advance one time step.

        Returns:
            obs     : dict[bus_id → state_vector]
            rewards : dict[bus_id → float]
            done    : bool
            info    : {
                "snapshot_T1": SnapshotDict,  # global state BEFORE this step (decision time)
                "snapshot_T2": SnapshotDict,  # global state AFTER this step  (arrival time)
                "snapshot":    SnapshotDict,  # alias for snapshot_T2 (backward compat)
                "t":           float,
            }

        snapshot_T1 / snapshot_T2 semantics (H2O+.md §3.2):
            T1 = the moment Ego Bus departs from station k (action is being issued).
            T2 = the moment Ego Bus arrives at station k+1 (next decision event).
            Discriminator input: D(obs_t, a_t, obs_{t+1}, z_t, z_{t+1})
            where z_t = extract_structured_context(snapshot_T1)
                  z_t1= extract_structured_context(snapshot_T2)

        Note: **kwargs passes through `render=`, `debug=` etc. from parent
              initialize_state() so this override stays backward-compatible.
        """
        # Capture T1: global network state at the moment the action is applied
        snapshot_T1 = self.capture_full_system_snapshot()

        # Build co-line bus positions from virtual 102X/705X scheduler
        # Mirrors MultiLineEnv.step() which populates co_line_buses from all other active lines.
        # Use current segment speed as conversion factor (mean across active buses, fallback 8 m/s)
        active_speeds = [b.current_speed for b in self.bus_all if b.on_route and b.current_speed > 0]
        seg_speed = float(np.mean(active_speeds)) if active_speeds else 8.0
        co_line_buses = self._co_scheduler.get_co_line_buses(
            self.current_time, seg_speed=seg_speed, target_headway=self.line_headway
        )

        state, reward, done = super().step(action_dict, co_line_buses=co_line_buses, **kwargs)

        # Capture T2: global network state after the simulation step
        snapshot_T2 = self.capture_full_system_snapshot()

        info = {
            "snapshot_T1": snapshot_T1,
            "snapshot_T2": snapshot_T2,
            "snapshot":    snapshot_T2,   # backward-compat alias
            "t":           self.current_time,
        }
        return state, reward, done, info

    def step_fast(self, action_dict: dict, **kwargs) -> tuple[dict, dict, bool]:
        """Lightweight step for training — skips snapshot capture."""
        co_line_buses = self._co_scheduler.get_co_line_buses(
            self.current_time, target_headway=self.line_headway
        )
        state, reward, done = super().step(action_dict, co_line_buses=co_line_buses, **kwargs)
        return state, reward, done

    def step_to_event(self, action_dict: dict, **kwargs) -> tuple[dict, dict, bool]:
        """Fast-forward through idle ticks until a bus emits obs, or done."""
        while True:
            state, reward, done = self.step_fast(action_dict, **kwargs)
            if done:
                return state, reward, done
            if any(v for v in state.values()):
                return state, reward, done

    # ------------------------------------------------------------------
    # Snapshot I/O
    # ------------------------------------------------------------------

    def capture_full_system_snapshot(self) -> dict:
        """
        Serialise the *current* simulation state into a SnapshotDict.

        Call this immediately *after* a decision event (i.e. after step())
        so that the returned snapshot represents the post-action state.

        Returns:
            SnapshotDict (see module docstring for schema).
        """
        route_length = sum(r.distance for r in self.routes)
        buses_data = []
        for bus in self.bus_all:
            if not bus.on_route:
                continue  # Skip retired buses — matches SUMO's active-only snapshot
            entry = {
                "bus_id":            bus.bus_id,
                "trip_id":           bus.trip_id,
                "sumo_trip_index":   getattr(bus, 'sumo_trip_index', bus.trip_id),
                "direction":         int(bus.direction),
                "absolute_distance": float(bus.absolute_distance),
                "current_speed":     float(bus.current_speed),
                "load":              int(len(bus.passengers)),
                "holding_time":      float(bus.holding_time),
                "forward_headway":   float(bus.forward_headway),
                "backward_headway":  float(bus.backward_headway),
                "last_station_id":   int(bus.last_station.station_id),
                "next_station_id":   int(bus.next_station.station_id),
                "state":             bus.state.name,
                "on_route":          bool(bus.on_route),
                "trip_id_list":      list(bus.trip_id_list),
                "next_station_dis":  float(bus.next_station_dis),
                "last_station_dis":  float(bus.last_station_dis),
            }
            # Add as `pos` for extract_structured_context compatibility
            entry["pos"] = float(bus.absolute_distance)
            entry["speed"] = float(bus.current_speed)
            entry["route_length"] = float(route_length)
            buses_data.append(entry)

        stations_data = []
        for st in self.stations:
            sname = st.station_name
            entry = {
                "station_id":    int(st.station_id),
                "station_name":  sname,
                "direction":     bool(st.direction),
                "waiting_count": int(len(st.waiting_passengers)),
                "pos":           float(self._station_linear_pos.get(sname, 0.0)),
                "route_length":  float(route_length),
            }
            stations_data.append(entry)

        return {
            "sim_time":       float(self.current_time),
            "current_time":   float(self.current_time),
            "all_buses":      buses_data,
            "all_stations":   stations_data,
            "launched_trips": [i for i, t in enumerate(self.timetables) if t.launched],
            "timetable_state":[bool(t.launched) for t in self.timetables],
        }

    def restore_full_system_snapshot(self, snapshot: dict) -> None:
        """
        God-mode state injection.  Overwrites the *current* simulation state
        with the given SnapshotDict.  Must be called after super().reset() so
        that objects are properly initialised before being overwritten.

        Accepts both:
          - Full-fidelity format (from bridge_to_full_snapshot / capture_full_system_snapshot)
          - Lightweight format  (from bridge_to_snapshot / raw_snapshot)

        Args:
            snapshot: SnapshotDict in either format.
        """
        # Accept both 'current_time' and 'sim_time' keys
        self.current_time = float(
            snapshot.get("current_time", snapshot.get("sim_time", 0.0))
        )

        # --- Mark timetable entries as launched and pre-launch buses ---
        # super().reset() empties bus_all=[]; buses only appear via launch_bus().
        # We must pre-launch buses for all timetable slots marked launched in snap.
        if "launched_trips" in snapshot:
            launched_set = set(snapshot["launched_trips"])
        else:
            # Lightweight snapshot: infer launched trips from current_time
            # Launch all timetable entries whose start_time <= current_time
            launched_set = set()
            for i, t in enumerate(self.timetables):
                start_t = getattr(t, "start_time", getattr(t, "time", float("inf")))
                if start_t <= self.current_time:
                    launched_set.add(i)
        for i, t in enumerate(self.timetables):
            t.launched = (i in launched_set)
            if i in launched_set:
                self.launch_bus(t)

        # After pre-launching, rebuild the bus_id lookup map
        bus_by_id: dict[int, object] = {b.bus_id: b for b in self.bus_all}
        station_by_id: dict[int, object] = {
            s.station_id: s for s in self.stations
        }

        # --- Restore bus states ---
        restored_bus_ids: set[int] = set()

        for bd in snapshot["all_buses"]:
            bid = bd["bus_id"]

            if bid in bus_by_id:
                bus = bus_by_id[bid]
            else:
                # Bus_id not matched (e.g. id assignment differs); try by index order
                unrestored = [b for b in self.bus_all if b.bus_id not in restored_bus_ids]
                if unrestored:
                    bus = unrestored[0]
                else:
                    continue

            # Core kinematics (accept both full-fidelity and lightweight field names)
            bus.trip_id            = bd.get("trip_id", 0)
            bus.trip_id_list       = list(bd.get("trip_id_list", [bus.trip_id]))
            # Inject SUMO trip index for embedding alignment (Phase 3 H2O+)
            bus.sumo_trip_index    = bd.get("sumo_trip_index", bus.trip_id)
            bus.direction          = bool(bd.get("direction", 0))
            bus.absolute_distance  = float(bd.get("absolute_distance", bd.get("pos", 0.0)))
            bus.current_speed      = float(bd.get("current_speed", bd.get("speed", 0.0)))
            bus.holding_time       = float(bd.get("holding_time", 0.0))
            bus.forward_headway    = float(bd.get("forward_headway", 360.0))
            bus.backward_headway   = float(bd.get("backward_headway", 360.0))
            bus.next_station_dis   = float(bd.get("next_station_dis", 0.0))
            bus.last_station_dis   = float(bd.get("last_station_dis", 0.0))
            bus.on_route           = bool(bd.get("on_route", True))

            # Station pointers (may be absent in lightweight snapshots)
            lst_id  = bd.get("last_station_id")
            nxt_id  = bd.get("next_station_id")
            if lst_id is not None and lst_id in station_by_id:
                bus.last_station = station_by_id[lst_id]
            if nxt_id is not None and nxt_id in station_by_id:
                bus.next_station = station_by_id[nxt_id]

            # Restore BusState enum
            state_name = bd.get("state", "TRAVEL")
            try:
                bus.state = BusState[state_name]
            except KeyError:
                bus.state = BusState.TRAVEL

            # Passengers: restore count
            target_load = int(bd.get("load", 0))
            current_load = len(bus.passengers)
            if current_load > target_load:
                bus.passengers = bus.passengers[:target_load]
            elif current_load < target_load:
                padding = np.array([None] * (target_load - current_load), dtype=object)
                bus.passengers = np.concatenate([bus.passengers, padding])

            bus.in_station = bus.state in (BusState.HOLDING, BusState.WAITING_ACTION, BusState.DWELLING)
            restored_bus_ids.add(bus.bus_id)

        # --- Restore station waiting passenger counts ---
        # We cannot restore individual Passenger objects (they carry OD info),
        # but we restore the *count* as empty slots. The stochastic passenger
        # arrival process will re-populate correctly within a few seconds.
        station_snap = {sd["station_id"]: sd for sd in snapshot.get("all_stations", [])}
        for st in self.stations:
            if st.station_id in station_snap:
                count = int(station_snap[st.station_id]["waiting_count"])
                # Keep existing passengers if count matches, else reset
                if len(st.waiting_passengers) != count:
                    st.waiting_passengers = [None] * count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_station_positions(self) -> dict[str, float]:
        """
        Pre-compute linear absolute position (m) of each station from route origin.

        Uses the route_news.xlsx distances loaded by the parent class.
        Matches the convention: upstream stations start at 0, downstream uses
        the same distances (env_bus stores all stations in a concat list).
        """
        pos_map: dict[str, float] = {}
        cumulative = 0.0

        # The first half of self.stations is upstream (direction=True),
        # the second half downstream (direction=False, reversed).
        # We compute positions for each half independently.

        # --- Upstream (direction=True) ---
        half = len(self.routes) // 2
        upstream_routes = self.routes[:half]
        upstream_stations = self.stations[:round(len(self.stations) / 2)]

        up_pos = 0.0
        for i, st in enumerate(upstream_stations):
            pos_map[f"{st.station_name}_up"] = up_pos
            # Use station_name as key since same name exists in both directions
            pos_map[st.station_name] = up_pos   # fallback key (direction ambiguous)
            if i < len(upstream_routes):
                up_pos += upstream_routes[i].distance

        # --- Downstream (direction=False) — mirror of upstream ---
        downstream_stations = self.stations[round(len(self.stations) / 2) - 1:]
        down_pos = 0.0
        for i, st in enumerate(downstream_stations):
            pos_map[f"{st.station_name}_down"] = down_pos
            if i < len(upstream_routes):
                down_pos += upstream_routes[i].distance   # symmetric route

        return pos_map

    def _get_station_pos(self, station_name: str, direction: bool) -> float:
        """Get the linear position of a station for use in snapshots."""
        key = f"{station_name}_{'up' if direction else 'down'}"
        return self._station_linear_pos.get(key, self._station_linear_pos.get(station_name, 0.0))


# =============================================================================
# Multi-Line Adapter (Phase 3+)
# =============================================================================

from sim_core.sim import MultiLineEnv as _MultiLineEnv  # noqa: E402


class MultiLineSimEnv(_MultiLineEnv):
    """
    Gym-compatible adapter around MultiLineEnv.

    Operates over all 12 SUMO lines simultaneously for z-feature computation,
    while exposing a BusSimEnv-compatible interface for the 7X policy rollout.

    For H2O+ training:
        - capture_full_system_snapshot() → all-lines snapshot for z
        - state/reward/step_to_event/reset → proxied to 7X line
        - Other lines step in the background with zero-hold actions
    """

    TARGET_LINE = "7X"

    def __init__(self, path: str, debug: bool = False, render: bool = False):
        super().__init__(path, debug=debug, render=render)
        if self.TARGET_LINE not in self.line_map:
            raise RuntimeError(
                f"Target line {self.TARGET_LINE} not found. "
                f"Available: {list(self.line_map.keys())}"
            )
        self._x7 = self.line_map[self.TARGET_LINE]
        # Override parent's aggregate values with 7X-only values
        self.max_agent_num = self._x7.max_agent_num
        self.action_space = self._x7.action_space

    # ------------------------------------------------------------------
    # BusSimEnv-compatible properties (proxy to 7X)
    # ------------------------------------------------------------------

    @property
    def state(self):
        return self._x7.state

    @state.setter
    def state(self, val):
        self._x7.state = val

    @property
    def reward(self):
        return self._x7.reward

    @property
    def stations(self):
        return self._x7.stations

    # ------------------------------------------------------------------
    # Reset: reset all lines, optionally inject snapshot into 7X
    # ------------------------------------------------------------------

    def reset(self, snapshot=None):
        """Reset all lines. If snapshot given, inject into 7X (buffer reset)."""
        super().reset()
        if snapshot is not None:
            # If this is an all-lines snapshot (from raw_snapshot), filter
            # to only buses/stations relevant to 7X before injecting.
            if any(b.get("line_id") for b in snapshot.get("all_buses", [])):
                x7_lines = {self.TARGET_LINE, self.TARGET_LINE.replace("X", "S")}
                filtered = dict(snapshot)
                filtered["all_buses"] = [
                    b for b in snapshot["all_buses"]
                    if b.get("line_id") in x7_lines or b.get("line_id") is None
                ]
                filtered["all_stations"] = [
                    s for s in snapshot.get("all_stations", [])
                    if s.get("line_id") in x7_lines or s.get("line_id") is None
                ]
                self._x7.restore_full_system_snapshot(filtered)
            else:
                self._x7.restore_full_system_snapshot(snapshot)
            return self._x7.state
        # Standard reset: step all lines until 7X has obs
        self._init_all_lines()
        return self._x7.state

    def _init_all_lines(self):
        """Step until 7X has at least one bus with obs."""
        actions = self._zero_actions()
        for _ in range(5000):
            state, reward, done = super().step(actions)
            if done:
                break
            x7_state = state.get(self.TARGET_LINE, {})
            if any(v for v in x7_state.values()):
                break

    # ------------------------------------------------------------------
    # Step: advance all lines, return only 7X state/reward
    # ------------------------------------------------------------------

    def step_to_event(self, action_dict, **kwargs):
        """
        Step all lines until 7X produces a decision event.

        action_dict: {bus_id: hold_time} for 7X only.
        Other lines get zero-hold.
        """
        full_actions = self._zero_actions()
        full_actions[self.TARGET_LINE] = action_dict
        while True:
            state, reward, done = super().step(full_actions, **kwargs)
            if done:
                return self._x7.state, self._x7.reward, True
            x7_state = state.get(self.TARGET_LINE, {})
            if any(v for v in x7_state.values()):
                return self._x7.state, self._x7.reward, False
            # Reset 7X actions to None while waiting
            full_actions[self.TARGET_LINE] = {
                k: None for k in range(self._x7.max_agent_num)
            }

    def step_fast(self, action_dict, **kwargs):
        """Single-tick step. action_dict is for 7X only."""
        full_actions = self._zero_actions()
        full_actions[self.TARGET_LINE] = action_dict
        state, reward, done = super().step(full_actions, **kwargs)
        return self._x7.state, self._x7.reward, done

    # ------------------------------------------------------------------
    # Snapshot: aggregate all lines for z-feature computation
    # ------------------------------------------------------------------

    def capture_full_system_snapshot(self) -> dict:
        """
        Build an all-lines snapshot for extract_structured_context().

        All buses from all 12 lines are included (matching SUMO's
        bridge_to_snapshot which iterates bridge.active_bus_ids across
        all lines). Each bus carries its line's route_length so
        extract_structured_context can normalise by fractional position.

        Stations from ALL lines are included with their own route_length.
        """
        all_buses = []
        for lid, le in self.line_map.items():
            # Compute this line's total route length
            route_len = sum(r.distance for r in le.routes)
            for bus in le.bus_all:
                if not bus.on_route:
                    continue
                all_buses.append({
                    "bus_id":       int(bus.bus_id),
                    "line_id":      lid,
                    "pos":          float(bus.absolute_distance),
                    "route_length": float(route_len),
                    "speed":        float(bus.current_speed),
                    "load":         int(len(bus.passengers)),
                    "direction":    int(0 if bus.direction else 1),
                })

        all_stations = []
        for lid, le in self.line_map.items():
            route_len = sum(r.distance for r in le.routes)
            # Compute cumulative station positions from routes
            n_routes = len(le.routes)
            n_stations = len(le.stations)
            # stations[0] is at distance 0, station[i] is at sum(routes[0..i-1].distance)
            cum_pos = 0.0
            for i, st in enumerate(le.stations):
                all_stations.append({
                    "station_id":    int(st.station_id),
                    "station_name":  st.station_name,
                    "line_id":       lid,
                    "pos":           float(cum_pos),
                    "route_length":  float(route_len),
                    "waiting_count": int(len(st.waiting_passengers)),
                })
                # Advance cumulative position
                if i < n_routes:
                    cum_pos += le.routes[i].distance

        return {
            "sim_time":     float(self.current_time),
            "current_time": float(self.current_time),
            "all_buses":    all_buses,
            "all_stations": all_stations,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def iter_bus_obs(self, state: dict):
        """Yield (line_id, bus_id, obs) for every bus with a non-empty state."""
        for lid, bd in state.items():
            for bid, v in bd.items():
                if v:
                    yield lid, bid, v[-1]

