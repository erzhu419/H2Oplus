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
            state, _, _ = self.initialize_state()
            return state
        else:
            # God-mode buffer reset
            super().reset()                      # reinitialise objects
            self.restore_full_system_snapshot(snapshot)
            # Return current obs (may be empty if no bus is at a decision point yet)
            return self.state

    def initialize_state(self, render: bool = False):
        """Override to handle the 4-tuple returned by BusSimEnv.step()."""
        def count_non_empty(lst):
            return sum(1 for v in lst if v)

        while count_non_empty(list(self.state.values())) == 0:
            state, reward, done, _ = self.step(self.action_dict, render=render)

        return self.state, self.reward, self.done

    def step(self, action_dict: dict, **kwargs) -> tuple[dict, dict, bool, dict]:
        """
        Advance one time step.

        Returns:
            obs     : dict[bus_id → state_vector]
            rewards : dict[bus_id → float]
            done    : bool
            info    : {"snapshot": SnapshotDict, "t": float}

        Note: **kwargs passes through `render=`, `debug=` etc. from parent
              initialize_state() so this override stays backward-compatible.
        """
        state, reward, done = super().step(action_dict, **kwargs)
        snapshot = self.capture_full_system_snapshot()
        info = {
            "snapshot": snapshot,
            "t": self.current_time,
        }
        return state, reward, done, info

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
        buses_data = []
        for bus in self.bus_all:
            entry = {
                "bus_id":            bus.bus_id,
                "trip_id":           bus.trip_id,
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

        This implements the "buffer reset" strategy described in H2O+.md §2.2:
        sample a real-world transition from the offline buffer and seed the
        simulator at the corresponding state T1, so the agent continues from a
        realistic distribution.

        Args:
            snapshot: SnapshotDict as produced by capture_full_system_snapshot().
        """
        self.current_time = float(snapshot["current_time"])

        # --- Mark timetable entries as launched and pre-launch buses ---
        # super().reset() empties bus_all=[]; buses only appear via launch_bus().
        # We must pre-launch buses for all timetable slots marked launched in snap.
        launched_set = set(snapshot.get("launched_trips", []))
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

            # Core kinematics
            bus.trip_id            = bd["trip_id"]
            bus.trip_id_list       = list(bd.get("trip_id_list", [bd["trip_id"]]))
            bus.direction          = bool(bd["direction"])
            bus.absolute_distance  = float(bd["absolute_distance"])
            bus.current_speed      = float(bd["current_speed"])
            bus.holding_time       = float(bd["holding_time"])
            bus.forward_headway    = float(bd["forward_headway"])
            bus.backward_headway   = float(bd["backward_headway"])
            bus.next_station_dis   = float(bd["next_station_dis"])
            bus.last_station_dis   = float(bd["last_station_dis"])
            bus.on_route           = bool(bd["on_route"])

            # Station pointers
            lst_id  = bd["last_station_id"]
            nxt_id  = bd["next_station_id"]
            if lst_id in station_by_id:
                bus.last_station = station_by_id[lst_id]
            if nxt_id in station_by_id:
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
                    st.waiting_passengers = np.array([None] * count, dtype=object)

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
    Thin gym-compatible adapter around MultiLineEnv.

    Provides the same interface as BusSimEnv but operates over all 10 SUMO
    lines simultaneously.  State/reward/action dicts are nested:
        state:   {line_id: {bus_id: [obs_list]}}
        reward:  {line_id: {bus_id: float}}
        actions: {line_id: {bus_id: float}}

    Usage:
        env = MultiLineSimEnv('calibrated_env')
        obs, reward, done = env.reset()
        for t in range(N):
            actions = {lid: {bid: 0.0 for bid in range(le.max_agent_num)}
                       for lid, le in env.line_map.items()}
            obs, reward, done = env.step(actions)
    """

    def __init__(self, path: str, debug: bool = False, render: bool = False):
        super().__init__(path, debug=debug, render=render)

    # All functionality inherited from MultiLineEnv.
    # Add helper for flat obs extraction compatible with data_utils:
    def iter_bus_obs(self, state: dict):
        """Yield (line_id, bus_id, obs) for every bus with a non-empty state."""
        for lid, bd in state.items():
            for bid, v in bd.items():
                if v:
                    yield lid, bid, v[-1]
