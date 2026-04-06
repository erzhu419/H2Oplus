"""
sumo_snapshot.py
================
Converts a live SumoRLBridge state into the SnapshotDict format used by
extract_structured_context() in common/data_utils.py.

The output schema exactly mirrors BusSimEnv.capture_full_system_snapshot():
  {
    "sim_time": float,
    "all_buses": [
        {"bus_id": str, "pos": float,  # linear absolute distance (m)
         "speed": float, "load": int, "direction": int}
    ],
    "all_stations": [
        {"station_id": str, "station_name": str,
         "pos": float, "waiting_count": int}
    ]
  }

Usage
-----
    from sumo_env.sumo_snapshot import bridge_to_snapshot
    from common.data_utils import extract_structured_context, build_edge_linear_map

    edge_map = build_edge_linear_map(EDGE_XML, line_id="7X")
    snapshot = bridge_to_snapshot(bridge, edge_map)
    z_t = extract_structured_context(snapshot)   # shape (30,)
"""

import os
import sys
from typing import Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Conditional SUMO import (not required at import time — only when actually
# calling bridge_to_snapshot with a live bridge object)
# ---------------------------------------------------------------------------

def bridge_to_snapshot(
    bridge,
    edge_map: Optional[Dict[str, float]] = None,
    *,
    all_edge_maps: Optional[Dict[str, Dict[str, float]]] = None,
    line_route_lengths: Optional[Dict[str, float]] = None,
) -> dict:
    """
    Convert the current state of a SumoRLBridge into an extract_structured_context-
    compatible SnapshotDict.

    Parameters
    ----------
    bridge : SumoRLBridge
        A fully initialised and stepped bridge. Must have
        ``active_bus_ids``, ``bus_obj_dic``, ``stop_obj_dic``, and
        ``line_stop_distances`` populated.
    edge_map : dict[str, float] (DEPRECATED — kept for backward compat)
        Single-line {edge_id → cumulative start distance (m)}.
        If ``all_edge_maps`` is provided, this parameter is ignored.
    all_edge_maps : dict[str, dict[str, float]]
        {line_id → {edge_id → cumulative start distance (m)}}.
        Each bus is mapped using its own line's edge_map.
    line_route_lengths : dict[str, float]
        {line_id → total route length (m)} for each line.
        Stored per-bus so extract_structured_context can normalise by
        route completion fraction.

    Returns
    -------
    dict
        SnapshotDict matching BusSimEnv.capture_full_system_snapshot() schema.
    """
    if all_edge_maps is None and edge_map is not None:
        # Legacy single-line fallback
        all_edge_maps = {"_default": edge_map}
    elif all_edge_maps is None:
        all_edge_maps = {}

    if line_route_lengths is None:
        line_route_lengths = {}

    # ── buses ────────────────────────────────────────────────────────────────
    all_buses = []
    for bus_id in bridge.active_bus_ids:
        bus_obj = bridge.bus_obj_dic.get(bus_id)
        if bus_obj is None:
            continue

        line_id = getattr(bus_obj, "belong_line_id_s", None) or "_default"

        # Select edge_map for this bus's line
        bus_edge_map = all_edge_maps.get(line_id, all_edge_maps.get("_default", {}))
        route_len = line_route_lengths.get(line_id, 0.0)

        # Linear position via edge_map + lane offset
        try:
            import traci as _traci
            road_id = _traci.vehicle.getRoadID(bus_id)
            lane_pos = _traci.vehicle.getLanePosition(bus_id)
            speed    = _traci.vehicle.getSpeed(bus_id)
        except Exception:
            road_id = ""
            lane_pos = 0.0
            speed = 0.0

        if road_id in bus_edge_map:
            pos = bus_edge_map[road_id] + lane_pos
        else:
            # Fallback: use known stop distance along this bus's own line
            stop_id = getattr(bus_obj, "current_stop_id", None)
            if line_id and stop_id and stop_id in bridge.line_stop_distances.get(line_id, {}):
                pos = bridge.line_stop_distances[line_id][stop_id]
            else:
                pos = 0.0

        direction = getattr(bus_obj, "direction_n", 1)

        # Passenger load
        load = getattr(bus_obj, "current_load_n", 0)
        if load == 0:
            jsd = getattr(bus_obj, "just_server_stop_data_d", {})
            load = int(sum(v[2] if len(v) > 2 else 0 for v in jsd.values()))

        all_buses.append({
            "bus_id"      : bus_id,
            "line_id"     : line_id,
            "pos"         : float(pos),
            "route_length": float(route_len) if route_len > 0 else None,
            "speed"       : float(speed),
            "load"        : int(load),
            "direction"   : int(direction),
            # Raw traci info for full-fidelity storage
            "road_id"     : road_id,
            "lane_pos"    : float(lane_pos),
        })

    # ── stations ─────────────────────────────────────────────────────────────
    all_stations = []
    for stop_id, stop_obj in bridge.stop_obj_dic.items():
        # Best-effort position: use first line that serves this stop
        pos = 0.0
        stop_line_id = None
        for lid, stop_dists in bridge.line_stop_distances.items():
            if stop_id in stop_dists:
                pos = stop_dists[stop_id]
                stop_line_id = lid
                break

        route_len = line_route_lengths.get(stop_line_id, 0.0) if stop_line_id else 0.0

        # Waiting passengers — original SUMO stop.py uses `passenger_num_n`
        # (set by traci.busstop.getPersonIDs in update_stop_state())
        waiting = getattr(stop_obj, "passenger_num_n",
                  getattr(stop_obj, "wait_passenger_num_n",
                  getattr(stop_obj, "waiting_passenger_num", 0)))

        all_stations.append({
            "station_id"   : stop_id,
            "station_name" : stop_id,
            "line_id"      : stop_line_id,
            "pos"          : float(pos),
            "route_length" : float(route_len) if route_len > 0 else None,
            "waiting_count": int(waiting),
        })

    return {
        "sim_time"    : float(bridge.current_time),
        "all_buses"   : all_buses,
        "all_stations": all_stations,
    }


# ---------------------------------------------------------------------------
# Full-fidelity snapshot — compatible with BusSimEnv.restore_full_system_snapshot
# ---------------------------------------------------------------------------

def bridge_to_full_snapshot(
    bridge,
    edge_map: Dict[str, float],
    sim_env=None,
    bus_index: Optional[Dict[str, int]] = None,
) -> Dict[str, dict]:
    """
    Build per-line SnapshotDicts from a live SumoRLBridge, compatible with
    BusSimEnv.restore_full_system_snapshot().

    Parameters
    ----------
    bridge : SumoRLBridge
    edge_map : dict   {edge_id → cumulative distance}
    sim_env  : MultiLineSimEnv (optional)
        Used to resolve station_id integers and timetable slot indices
        from the corresponding line env.
    bus_index : dict (optional)
        {sumo_trip_id_str → int}  -- stable index matching
        rl_env._bus_index, used to embed `sumo_trip_index` in snapshot
        for Phase 3 God-mode reset.

    Returns
    -------
    dict[str, dict]
        {line_id → SnapshotDict} for each line that has active buses.
    """
    import traci as _traci

    # Group bridge buses by line
    line_buses: Dict[str, list] = {}
    for bus_id in bridge.active_bus_ids:
        bus_obj = bridge.bus_obj_dic.get(bus_id)
        if bus_obj is None:
            continue
        line_id = getattr(bus_obj, "belong_line_id_s", None)
        if not line_id:
            continue
        line_buses.setdefault(line_id, []).append((bus_id, bus_obj))

    result: Dict[str, dict] = {}

    for line_id, bus_list in line_buses.items():
        # Resolve sim_env line environment for this line
        line_env = None
        if sim_env is not None:
            line_env = sim_env.line_map.get(line_id)

        # Build station position lookup for this line
        stop_dists = bridge.line_stop_distances.get(line_id, {})
        # Sort stops by distance to build ordered station list
        sorted_stops = sorted(stop_dists.items(), key=lambda x: x[1])
        stop_to_idx = {stop_id: i for i, (stop_id, _) in enumerate(sorted_stops)}

        all_buses_data = []
        launched_trip_indices = set()

        for bus_id, bus_obj in bus_list:
            fb = bridge.fleet_obj_dic.get(bus_id)

            # ── Position via traci ───────────────────────────────────
            try:
                road_id  = _traci.vehicle.getRoadID(bus_id)
                lane_pos = _traci.vehicle.getLanePosition(bus_id)
                speed    = _traci.vehicle.getSpeed(bus_id)
            except Exception:
                road_id = ""
                lane_pos = 0.0
                speed = 0.0

            if road_id in edge_map:
                pos = edge_map[road_id] + lane_pos
            else:
                pos = stop_dists.get(
                    getattr(bus_obj, "current_stop_id", ""), 0.0)

            # ── Trip/fleet info ──────────────────────────────────────
            fleet_idx = -1
            trip_ids_served = []
            if fb is not None:
                try:
                    fleet_idx = int(fb.fleet_id.rsplit('_', 1)[-1])
                except (ValueError, IndexError):
                    fleet_idx = 0
                trip_ids_served = list(range(len(fb.trip_ids)))

            # ── Station pointers ─────────────────────────────────────
            # Find closest station behind and ahead based on position
            last_sid = 0
            next_sid = 0
            last_s_dis = pos
            next_s_dis = 0.0
            for i, (sid, sdist) in enumerate(sorted_stops):
                if sdist <= pos:
                    last_sid = i
                    last_s_dis = pos - sdist
                else:
                    next_sid = i
                    next_s_dis = sdist - pos
                    break
            else:
                # past all stations
                if sorted_stops:
                    next_sid = len(sorted_stops) - 1
                    next_s_dis = 0.0

            # ── State ────────────────────────────────────────────────
            bus_state_str = getattr(bus_obj, "bus_state_s", "Running")
            if bus_state_str == "Stop":
                state_name = "HOLDING"
            else:
                state_name = "TRAVEL"

            # ── Headway (from last decision event or fallback) ───────
            fwd_hw = getattr(bus_obj, "last_forward_headway",
                             bridge.line_headways.get(line_id, 360.0))
            if fwd_hw is None:
                fwd_hw = bridge.line_headways.get(line_id, 360.0)
            bwd_hw = fwd_hw  # symmetric fallback

            direction = 1 if line_id.endswith('S') else 0

            # ── Mark launched timetable slots ────────────────────────
            # Use trip count to infer which timetable entries were used
            trip_id = len(fb.trip_ids) - 1 if fb and fb.trip_ids else 0
            if fb:
                for t_idx in range(len(fb.trip_ids)):
                    launched_trip_indices.add(t_idx)

            load = getattr(bus_obj, "current_load_n", 0)
            on_route = fb.on_route if fb else True

            all_buses_data.append({
                "bus_id":            fleet_idx,
                "trip_id":           trip_id,
                "trip_id_list":      trip_ids_served if trip_ids_served else [trip_id],
                "sumo_trip_index":   bus_index.get(bus_id, -1) if bus_index else -1,
                "direction":         direction,
                "absolute_distance": float(pos),
                "current_speed":     float(speed),
                "load":              int(load),
                "holding_time":      0.0,
                "forward_headway":   float(fwd_hw),
                "backward_headway":  float(bwd_hw),
                "last_station_id":   last_sid,
                "next_station_id":   next_sid,
                "next_station_dis":  float(next_s_dis),
                "last_station_dis":  float(last_s_dis),
                "state":             state_name,
                "on_route":          bool(on_route),
                "pos":               float(pos),
                "speed":             float(speed),
            })

        # ── Stations ─────────────────────────────────────────────────
        all_stations_data = []
        for i, (stop_id, sdist) in enumerate(sorted_stops):
            stop_obj = bridge.stop_obj_dic.get(stop_id)
            waiting = 0
            if stop_obj:
                waiting = getattr(stop_obj, "passenger_num_n",
                          getattr(stop_obj, "wait_passenger_num_n",
                          getattr(stop_obj, "waiting_passenger_num", 0)))
            all_stations_data.append({
                "station_id":    i,
                "station_name":  stop_id,
                "direction":     True if line_id.endswith('S') else False,
                "waiting_count": int(waiting),
                "pos":           float(sdist),
            })

        result[line_id] = {
            "sim_time":       float(bridge.current_time),
            "current_time":   float(bridge.current_time),
            "all_buses":      all_buses_data,
            "all_stations":   all_stations_data,
            "launched_trips": sorted(launched_trip_indices),
        }

    return result


# ---------------------------------------------------------------------------
# Mock helper — lets tests verify the schema without a live SUMO session
# ---------------------------------------------------------------------------

def make_mock_snapshot(
    sim_time: float = 1000.0,
    n_buses: int = 3,
    n_stops: int = 10,
    route_length: float = 13000.0,
) -> dict:
    """Return a synthetic snapshot for unit-testing extract_structured_context."""
    rng = np.random.default_rng(42)
    buses = [
        {
            "bus_id"   : str(i),
            "pos"      : float(rng.uniform(0, route_length)),
            "speed"    : float(rng.uniform(0, 15)),
            "load"     : int(rng.integers(0, 40)),
            "direction": int(rng.integers(0, 2)),
        }
        for i in range(n_buses)
    ]
    positions = np.linspace(0, route_length, n_stops + 2)[1:-1]
    stations = [
        {
            "station_id"   : f"S{k:02d}",
            "station_name" : f"S{k:02d}",
            "pos"          : float(positions[k]),
            "waiting_count": int(rng.integers(0, 30)),
        }
        for k in range(n_stops)
    ]
    return {"sim_time": sim_time, "all_buses": buses, "all_stations": stations}
