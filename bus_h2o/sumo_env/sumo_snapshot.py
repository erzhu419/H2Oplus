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

def bridge_to_snapshot(bridge, edge_map: Dict[str, float]) -> dict:
    """
    Convert the current state of a SumoRLBridge into an extract_structured_context-
    compatible SnapshotDict.

    Parameters
    ----------
    bridge : SumoRLBridge
        A fully initialised and stepped bridge. Must have
        ``active_bus_ids``, ``bus_obj_dic``, ``stop_obj_dic``, and
        ``line_stop_distances`` populated.
    edge_map : dict[str, float]
        {edge_id → cumulative start distance (m)} from build_edge_linear_map().
        Used to convert traci road_id + lane_pos into linear absolute position.

    Returns
    -------
    dict
        SnapshotDict matching BusSimEnv.capture_full_system_snapshot() schema.
    """
    # ── buses ────────────────────────────────────────────────────────────────
    all_buses = []
    for bus_id in bridge.active_bus_ids:
        bus_obj = bridge.bus_obj_dic.get(bus_id)
        if bus_obj is None:
            continue

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

        if road_id in edge_map:
            pos = edge_map[road_id] + lane_pos
        else:
            # Fallback: use known stop distance if bus just arrived
            line_id  = getattr(bus_obj, "belong_line_id_s", None)
            stop_id  = getattr(bus_obj, "current_stop_id", None)
            if line_id and stop_id and stop_id in bridge.line_stop_distances.get(line_id, {}):
                pos = bridge.line_stop_distances[line_id][stop_id]
            else:
                pos = 0.0

        direction = getattr(bus_obj, "direction_n", 1)

        # Passenger load: count passengers currently on bus (approximated)
        load = getattr(bus_obj, "current_load_n", 0)
        if load == 0:
            # Fallback: use boarding count from stop data
            jsd = getattr(bus_obj, "just_server_stop_data_d", {})
            load = int(sum(v[2] if len(v) > 2 else 0 for v in jsd.values()))

        all_buses.append({
            "bus_id"   : bus_id,
            "pos"      : float(pos),
            "speed"    : float(speed),
            "load"     : int(load),
            "direction": int(direction),
        })

    # ── stations ─────────────────────────────────────────────────────────────
    all_stations = []
    for stop_id, stop_obj in bridge.stop_obj_dic.items():
        # Best-effort position: use first line that serves this stop
        pos = 0.0
        for line_id, stop_dists in bridge.line_stop_distances.items():
            if stop_id in stop_dists:
                pos = stop_dists[stop_id]
                break

        # Waiting passengers — stop_obj should have this after update_stop_state()
        waiting = getattr(stop_obj, "wait_passenger_num_n", None)
        if waiting is None:
            # older attribute name
            waiting = getattr(stop_obj, "waiting_passenger_num", 0)

        all_stations.append({
            "station_id"   : stop_id,
            "station_name" : stop_id,     # SUMO stop_id doubles as name
            "pos"          : float(pos),
            "waiting_count": int(waiting),
        })

    return {
        "sim_time"    : float(bridge.current_time),
        "all_buses"   : all_buses,
        "all_stations": all_stations,
    }


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
