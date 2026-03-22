"""
extract_sumo_network.py
=======================
Phase 2 — Parse SUMO Ruiguang XML files and generate per-line calibrated data
for all 12 bus lines (7X, 7S, 102X, 102S, 122X, 122S, 311X, 311S, 406X, 406S, 705X, 705S).

Output directory structure:
    calibrated_env/data/{line_id}/
        stop_news.xlsx      — stop index, stop name, cumulative distance (m)
        route_news.xlsx     — route_id, start_stop, end_stop, distance, V_max, OD counts per hour
        time_table.xlsx     — launch_time (s), direction (1=upstream 0=downstream or 0 for one-way)
        passenger_OD.xlsx   — (stop, period) x dest_stop boarding counts

Usage
-----
    cd H2Oplus/bus_h2o
    python data/extract_sumo_network.py \\
        --sumo_root /home/erzhu419/mine_code/sumo-rl/SUMO_ruiguang \\
        --out_dir   calibrated_env/data

Dependencies: pandas, openpyxl, xml.etree.ElementTree (stdlib)
"""

import argparse
import os
import sys
import re
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from collections import defaultdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_busline_edge_cumdist(edge_xml: str) -> dict[str, tuple[dict, float]]:
    """
    Parse a_sorted_busline_edge.xml.
    Returns {line_id: ({edge_id: cum_start_m}, total_length_m)}.
    """
    tree = ET.parse(edge_xml)
    result = {}
    for bl in tree.getroot().findall('busline'):
        lid = bl.get('id')
        cum = 0.0
        edge_map = {}
        for elem in bl.findall('element'):
            eid    = elem.get('id')
            length = float(elem.get('length', 0))
            if eid not in edge_map:
                edge_map[eid] = cum
            cum += length
        result[lid] = (edge_map, cum)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_bus_stops(station_add_xml: str) -> dict:
    """Return {stop_id: {'name': str, 'edge_id': str, 'end_pos': float}}."""
    tree = ET.parse(station_add_xml)
    stops = {}
    for bs in tree.getroot().findall('.//busStop'):
        sid = bs.get('id')
        lane = bs.get('lane', '')
        edge = lane.rsplit('_', 1)[0] if '_' in lane else lane
        stops[sid] = {
            'name'    : bs.get('name', sid),
            'edge_id' : edge,
            'end_pos' : float(bs.get('endPos', 0)),
            'lane'    : lane,
        }
    return stops


def _parse_edge_lengths(net_xml: str) -> dict:
    """Return {edge_id: length_m} from the SUMO network file."""
    tree = ET.parse(net_xml)
    lengths = {}
    for edge in tree.getroot().findall('.//edge'):
        eid = edge.get('id')
        if eid and not eid.startswith(':'):  # skip junction edges
            for lane in edge.findall('lane'):
                lengths[eid] = float(lane.get('length', 0))
                break   # take first lane length
    return lengths


def _parse_bus_routes(rou_xml: str) -> dict:
    """Return {vehicle_id: {'depart': float, 'stops': [stop_id, ...], 'edges': [edge, ...]}}.
    Line id is inferred from the first stop prefix (e.g. '7X01_7S26' → '7X').
    """
    tree = ET.parse(rou_xml)
    root = tree.getroot()
    vehicles = {}
    for veh in root.findall('.//vehicle'):
        vid  = veh.get('id')
        dept = float(veh.get('depart', 0))
        route_el = veh.find('route')
        edges = route_el.get('edges', '').split() if route_el is not None else []
        stops = [s.get('busStop') for s in veh.findall('.//stop') if s.get('busStop')]
        vehicles[vid] = {'depart': dept, 'stops': stops, 'edges': edges}
    return vehicles


def _infer_line_id(first_stop: str, all_lines: list[str]) -> str:
    """From a stop name like '7X01_7S26', infer the primary line (e.g. '7X').
    Try all known line IDs sorted by length descending to avoid prefix conflicts.
    """
    for line_id in sorted(all_lines, key=len, reverse=True):
        if first_stop.startswith(line_id):
            return line_id
    return 'UNKNOWN'


def _line_stop_prefix(line_id: str, stop_id: str) -> bool:
    """True if stop_id belongs to line_id (prefix match)."""
    return stop_id.startswith(line_id)


def _cumulative_distances_from_edgemap(
        stop_sequence: list[str],
        stop_meta: dict,
        line_edge_cumdist: tuple[dict, float]) -> list[float]:
    """
    Convert stop sequence to cumulative distances using a_sorted_busline_edge.xml.
    For each stop its linear position = edge_map[edge_id] + end_pos_on_edge.
    """
    edge_map, total = line_edge_cumdist
    # Fallback segment if edge not found: total / n_segments
    n = max(len(stop_sequence) - 1, 1)
    seg_fallback = total / n

    pos = []
    for sid in stop_sequence:
        meta = stop_meta.get(sid, {})
        eid  = meta.get('edge_id', '')
        epos = meta.get('end_pos', 0.0)
        if eid in edge_map:
            pos.append(edge_map[eid] + epos)
        else:
            pos.append(None)   # resolve below

    # Fill missing with linear interpolation
    for i, p in enumerate(pos):
        if p is None:
            prev = next((pos[j] for j in range(i-1, -1, -1) if pos[j] is not None), 0.0)
            nxt  = next((pos[j] for j in range(i+1, len(pos)) if pos[j] is not None), total)
            pos[i] = (prev + nxt) / 2.0

    # Ensure monotone (SUMO route is ordered)
    for i in range(1, len(pos)):
        if pos[i] <= pos[i-1]:
            pos[i] = pos[i-1] + seg_fallback

    # Convert to cumulative from 0
    base = pos[0]
    cum  = [p - base for p in pos]
    return cum


def _passenger_od_placeholder(stop_ids: list[str], hours: list[int]) -> pd.DataFrame:
    """Return a placeholder OD table with uniform low demand."""
    rng = np.random.default_rng(42)
    idx = pd.MultiIndex.from_product([stop_ids, hours], names=['stop_name', 'period'])
    data = {}
    for dst in stop_ids:
        data[dst] = rng.integers(1, 8, size=len(idx)).astype(float)
    df = pd.DataFrame(data, index=idx)
    # Zero out diagonal (no self-demand)
    for s in stop_ids:
        if s in df.columns:
            df.loc[(s, slice(None)), s] = 0.0
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

KNOWN_LINES = ['7X', '7S', '102X', '102S', '122X', '122S',
               '311X', '311S', '406X', '406S', '705X', '705S']
HOURS = list(range(6, 20))           # 6:00 – 19:00
DEFAULT_V_MAX = 10.0                 # m/s (~36 km/h), calibrated average


def extract_one_line(line_id: str,
                     vehicles_for_line: list[dict],
                     stop_meta: dict,
                     line_cumdist: tuple[dict, float],   # from a_sorted_busline_edge.xml
                     out_dir: str) -> None:
    """Generate the four xlsx files for one line."""
    os.makedirs(out_dir, exist_ok=True)

    # --- build stop sequence from first vehicle ---
    ref_veh = sorted(vehicles_for_line, key=lambda v: v['depart'])[0]
    raw_stops = ref_veh['stops']

    # Keep only stops whose name starts with line_id OR shared stops (contains line_id)
    def stop_belongs_to_any(s):
        return any(s.startswith(l) for l in KNOWN_LINES)

    line_stops = [s for s in raw_stops if s.startswith(line_id) or not stop_belongs_to_any(s)]
    if not line_stops:
        line_stops = [s for s in raw_stops if line_id in s]
    if not line_stops:
        line_stops = raw_stops

    cum_dists = _cumulative_distances_from_edgemap(line_stops, stop_meta, line_cumdist)

    # ---------- stop_news.xlsx ----------
    stop_records = []
    for idx, (sid, d) in enumerate(zip(line_stops, cum_dists)):
        stop_records.append({'stop_id': idx,
                             'stop_name': sid,
                             'cumulative_dist_m': d})
    stops_df = pd.DataFrame(stop_records)
    stops_df.to_excel(os.path.join(out_dir, 'stop_news.xlsx'), index=False)

    # ---------- route_news.xlsx ----------
    route_records = []
    for i in range(len(line_stops) - 1):
        dist = cum_dists[i + 1] - cum_dists[i]
        row = {
            'route_id'  : i,
            'start_stop': line_stops[i],
            'end_stop'  : line_stops[i + 1],
            'distance'  : max(dist, 50.0),   # guard zero-length edges
            'V_max'     : DEFAULT_V_MAX,
        }
        # Placeholder hourly travel counts (pax)
        for hr in HOURS:
            row[f'{hr:02d}:00:00'] = int(np.random.default_rng(42 + i + hr).integers(3, 12))
        route_records.append(row)
    routes_df = pd.DataFrame(route_records)
    routes_df.to_excel(os.path.join(out_dir, 'route_news.xlsx'), index=False)

    # ---------- time_table.xlsx ----------
    tt_records = []
    for veh in sorted(vehicles_for_line, key=lambda v: v['depart']):
        tt_records.append({'launch_time': veh['depart'], 'direction': 1})
    tt_df = pd.DataFrame(tt_records)
    tt_df.to_excel(os.path.join(out_dir, 'time_table.xlsx'), index=False)

    # ---------- passenger_OD.xlsx ----------
    od_df = _passenger_od_placeholder(line_stops, HOURS)
    od_df.to_excel(os.path.join(out_dir, 'passenger_OD.xlsx'))

    print(f"  {line_id:6s}: {len(line_stops):3d} stops, {len(line_stops)-1:3d} routes, "
          f"{len(tt_records):3d} trips, route_len={cum_dists[-1]:.0f}m → {out_dir}")


def main(args):
    sumo_root = args.sumo_root

    station_xml = os.path.join(sumo_root, 'b_network', '3_bus_station.add.xml')
    net_xml     = os.path.join(sumo_root, 'b_network', '5g_changsha_bus_network_with_signal_d.net.xml')
    rou_xml     = os.path.join(sumo_root, 'd_bus_rou', '2_bus_timetable.rou.xml')
    edge_xml    = os.path.join(sumo_root, 'online_control', 'intersection_delay',
                               'a_sorted_busline_edge.xml')

    print("[1] Parsing bus stop definitions...")
    stop_meta = _parse_bus_stops(station_xml)
    print(f"    {len(stop_meta)} bus stops loaded")

    print("[2] Parsing busline edge cumulative distances...")
    all_cumdist = _parse_busline_edge_cumdist(edge_xml)
    for lid, (_, total) in sorted(all_cumdist.items()):
        print(f"    {lid:6s}: {total:.0f}m")

    print("[3] Parsing bus timetable routes...")
    all_vehicles = _parse_bus_routes(rou_xml)
    print(f"    {len(all_vehicles)} vehicles")

    # Assign each vehicle to a line by first stop prefix
    line_vehicles: dict[str, list] = defaultdict(list)
    unknown = 0
    for vid, veh in all_vehicles.items():
        if not veh['stops']:
            unknown += 1
            continue
        lid = _infer_line_id(veh['stops'][0], KNOWN_LINES)
        if lid == 'UNKNOWN' and len(veh['stops']) > 1:
            lid = _infer_line_id(veh['stops'][1], KNOWN_LINES)
        if lid == 'UNKNOWN':
            # Try scanning all stops for any line prefix
            for s in veh['stops'][2:]:
                lid2 = _infer_line_id(s, KNOWN_LINES)
                if lid2 != 'UNKNOWN':
                    lid = lid2; break
        line_vehicles[lid].append(veh)
    if unknown:
        print(f"    NOTE: {unknown} vehicles with no stops (skipped)")
    if line_vehicles.get('UNKNOWN'):
        print(f"    NOTE: {len(line_vehicles['UNKNOWN'])} vehicles could not be assigned to a line")

    print(f"\n[4] Generating calibrated data per line → {args.out_dir}")
    for line_id in KNOWN_LINES:
        vehs = line_vehicles.get(line_id, [])
        if not vehs:
            print(f"  {line_id:6s}: NO VEHICLES FOUND — skipping")
            continue
        cumdist = all_cumdist.get(line_id, ({}, 1.0))
        out = os.path.join(args.out_dir, line_id)
        extract_one_line(line_id, vehs, stop_meta, cumdist, out)

    print("\n✅ extract_sumo_network.py done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract SUMO network data for SimpleSim')
    parser.add_argument('--sumo_root', type=str,
                        default='/home/erzhu419/mine_code/sumo-rl/SUMO_ruiguang',
                        help='Root of SUMO_ruiguang directory')
    parser.add_argument('--out_dir', type=str,
                        default='calibrated_env/data',
                        help='Output base dir for per-line xlsx files')
    args = parser.parse_args()
    main(args)
