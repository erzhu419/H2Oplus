"""
generate_calibrated_data.py
============================
Generates calibrated SimpleSim data files based on the real 7X/7S network.

Source of truth:
    - Stop sequence: a_sorted_busline_edge.xml (type="s" elements)
    - Inter-stop distances: cumulative edge lengths from XML
    - TL delays: sum of E[delay] = phase_time/2 per intersection in segment
    - Effective V_max per segment: dist / (expected_travel_time)
      where expected_travel_time = dist / free_flow_speed + TL_delay
    - OD demand: uniform approximation based on 30-person trip average
      (real SUMO passenger generation uses Poisson arrival)
    - Timetable: 7X and 7S, headway from real schedule

Output files (in H2Oplus/bus_h2o/data/):
    - stop_news.xlsx      — stop_id, stop_name
    - route_news.xlsx     — route_id, start_stop, end_stop, distance, V_max,
                            + hourly speed limits (6:00-19:00)
    - time_table.xlsx     — launch_time, direction (1=7X outbound, 0=7S inbound)
    - passenger_OD.xlsx   — OD demand matrix (per hour)

Run:
    python3 generate_calibrated_data.py
"""

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import os

XML = os.path.abspath(os.path.join(os.path.dirname(__file__), 
    '../../../SUMO_ruiguang/online_control/intersection_delay/a_sorted_busline_edge.xml'))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))  # = bus_h2o/data/
os.makedirs(OUT_DIR, exist_ok=True)

FREE_FLOW_SPEED = 10.0    # m/s ≈ 36 km/h (urban free-flow)
STOP_DWELL     = 15.0     # s   (average dwell at each stop, bundled into V_max)

# ---------------------------------------------------------------------------
# 1. Parse XML to extract stops and inter-stop segments
# ---------------------------------------------------------------------------

def parse_line(root, line_id):
    """Returns list of stop dicts with cumulative_m, dist_m, tl_delay_s."""
    bl = next(b for b in root.findall('busline') if b.get('id') == line_id)
    stops = []
    cumulative = 0.0
    last_stop_cum = 0.0
    current_tls = []    # (phase_time, cycle_time) per intersection in segment

    for elem in bl.findall('element'):
        etype = elem.get('type')
        length = float(elem.get('length', 0))

        if etype == 's':
            stop_id = elem.get('stop_id', '')
            # primary stop name = first component (e.g. "7X01_7S26" → "7X01")
            primary = stop_id.split('_')[0]
            dist_m = cumulative - last_stop_cum
            # expected TL delay = sum(phase_time / 2) for each TL in segment
            tl_delay = sum(pt / 2 for pt, ct in current_tls)
            stops.append({
                'stop_id': stop_id,
                'primary': primary,
                'cumulative_m': cumulative,
                'dist_m': dist_m,
                'tl_delay_s': tl_delay,
                'n_tls': len(current_tls),
            })
            last_stop_cum = cumulative
            current_tls = []
        elif etype == 'i':
            phase = float(elem.get('phase_time', 0))
            cycle = float(elem.get('cycle_time', 90))
            current_tls.append((phase, cycle))

        cumulative += length

    return stops, cumulative


def effective_vmax(dist_m, tl_delay_s, stop_dwell=STOP_DWELL, free_flow=FREE_FLOW_SPEED):
    """
    Compute effective speed limit so that SimpleSim travel time matches reality.
    
    travel_time = dist_m / free_flow + tl_delay_s + stop_dwell
    V_max_eff   = dist_m / (travel_time - stop_dwell)  # dwell handled separately by sim
    """
    if dist_m <= 0:
        return free_flow
    travel_time = dist_m / free_flow + tl_delay_s
    if travel_time <= 0:
        return free_flow
    # Return the effective speed (without dwell, which sim adds separately)
    return dist_m / travel_time


print("Parsing XML...")
tree = ET.parse(XML)
root = tree.getroot()

stops_7X, route_len_7X = parse_line(root, '7X')
stops_7S, route_len_7S = parse_line(root, '7S')

print(f"7X: {len(stops_7X)} stops, total {route_len_7X:.0f} m")
print(f"7S: {len(stops_7S)} stops, total {route_len_7S:.0f} m")

# ---------------------------------------------------------------------------
# 2. Build stop_news.xlsx
# ---------------------------------------------------------------------------
# Convention: sim uses integer stop_id, stop_name
# Upstream (direction=True): 7X01..7X25 + terminals
# Downstream (direction=False): 7S01..7S26 + terminals (reversed internally by sim.set_stations)

# We keep the same schema as the legacy env:
#   stop_id  0=Terminal_up, 1=first_stop, ..., N=Terminal_down
# Use 7X stop names for outbound, terminal names match

all_stop_names = ['Terminal_up'] + [s['primary'] for s in stops_7X] + ['Terminal_down']
stop_news_df = pd.DataFrame({
    'stop_id': list(range(len(all_stop_names))),
    'stop_name': all_stop_names
})
stop_news_path = os.path.join(OUT_DIR, 'stop_news.xlsx')
stop_news_df.to_excel(stop_news_path, index=False)
print(f"Saved {stop_news_path}  ({len(stop_news_df)} stops)")

# ---------------------------------------------------------------------------
# 3. Build route_news.xlsx
#    Columns: route_id, start_stop, end_stop, distance, V_max, 06:00 .. 19:00
# ---------------------------------------------------------------------------
# Hourly speed factors (peak/off-peak relative to free_flow)
# Based on typical urban patterns: 0.7 peak, 1.0 off-peak
HOUR_FACTORS = {
     6: 0.90,  # early morning, good flow
     7: 0.70,  # AM peak
     8: 0.65,  # heavy AM peak
     9: 0.80,
    10: 0.90,
    11: 0.95,
    12: 0.90,
    13: 0.90,
    14: 0.95,
    15: 0.90,
    16: 0.75,  # PM peak building
    17: 0.68,  # heavy PM peak
    18: 0.72,
    19: 0.85,
}
HOURS = list(HOUR_FACTORS.keys())
import datetime
HOUR_COLS = [datetime.time(h, 0) for h in HOURS]

def build_route_rows(stops, stop_names, start_idx):
    """Build route_news rows for one direction."""
    rows = []
    n = len(stop_names)
    for i in range(n - 1):
        start_s = stop_names[i]
        end_s   = stop_names[i + 1]
        if i < len(stops):
            dist_m     = stops[i]['dist_m']
            tl_delay_s = stops[i]['tl_delay_s']
        else:
            dist_m     = 500.0
            tl_delay_s = 0.0
        
        v_eff = effective_vmax(dist_m, tl_delay_s)
        row = {
            'route_id': start_idx + i,
            'start_stop': start_s,
            'end_stop': end_s,
            'distance': round(dist_m),
            'V_max': round(v_eff, 2),
        }
        for h in HOURS:
            row[datetime.time(h, 0)] = round(v_eff * HOUR_FACTORS[h], 2)
        rows.append(row)
    return rows

stop_names_up   = all_stop_names                    # Terminal_up → X01..X25 → Terminal_down
stop_names_down = list(reversed(all_stop_names))    # Terminal_down → X25..X01 → Terminal_up

rows_up   = build_route_rows(stops_7X, stop_names_up,   start_idx=0)
n_up      = len(rows_up)
# For downstream, use 7S stops distances (different geometry, not just reversed)
stop_names_down_s = ['Terminal_down'] + [s['primary'] for s in stops_7S] + ['Terminal_up']
rows_down = build_route_rows(stops_7S, stops_7S, start_idx=n_up)
for i, r in enumerate(rows_down):
    r['route_id'] = n_up + i
    r['start_stop'] = stop_names_down_s[i]
    r['end_stop']   = stop_names_down_s[i + 1]

all_route_rows = rows_up + rows_down
route_news_df = pd.DataFrame(all_route_rows)
route_news_path = os.path.join(OUT_DIR, 'route_news.xlsx')
route_news_df.to_excel(route_news_path, index=False)
print(f"Saved {route_news_path}  ({len(route_news_df)} route segments)")

# ---------------------------------------------------------------------------
# 4. Build time_table.xlsx
#    Real 7X/7S timetable: headway ~6 min in peak, ~10 min off-peak
#    Simulate 6:00 to 20:00 operation
# ---------------------------------------------------------------------------
HEADWAY_PEAK     = 360   # 6 min in seconds
HEADWAY_OFFPEAK  = 600   # 10 min
PEAK_HOURS       = {7, 8, 17, 18}  # peak hours
OP_START         = 6 * 3600        # 06:00
OP_END           = 20 * 3600       # 20:00
DIRECTION_UP     = 1               # 7X outbound
DIRECTION_DOWN   = 0               # 7S inbound

tt_rows = []
t = OP_START
while t < OP_END:
    hour = t // 3600
    headway = HEADWAY_PEAK if hour in PEAK_HOURS else HEADWAY_OFFPEAK
    tt_rows.append({'launch_time': t, 'direction': DIRECTION_UP})
    tt_rows.append({'launch_time': t, 'direction': DIRECTION_DOWN})
    t += headway

tt_df = pd.DataFrame(tt_rows).sort_values('launch_time').reset_index(drop=True)
tt_path = os.path.join(OUT_DIR, 'time_table.xlsx')
tt_df.to_excel(tt_path, index=False)
print(f"Saved {tt_path}  ({len(tt_df)} timetable entries, {len(tt_df)//2} trips each direction)")

# ---------------------------------------------------------------------------
# 5. Build passenger_OD.xlsx
#    Legacy format: rows = MultiIndex(stop_name, HH:MM:SS), columns = dest_stop
#    Loaded by sim.py as: pd.read_excel(..., index_col=[1, 0])
#    which gives MultiIndex [(stop_name, time_period), ...]
# ---------------------------------------------------------------------------
effective_stops = [row['stop_name'] for _, row in stop_news_df.iterrows()
                   if row['stop_name'] not in ('Terminal_up', 'Terminal_down')]
n_eff = len(effective_stops)

BASE_DEMAND = 4.0         # pax/hr baseline per O-D pair (conservative for 25-stop line)
HOUR_DEMAND_SCALE = {
     6: 0.8, 7: 1.5, 8: 1.8, 9: 1.2, 10: 0.9, 11: 0.8,
    12: 0.9, 13: 0.8, 14: 0.8, 15: 0.9, 16: 1.2, 17: 1.6,
    18: 1.4, 19: 0.9,
}

hour_strs = [f"{h:02d}:00:00" for h in HOURS]

# Build rows: one block per time_period, one row per origin stop
# Written columns: [stop_name, time_period, dest1, dest2, ...]
# When read with index_col=[1, 0], col 0=stop_name, col 1=time_period become the MultiIndex
rows = []
for h_str, h in zip(hour_strs, HOURS):
    scale = HOUR_DEMAND_SCALE.get(h, 1.0)
    for origin in effective_stops:
        # NOTE: sim.py calls pd.read_excel(..., index_col=[1, 0])
        # That means col[1] → MI level 0, col[0] → MI level 1.
        # So col0=time_period, col1=stop_name gives:
        #   MI level 0 = stop_name, MI level 1 = time_period  ← matches legacy
        row = {'time_period': h_str, 'stop_name': origin}
        for dest in effective_stops:
            row[dest] = 0 if dest == origin else round(BASE_DEMAND * scale)
        rows.append(row)

od_df = pd.DataFrame(rows)
# When pd.read_excel reads with index_col=[1, 0] it reads cols [1,0] as the MI.
# Our file has: col0=stop_name, col1=time_period → with index_col=[1,0] we get (time_period,stop_name).
# Legacy actually has (stop_name, time_period) in the index.
# To mimic exactly: the FIRST non-index col is the first index level when read_excel stores them.
# Simplest: write without explicit multi-index, let read_excel pick up cols 0 and 1 as MI.
od_path = os.path.join(OUT_DIR, 'passenger_OD.xlsx')
od_df.to_excel(od_path, index=False)
print(f"Saved {od_path}  ({n_eff} stops × {len(hour_strs)} hours = {n_eff * len(hour_strs)} rows)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== CALIBRATION SUMMARY ===")
print(f"7X: {len(stops_7X)} stops, route = {route_len_7X:.0f} m")
print(f"7S: {len(stops_7S)} stops, route = {route_len_7S:.0f} m")
print(f"Effective stop names: {all_stop_names[:5]} ... {all_stop_names[-3:]}")
print(f"Route V_max range: {route_news_df['V_max'].min():.1f} – {route_news_df['V_max'].max():.1f} m/s")
print(f"Timetable: {len(tt_df)//2} trips per direction, {len(tt_df)} total")
print(f"OD: {n_eff} stops × {len(hour_strs)} time periods = {n_eff * len(hour_strs)} rows")
print(f"\nOutput directory: {OUT_DIR}")
print("All files ready. Update BusSimEnv path= to point to bus_h2o/data/")

