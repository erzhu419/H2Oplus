"""
plot_network.py
===============
Plot the full Changsha SUMO network topology with all 12 bus lines colored
distinctly. Line 7X is highlighted only as a visual reference; the policy
controls all 12 directional services.

Output: paper/figures/network_topology.pdf (and .png)
"""

import os, re, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sumolib

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

SUMO = "/home/erzhu419/mine_code/sumo-rl/SUMO_ruiguang"
NET  = os.path.join(SUMO, "b_network", "5g_changsha_bus_network_with_signal_d.net.xml")
BUS  = os.path.join(SUMO, "d_bus_rou", "2_bus_timetable.rou.xml")
STOP = os.path.join(SUMO, "b_network", "3_bus_station.add.xml")
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "network_topology")

LINE_ORDER = ["7X", "7S", "102X", "102S", "311X", "311S",
              "122X", "122S", "406X", "406S", "705X", "705S"]
LINE_COLORS = {
    "7X":   "#d62728",   # bright red — reference line (longest)
    "7S":   "#ff7f0e",   # orange
    "102X": "#2ca02c",
    "102S": "#98df8a",
    "311X": "#1f77b4",
    "311S": "#aec7e8",
    "122X": "#9467bd",
    "122S": "#c5b0d5",
    "406X": "#8c564b",
    "406S": "#c49c94",
    "705X": "#e377c2",
    "705S": "#f7b6d2",
}

print("Loading network (may take a few seconds)...")
net = sumolib.net.readNet(NET)
print(f"  {len(net.getEdges())} edges, {len(net.getNodes())} nodes")

# ── 1. Extract background edges (light grey) ─────────────────────────────────
bg_xs, bg_ys = [], []
for edge in net.getEdges():
    for shape_pt, next_pt in zip(edge.getShape()[:-1], edge.getShape()[1:]):
        bg_xs.append([shape_pt[0], next_pt[0]])
        bg_ys.append([shape_pt[1], next_pt[1]])

# ── 2. Parse bus routes: for each line get the sequence of edges ──────────────
print("Parsing bus routes...")
line_edges = defaultdict(set)  # line -> set of edge IDs (one sample bus per line is enough)
line_first_bus_seen = set()
with open(BUS) as f:
    current_bus_id = None
    for line_txt in f:
        m = re.search(r'<vehicle id="([^"]+)"', line_txt)
        if m:
            current_bus_id = m.group(1)
            # extract line name (everything before last '_<digits>')
            m2 = re.match(r"^(.+)_\d+$", current_bus_id)
            current_line = m2.group(1) if m2 else current_bus_id
            continue
        m = re.search(r'<route edges="([^"]+)"', line_txt)
        if m and current_bus_id and current_line not in line_first_bus_seen:
            edges = m.group(1).split()
            for e in edges:
                line_edges[current_line].add(e)
            line_first_bus_seen.add(current_line)

for k in LINE_ORDER:
    print(f"  {k}: {len(line_edges[k])} edges")

# ── 3. Parse bus stops for markers ────────────────────────────────────────────
stops = []
import xml.etree.ElementTree as ET
tree = ET.parse(STOP)
for bs in tree.getroot().iter("busStop"):
    lane = bs.get("lane")
    try:
        edge_id = lane.rsplit("_", 1)[0]
        edge = net.getEdge(edge_id)
        # Position along edge
        start = float(bs.get("startPos", 0))
        end = float(bs.get("endPos", start + 10))
        pos = (start + end) / 2
        lane_obj = edge.getLane(int(lane.rsplit("_", 1)[1]))
        coords = lane_obj.getShape()
        # approximate by proportion
        prop = min(pos / lane_obj.getLength(), 1.0)
        idx = int(prop * (len(coords) - 1))
        stops.append((coords[idx][0], coords[idx][1], bs.get("id", "")))
    except Exception:
        pass
print(f"  {len(stops)} bus stops placed")

# Bucket stops by line prefix
def stop_line_prefix(stop_id):
    # Stops are named like "7X01_7S26" — first prefix before the digits
    import re as _re
    m = _re.match(r"^([0-9]+[XS])", stop_id)
    return m.group(1) if m else None

stops_by_line = {k: [] for k in LINE_ORDER}
for x, y, sid in stops:
    pref = stop_line_prefix(sid)
    if pref in stops_by_line:
        stops_by_line[pref].append((x, y, sid))
for k in LINE_ORDER:
    print(f"  stops for {k}: {len(stops_by_line[k])}")

# ── 4. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 9))

# Background: all roads in grey
from matplotlib.collections import LineCollection
segs = list(zip([list(zip(x, y)) for x, y in zip(bg_xs, bg_ys)]))
# Simpler: just matplotlib.plot a line per edge
for xs, ys in zip(bg_xs, bg_ys):
    ax.plot(xs, ys, color="#cccccc", lw=0.3, zorder=1)

# Plot each line
def plot_line(line_key, zorder, lw, alpha=0.85):
    color = LINE_COLORS[line_key]
    edge_ids = line_edges[line_key]
    for eid in edge_ids:
        try:
            edge = net.getEdge(eid)
        except KeyError:
            continue
        shape = edge.getShape()
        xs = [p[0] for p in shape]
        ys = [p[1] for p in shape]
        ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=zorder, solid_capstyle="round")

# Draw the other services first so the reference service remains legible.
for k in LINE_ORDER:
    if k == "7X": continue
    plot_line(k, zorder=2, lw=1.2, alpha=0.75)
    # Small stop markers for the other services.
    k_stops = stops_by_line.get(k, [])
    if k_stops:
        xs = [s[0] for s in k_stops]
        ys = [s[1] for s in k_stops]
        ax.scatter(xs, ys, s=8, c=LINE_COLORS[k], edgecolors="black",
                   linewidths=0.3, zorder=3, marker="o", alpha=0.85)

# Reference service on top (thicker for visual legibility only).
plot_line("7X", zorder=5, lw=2.8, alpha=1.0)

# Larger markers identify the visually highlighted reference service.
line7x_stops = stops_by_line.get("7X", [])
if line7x_stops:
    xs = [s[0] for s in line7x_stops]
    ys = [s[1] for s in line7x_stops]
    ax.scatter(xs, ys, s=36, c=LINE_COLORS["7X"], edgecolors="black",
               linewidths=0.8, zorder=6, marker="o", label="Line 7X reference stops")

# Legend
patches = [mpatches.Patch(color=LINE_COLORS[k], label=f"Line {k}")
           for k in LINE_ORDER]
patches.append(mpatches.Patch(color="#cccccc", label="Road network"))
ax.legend(handles=patches, loc="lower left", fontsize=8, ncol=2, framealpha=0.9)

ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
ax.spines[["top","right","bottom","left"]].set_visible(False)

plt.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT + ".pdf", bbox_inches="tight")
plt.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUT}.pdf / .png")
