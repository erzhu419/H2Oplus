"""
verify_phase0.py
================
Standalone verification for Phase 0 (common/data_utils.py).
No SUMO installation required — pure Python + PyTorch.

Run:
    cd /home/erzhu419/mine_code/sumo-rl/H2Oplus/bus_h2o
    python verify_phase0.py
"""

import sys
import os

# Make sure we can import from the local package
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from common.data_utils import (
    build_edge_linear_map,
    sumo_pos_to_linear,
    extract_structured_context,
    set_route_length,
    SimpleTransitionDiscriminator,
    compute_importance_weight,
)

EDGE_XML = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "network_data/a_sorted_busline_edge.xml",
))

PASSED = []
FAILED = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ✅ {name}")
        PASSED.append(name)
    else:
        msg = f"  ❌ {name}"
        if detail:
            msg += f"  →  {detail}"
        print(msg)
        FAILED.append(name)


# ---------------------------------------------------------------------------
# Test 1: Edge -> Linear Map (7X line)
# ---------------------------------------------------------------------------
print("\n[Test 1] build_edge_linear_map — line 7X")

edge_map_7X = build_edge_linear_map(EDGE_XML, "7X")
print(f"  Loaded {len(edge_map_7X)} edge entries for line 7X")

# First edge should start at 0
first_edge = next(iter(edge_map_7X))
check("First edge starts at 0.0", edge_map_7X[first_edge] == 0.0,
      f"got {edge_map_7X[first_edge]}")

# Route total should be somewhere in plausible range for a city bus line (5–50 km)
total_length = max(edge_map_7X.values())
check("Route length in plausible range (5 km – 50 km)",
      5_000 <= total_length <= 50_000,
      f"got {total_length:.0f} m")

print(f"  Route 7X total length estimate: {total_length:.1f} m")

# Test sumo_pos_to_linear
pos = sumo_pos_to_linear(first_edge, 10.0, edge_map_7X)
check("sumo_pos_to_linear: first edge offset 10 m → 10.0 m",
      abs(pos - 10.0) < 1e-6, f"got {pos}")

unknown_edge_pos = sumo_pos_to_linear("E_NONEXISTENT", 5.0, edge_map_7X)
check("sumo_pos_to_linear: unknown edge falls back to 0 + offset",
      abs(unknown_edge_pos - 5.0) < 1e-6, f"got {unknown_edge_pos}")


# ---------------------------------------------------------------------------
# Test 2: build_edge_linear_map — line 7S
# ---------------------------------------------------------------------------
print("\n[Test 2] build_edge_linear_map — line 7S")

edge_map_7S = build_edge_linear_map(EDGE_XML, "7S")
total_7S = max(edge_map_7S.values())
check("Route 7S length in plausible range (5 km – 50 km)",
      5_000 <= total_7S <= 50_000,
      f"got {total_7S:.0f} m")
print(f"  Route 7S total length estimate: {total_7S:.1f} m")


# ---------------------------------------------------------------------------
# Test 3: extract_structured_context with mock snapshot
# ---------------------------------------------------------------------------
print("\n[Test 3] extract_structured_context — mock snapshot")

# Use 7X route length
set_route_length(total_length)

# 3 buses at different positions along the route
mock_snapshot = {
    "all_buses": [
        {"pos": total_length * 0.1, "speed": 8.0,  "load": 20},
        {"pos": total_length * 0.4, "speed": 12.0, "load": 35},
        {"pos": total_length * 0.8, "speed": 5.0,  "load": 10},
    ],
    "all_stations": [
        {"pos": total_length * 0.15, "waiting_count": 8},
        {"pos": total_length * 0.35, "waiting_count": 15},
        {"pos": total_length * 0.60, "waiting_count": 3},
        {"pos": total_length * 0.85, "waiting_count": 0},
    ],
}

z = extract_structured_context(mock_snapshot, num_segments=10)

check("Output shape == (30,)",     z.shape == (30,),     f"got {z.shape}")
check("Output dtype == float32",   z.dtype == np.float32, f"got {z.dtype}")
check("No NaN in output",          not np.any(np.isnan(z)), str(z))
check("Speed slice in [0, 2]",     np.all(z[:10] >= 0) and np.all(z[:10] <= 2.0),
      f"max speed val = {z[:10].max():.3f}")
check("Density slice >= 0",        np.all(z[10:20] >= 0))
check("Waiting slice >= 0",        np.all(z[20:30] >= 0))
print(f"  z[:10] (speed)  : {z[:10].round(3)}")
print(f"  z[10:20] (dens) : {z[10:20].round(3)}")
print(f"  z[20:30] (wait) : {z[20:30].round(3)}")

# Empty snapshot → all zeros except default speed
z_empty = extract_structured_context({"all_buses": [], "all_stations": []})
check("Empty snapshot → speed defaults to 30/30=1.0",
      np.all(z_empty[:10] == 1.0), f"{z_empty[:10]}")
check("Empty snapshot → density == 0",
      np.all(z_empty[10:20] == 0.0))
check("Empty snapshot → waiting == 0",
      np.all(z_empty[20:30] == 0.0))


# ---------------------------------------------------------------------------
# Test 4: SimpleTransitionDiscriminator forward pass
# ---------------------------------------------------------------------------
print("\n[Test 4] SimpleTransitionDiscriminator — forward pass")

OBS_DIM = 7   # typical state_dim from sim.py
ACT_DIM = 1   # holding time
BATCH   = 8

D = SimpleTransitionDiscriminator(obs_dim=OBS_DIM, act_dim=ACT_DIM)

obs      = torch.randn(BATCH, OBS_DIM)
act      = torch.randn(BATCH, ACT_DIM)
next_obs = torch.randn(BATCH, OBS_DIM)
z_t      = torch.randn(BATCH, 30)
z_t1     = torch.randn(BATCH, 30)

logit = D(obs, act, next_obs, z_t, z_t1)

check("Output shape == (8, 1)",   logit.shape == (BATCH, 1),  f"got {logit.shape}")
check("No NaN in output",         not torch.any(torch.isnan(logit)).item())
check("Output requires grad",     logit.requires_grad)

# Importance weight helper
w = compute_importance_weight(D, obs, act, next_obs, z_t, z_t1)
check("Importance weight shape == (8, 1)", w.shape == (BATCH, 1))
check("Importance weight >= 0",            torch.all(w >= 0).item())
check("Importance weight not requires_grad (detached)", not w.requires_grad)

# Loss computation (BCEWithLogitsLoss)
criterion   = torch.nn.BCEWithLogitsLoss()
real_labels = torch.full_like(logit, 0.9)
sim_labels  = torch.full_like(logit, 0.1)
loss_real   = criterion(logit, real_labels)
loss_sim    = criterion(logit, sim_labels)
total_loss  = loss_real + loss_sim
total_loss.backward()
check("Backward pass succeeds (no error)", True)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*50}")
total = len(PASSED) + len(FAILED)
if not FAILED:
    print(f"✅ All {total} checks passed.  Phase 0 verification complete.")
else:
    print(f"❌ {len(FAILED)}/{total} checks FAILED:")
    for f in FAILED:
        print(f"   - {f}")
    sys.exit(1)
