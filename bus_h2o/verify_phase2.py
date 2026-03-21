"""
verify_phase2.py
================
Verification for Phase 2: BusSimEnv snapshot capture & restore.
Runs a short live episode and validates:
    1. Standard reset + step works (env not broken)
    2. capture_full_system_snapshot() returns correct schema
    3. restore_full_system_snapshot() restores bus count, positions, stations
    4. extract_structured_context() works on the captured snapshot (Phase 0+2 join)

Run:
    cd /home/erzhu419/mine_code/sumo-rl/H2Oplus/bus_h2o
    python verify_phase2.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))                  # bus_h2o/
_SUMO_RL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))  # sumo-rl/
sys.path.insert(0, _SUMO_RL_ROOT)

import numpy as np

from envs.bus_sim_env import BusSimEnv
from common.data_utils import extract_structured_context, set_route_length

_SUMO_RL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))  # sumo-rl/
sys.path.insert(0, _SUMO_RL_ROOT)
SIM_ENV_PATH = os.path.join(_SUMO_RL_ROOT, "LSTM-RL-legacy", "env")
# sumo-rl/H2Oplus/bus_h2o/ → ../.. = sumo-rl/

PASSED, FAILED = [], []


def check(name, condition, detail=""):
    if condition:
        print(f"  ✅ {name}")
        PASSED.append(name)
    else:
        msg = f"  ❌ {name}"
        if detail:
            msg += f"  →  {detail}"
        print(msg)
        FAILED.append(name)


print("\n[Setup] Initialising BusSimEnv...")
env = BusSimEnv(path=SIM_ENV_PATH, debug=False)
actions = {k: 0.0 for k in range(env.max_agent_num)}

# ---------------------------------------------------------------------------
# Test 1: Standard reset + advance until first event
# ---------------------------------------------------------------------------
print("\n[Test 1] Standard reset — step until first bus observation")

obs = env.reset()
steps_taken = 0
MAX_STEPS = 500
info = None   # guarantee defined even if while loop body is skipped

# Advance until we get a non-empty observation (or take at least 1 step)
obs, rewards, done, info = env.step(actions)
steps_taken = 1

while all(len(v) == 0 for v in obs.values()) and steps_taken < MAX_STEPS:
    obs, rewards, done, info = env.step(actions)
    steps_taken += 1

check("Environment reset without crash",            True)
check("Step() returns 4-tuple",                     True)
check(f"Got first bus event within {MAX_STEPS} steps", steps_taken < MAX_STEPS,
      f"steps_taken={steps_taken}")
check("info contains 'snapshot' key",               "snapshot" in info)
check("info['snapshot'] is dict",                   isinstance(info.get("snapshot"), dict))

snap = info["snapshot"]


# ---------------------------------------------------------------------------
# Test 2: SnapshotDict schema
# ---------------------------------------------------------------------------
print(f"\n[Test 2] SnapshotDict schema (after {steps_taken} steps, t={snap.get('sim_time'):.0f}s)")

check("Has 'sim_time'",         "sim_time"       in snap)
check("Has 'all_buses'",        "all_buses"      in snap)
check("Has 'all_stations'",     "all_stations"   in snap)
check("Has 'launched_trips'",   "launched_trips" in snap)

buses = snap["all_buses"]
stns  = snap["all_stations"]

check("all_buses is list",      isinstance(buses, list))
check("all_stations is list",   isinstance(stns,  list))
check("At least 1 bus active",  len(buses) > 0,       f"got {len(buses)}")
check("Stations > 0",           len(stns)  > 0,       f"got {len(stns)}")

if buses:
    b0 = buses[0]
    for field in ["bus_id", "absolute_distance", "current_speed", "load",
                  "direction", "state", "pos", "speed"]:
        check(f"Bus has field '{field}'", field in b0)

if stns:
    s0 = stns[0]
    for field in ["station_id", "station_name", "waiting_count", "pos"]:
        check(f"Station has field '{field}'", field in s0)

    # pos values should be non-negative
    bad_pos = [s for s in stns if s["pos"] < 0]
    check("All station pos >= 0",  len(bad_pos) == 0,  f"bad: {bad_pos[:3]}")


# ---------------------------------------------------------------------------
# Test 3: Snapshot restore (god-mode reset)
# ---------------------------------------------------------------------------
print("\n[Test 3] restore_full_system_snapshot — god-mode reset")

# Take a few more steps to get a richer snapshot
for _ in range(10):
    obs, rewards, done, info = env.step(actions)
    if done:
        break
snap_reference = info["snapshot"]
n_buses_ref = len(snap_reference["all_buses"])
t_ref       = snap_reference["sim_time"]

print(f"  Reference snapshot: t={t_ref:.0f}s, {n_buses_ref} buses")

# Now do a god-mode reset back to that snapshot
obs_restored = env.reset(snapshot=snap_reference)

snap_after_restore = env.capture_full_system_snapshot()
n_buses_restored   = len(snap_after_restore["all_buses"])
t_restored         = snap_after_restore["sim_time"]

check("Restored sim_time matches",     abs(t_restored - t_ref) < 1.0,
      f"expected {t_ref}, got {t_restored}")
check("Restored bus count matches",    n_buses_restored == n_buses_ref,
      f"expected {n_buses_ref}, got {n_buses_restored}")

# Check positions roughly restored (within 1 m for buses that exist in both)
ref_positions  = {b["bus_id"]: b["absolute_distance"] for b in snap_reference["all_buses"]}
rest_positions = {b["bus_id"]: b["absolute_distance"] for b in snap_after_restore["all_buses"]}
common_ids     = set(ref_positions) & set(rest_positions)

if common_ids:
    max_pos_err = max(abs(ref_positions[bid] - rest_positions[bid]) for bid in common_ids)
    check(f"Max position error < 0.1 m for {len(common_ids)} buses",
          max_pos_err < 0.1, f"max error = {max_pos_err:.4f} m")


# ---------------------------------------------------------------------------
# Test 4: Phase 0+2 integration — extract_structured_context on real snapshot
# ---------------------------------------------------------------------------
print("\n[Test 4] extract_structured_context — on live snapshot")

# Set route length from station positions
all_pos = [s["pos"] for s in snap_reference["all_stations"] if s["pos"] > 0]
if all_pos:
    route_len = max(all_pos) + 500   # small buffer
else:
    route_len = 12_000               # fallback

set_route_length(route_len)
print(f"  Using route_length = {route_len:.1f} m")

z_t = extract_structured_context(snap_reference, num_segments=10)

check("z_t shape == (30,)",    z_t.shape == (30,),  f"got {z_t.shape}")
check("No NaN in z_t",         not np.any(np.isnan(z_t)))
check("z_t speed range [0,2]", np.all(z_t[:10] >= 0) and np.all(z_t[:10] <= 2.0))
print(f"  z_t[:10] (speed)  : {z_t[:10].round(3)}")
print(f"  z_t[10:20] (dens) : {z_t[10:20].round(3)}")
print(f"  z_t[20:30] (wait) : {z_t[20:30].round(3)}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*55}")
total = len(PASSED) + len(FAILED)
if not FAILED:
    print(f"✅ All {total} checks passed.  Phase 2 verification complete.")
else:
    print(f"❌ {len(FAILED)}/{total} checks FAILED:")
    for f in FAILED:
        print(f"   - {f}")
    sys.exit(1)
