"""
test_disc_drift.py
==================
Test how discriminator output evolves as SIM rollout progresses
from a SUMO snapshot reset point.

Hypothesis
----------
Immediately after reset, SIM state == SUMO state,
so discriminator can't tell them apart (w ~ 1.0).
As SIM runs forward, its trajectory diverges from SUMO-like dynamics,
and discriminator should increasingly detect this (w decreasing, P(real) dropping).

Procedure
---------
Phase 1:  Collect SIM z data (fresh-reset episodes with zero-hold policy)
Phase 2:  Train discriminator on offline z (real=SUMO) vs SIM z (sim)
Phase 3:  Drift test — reset SIM from N offline snapshots, run forward
          K steps each, track discriminator output at every bus arrival
Phase 4:  Plot  w / P(real) / logit  vs  step-since-reset

Usage
-----
    cd H2Oplus/SimpleSAC
    python test_disc_drift.py \
        --n_snapshots 10 \
        --max_steps 50 \
        --disc_train_epochs 200 \
        --sim_warmup_episodes 20 \
        --device cpu
"""

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import torch

# ── Path setup ────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_H2O_ROOT = os.path.dirname(_HERE)
_BUS_H2O = os.path.join(_H2O_ROOT, "bus_h2o")
sys.path.insert(0, _HERE)
sys.path.insert(0, _BUS_H2O)

from bus_replay_buffer import BusMixedReplayBuffer
from snapshot_store import SnapshotStore
from common.data_utils import (
    ZOnlyDiscriminator,
    compute_z_importance_weight,
    extract_structured_context,
    set_route_length,
    build_edge_linear_map,
)

# Matplotlib — non-interactive backend so it works on headless servers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ======================================================================
#  Helper: extract active buses from env.state
# ======================================================================

def _extract_active_buses(state_dict):
    """
    Parse the env state dict into (bus_id, obs_vec) pairs.

    state_dict: {int: [[15-float list]] or []}
    Returns:    list of (bus_id, np.ndarray(15,))
    """
    active = []
    for bus_id, obs_list in state_dict.items():
        if not obs_list:
            continue
        inner = obs_list[-1]
        if isinstance(inner, (list, np.ndarray)):
            vec = inner
            if isinstance(vec, list) and vec and isinstance(vec[0], list):
                vec = vec[-1]
            if vec:
                active.append((bus_id, np.array(vec, dtype=np.float32)))
    return active


# ======================================================================
#  Phase 1:  Collect SIM z data for discriminator training
# ======================================================================

def collect_sim_z_data(env, n_episodes, max_events_per_ep=100):
    """
    Run SIM with zero-hold policy and collect (z_t, z_t1) pairs.

    Returns
    -------
    z_pairs : list of (z_t, z_t1)  as numpy float32 arrays of shape (30,)
    """
    z_pairs = []

    for ep in range(n_episodes):
        env.reset()

        # Fast-forward until first decision event
        null_act = {k: None for k in range(env.max_agent_num)}
        done = False
        for _ in range(10_000):
            state, reward, done = env.step_fast(null_act)
            if done:
                break
            if any(v for v in state.values()):
                break
        if done:
            continue

        prev_z = extract_structured_context(
            env.capture_full_system_snapshot()
        )

        for ev_idx in range(max_events_per_ep):
            # Build zero-hold action for every active bus
            active = _extract_active_buses(state)
            action_dict = {k: None for k in range(env.max_agent_num)}
            for bus_id, _ in active:
                action_dict[bus_id] = [0.0, 1.0]  # hold=0, speed=1

            state, reward, done = env.step_to_event(action_dict)
            z_now = extract_structured_context(
                env.capture_full_system_snapshot()
            )
            z_pairs.append((prev_z.copy(), z_now.copy()))
            prev_z = z_now.copy()

            if done:
                break

        if (ep + 1) % 5 == 0:
            print(
                f"  [Phase 1] {len(z_pairs):,} z pairs "
                f"from {ep + 1}/{n_episodes} episodes"
            )

    return z_pairs


# ======================================================================
#  Phase 2:  Train discriminator
# ======================================================================

def train_discriminator(
    disc, real_z_t, real_z_t1, sim_z_t, sim_z_t1,
    n_epochs=200, batch_size=256, lr=3e-4, device="cpu",
):
    """
    Train ZOnlyDiscriminator on real (offline / SUMO) vs sim z pairs.

    Uses the same label-smoothing as H2OPlusBus._train_discriminator:
        real label = 0.9,  sim label = 0.1
    """
    disc.to(device)
    disc.train()

    optimizer = torch.optim.Adam(disc.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    n_real = real_z_t.shape[0]
    n_sim = sim_z_t.shape[0]
    print(
        f"  [Phase 2] Training discriminator: "
        f"{n_real:,} real pairs, {n_sim:,} sim pairs, "
        f"{n_epochs} epochs"
    )

    for epoch in range(n_epochs):
        real_idx = np.random.randint(0, n_real, size=batch_size)
        sim_idx = np.random.randint(0, n_sim, size=batch_size)

        r_zt = torch.FloatTensor(real_z_t[real_idx]).to(device)
        r_zt1 = torch.FloatTensor(real_z_t1[real_idx]).to(device)
        s_zt = torch.FloatTensor(sim_z_t[sim_idx]).to(device)
        s_zt1 = torch.FloatTensor(sim_z_t1[sim_idx]).to(device)

        real_logits = disc(r_zt, r_zt1)
        sim_logits = disc(s_zt, s_zt1)

        loss_real = criterion(real_logits, torch.full_like(real_logits, 0.9))
        loss_sim = criterion(sim_logits, torch.full_like(sim_logits, 0.1))
        loss = loss_real + loss_sim

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                real_acc = (
                    (torch.sigmoid(real_logits) > 0.5).float().mean().item()
                )
                sim_acc = (
                    (torch.sigmoid(sim_logits) < 0.5).float().mean().item()
                )
            print(
                f"    epoch {epoch + 1:>4d}/{n_epochs}: "
                f"loss={loss.item():.4f}  "
                f"real_acc={real_acc:.3f}  sim_acc={sim_acc:.3f}"
            )

    disc.eval()
    return disc


# ======================================================================
#  Phase 3:  Drift test
# ======================================================================

def run_drift_test(env, replay_buffer, disc, n_snapshots, max_steps, device):
    """
    For each sampled snapshot:
      1. Reset SIM to it
      2. Run forward with zero-hold policy
      3. At each bus-arrival step, compute discriminator output

    Returns
    -------
    all_w_by_step       : {step_idx: [w_values across snapshots]}
    all_logit_by_step   : {step_idx: [logit values]}
    all_prob_by_step    : {step_idx: [P(real) values]}
    all_sim_time_by_step: {step_idx: [sim_time offsets from reset]}
    """
    all_w_by_step = {}
    all_logit_by_step = {}
    all_prob_by_step = {}
    all_sim_time_by_step = {}

    for snap_i in range(n_snapshots):
        snapshot, obs, z_t_offline = replay_buffer.sample_snapshot()
        reset_sim_time = snapshot.get("sim_time", 0.0)

        env.reset(snapshot=snapshot)

        # Fast-forward until first decision event
        null_act = {k: None for k in range(env.max_agent_num)}
        done = False
        for _ in range(10_000):
            state, reward, done = env.step_fast(null_act)
            if done:
                break
            if any(v for v in state.values()):
                break
        if done:
            print(f"  [Snapshot {snap_i + 1}] ended during fast-forward, skip")
            continue

        prev_z = extract_structured_context(
            env.capture_full_system_snapshot()
        )

        steps_collected = 0
        for step in range(max_steps):
            active = _extract_active_buses(state)
            action_dict = {k: None for k in range(env.max_agent_num)}
            for bus_id, _ in active:
                action_dict[bus_id] = [0.0, 1.0]

            state, reward, done = env.step_to_event(action_dict)

            snap_now = env.capture_full_system_snapshot()
            z_now = extract_structured_context(snap_now)
            sim_time_now = snap_now.get("sim_time", 0.0)

            # Discriminator inference
            with torch.no_grad():
                z_t_tensor = torch.FloatTensor(prev_z).unsqueeze(0).to(device)
                z_t1_tensor = torch.FloatTensor(z_now).unsqueeze(0).to(device)
                logit = disc(z_t_tensor, z_t1_tensor).item()
                prob = torch.sigmoid(torch.tensor(logit)).item()
                w = prob / (1.0 - prob + 1e-8)

            all_w_by_step.setdefault(step, []).append(w)
            all_logit_by_step.setdefault(step, []).append(logit)
            all_prob_by_step.setdefault(step, []).append(prob)
            all_sim_time_by_step.setdefault(step, []).append(
                sim_time_now - reset_sim_time
            )

            prev_z = z_now.copy()
            steps_collected += 1

            if done:
                break

        print(
            f"  [Snapshot {snap_i + 1}/{n_snapshots}] "
            f"{steps_collected} steps, "
            f"reset_time={reset_sim_time:.0f}s"
        )

    return all_w_by_step, all_logit_by_step, all_prob_by_step, all_sim_time_by_step


# ======================================================================
#  Phase 3-extra:  Offline baseline (D on real z pairs)
# ======================================================================

def compute_offline_baseline(disc, replay_buffer, n_samples, device):
    """
    Compute discriminator output on random offline (SUMO) z pairs
    to establish the 'real' reference level.
    """
    n = replay_buffer.fixed_dataset_size
    indices = np.random.randint(0, n, size=n_samples)

    z_t = torch.FloatTensor(replay_buffer.z_t[indices]).to(device)
    z_t1 = torch.FloatTensor(replay_buffer.z_t1[indices]).to(device)

    with torch.no_grad():
        logits = disc(z_t, z_t1).squeeze()       # (N,)
        probs = torch.sigmoid(logits)              # (N,)
        ws = probs / (1.0 - probs + 1e-8)         # (N,)

    return {
        "mean_w": float(ws.mean().item()),
        "std_w": float(ws.std().item()),
        "mean_prob": float(probs.mean().item()),
        "std_prob": float(probs.std().item()),
        "mean_logit": float(logits.mean().item()),
    }


# ======================================================================
#  Phase 4:  Plotting
# ======================================================================

def plot_results(
    all_w, all_logit, all_prob, all_sim_time,
    baseline, output_path, min_samples=2,
):
    """
    Plot discriminator metrics vs steps-since-reset.
    Adds horizontal reference lines from the offline baseline.
    """
    steps = sorted(all_w.keys())
    steps = [s for s in steps if len(all_w[s]) >= min_samples]
    if not steps:
        print("[Phase 4] Not enough data to plot!")
        return

    mean_w = [np.mean(all_w[s]) for s in steps]
    std_w = [np.std(all_w[s]) for s in steps]
    mean_prob = [np.mean(all_prob[s]) for s in steps]
    std_prob = [np.std(all_prob[s]) for s in steps]
    mean_logit = [np.mean(all_logit[s]) for s in steps]
    std_logit = [np.std(all_logit[s]) for s in steps]
    mean_dt = [np.mean(all_sim_time[s]) for s in steps]
    n_samples = [len(all_w[s]) for s in steps]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # ── Panel 1: Importance weight w ──────────────────────────────────
    ax = axes[0]
    ax.plot(steps, mean_w, "b-o", linewidth=2, markersize=3, label="SIM: mean w")
    ax.fill_between(
        steps,
        [m - s for m, s in zip(mean_w, std_w)],
        [m + s for m, s in zip(mean_w, std_w)],
        alpha=0.15, color="blue",
    )
    ax.axhline(
        y=baseline["mean_w"], color="green", linestyle="--",
        linewidth=1.5, label=f"Offline baseline: w={baseline['mean_w']:.2f}",
    )
    ax.set_ylabel("Importance Weight  w = P(real)/(1-P(real))")
    ax.set_title("Discriminator Drift: SIM rollout from SUMO snapshot")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── Panel 2: P(real) ──────────────────────────────────────────────
    ax = axes[1]
    ax.plot(
        steps, mean_prob, "r-o", linewidth=2, markersize=3,
        label="SIM: mean P(real)",
    )
    ax.fill_between(
        steps,
        [m - s for m, s in zip(mean_prob, std_prob)],
        [m + s for m, s in zip(mean_prob, std_prob)],
        alpha=0.15, color="red",
    )
    ax.axhline(
        y=baseline["mean_prob"], color="green", linestyle="--",
        linewidth=1.5,
        label=f"Offline baseline: P(real)={baseline['mean_prob']:.3f}",
    )
    ax.axhline(
        y=0.5, color="gray", linestyle=":", alpha=0.5, label="chance (0.5)",
    )
    ax.set_ylabel("P(real) = sigmoid(D)")
    ax.set_title("Discriminator Probability")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── Panel 3: sample count + mean sim-time offset ──────────────────
    ax = axes[2]
    ax.bar(steps, n_samples, color="steelblue", alpha=0.4, label="# snapshots")
    ax.set_ylabel("# snapshots (bars)", color="steelblue")
    ax.set_xlabel("Decision-event steps since snapshot reset")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(
        steps, mean_dt, "k-s", markersize=3, linewidth=1,
        label="mean elapsed sim-time (s)",
    )
    ax2.set_ylabel("Mean elapsed sim-time (s)")

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n[Phase 4] Plot saved to {output_path}")


# ======================================================================
#  Main
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Test discriminator drift over SIM rollout from SUMO snapshots"
    )
    p.add_argument(
        "--dataset_file", type=str,
        default=os.path.join(_BUS_H2O, "datasets_v2", "merged_all_v2.h5"),
    )
    p.add_argument(
        "--sim_env_path", type=str,
        default=os.path.join(_BUS_H2O, "calibrated_env"),
    )
    p.add_argument(
        "--edge_xml", type=str,
        default=os.path.join(_BUS_H2O, "network_data", "a_sorted_busline_edge.xml"),
    )
    p.add_argument("--line_id", type=str, default="7X")
    p.add_argument(
        "--n_snapshots", type=int, default=10,
        help="Number of random SUMO snapshots to test in drift phase",
    )
    p.add_argument(
        "--max_steps", type=int, default=50,
        help="Max decision events per drift rollout",
    )
    p.add_argument(
        "--disc_train_epochs", type=int, default=200,
        help="Epochs to train discriminator",
    )
    p.add_argument(
        "--sim_warmup_episodes", type=int, default=20,
        help="Number of fresh-reset SIM episodes to collect for disc training",
    )
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output", type=str, default="disc_drift_results.png",
        help="Output plot path",
    )
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"\n{'='*60}")
    print(f"  Discriminator Drift Test  [{ts}]")
    print(f"{'='*60}")

    # ── Route length (must be set before any z extraction) ────────────
    if os.path.exists(args.edge_xml):
        edge_map = build_edge_linear_map(args.edge_xml, args.line_id)
        route_length = max(edge_map.values()) if edge_map else 13119.0
    else:
        route_length = 13119.0
    set_route_length(route_length)
    print(f"[Setup] Route length = {route_length:.1f} m")

    # ── Load offline data ─────────────────────────────────────────────
    print(f"[Setup] Loading offline data: {args.dataset_file}")
    replay_buffer = BusMixedReplayBuffer(
        state_dim=17,
        action_dim=2,
        context_dim=30,
        dataset_file=args.dataset_file,
        device=args.device,
        buffer_ratio=1.0,
        reward_scale=1.0,
        reward_bias=0.0,
        action_scale=1.0,
        action_bias=0.0,
    )
    print(
        f"[Setup] {replay_buffer.fixed_dataset_size:,} offline transitions, "
        f"{len(replay_buffer._valid_snap_indices):,} valid snapshots"
    )

    # ── Snapshot store (lazy HDF5 loading) ────────────────────────────
    if getattr(replay_buffer, "_has_lazy_snap_index", False):
        manifest_path = os.path.join(
            os.path.dirname(args.dataset_file), "file_manifest.json"
        )
        if os.path.exists(manifest_path):
            with open(manifest_path) as mf:
                file_manifest = json.load(mf)
            snap_store = SnapshotStore(
                archive_dir=os.path.dirname(args.dataset_file),
                file_manifest=file_manifest,
                cache_size=256,
                snapshot_key="snapshot_T1",  # SIM format (has current_time, trip_id etc.)
            )
            replay_buffer.set_snapshot_store(snap_store)
            print(
                f"[Setup] SnapshotStore: {len(file_manifest)} archive files"
            )
        else:
            print(f"[Setup] WARNING: file_manifest.json not found at {manifest_path}")

    # ── Create SIM env ────────────────────────────────────────────────
    print(f"[Setup] Creating MultiLineSimEnv from {args.sim_env_path}")
    from envs.bus_sim_env import MultiLineSimEnv
    sim_env = MultiLineSimEnv(path=args.sim_env_path, debug=False)

    # ==================================================================
    #  Phase 1:  Collect SIM z data
    # ==================================================================
    print(f"\n--- Phase 1: Collect SIM z data ({args.sim_warmup_episodes} episodes) ---")
    t0 = time.time()
    sim_z_pairs = collect_sim_z_data(
        sim_env, args.sim_warmup_episodes, max_events_per_ep=100
    )
    print(
        f"[Phase 1] Done: {len(sim_z_pairs):,} z pairs "
        f"in {time.time() - t0:.1f}s"
    )
    if len(sim_z_pairs) < 100:
        print("[Phase 1] ERROR: Not enough SIM z pairs. Increase --sim_warmup_episodes.")
        return

    sim_z_t_arr = np.array([p[0] for p in sim_z_pairs])
    sim_z_t1_arr = np.array([p[1] for p in sim_z_pairs])

    # Real z from offline buffer
    real_z_t_arr = replay_buffer.z_t[: replay_buffer.fixed_dataset_size]
    real_z_t1_arr = replay_buffer.z_t1[: replay_buffer.fixed_dataset_size]

    # ==================================================================
    #  Phase 2:  Train discriminator
    # ==================================================================
    print(f"\n--- Phase 2: Train discriminator ({args.disc_train_epochs} epochs) ---")
    t0 = time.time()
    disc = ZOnlyDiscriminator(context_dim=30)
    disc = train_discriminator(
        disc,
        real_z_t_arr,
        real_z_t1_arr,
        sim_z_t_arr,
        sim_z_t1_arr,
        n_epochs=args.disc_train_epochs,
        batch_size=args.batch_size,
        lr=3e-4,
        device=args.device,
    )
    print(f"[Phase 2] Done in {time.time() - t0:.1f}s")

    # Offline baseline
    print("[Phase 2] Computing offline baseline...")
    baseline = compute_offline_baseline(disc, replay_buffer, n_samples=2000, device=args.device)
    print(
        f"[Phase 2] Baseline: "
        f"w={baseline['mean_w']:.3f} +/- {baseline['std_w']:.3f}, "
        f"P(real)={baseline['mean_prob']:.3f} +/- {baseline['std_prob']:.3f}"
    )

    # ==================================================================
    #  Phase 3:  Drift test
    # ==================================================================
    print(
        f"\n--- Phase 3: Drift test "
        f"({args.n_snapshots} snapshots x {args.max_steps} steps) ---"
    )
    t0 = time.time()
    all_w, all_logit, all_prob, all_sim_time = run_drift_test(
        sim_env, replay_buffer, disc,
        args.n_snapshots, args.max_steps, args.device,
    )
    print(f"[Phase 3] Done in {time.time() - t0:.1f}s")

    # ==================================================================
    #  Phase 4:  Plot & summary
    # ==================================================================
    print(f"\n--- Phase 4: Plot ---")
    plot_results(
        all_w, all_logit, all_prob, all_sim_time,
        baseline, args.output, min_samples=2,
    )

    # ── Text summary ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("DISCRIMINATOR DRIFT SUMMARY")
    print(f"{'='*70}")
    print(
        f"  Offline baseline:  w = {baseline['mean_w']:.3f},  "
        f"P(real) = {baseline['mean_prob']:.3f}"
    )
    print(f"{'─'*70}")
    print(
        f"{'step':>6}  {'mean_w':>10}  {'mean_P(real)':>12}  "
        f"{'mean_dt(s)':>10}  {'n':>4}"
    )
    print(f"{'─'*70}")
    for step in sorted(all_w.keys()):
        n = len(all_w[step])
        if n >= 1:
            print(
                f"{step:>6d}  {np.mean(all_w[step]):>10.4f}  "
                f"{np.mean(all_prob[step]):>12.4f}  "
                f"{np.mean(all_sim_time[step]):>10.0f}  "
                f"{n:>4d}"
            )
    print(f"{'='*70}")

    # ── Save raw data as .npz for further analysis ────────────────────
    npz_path = os.path.splitext(args.output)[0] + ".npz"
    np.savez(
        npz_path,
        steps=np.array(sorted(all_w.keys())),
        w_by_step={str(k): np.array(v) for k, v in all_w.items()},
        prob_by_step={str(k): np.array(v) for k, v in all_prob.items()},
        logit_by_step={str(k): np.array(v) for k, v in all_logit.items()},
        sim_time_by_step={str(k): np.array(v) for k, v in all_sim_time.items()},
        baseline_mean_w=baseline["mean_w"],
        baseline_mean_prob=baseline["mean_prob"],
    )
    print(f"[Phase 4] Raw data saved to {npz_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
