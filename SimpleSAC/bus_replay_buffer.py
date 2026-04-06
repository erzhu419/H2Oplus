"""
bus_replay_buffer.py
====================
Phase 3: Mixed Replay Buffer for H2O+ Bus Sim-to-Real.

Supports two loading modes:
    A. Single merged HDF5 (recommended):  dataset_file="merged_all.h5"
       - No snapshot fields, ~42 MB for 3.37M transitions
       - Fields: observations, next_observations, actions, rewards,
                 terminals, timeouts, z_t, z_t1, policy_id, seed
    B. Directory of per-policy HDF5 files (legacy): dataset_dir + dataset_glob

Maintains two partitions:
    - Fixed offline partition [0, fixed_dataset_size): real SUMO data
    - Growing online partition [fixed_dataset_size, size): sim rollouts

Key API:
    buffer.sample(batch_size, scope="real"|"sim"|None)  → dict with z_t, z_t1
    buffer.sample_snapshot()  → (snapshot_bytes, obs, z_t)  [legacy only]
    buffer.append(obs, action, reward, next_obs, done, z_t, z_t1)
"""

import os
import glob
import logging
import pickle

import h5py
import numpy as np
import torch


class BusMixedReplayBuffer:
    """Mixed replay buffer for bus H2O+ with offline/online partitions."""

    def __init__(
        self,
        state_dim: int = 15,
        action_dim: int = 1,
        context_dim: int = 30,
        dataset_file: str = None,
        dataset_dir: str = None,
        dataset_glob: str = "sumo_*.h5",
        device: str = "cuda",
        buffer_ratio: float = 10.0,
        action_scale: float = 1.0,
        action_bias: float = 0.0,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
        skip_snapshots: bool = False,
    ):
        """
        Args:
            state_dim:     Observation dimension (15 for SUMO obs).
            action_dim:    Action dimension (2 for hold + speed).
            context_dim:   z feature dimension (30).
            dataset_file:  Path to single merged HDF5 file (preferred).
            dataset_dir:   Directory containing HDF5 files (legacy).
            dataset_glob:  Glob pattern to match HDF5 files.
            device:        Torch device string.
            buffer_ratio:  Ratio of max online buffer to offline size.
            action_scale:  (deprecated) Kept for API compat; default 1.0 (no-op).
            action_bias:   (deprecated) Kept for API compat; default 0.0 (no-op).
            reward_scale:  Multiply rewards by this factor.
            reward_bias:   Add this to rewards after scaling.
            skip_snapshots: If True, skip loading snapshot_T1 fields.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.context_dim = context_dim
        self.device = torch.device(device)
        self.buffer_ratio = buffer_ratio
        self.action_scale = action_scale
        self.action_bias = action_bias
        self.reward_scale = reward_scale
        self.reward_bias = reward_bias
        self.skip_snapshots = skip_snapshots

        # ------------------------------------------------------------------
        # 1. Load offline data from HDF5
        # ------------------------------------------------------------------
        all_snap_bytes = []   # list of bytes|None (only populated in legacy mode)

        if dataset_file is not None:
            # ── Mode A: single merged HDF5 (fast, no in-memory snapshots) ───
            logging.info(f"[BusMixedReplayBuffer] Loading merged file {dataset_file}")
            with h5py.File(dataset_file, "r") as f:
                n = f["observations"].shape[0]
                s = np.array(f["observations"], dtype=np.float32)
                a = np.array(f["actions"], dtype=np.float32).reshape(n, -1)
                r = np.array(f["rewards"], dtype=np.float32).reshape(n, 1)
                s_ = np.array(f["next_observations"], dtype=np.float32)
                done = np.array(f["terminals"], dtype=np.float32).reshape(n, 1)
                zt = np.array(f["z_t"], dtype=np.float32)
                zt1 = np.array(f["z_t1"], dtype=np.float32)

                # Lazy snapshot index (from merge_v2_lazy.py)
                if "snap_file_id" in f and "snap_row_id" in f:
                    self._snap_file_id = np.array(f["snap_file_id"], dtype=np.uint8)
                    self._snap_row_id = np.array(f["snap_row_id"], dtype=np.uint32)
                    self._has_lazy_snap_index = True
                    logging.info(
                        "[BusMixedReplayBuffer] Loaded lazy snapshot index "
                        f"(snap_file_id + snap_row_id, {n:,} entries)"
                    )
                else:
                    self._snap_file_id = None
                    self._snap_row_id = None
                    self._has_lazy_snap_index = False

            # Actions stored in raw tanh [-1, 1] space
            # (matches ensemble checkpoint + Residual Control output)
            r = r * self.reward_scale + self.reward_bias

            fixed_dataset_size = n
            all_snap_bytes = []  # no in-memory snapshots in merged mode
            logging.info(
                f"[BusMixedReplayBuffer] Loaded {fixed_dataset_size:,} offline transitions "
                f"from merged file"
            )

        elif dataset_dir is not None:
            # ── Mode B: directory of per-policy HDF5 files (legacy) ────
            all_obs, all_act, all_rew, all_next_obs, all_term = [], [], [], [], []
            all_zt, all_zt1 = [], []

            h5_files = sorted(glob.glob(os.path.join(dataset_dir, dataset_glob)))
            if not h5_files:
                raise FileNotFoundError(
                    f"No HDF5 files matching '{dataset_glob}' in {dataset_dir}"
                )
            for fpath in h5_files:
                logging.info(f"[BusMixedReplayBuffer] Loading {os.path.basename(fpath)}")
                with h5py.File(fpath, "r") as f:
                    n = f["observations"].shape[0]
                    all_obs.append(np.array(f["observations"], dtype=np.float32))
                    all_act.append(np.array(f["actions"], dtype=np.float32).reshape(n, -1))
                    all_rew.append(np.array(f["rewards"], dtype=np.float32).reshape(n, 1))
                    all_next_obs.append(np.array(f["next_observations"], dtype=np.float32))
                    all_term.append(np.array(f["terminals"], dtype=np.float32).reshape(n, 1))
                    all_zt.append(np.array(f["z_t"], dtype=np.float32))
                    all_zt1.append(np.array(f["z_t1"], dtype=np.float32))

                    # Snapshot loading (optional — skipped for merged workflow)
                    if not self.skip_snapshots and "raw_snapshot" in f:
                        snap_ds = f["raw_snapshot"]
                        for i in range(n):
                            all_snap_bytes.append(bytes(snap_ds[i]))
                    else:
                        all_snap_bytes.extend([None] * n)

            s = np.concatenate(all_obs, axis=0)
            a = np.concatenate(all_act, axis=0)
            r = np.concatenate(all_rew, axis=0)
            s_ = np.concatenate(all_next_obs, axis=0)
            done = np.concatenate(all_term, axis=0)
            zt = np.concatenate(all_zt, axis=0)
            zt1 = np.concatenate(all_zt1, axis=0)

            # Actions stored in raw tanh [-1, 1] space
            r = r * self.reward_scale + self.reward_bias

            fixed_dataset_size = s.shape[0]
            logging.info(
                f"[BusMixedReplayBuffer] Loaded {fixed_dataset_size:,} offline transitions "
                f"from {len(h5_files)} files"
            )
        else:
            # No offline data (pure online mode — unlikely but safe)
            fixed_dataset_size = 0
            s = np.zeros((0, state_dim), dtype=np.float32)
            a = np.zeros((0, action_dim), dtype=np.float32)
            r = np.zeros((0, 1), dtype=np.float32)
            s_ = np.zeros((0, state_dim), dtype=np.float32)
            done = np.zeros((0, 1), dtype=np.float32)
            zt = np.zeros((0, context_dim), dtype=np.float32)
            zt1 = np.zeros((0, context_dim), dtype=np.float32)

        self.fixed_dataset_size = fixed_dataset_size
        self.max_size = int((self.buffer_ratio + 1) * max(fixed_dataset_size, 1))

        # ------------------------------------------------------------------
        # 1b. Renormalize offline z-features: density/5.0 → fractional density
        #     Old format: z[10:20] = raw_count / 5.0  (scale-dependent)
        #     New format: z[10:20] = raw_count / total_count  (scale-invariant)
        #     This bridges the SIM-REAL bus-count gap (~9 vs ~67 active buses).
        # ------------------------------------------------------------------
        if fixed_dataset_size > 0:
            from common.data_utils import renormalize_z_density
            logging.info("[BusMixedReplayBuffer] Renormalizing offline z_t/z_t1 density channel "
                         "(absolute → fractional)")
            zt = renormalize_z_density(zt)
            zt1 = renormalize_z_density(zt1)

        online_capacity = self.max_size - fixed_dataset_size

        # ------------------------------------------------------------------
        # 2. Allocate arrays: [offline | online_padding]
        # ------------------------------------------------------------------
        self.state = np.vstack([s, np.zeros((online_capacity, state_dim), dtype=np.float32)])
        self.action = np.vstack([a, np.zeros((online_capacity, action_dim), dtype=np.float32)])
        self.reward = np.vstack([r, np.zeros((online_capacity, 1), dtype=np.float32)])
        self.next_state = np.vstack([s_, np.zeros((online_capacity, state_dim), dtype=np.float32)])
        self.done = np.vstack([done, np.zeros((online_capacity, 1), dtype=np.float32)])
        self.z_t = np.vstack([zt, np.zeros((online_capacity, context_dim), dtype=np.float32)])
        self.z_t1 = np.vstack([zt1, np.zeros((online_capacity, context_dim), dtype=np.float32)])

        # Snapshot bytes are only stored for offline data (used for buffer reset)
        self._snapshot_bytes = all_snap_bytes   # list of bytes|None, len=fixed_dataset_size

        # Lazy snapshot store (injected externally via set_snapshot_store())
        self.snapshot_store = None

        # Build index of valid snapshots:
        #   - If in-memory snapshots exist (legacy mode): use those
        #   - If lazy index exists (merged mode): ALL offline indices are valid
        #   - Otherwise: no snapshots available
        if self._snapshot_bytes:
            self._valid_snap_indices = [
                i for i, sb in enumerate(self._snapshot_bytes) if sb is not None
            ]
        elif getattr(self, '_has_lazy_snap_index', False):
            # All offline transitions have snapshot pointers
            self._valid_snap_indices = list(range(fixed_dataset_size))
        else:
            self._valid_snap_indices = []

        # ------------------------------------------------------------------
        # 2b. Retroactively apply tanh large-deviation penalty to offline rewards.
        #     The offline SUMO data was collected without this penalty, but the
        #     online sim env includes it. Apply it here to align reward scales.
        #     Obs layout (first 15 dims of augmented 17-dim obs):
        #       obs[5] = forward_headway (s)
        #       obs[6] = backward_headway (s)
        #       obs[8] = target_headway (static median, typically 360s)
        # ------------------------------------------------------------------
        if fixed_dataset_size > 0:
            self._apply_tanh_penalty(fixed_dataset_size)

        self.ptr = fixed_dataset_size
        self.size = fixed_dataset_size

        # ------------------------------------------------------------------
        # 3. State normalisation stats (from offline data only)
        # ------------------------------------------------------------------
        if fixed_dataset_size > 0:
            self.state_mean = self.state[:fixed_dataset_size].mean(0, keepdims=True)
            self.state_std = self.state[:fixed_dataset_size].std(0, keepdims=True) + 1e-3
        else:
            self.state_mean = np.zeros((1, state_dim), dtype=np.float32)
            self.state_std = np.ones((1, state_dim), dtype=np.float32)

    # ------------------------------------------------------------------
    # Offline reward correction
    # ------------------------------------------------------------------

    def _apply_tanh_penalty(self, n: int):
        """Retroactively apply the tanh large-deviation penalty to offline rewards.

        The offline SUMO data was collected without this penalty, but the online
        sim env includes it (see sim_core/bus.py _compute_reward_linear).

        Formula (per transition):
            fwd_dev = |fwd_hw - target|
            f_pen   = max(0, 20 * tanh((fwd_dev - 0.5 * target) / 30))
            bwd_dev = |bwd_hw - target|
            b_pen   = max(0, 20 * tanh((bwd_dev - 0.5 * target) / 30))
            reward  -= (f_pen + b_pen)

        Obs indices (valid for both 15-dim base and 17-dim augmented obs):
            obs[:, 5] = forward_headway (s)
            obs[:, 6] = backward_headway (s)
            obs[:, 8] = target_headway (static median, typically 360s)
        """
        obs = self.state[:n]        # (N, obs_dim)
        fwd_hw = obs[:, 5]          # forward headway
        bwd_hw = obs[:, 6]          # backward headway
        target = obs[:, 8]          # target headway

        # Forward penalty
        fwd_dev = np.abs(fwd_hw - target)
        f_pen = 20.0 * np.tanh((fwd_dev - 0.5 * target) / 30.0)
        f_pen = np.maximum(0.0, f_pen)

        # Backward penalty
        bwd_dev = np.abs(bwd_hw - target)
        b_pen = 20.0 * np.tanh((bwd_dev - 0.5 * target) / 30.0)
        b_pen = np.maximum(0.0, b_pen)

        penalty = f_pen + b_pen
        self.reward[:n, 0] -= penalty

        logging.info(
            f"[BusMixedReplayBuffer] Applied tanh penalty to {n:,} offline rewards: "
            f"mean_penalty={penalty.mean():.2f}, max={penalty.max():.2f}, "
            f"nonzero={int((penalty > 0).sum()):,}/{n:,}"
        )

    # ------------------------------------------------------------------
    # Append online (sim) data
    # ------------------------------------------------------------------

    def append(self, obs, action, reward, next_obs, done, z_t, z_t1):
        """Append a single sim transition to the online partition.

        Actions should be in raw tanh [-1, 1] space (matching policy output).
        """
        self.state[self.ptr] = obs
        self.action[self.ptr] = action  # raw tanh [-1, 1] space
        self.reward[self.ptr] = reward * self.reward_scale + self.reward_bias
        self.next_state[self.ptr] = next_obs
        self.done[self.ptr] = done
        self.z_t[self.ptr] = z_t
        self.z_t1[self.ptr] = z_t1

        # Circular within online partition
        self.ptr = (
            (self.ptr + 1 - self.fixed_dataset_size)
            % (self.max_size - self.fixed_dataset_size)
            + self.fixed_dataset_size
        )
        self.size = min(self.size + 1, self.max_size)

    def append_traj(self, observations, actions, rewards, next_observations,
                    dones, z_ts, z_t1s):
        """Append a batch of sim transitions."""
        for o, a, r, no, d, zt, zt1 in zip(
            observations, actions, rewards, next_observations,
            dones, z_ts, z_t1s
        ):
            self.append(o, a, r, no, d, zt, zt1)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, batch_size, scope=None, type=None):
        """
        Sample a batch from the buffer.

        Args:
            batch_size: Number of transitions to sample.
            scope:      "real" → offline only, "sim" → online only, None → all.
            type:       None → full dict, "sas" → (obs, act, next_obs) only.

        Returns:
            Dict with torch tensors: observations, actions, rewards,
            next_observations, dones, z_t, z_t1.
        """
        if scope is None:
            ind = np.random.randint(0, self.size, size=batch_size)
        elif scope == "real":
            if self.fixed_dataset_size == 0:
                raise RuntimeError("No offline data loaded")
            ind = np.random.randint(0, self.fixed_dataset_size, size=batch_size)
        elif scope == "sim":
            if self.size <= self.fixed_dataset_size:
                raise RuntimeError("No online data in buffer yet")
            ind = np.random.randint(self.fixed_dataset_size, self.size, size=batch_size)
        else:
            raise ValueError(f"Unknown scope: {scope}")

        batch = {
            "observations": torch.FloatTensor(self.state[ind]).to(self.device),
            "actions": torch.FloatTensor(self.action[ind]).to(self.device),
            "next_observations": torch.FloatTensor(self.next_state[ind]).to(self.device),
            "z_t": torch.FloatTensor(self.z_t[ind]).to(self.device),
            "z_t1": torch.FloatTensor(self.z_t1[ind]).to(self.device),
            "_indices": torch.LongTensor(ind).to(self.device),  # JTT: for PriorityIndex
        }

        if type == "sas":
            return batch

        batch["rewards"] = torch.FloatTensor(self.reward[ind]).to(self.device)
        batch["dones"] = torch.FloatTensor(self.done[ind]).to(self.device)
        return batch

    # ------------------------------------------------------------------
    # Buffer reset: sample a snapshot for god-mode reset
    # ------------------------------------------------------------------

    def sample_snapshot(self):
        """
        Sample a random offline transition that has a valid snapshot.

        Supports three modes:
            1. In-memory snapshot bytes (legacy per-file loading)
            2. Lazy SnapshotStore (merged mode with snap_file_id/snap_row_id)
            3. No snapshots → RuntimeError

        Returns:
            (snapshot_dict, obs, z_t)
        """
        if not self._valid_snap_indices:
            raise RuntimeError("No valid snapshots in offline data")

        idx = self._valid_snap_indices[
            np.random.randint(0, len(self._valid_snap_indices))
        ]
        return self._load_snapshot_at(idx)

    def sample_snapshot_by_idx(self, idx: int):
        """Load snapshot at a specific offline index (for JTT targeted reset).

        If the index has no valid snapshot, falls back to the nearest
        offline index that does have one.
        """
        idx = max(0, min(idx, self.fixed_dataset_size - 1))

        # Check if this specific index has a snapshot
        has_snap = False
        if self.snapshot_store is not None and self._has_lazy_snap_index:
            has_snap = True  # lazy mode: all indices are valid
        elif idx < len(self._snapshot_bytes) and self._snapshot_bytes[idx] is not None:
            has_snap = True  # in-memory mode

        if not has_snap and self._valid_snap_indices:
            # Fallback: find nearest valid snapshot index
            distances = np.abs(np.array(self._valid_snap_indices) - idx)
            nearest = self._valid_snap_indices[int(np.argmin(distances))]
            idx = nearest
        elif not has_snap:
            raise RuntimeError("No valid snapshots in offline data")

        return self._load_snapshot_at(idx)

    def _load_snapshot_at(self, idx: int):
        """Internal: load snapshot at idx via best available method.

        Returns:
            (snapshot_dict, obs, z_t)
        """
        # Method 1: Lazy SnapshotStore (preferred for merged mode)
        if self.snapshot_store is not None and self._has_lazy_snap_index:
            snapshot_dict = self.snapshot_store.get_by_buffer_idx(
                self._snap_file_id, self._snap_row_id, idx
            )
        # Method 2: In-memory snapshot bytes (legacy)
        elif idx < len(self._snapshot_bytes) and self._snapshot_bytes[idx] is not None:
            snapshot_dict = pickle.loads(self._snapshot_bytes[idx])
        else:
            raise RuntimeError(f"No snapshot available at index {idx}")

        obs = self.state[idx].copy()
        z_t = self.z_t[idx].copy()
        return snapshot_dict, obs, z_t

    def set_snapshot_store(self, store):
        """Inject a SnapshotStore for lazy snapshot loading.

        Called from h2o+_bus_main.py after creating the SnapshotStore.
        """
        self.snapshot_store = store
        if self._has_lazy_snap_index and not self._valid_snap_indices:
            # Rebuild valid indices now that store is available
            self._valid_snap_indices = list(range(self.fixed_dataset_size))
        logging.info(
            f"[BusMixedReplayBuffer] SnapshotStore attached: "
            f"{len(self._valid_snap_indices):,} valid snapshot indices"
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_mean_std(self):
        """Return (mean, std) tensors for state normalisation."""
        return (
            torch.FloatTensor(self.state_mean).to(self.device),
            torch.FloatTensor(self.state_std).to(self.device),
        )

    def get_reward_stats(self):
        """Return (mean, std) of offline rewards for global normalization."""
        offline_r = self.reward[:self.fixed_dataset_size].flatten()
        return float(np.mean(offline_r)), float(np.std(offline_r) + 1e-6)

    @property
    def online_size(self):
        """Number of transitions in the online (sim) partition."""
        return max(0, self.size - self.fixed_dataset_size)

    def has_online_data(self):
        """True if at least one sim transition has been appended."""
        return self.size > self.fixed_dataset_size
