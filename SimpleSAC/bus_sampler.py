"""
bus_sampler.py
==============
Phase 3: Event-driven bus environment samplers for H2O+ training.

BusStepSampler:
    - Wraps BusSimEnv for online rollout during training.
    - Supports buffer reset (snapshot injection) with probability P_RESET.
    - Implements discriminator-guided early truncation.
    - Handles async multi-agent transitions via pending cache.

BusEvalSampler:
    - Runs BusSimEnv without buffer reset for evaluation.
    - Collects total episode reward for performance tracking.

Key design: each loop iteration == one "decision event" (a bus arrives at a
station and needs a hold-time decision).  Between events, the env is fast-
forwarded with `step_to_event()` so we don't burn iterations on idle ticks.
"""

import os
import sys
import numpy as np
import torch

# Ensure bus_h2o is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_BUS_H2O = os.path.join(os.path.dirname(_HERE), "bus_h2o")
if _BUS_H2O not in sys.path:
    sys.path.insert(0, _BUS_H2O)

from common.data_utils import (extract_structured_context, TransitionDiscriminator,
                                DynamicsDiscriminator, ContrastiveDynamicsDiscriminator)  # noqa: E402


# ── Original action mapping (matches ensemble checkpoint training) ─────────
def _map_raw_to_env(action_raw):
    """Convert raw tanh [-1,1]×2 → env [hold_time, speed_ratio].

    Matches sac_ensemble_SUMO_linear_penalty.py PolicyNetwork.get_action():
        hold  = 30 * tanh(mean) + 30   → [0, 60]s
        speed = 0.2 * tanh(mean) + 1.0 → [0.8, 1.2]

    NOTE: The previous implementation used --use_residual_control mapping
    (hold = (a+1)*60, range [0,120]), which was INCORRECT and caused 2x
    hold time inflation, degrading reward from -660K to -1.18M.

    Args:
        action_raw: np.ndarray shape (2,), values in [-1, 1]

    Returns:
        (hold_time, speed_ratio) ready for sim env.
    """
    a_hold = float(action_raw[0])
    a_speed = float(action_raw[1])

    # Original mapping: hold = 30*tanh + 30 → [0, 60]
    hold = float(np.clip(30.0 * a_hold + 30.0, 0.0, 60.0))

    # Bang-Bang deterministic speed mapping (5 tiers)
    if a_speed > 0.6:
        speed = 1.2
    elif a_speed > 0.2:
        speed = 1.1
    elif a_speed > -0.2:
        speed = 1.0
    elif a_speed > -0.6:
        speed = 0.9
    else:
        speed = 0.8

    return hold, speed


def _extract_active_buses(state_dict):
    """
    Parse the env state dict into (bus_id, obs_vec) pairs.

    The env returns state[bus_id] as:
        [[15-float list]]   — list containing one inner 15-element list
        or [] / None        — bus has no pending decision

    Returns:
        list of (bus_id: int, obs_vec: np.ndarray shape (15,))
    """
    active = []
    for bus_id, obs_list in state_dict.items():
        if not obs_list:
            continue
        # obs_list is e.g. [[11.0, 0.0, 6.0, ...]]
        inner = obs_list[-1]
        if isinstance(inner, (list, np.ndarray)):
            vec = inner
            # Handle nested lists: [[...]] → [...]
            if isinstance(vec, list) and vec and isinstance(vec[0], list):
                vec = vec[-1]
            if vec:
                active.append((bus_id, np.array(vec, dtype=np.float32)))
    return active


class BusStepSampler:
    """
    Online rollout sampler for H2O+ bus training.

    At each call to `sample()`, runs the sim env for `n_steps` decision events
    and stores transitions into the replay buffer.

    Features:
        - Buffer reset: with prob p_reset, reset from an offline snapshot.
        - Early truncation: if discriminator w < threshold, truncate rollout.
        - Multi-agent pending cache: tracks (obs, action, z_t) pending settlement.
    """

    def __init__(
        self,
        env,
        replay_buffer,
        max_traj_events: int = 100,
        p_reset: float = 0.5,
        h_rollout: int = 30,
        w_threshold: float = 0.3,
        warmup_episodes: int = 20,
        action_dim: int = 2,
    ):
        """
        Args:
            env:              BusSimEnv instance (single line).
            replay_buffer:    BusMixedReplayBuffer instance.
            max_traj_events:  Max decision events per episode (not sim ticks).
            p_reset:          Probability of buffer reset vs standard reset.
            h_rollout:        Max decision events after a buffer reset episode.
            w_threshold:      Discriminator threshold for early truncation.
            warmup_episodes:  Skip early truncation for this many episodes.
            action_dim:       Action dimension (2 for hold + speed).
        """
        self.env = env
        self.replay_buffer = replay_buffer
        self.max_traj_events = max_traj_events
        self.p_reset = p_reset
        self.h_rollout = h_rollout
        self.w_threshold = w_threshold
        self.warmup_episodes = warmup_episodes
        self.action_dim = action_dim

        self._episode_count = 0
        self._pending = {}   # {bus_id: {obs_aug, action_raw, z_t, station_idx}}
        self._last_action = {}  # {bus_id: np.ndarray(action_dim)}

        # JTT: targeted reset support (injected by main)
        self.priority_index = None
        self.current_temperature = float('inf')  # inf = uniform (Phase 1)

    def sample(self, policy, n_steps, deterministic=False, discriminator=None):
        """
        Run the sim env for approximately `n_steps` decision events.

        Args:
            policy:         SamplerPolicy — callable(obs_batch) → action_batch.
            n_steps:        Target number of decision events to collect.
            deterministic:  Whether policy acts deterministically.
            discriminator:  ZOnlyDiscriminator (or None to skip early truncation).

        Returns:
            dict with statistics: n_transitions, n_truncated, mean_w, etc.
        """
        total_events = 0
        total_transitions = 0
        total_truncated = 0
        w_values = []
        # JTT reset tracking
        jtt_resets = 0         # resets that used priority-weighted sampling
        uniform_resets = 0     # resets that used uniform random
        fresh_resets = 0       # resets from scratch (no snapshot)
        jtt_reset_indices = [] # which buffer indices were targeted

        while total_events < n_steps:
            # ── Episode start ──────────────────────────────────────────
            # Check snapshot availability: SUMO pool or offline buffer
            has_sumo_snaps = (hasattr(self.env, 'get_random_snapshot')
                              and self.env.get_random_snapshot() is not None)
            has_buffer_snaps = bool(self.replay_buffer._valid_snap_indices)

            use_buffer_reset = (
                np.random.random() < self.p_reset
                and (has_sumo_snaps or has_buffer_snaps)
            )

            if use_buffer_reset:
                if has_sumo_snaps:
                    # ── SUMO snapshot pool reset ───────────────────────
                    if (self.priority_index is not None
                            and self._episode_count > self.warmup_episodes
                            and hasattr(self.env, 'get_prioritized_snapshot')):
                        # JTT: use priority-weighted snapshot selection
                        snap_path = self.env.get_prioritized_snapshot(
                            temperature=getattr(self, 'current_temperature', 1.0))
                        jtt_resets += 1
                    else:
                        snap_path = self.env.get_random_snapshot()
                        uniform_resets += 1
                    self._last_reset_snapshot = snap_path  # track for priority update
                    self.env.reset(snapshot=snap_path)
                elif has_buffer_snaps:
                    # ── Offline buffer snapshot reset ──────────────────
                    if (
                        self.priority_index is not None
                        and self._episode_count > self.warmup_episodes
                    ):
                        idx = self.priority_index.sample_reset_idx(
                            temperature=self.current_temperature,
                            valid_indices=self.replay_buffer._valid_snap_indices,
                        )
                        snapshot, _, _ = self.replay_buffer.sample_snapshot_by_idx(idx)
                        jtt_resets += 1
                        jtt_reset_indices.append(int(idx))
                    else:
                        snapshot, _, _ = self.replay_buffer.sample_snapshot()
                        uniform_resets += 1
                    self.env.reset(snapshot=snapshot)
                max_events = self.h_rollout
            else:
                self.env.reset()
                max_events = self.max_traj_events
                fresh_resets += 1

            # Initialize state: step until at least one bus has obs
            self._init_env_state()

            self._pending.clear()
            self._last_action.clear()
            self._episode_count += 1
            event_in_episode = 0
            last_z_t = None   # for early truncation z-pair tracking
            last_settled_obs = np.zeros(17, dtype=np.float32)    # obs_aug dim (15+2)
            last_settled_action = np.zeros(self.action_dim, dtype=np.float32)

            # ── Decision event loop ────────────────────────────────────
            for ev_idx in range(max_events):
                # Get current active buses (those with pending decisions)
                active_buses = _extract_active_buses(self.env.state)

                if not active_buses:
                    break

                # ── Capture current snapshot for z features ───────────
                snapshot_now = self.env.capture_full_system_snapshot()
                z_now = extract_structured_context(snapshot_now)

                action_dict = {}

                for bus_id, obs_vec in active_buses:
                    total_events += 1
                    station_idx = int(obs_vec[2]) if len(obs_vec) > 2 else -1
                    reward_val = self.env.reward.get(bus_id, 0.0)

                    # ── Augment obs with last_action ───────────────────
                    prev_a = self._last_action.get(
                        bus_id, np.zeros(self.action_dim, dtype=np.float32)
                    )
                    obs_aug = np.concatenate([obs_vec, prev_a])  # 15 + 2 = 17

                    # ── Settle pending transition ─────────────────────
                    if bus_id in self._pending:
                        prev = self._pending.pop(bus_id)
                        if station_idx != prev["station_idx"]:
                            # Buffer stores raw [0,60] actions
                            self.replay_buffer.append(
                                obs=prev["obs_aug"],
                                action=prev["action_raw"],
                                reward=reward_val,
                                next_obs=obs_aug,
                                done=0.0,
                                z_t=prev["z_t"],
                                z_t1=z_now.copy(),
                            )
                            total_transitions += 1

                    # ── Select action for this bus ─────────────────────
                    obs_tensor = np.expand_dims(obs_aug, 0)
                    # Policy outputs raw tanh [-1, 1] (BusSamplerPolicy)
                    action_raw = policy(obs_tensor, deterministic=deterministic)[0]
                    # Map to env action via Residual Control + Bang-Bang
                    hold, speed = _map_raw_to_env(action_raw)
                    action_dict[bus_id] = [hold, speed]

                    # Track last_action as raw tanh (for obs augmentation)
                    self._last_action[bus_id] = action_raw.copy()

                    # ── Cache as pending ───────────────────────────────
                    self._pending[bus_id] = {
                        "obs_aug": obs_aug.copy(),
                        "action_raw": action_raw.copy(),
                        "z_t": z_now.copy(),
                        "station_idx": station_idx,
                    }

                # ── Track last settled transition for TransitionDiscriminator ─
                # (obs_aug and action_raw of the most recent settled bus)
                if bus_id in self._last_action:
                    last_settled_obs = obs_aug.copy()
                    last_settled_action = action_raw.copy()

                # ── Early truncation check ────────────────────────────
                truncate = False
                if (
                    discriminator is not None
                    and event_in_episode > 2
                    and self._episode_count > self.warmup_episodes
                    and self.replay_buffer.has_online_data()
                    and last_z_t is not None
                ):
                    with torch.no_grad():
                        dev = self.replay_buffer.device
                        z_t_tensor = torch.FloatTensor(last_z_t).unsqueeze(0).to(dev)
                        z_t1_tensor = torch.FloatTensor(z_now).unsqueeze(0).to(dev)
                        obs_t = torch.FloatTensor(last_settled_obs).unsqueeze(0).to(dev)
                        act_t = torch.FloatTensor(last_settled_action).unsqueeze(0).to(dev)
                        nobs_t = torch.FloatTensor(obs_aug).unsqueeze(0).to(dev)

                        if isinstance(discriminator, (DynamicsDiscriminator, ContrastiveDynamicsDiscriminator)):
                            w = discriminator.compute_weight(obs_t, act_t, nobs_t).item()
                        elif isinstance(discriminator, TransitionDiscriminator):
                            logit = discriminator(obs_t, act_t, nobs_t, z_t_tensor, z_t1_tensor)
                            prob_real = torch.sigmoid(logit).item()
                            w = prob_real / (1.0 - prob_real + 1e-8)
                        else:
                            logit = discriminator(z_t_tensor, z_t1_tensor)
                            prob_real = torch.sigmoid(logit).item()
                            w = prob_real / (1.0 - prob_real + 1e-8)
                        w_values.append(w)

                        if w < self.w_threshold:
                            truncate = True
                            total_truncated += 1

                last_z_t = z_now.copy()

                if truncate:
                    for bus_id, prev in self._pending.items():
                        self.replay_buffer.append(
                            obs=prev["obs_aug"],
                            action=prev["action_raw"],
                            reward=0.0,
                            next_obs=prev["obs_aug"],
                            done=1.0,
                            z_t=prev["z_t"],
                            z_t1=z_now.copy(),
                        )
                        total_transitions += 1
                    self._pending.clear()
                    break

                # ── Fast-forward to next decision event ───────────────
                state, reward, done = self.env.step_to_event(action_dict)
                event_in_episode += 1

                if done:
                    for bus_id, prev in self._pending.items():
                        snap_end = self.env.capture_full_system_snapshot()
                        z_end = extract_structured_context(snap_end)
                        self.replay_buffer.append(
                            obs=prev["obs_aug"],
                            action=prev["action_raw"],
                            reward=0.0,
                            next_obs=prev["obs_aug"],
                            done=1.0,
                            z_t=prev["z_t"],
                            z_t1=z_end.copy(),
                        )
                        total_transitions += 1
                    self._pending.clear()
                    break

            # ── Update SUMO snapshot priority after episode ────────
            # Higher reward magnitude = harder state = higher priority for JTT
            if (hasattr(self, '_last_reset_snapshot')
                    and self._last_reset_snapshot is not None
                    and hasattr(self.env, 'update_snapshot_priority')
                    and total_transitions > 0):
                # Use mean absolute reward as priority (harder episodes = more negative = higher priority)
                episode_priority = abs(total_events) * 0.01  # simple heuristic
                self.env.update_snapshot_priority(self._last_reset_snapshot, episode_priority)
                self._last_reset_snapshot = None

        # Snapshot store cache stats (if available)
        snap_cache_stats = {}
        if hasattr(self.replay_buffer, 'snapshot_store') and self.replay_buffer.snapshot_store is not None:
            snap_cache_stats = self.replay_buffer.snapshot_store.cache_stats

        return {
            "n_events": total_events,
            "n_transitions": total_transitions,
            "n_truncated": total_truncated,
            "mean_w": float(np.mean(w_values)) if w_values else 0.0,
            "n_episodes": self._episode_count,
            # ── JTT reset diagnostics ──
            "jtt_resets": jtt_resets,
            "uniform_resets": uniform_resets,
            "fresh_resets": fresh_resets,
            "jtt_reset_ratio": jtt_resets / max(jtt_resets + uniform_resets + fresh_resets, 1),
            "jtt_temperature": self.current_temperature,
            "jtt_priority_index_attached": self.priority_index is not None,
            "jtt_past_warmup": self._episode_count > self.warmup_episodes,
            "jtt_valid_snap_count": len(self.replay_buffer._valid_snap_indices),
            # Snapshot cache stats
            "snap_cache_hits": snap_cache_stats.get("hits", 0),
            "snap_cache_misses": snap_cache_stats.get("misses", 0),
            "snap_cache_hit_rate": snap_cache_stats.get("hit_rate", 0.0),
        }

    def _init_env_state(self):
        """Step env until at least one bus produces observations."""
        action_dict = {}
        for _ in range(10000):  # safety limit
            state, reward, done = self.env.step_fast(action_dict)
            if done:
                break
            if any(v for v in state.values()):
                break


class BusEvalSampler:
    """
    Evaluation sampler for bus H2O+.

    Runs the sim env from standard reset (no buffer reset) and
    collects total episode reward for performance tracking.
    """

    def __init__(
        self,
        env,
        max_traj_events: int = 200,
        action_dim: int = 2,
    ):
        self.env = env
        self.max_traj_events = max_traj_events
        self.action_dim = action_dim

    def sample(self, policy, n_trajs, deterministic=True):
        """
        Run n_trajs episodes, collect trajectory data.

        Returns:
            list of dicts, each with 'rewards' (list of floats).
        """
        trajs = []
        for _ in range(n_trajs):
            self.env.reset()
            self._init_env_state()

            rewards_list = []
            pending = {}
            _last_action = {}  # per-bus last_action tracking
            done = self.env.done

            for ev_idx in range(self.max_traj_events):
                if done:
                    break

                active_buses = _extract_active_buses(self.env.state)

                if not active_buses:
                    # No decision events yet — step forward
                    action_dict = {}
                    state, reward, done = self.env.step_to_event(action_dict)
                    continue

                action_dict = {}
                for bus_id, obs_vec in active_buses:
                    station_idx = int(obs_vec[2]) if len(obs_vec) > 2 else -1
                    reward_val = self.env.reward.get(bus_id, 0.0)

                    # Settle pending
                    if bus_id in pending:
                        prev = pending.pop(bus_id)
                        if station_idx != prev["station_idx"]:
                            rewards_list.append(reward_val)

                    # Augment obs with last_action
                    prev_a = _last_action.get(
                        bus_id, np.zeros(self.action_dim, dtype=np.float32)
                    )
                    obs_aug = np.concatenate([obs_vec, prev_a])  # 15 + 2 = 17

                    # Select action (policy outputs raw tanh [-1, 1])
                    obs_tensor = np.expand_dims(obs_aug, 0)
                    action_raw = policy(obs_tensor, deterministic=deterministic)[0]
                    # Map to env action via Residual Control + Bang-Bang
                    hold, speed = _map_raw_to_env(action_raw)
                    action_dict[bus_id] = [hold, speed]

                    _last_action[bus_id] = action_raw.copy()

                    pending[bus_id] = {
                        "obs": obs_aug.copy(),
                        "station_idx": station_idx,
                    }

                state, reward, done = self.env.step_to_event(action_dict)

            trajs.append({"rewards": rewards_list})

        return trajs

    def _init_env_state(self):
        action_dict = {}
        for _ in range(10000):
            state, reward, done = self.env.step_fast(action_dict)
            if done:
                break
            if any(v for v in state.values()):
                break
