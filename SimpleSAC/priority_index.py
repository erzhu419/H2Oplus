"""
priority_index.py
=================
Phase 5: JTT-Inspired Priority Index for targeted snapshot reset.

Maintains per-transition priority scores for offline data, using three
fragility signals (TD error, Q-disagreement, discriminator drift) that
are already computed during H2O+ training.

Key idea (from JTT paper):
    In supervised learning, JTT upweights misclassified samples.
    In RL with snapshot reset, we *generate new counterfactual trajectories*
    by resetting the simulator to exactly those fragile states.
    This is strictly more powerful than JTT's upweighting.

Reference:
    Liu et al., "Just Train Twice: Improving Group Robustness without
    Training Group Information", ICML 2021.
"""

import numpy as np


class PriorityIndex:
    """Maintains per-transition priority scores for offline (real) data.

    Design principles:
    - Only tracks offline (real) data — only offline transitions have
      corresponding snapshots for reset.
    - Uses EMA smoothing to dampen noise from single-batch estimates.
    - Supports temperature-controlled probability sampling, progressing
      from uniform (T→∞) to greedy (T→0) over training.

    Usage:
        priority = PriorityIndex(n_offline=100000)
        # During training, after computing Q-losses:
        priority.update(indices, td_err, q_dis, d_drift)
        # During rollout, for snapshot reset:
        idx = priority.sample_reset_idx(temperature=1.0)
    """

    def __init__(
        self,
        n_offline: int,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        ema_decay: float = 0.1,
    ):
        """
        Args:
            n_offline:  Number of offline transitions to track.
            alpha:      Weight for TD error in priority score.
            beta:       Weight for Q-disagreement.
            gamma:      Weight for discriminator drift.
            ema_decay:  EMA smoothing coefficient for priority updates.
        """
        self.n = n_offline
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ema_decay = ema_decay

        # Per-transition EMA-smoothed priority signals
        self.td_error = np.zeros(n_offline, dtype=np.float32)
        self.q_disagree = np.zeros(n_offline, dtype=np.float32)
        self.disc_drift = np.zeros(n_offline, dtype=np.float32)
        self.update_count = np.zeros(n_offline, dtype=np.int32)

    def update(self, indices, td_err, q_dis, d_drift):
        """Update priority scores for sampled offline transitions.

        Called after every training batch — scores are EMA-smoothed
        to avoid noise from single-batch estimates.

        Args:
            indices:   np.ndarray of offline buffer indices (int).
            td_err:    np.ndarray of absolute TD errors.
            q_dis:     np.ndarray of |Q1 - Q2| disagreement.
            d_drift:   np.ndarray of (1 - w/w_max) discriminator drift.
        """
        # Ensure numpy arrays
        if hasattr(indices, 'cpu'):
            indices = indices.cpu().numpy()
        indices = np.asarray(indices, dtype=np.int64)

        if hasattr(td_err, 'cpu'):
            td_err = td_err.cpu().numpy()
        if hasattr(q_dis, 'cpu'):
            q_dis = q_dis.cpu().numpy()
        if hasattr(d_drift, 'cpu'):
            d_drift = d_drift.cpu().numpy()

        td_err = np.asarray(td_err, dtype=np.float32).ravel()
        q_dis = np.asarray(q_dis, dtype=np.float32).ravel()
        d_drift = np.asarray(d_drift, dtype=np.float32).ravel()

        # Only update indices within valid range
        mask = (indices >= 0) & (indices < self.n)
        if not mask.all():
            indices = indices[mask]
            td_err = td_err[mask]
            q_dis = q_dis[mask]
            d_drift = d_drift[mask]

        if len(indices) == 0:
            return

        ema = self.ema_decay
        for arr, new_val in [
            (self.td_error, td_err),
            (self.q_disagree, q_dis),
            (self.disc_drift, d_drift),
        ]:
            old = arr[indices]
            arr[indices] = (1 - ema) * old + ema * new_val

        self.update_count[indices] += 1

    def sample_reset_idx(self, temperature: float = 1.0, valid_indices=None) -> int:
        """Sample one offline index with probability ∝ priority^(1/T).

        Args:
            temperature: Controls sampling sharpness.
                1.0  → standard priority sampling.
                ∞    → uniform random (recovers standard H2O+).
                →0   → always pick highest priority (greedy).
            valid_indices: Optional list/array of indices that have valid
                snapshots. If provided, sampling is restricted to these.

        Returns:
            int: Offline buffer index for snapshot reset.
        """
        # Compute combined priority scores
        p = (
            self.alpha * self.td_error
            + self.beta * self.q_disagree
            + self.gamma * self.disc_drift
        )

        if valid_indices is not None:
            valid_indices = np.asarray(valid_indices)
            # Restrict to valid subset
            p_sub = p[valid_indices].copy()
            counts_sub = self.update_count[valid_indices]

            # Zero out un-visited transitions
            p_sub[counts_sub == 0] = 0.0
            p_sub = np.clip(p_sub, 0, None)

            if p_sub.sum() < 1e-12:
                # No useful priority info — fall back to uniform
                return valid_indices[np.random.randint(len(valid_indices))]

            # Temperature-scaled sampling
            if temperature > 50.0:
                # Effectively uniform
                return valid_indices[np.random.randint(len(valid_indices))]

            p_sub = p_sub ** (1.0 / max(temperature, 1e-4))
            p_sub = p_sub / (p_sub.sum() + 1e-8)

            local_idx = np.random.choice(len(valid_indices), p=p_sub)
            return int(valid_indices[local_idx])
        else:
            # Use all offline transitions
            valid = self.update_count > 0
            if not valid.any():
                return np.random.randint(self.n)

            p = np.clip(p, 0, None)
            p[~valid] = 0.0

            if p.sum() < 1e-12:
                return np.random.randint(self.n)

            if temperature > 50.0:
                return np.random.randint(self.n)

            p = p ** (1.0 / max(temperature, 1e-4))
            p = p / (p.sum() + 1e-8)

            return int(np.random.choice(self.n, p=p))

    @property
    def priority_scores(self):
        """Return the current weighted priority score vector (for logging)."""
        return (
            self.alpha * self.td_error
            + self.beta * self.q_disagree
            + self.gamma * self.disc_drift
        )

    def get_stats(self):
        """Return a dict of summary statistics for logging."""
        scores = self.priority_scores
        visited = self.update_count > 0
        n_visited = int(visited.sum())

        if n_visited == 0:
            return {
                "n_visited": 0,
                "coverage": 0.0,
                "mean_priority": 0.0,
                "std_priority": 0.0,
                "max_priority": 0.0,
                "error_set_size": 0,
            }

        valid_scores = scores[visited]
        threshold = np.percentile(valid_scores, 80) if len(valid_scores) > 0 else 0.0

        return {
            "n_visited": n_visited,
            "coverage": n_visited / self.n,
            "mean_priority": float(valid_scores.mean()),
            "std_priority": float(valid_scores.std()),
            "max_priority": float(valid_scores.max()),
            "error_set_size": int((valid_scores > threshold).sum()),
        }
