"""
daganzo_policy.py
=================
Analytical cooperative-holding baseline (Daganzo 2009 / Xuan 2011).

Per-bus rule (computed at every stop arrival):

    h_i = max(0, alpha * (H_target - max(h_prev, h_next)))

Where:
    H_target  = scheduled headway for the line (from timetable)
    h_prev    = forward (gap to preceding bus that already departed) headway
    h_next    = backward (gap to following bus) headway
    alpha     = cooperation strength in [0, 1]; Daganzo recommends ~0.6

Rationale:
    If the bus is "running late" (large forward gap), holding makes things
    worse, so h_i = 0.  If it's "running early" (small forward gap or
    closing in on the leader), holding restores the schedule.  Using
    max(h_prev, h_next) is the symmetric two-sided variant which helps
    even when the trailing bus is the slower one.

Action interface (matches `eval_with_metrics.py` / `eval_offline_on_sumo.py`
and the trained RL policies):

    obs = event_to_obs(...) is a 15-dim vector with
        obs[5] = forward_headway   (h_prev, observed)
        obs[6] = backward_headway  (h_next, observed)
        obs[8] = target_hw         (H_target)

    raw_action = [hold_norm, speed_norm] in [-1, 1]
    The eval harness then maps:
        hold_seconds = clip(30*hold_norm + 30, 0, 60)
        speed_mult   = piecewise on speed_norm

So given Daganzo h_i seconds, we invert: hold_norm = (h_i - 30) / 30.
Speed is held neutral (speed_norm = 0.0  =>  1.0x).

Reference: Daganzo CF (2009) "A headway-based approach to eliminate bus
bunching: Systematic analysis and comparisons", Transp. Res. B 43(10).
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np


# ── Default candidate locations for the timetable ─────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_TIMETABLE_DIR = os.path.join(_HERE, "calibrated_env", "data")
_LEGACY_TIMETABLE = os.path.join(_HERE, "data", "time_table.xlsx")


def _scheduled_headway_from_excel(xlsx_path: str) -> Optional[float]:
    """Median launch-time gap from a per-line time_table.xlsx.

    Returns None if the file is missing/unreadable so the caller can fall
    back to the env-provided value.
    """
    try:
        import pandas as pd  # local import keeps module light at import time
    except ImportError:
        return None
    if not os.path.isfile(xlsx_path):
        return None
    try:
        df = pd.read_excel(xlsx_path)
        col = "launch_time" if "launch_time" in df.columns else df.columns[0]
        times = np.sort(df[col].astype(float).to_numpy())
        if times.size < 2:
            return None
        diffs = np.diff(times)
        diffs = diffs[diffs > 0]  # drop dup launches at t=0 across directions
        if diffs.size == 0:
            return None
        return float(np.median(diffs))
    except Exception:
        return None


def _load_default_headways(timetable_dir: str = _DEFAULT_TIMETABLE_DIR) -> Dict[str, float]:
    """Scan calibrated_env/data/{LINE}/time_table.xlsx and build a dict."""
    out: Dict[str, float] = {}
    if os.path.isdir(timetable_dir):
        for line in sorted(os.listdir(timetable_dir)):
            sub = os.path.join(timetable_dir, line)
            if os.path.isdir(sub):
                hw = _scheduled_headway_from_excel(os.path.join(sub, "time_table.xlsx"))
                if hw is not None:
                    out[line] = hw
    return out


class DaganzoPolicy:
    """Cooperative-holding rule policy with the same call interface as the
    trained RL policies used by the H2O+ SUMO eval harness.

    Args:
        alpha: cooperation strength in [0, 1].  Default 0.6 (Daganzo).
        line_headways: optional pre-computed dict {line_id: H_target_sec}.
            If None, loaded from `bus_h2o/calibrated_env/data/{LINE}/time_table.xlsx`.
        default_headway: fallback when a line is not in the table.
        max_hold_sec: cap matching the env's action mapping (default 60).
        use_two_sided: if True (default), use max(h_prev, h_next).
            If False, single-side variant uses only h_prev (textbook rule
            when the trailing bus's position is unknown).
    """

    def __init__(
        self,
        alpha: float = 0.6,
        line_headways: Optional[Dict[str, float]] = None,
        default_headway: float = 360.0,
        max_hold_sec: float = 60.0,
        use_two_sided: bool = True,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = float(alpha)
        self.default_headway = float(default_headway)
        self.max_hold_sec = float(max_hold_sec)
        self.use_two_sided = bool(use_two_sided)
        if line_headways is None:
            line_headways = _load_default_headways()
        self.line_headways: Dict[str, float] = dict(line_headways)

    # ── interface used by run_episode(...) in eval_offline_on_sumo.py ──
    # Signature: policy_fn(ev, obs, bid, last_action) -> raw_action[2] in [-1,1]
    def __call__(self, ev, obs, bid=None, last_action=None) -> np.ndarray:
        return self._action_from_obs(ev, obs)

    # ── interface used by eval_with_metrics.py (drop-in for `policy(...)`) ─
    def forward(self, obs, deterministic: bool = True):
        # Accept either a torch tensor or a numpy array; mimic the
        # (action, log_prob) tuple returned by BusEmbeddingPolicy.
        try:
            import torch
            is_torch = isinstance(obs, torch.Tensor)
        except ImportError:
            torch = None
            is_torch = False

        if is_torch:
            obs_np = obs.detach().cpu().numpy()
        else:
            obs_np = np.asarray(obs, dtype=np.float32)

        if obs_np.ndim == 1:
            obs_np = obs_np[None, :]
        actions = np.stack([self._action_from_obs(None, o) for o in obs_np], axis=0)

        if is_torch and torch is not None:
            return torch.from_numpy(actions.astype(np.float32)), None
        return actions, None

    # Allow `policy(obs)` like the trained policy.
    def __getattr__(self, name):  # only used if not found
        if name == "eval":
            return lambda: self
        raise AttributeError(name)

    # ── internals ──────────────────────────────────────────────────────
    def _hold_seconds(
        self,
        h_prev: float,
        h_next: float,
        h_target: float,
        forward_present: bool = True,
        backward_present: bool = True,
    ) -> float:
        """Daganzo cooperative-holding rule (seconds)."""
        if not forward_present and not backward_present:
            return 0.0
        if self.use_two_sided and forward_present and backward_present:
            ref = max(float(h_prev), float(h_next))
        elif forward_present:
            ref = float(h_prev)
        else:
            ref = float(h_next)
        h_i = self.alpha * (float(h_target) - ref)
        return float(np.clip(h_i, 0.0, self.max_hold_sec))

    def _action_from_obs(self, ev, obs) -> np.ndarray:
        """Build [hold_norm, speed_norm] in [-1, 1] from a 15-dim obs.

        Uses ev.target_forward_headway and presence flags when ev is provided
        (matches the dynamic per-stop target used by the env), otherwise
        falls back to obs[8] / line_headways table.
        """
        obs = np.asarray(obs, dtype=np.float32)
        h_prev = float(obs[5])
        h_next = float(obs[6])
        h_target = float(obs[8]) if obs.size > 8 else self.default_headway

        forward_present = True
        backward_present = True
        if ev is not None:
            h_target = float(getattr(ev, "target_forward_headway", h_target))
            forward_present = bool(getattr(ev, "forward_bus_present", True))
            backward_present = bool(getattr(ev, "backward_bus_present", True))
            line_id = getattr(ev, "line_id", None)
            if line_id is not None and line_id in self.line_headways:
                # Prefer the env's dynamic per-stop target if present, else
                # fall back to the timetable median for that line.
                if not hasattr(ev, "target_forward_headway"):
                    h_target = self.line_headways[line_id]

        hold = self._hold_seconds(
            h_prev, h_next, h_target,
            forward_present=forward_present,
            backward_present=backward_present,
        )
        # Invert env mapping: hold_seconds = clip(30*hold_norm + 30, 0, 60)
        hold_norm = (hold - 30.0) / 30.0
        hold_norm = float(np.clip(hold_norm, -1.0, 1.0))
        speed_norm = 0.0  # neutral 1.0x; Daganzo controls only holding.
        return np.array([hold_norm, speed_norm], dtype=np.float32)


def build_default(alpha: float = 0.6) -> DaganzoPolicy:
    """Convenience factory used by eval scripts."""
    return DaganzoPolicy(alpha=alpha)
