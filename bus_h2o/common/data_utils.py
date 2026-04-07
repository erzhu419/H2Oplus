"""
common/data_utils.py
====================
Phase 0: Core data protocols for H2O+ Bus Sim-to-Real framework.

Provides:
    - build_edge_linear_map()       : Edge ID -> cumulative linear distance mapper
    - sumo_pos_to_linear()          : Convert SUMO (edge_id, lane_offset) to metres
    - extract_structured_context()  : Snapshot -> 30-dim spatial fingerprint z
    - ZOnlyDiscriminator            : Phase 3 primary: D(z_t, z_t+1) -> logit  [60-dim]
    - SimpleTransitionDiscriminator : Legacy: D(obs,act,obs',z_t,z_t+1) -> logit

Coordinate convention
---------------------
All `pos` fields in SnapshotDict must be **linear absolute distance from route
origin (m)**, consistent with LSTM-RL/env/bus.py `absolute_distance`.

For SUMO Real side:
    linear_pos = ROUTE_EDGE_MAP[road_id] + lane_position

For SimBus side:
    bus.absolute_distance is already in this format — no conversion needed.

For stops:
    bridge.line_stop_distances[line_id][stop_id] is already cumulative — use directly.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 0.0  Edge -> Linear Distance Mapper
# ---------------------------------------------------------------------------

def build_edge_linear_map(xml_path: str, line_id: str) -> dict[str, float]:
    """
    Parse `a_sorted_busline_edge.xml` and build a mapping from edge ID to
    the cumulative distance from the route origin (m) for *one* bus line.

    The XML lists elements in travel order with a `length` attribute.
    We accumulate lengths to get the start position of each edge.

    Args:
        xml_path : Absolute path to a_sorted_busline_edge.xml
        line_id  : Bus line ID as in XML, e.g. "7X" or "7S"

    Returns:
        dict {edge_id (str): cumulative_start_distance_m (float)}
        The edge_id is the element `id` attribute (e.g. "E_J393_J395").

    Example:
        edge_map = build_edge_linear_map(".../a_sorted_busline_edge.xml", "7X")
        linear_pos = sumo_pos_to_linear("E_J393_J395", 50.0, edge_map)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find the matching <busline> element
    busline_elem = None
    for bl in root.findall("busline"):
        if bl.get("id") == line_id:
            busline_elem = bl
            break
    if busline_elem is None:
        raise ValueError(
            f"Line ID '{line_id}' not found in {xml_path}. "
            f"Available: {[b.get('id') for b in root.findall('busline')]}"
        )

    edge_map: dict[str, float] = {}
    cumulative = 0.0

    for elem in busline_elem.findall("element"):
        eid = elem.get("id")
        length = float(elem.get("length", 0.0))
        # Only map edge-level IDs (skip duplicates — XML may list the same edge
        # multiple times for sub-sections; the FIRST occurrence is the entry point)
        if eid not in edge_map:
            edge_map[eid] = cumulative
        cumulative += length

    return edge_map


def sumo_pos_to_linear(
    edge_id: str,
    lane_offset: float,
    edge_map: dict[str, float],
) -> float:
    """
    Convert a SUMO vehicle position (road_id from traci, lanePosition from traci)
    to a linear absolute distance from the route origin (m).

    Usage:
        road_id = traci.vehicle.getRoadID(bus_id)
        lane_pos = traci.vehicle.getLanePosition(bus_id)
        linear_pos = sumo_pos_to_linear(road_id, lane_pos, ROUTE_EDGE_MAP["7X"])

    If edge_id is not in edge_map (e.g. bus is on a connector edge during
    intersection), returns `edge_map.get(edge_id, 0.0) + lane_offset`.
    This is a safe fallback — treat unknown edges as being at position 0.
    """
    return edge_map.get(edge_id, 0.0) + lane_offset


# ---------------------------------------------------------------------------
# 0.2  Snapshot Context Extractor
# ---------------------------------------------------------------------------

# Route total length in metres.
# Must be set at startup by calling set_route_length() or assigning directly.
ROUTE_LENGTH: float = 1.0  # placeholder; override before use


def set_route_length(length_m: float) -> None:
    """Set the global route length used by extract_structured_context."""
    global ROUTE_LENGTH
    ROUTE_LENGTH = float(length_m)


def extract_structured_context(
    snapshot: dict,
    num_segments: int = 10,
    zero_waiting: bool = True,
) -> np.ndarray:
    """
    Convert a full SnapshotDict into a compact 30-dim spatial fingerprint z.

    The *abstract* route is divided into `num_segments` equal fractional bins
    (each covering 1/num_segments of the route).  For each bin we compute:
        - mean speed of buses in that bin  (normalised by 30 m/s)
        - bus density                      (normalised by 5 buses/bin)
        - total waiting passengers at stops in that bin  (normalised by 20)

    Each bus/station is placed into a segment using its **fractional position**
    along its own line's route:

        fraction = pos / route_length          (clamped to [0, 1))
        segment  = int(fraction * num_segments)

    If a bus/station carries ``route_length`` in its dict entry, that value is
    used.  Otherwise the global ``ROUTE_LENGTH`` is used as fallback.

    Args:
        snapshot    : SnapshotDict with `all_buses` and `all_stations`.
                      Bus entries: pos, speed, [route_length]
                      Station entries: pos, waiting_count, [route_length]
        num_segments: Number of spatial bins (default 10 → output dim = 30)
        zero_waiting: If True, zero out waiting channel (debug/mitigation)

    Returns:
        z : np.ndarray, shape = (num_segments * 3,), dtype float32
            Concatenation of [vec_speed, vec_density, vec_waiting]
    """
    if ROUTE_LENGTH <= 0:
        raise RuntimeError(
            "ROUTE_LENGTH is not set. Call set_route_length(total_m) at startup."
        )

    # Per-segment accumulators
    seg_speeds:  list[list[float]] = [[] for _ in range(num_segments)]
    seg_counts:  list[int]         = [0] * num_segments
    seg_waiting: list[int]         = [0] * num_segments

    def _segment_idx(pos: float, route_len: float) -> int:
        """Map absolute position to segment index via fractional position."""
        rl = route_len if route_len and route_len > 0 else ROUTE_LENGTH
        frac = max(0.0, min(pos / rl, 1.0 - 1e-9))
        return min(int(frac * num_segments), num_segments - 1)

    # --- Bin buses ---
    for bus in snapshot.get("all_buses", []):
        pos = float(bus.get("pos", 0.0))
        rl  = bus.get("route_length", None)
        idx = _segment_idx(pos, rl)
        seg_speeds[idx].append(float(bus.get("speed", 0.0)))
        seg_counts[idx] += 1

    # --- Bin station waiting passengers ---
    for st in snapshot.get("all_stations", []):
        pos = float(st.get("pos", 0.0))
        rl  = st.get("route_length", None)
        idx = _segment_idx(pos, rl)
        seg_waiting[idx] += int(st.get("waiting_count", 0))

    # --- Normalise ---
    # Speed channel: **moving-fraction** per segment.
    #   value = (# buses moving) / (# buses in segment), or 0 if empty.
    # A bus is "moving" if speed > MOVING_THRESHOLD m/s.
    # Both SUMO and Sim have ~30% stopped ratio, so this indicator is
    # inherently aligned across the two environments without any absolute
    # speed calibration.  It captures the operationally meaningful signal
    # (where buses are stuck vs flowing) while being scale-invariant.
    MOVING_THRESHOLD = 0.5  # m/s; below this a bus is "stopped"
    vec_speed = np.zeros(num_segments, dtype=np.float32)
    for i, speeds in enumerate(seg_speeds):
        if speeds:
            n_moving = sum(1 for s in speeds if s > MOVING_THRESHOLD)
            vec_speed[i] = n_moving / len(speeds)

    # Density channel: **fractional density** per segment.
    #   value = (# buses in segment) / (total # buses), or 0 if no buses at all.
    # This is scale-invariant: unaffected by total bus count differences
    # between SUMO (~67 active buses across 12 lines) and SIM (~9 buses).
    raw_counts = np.array(seg_counts, dtype=np.float32)
    total_buses = raw_counts.sum()
    if total_buses > 0:
        vec_density = raw_counts / total_buses
    else:
        vec_density = np.zeros(num_segments, dtype=np.float32)

    if zero_waiting:
        vec_waiting = np.zeros(num_segments, dtype=np.float32)
    else:
        vec_waiting = np.array(seg_waiting, dtype=np.float32) / 20.0

    z = np.concatenate([vec_speed, vec_density, vec_waiting])
    return z.astype(np.float32)


def renormalize_z_density(z: np.ndarray, num_segments: int = 10) -> np.ndarray:
    """Convert old-format z-features (density/5.0) to fractional density.

    The old format stores density as raw_count / 5.0 in dims [10:20].
    This function converts it to raw_count / total_count (fractional density),
    making it invariant to total bus count.

    Works on both 1-D (single z) and 2-D (batch of z) inputs.
    """
    z = z.copy()
    if z.ndim == 1:
        density_old = z[num_segments: 2 * num_segments]
        raw_counts = density_old * 5.0
        total = raw_counts.sum()
        if total > 0:
            z[num_segments: 2 * num_segments] = raw_counts / total
        else:
            z[num_segments: 2 * num_segments] = 0.0
    elif z.ndim == 2:
        density_old = z[:, num_segments: 2 * num_segments]
        raw_counts = density_old * 5.0
        totals = raw_counts.sum(axis=1, keepdims=True)
        # Avoid division by zero
        safe_totals = np.where(totals > 0, totals, 1.0)
        z[:, num_segments: 2 * num_segments] = raw_counts / safe_totals
    return z


# ---------------------------------------------------------------------------
# Phase 3 (defined here for single-source-of-truth)
# 3.3  Simple Transition Discriminator (MLP)
# ---------------------------------------------------------------------------

class SimpleTransitionDiscriminator(nn.Module):
    """
    MLP-based Discriminator for H2O+ context-aware transition evaluation.

    Evaluates how "real" a sim transition is, given:
        - micro-level: (obs_T1, action, obs_T2)
        - macro-level: spatial context z_t at T1, z_t1 at T2

    Input concatenation:
        [obs(T1) | action | obs(T2) | z_t(30) | z_t1(30)]
        total dim = obs_dim + act_dim + obs_dim + 30 + 30

    Output:
        logit (B, 1)  — apply sigmoid to get P(real ∈ [0,1])

    Usage:
        D = SimpleTransitionDiscriminator(obs_dim=7, act_dim=1)
        logit = D(obs, act, next_obs, z_t, z_t1)
        w = torch.sigmoid(logit) / (1 - torch.sigmoid(logit) + 1e-8)

    Loss (in train.py, Appendix C):
        Use BCEWithLogitsLoss.
        Real labels  = 0.9  (label smoothing)
        Sim labels   = 0.1  (label smoothing)
        Add 5% Gaussian noise to z_t and z_t1 for real samples (data augmentation).

    Architecture note:
        First-version MLP. Upgrade to Cross-Attention (Appendix A) later if
        data quantity allows and MLP discriminator saturates.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        context_dim: int = 30,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        # obs_T1 + act + obs_T2 + z_t + z_t1
        input_dim = obs_dim + act_dim + obs_dim + context_dim + context_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),   # raw logit; no sigmoid here
        )

    def forward(
        self,
        obs:      torch.Tensor,   # (B, obs_dim)
        act:      torch.Tensor,   # (B, act_dim)
        next_obs: torch.Tensor,   # (B, obs_dim)
        z_t:      torch.Tensor,   # (B, 30)  — T1 spatial fingerprint
        z_t1:     torch.Tensor,   # (B, 30)  — T2 spatial fingerprint
    ) -> torch.Tensor:            # (B, 1)
        """
        All inputs must be (Batch, Dim) float tensors on the same device.
        Returns raw logit.  Use sigmoid externally for probabilities.
        """
        x = torch.cat([obs, act, next_obs, z_t, z_t1], dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Convenience: compute importance weight from discriminator output
# ---------------------------------------------------------------------------

def compute_importance_weight(
    discriminator: SimpleTransitionDiscriminator,
    obs:      torch.Tensor,
    act:      torch.Tensor,
    next_obs: torch.Tensor,
    z_t:      torch.Tensor,
    z_t1:     torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute H2O importance weight w = P(real) / P(sim) for sim transitions.

    w = sigmoid(logit) / (1 - sigmoid(logit) + eps)

    Returns:
        w : (B, 1) tensor, detached from computation graph.
    """
    with torch.no_grad():
        logit     = discriminator(obs, act, next_obs, z_t, z_t1)
        prob_real = torch.sigmoid(logit)
        w         = prob_real / (1.0 - prob_real + eps)
    return w


# ---------------------------------------------------------------------------
# Phase 3 PRIMARY: z-only discriminator  (obs-dim-agnostic)
# ---------------------------------------------------------------------------

class ZOnlyDiscriminator(nn.Module):
    """
    Context-only discriminator for H2O+ Phase 3.

    Input: (z_t, z_t+1) concatenated  →  60 float features.
    Dimension-agnostic w.r.t. the observation space, so it works with
    both SUMO (obs_dim=15) and BusSimEnv (obs_dim=32) without adaptation.

    Output: scalar logit per sample.

    Training labels (recommended):
        real (SUMO)   →  0.9  (label-smoothed positive)
        sim  (SimBus) →  0.1  (label-smoothed negative)

    Usage:
        D = ZOnlyDiscriminator(context_dim=30)
        logit = D(z_t, z_t1)                    # (B,1)
        w = compute_z_importance_weight(D, z_t, z_t1)
    """

    def __init__(
        self,
        context_dim: int = 30,
        hidden_dim: int  = 256,
        n_hidden: int    = 2,
    ) -> None:
        super().__init__()
        input_dim = context_dim * 2   # z_t || z_t+1
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        z_t:  torch.Tensor,   # (B, context_dim)
        z_t1: torch.Tensor,   # (B, context_dim)
    ) -> torch.Tensor:        # (B, 1)
        return self.net(torch.cat([z_t, z_t1], dim=-1))


def compute_z_importance_weight(
    discriminator,
    z_t:  torch.Tensor,
    z_t1: torch.Tensor,
    eps: float = 1e-8,
    obs: torch.Tensor = None,
    action: torch.Tensor = None,
    next_obs: torch.Tensor = None,
) -> torch.Tensor:
    """
    Importance weight from discriminator.

    Supports three discriminator types:
    - DynamicsDiscriminator: w = exp(-dynamics_error / temp) — dynamics-based
    - TransitionDiscriminator: w = sigmoid(D) / (1-sigmoid(D)) — classification-based
    - ZOnlyDiscriminator: w = sigmoid(D) / (1-sigmoid(D)) — z-only

    Returns detached (B, 1) tensor.
    """
    with torch.no_grad():
        if isinstance(discriminator, (ContrastiveDynamicsDiscriminator,
                                       DomainAdaptiveDiscriminator,
                                       FactoredDynamicsDiscriminator,
                                       TransitionVAE,
                                       VAEDomainDiscriminator)):
            w = discriminator.compute_weight(obs, action, next_obs)
            return w.unsqueeze(-1) if w.dim() == 1 else w
        elif isinstance(discriminator, DynamicsDiscriminator):
            w = discriminator.compute_weight(obs, action, next_obs)
            return w.unsqueeze(-1)
        elif isinstance(discriminator, TransitionDiscriminator):
            logit = discriminator(obs, action, next_obs, z_t, z_t1)
        else:
            logit = discriminator(z_t, z_t1)
        prob_real = torch.sigmoid(logit)
        w         = prob_real / (1.0 - prob_real + eps)
    return w


# ---------------------------------------------------------------------------
# Phase 3 IMPROVED: Transition discriminator (obs + action + z)
# ---------------------------------------------------------------------------

class TransitionDiscriminator(nn.Module):
    """
    Full-transition discriminator for H2O+.

    Input: (obs, action, next_obs, z_t[0:20], z_t1[0:20]) concatenated.
    Drops the waiting channel (dims 20:30) of z which is always zero.

    This gives the discriminator access to microstate transitions (individual
    bus obs/action) in addition to macrostate context (spatial fingerprint).

    When use_spectral_norm=True, all Linear layers are wrapped with spectral
    normalization to prevent discriminator oscillation / collapse.
    """

    def __init__(
        self,
        obs_dim: int = 17,
        action_dim: int = 2,
        context_dim: int = 30,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        z_effective_dim: int = 20,  # only speed(10) + density(10), drop waiting(10)
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        from torch.nn.utils import spectral_norm as _sn

        self.z_effective_dim = z_effective_dim
        input_dim = obs_dim + action_dim + obs_dim + z_effective_dim * 2

        def _maybe_sn(layer):
            return _sn(layer) if use_spectral_norm else layer

        layers: list[nn.Module] = [_maybe_sn(nn.Linear(input_dim, hidden_dim)), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers += [_maybe_sn(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU()]
        layers.append(_maybe_sn(nn.Linear(hidden_dim, 1)))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        obs: torch.Tensor,       # (B, obs_dim)
        action: torch.Tensor,    # (B, action_dim)
        next_obs: torch.Tensor,  # (B, obs_dim)
        z_t: torch.Tensor,       # (B, context_dim)  — only [:20] used
        z_t1: torch.Tensor,      # (B, context_dim)  — only [:20] used
    ) -> torch.Tensor:           # (B, 1)
        z_t_eff = z_t[:, :self.z_effective_dim]
        z_t1_eff = z_t1[:, :self.z_effective_dim]
        return self.net(torch.cat([obs, action, next_obs, z_t_eff, z_t1_eff], dim=-1))


# ---------------------------------------------------------------------------
# Dynamics-based discriminator: judges environment similarity, not data source
# ---------------------------------------------------------------------------

class DynamicsDiscriminator(nn.Module):
    """Dynamics-aware importance weighting for H2O+.

    Learns SUMO dynamics on dynamics-sensitive features only, then scores
    online transitions by how well they match.

    Key design: only predicts and compares **dynamics-sensitive** features
    (fwd_hw, bwd_hw, gap) — not co-line headways, sim_time, or other
    features that are noisy/unpredictable regardless of dynamics fidelity.

    Obs layout (17-dim, indices 0-16):
      [0-4]  categorical (line_id, bus_id, station_id, time_period, direction)
      [5]    fwd_headway  ← DYNAMICS-SENSITIVE
      [6]    bwd_headway  ← DYNAMICS-SENSITIVE
      [7]    waiting_pax
      [8]    target_hw
      [9]    base_stop_duration
      [10]   sim_time
      [11]   gap           ← DYNAMICS-SENSITIVE
      [12]   co_fwd_hw     (noisy, skip)
      [13]   co_bwd_hw     (noisy, skip)
      [14]   seg_speed
      [15]   last_action_0
      [16]   last_action_1

    Weight: w = sigmoid(margin) where margin = (baseline_err - actual_err) / scale
    - actual_err ≈ baseline_err → w ≈ 0.5 (uncertain)
    - actual_err << baseline_err → w → 1.0 (consistent with SUMO)
    - actual_err >> baseline_err → w → 0.0 (inconsistent)
    """

    # Indices of dynamics-sensitive features in the continuous part (after n_cat)
    # fwd_hw=5, bwd_hw=6, gap=11 → after removing 5 cat → indices 0, 1, 6
    DYNAMICS_INDICES = [0, 1, 6]  # fwd_hw, bwd_hw, gap (relative to n_cat offset)

    def __init__(
        self,
        obs_dim: int = 17,
        action_dim: int = 2,
        hidden_dim: int = 128,
        n_hidden: int = 2,
        temperature: float = 1.0,
        n_cat: int = 5,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_cat = n_cat
        self.temperature = temperature

        # Only predict dynamics-sensitive features
        n_dyn = len(self.DYNAMICS_INDICES)
        input_dim = obs_dim + action_dim

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, n_dyn))
        self.forward_model = nn.Sequential(*layers)

        # Per-feature running stats for normalized error
        self.register_buffer('feat_mean', torch.zeros(n_dyn))
        self.register_buffer('feat_var', torch.ones(n_dyn))
        self.register_buffer('baseline_err', torch.tensor(1.0))  # EMA of training error
        self._ema_count = 0

    def _extract_dyn_features(self, next_obs: torch.Tensor) -> torch.Tensor:
        """Extract dynamics-sensitive features from next_obs."""
        cont = next_obs[:, self.n_cat:]  # skip categorical
        return cont[:, self.DYNAMICS_INDICES]  # (B, n_dyn)

    def forward(self, obs: torch.Tensor, action: torch.Tensor,
                next_obs: torch.Tensor = None, *args, **kwargs) -> torch.Tensor:
        """Forward pass: returns dynamics error (lower = more real-like)."""
        pred = self.forward_model(torch.cat([obs, action], dim=-1))
        if next_obs is not None:
            target = self._extract_dyn_features(next_obs)
            # Normalized per-feature error
            norm_err = ((pred - target) / (self.feat_var.sqrt() + 1e-6)).pow(2).mean(dim=-1, keepdim=True)
            return norm_err  # (B, 1)
        return pred

    def predict_next_obs(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.forward_model(torch.cat([obs, action], dim=-1))

    def dynamics_error(self, obs, action, next_obs) -> torch.Tensor:
        """Per-sample normalized dynamics error on dynamics-sensitive features."""
        pred = self.predict_next_obs(obs, action)
        target = self._extract_dyn_features(next_obs)
        # Normalize each feature by its variance, then average
        norm_err = ((pred - target) / (self.feat_var.sqrt() + 1e-6)).pow(2).mean(dim=-1)
        return norm_err  # (B,)

    def compute_weight(self, obs, action, next_obs) -> torch.Tensor:
        """Compute IS weight based on dynamics consistency.

        Uses sigmoid on (baseline - actual) margin for smooth, calibrated weights.
        """
        with torch.no_grad():
            error = self.dynamics_error(obs, action, next_obs)  # (B,)

            # Margin-based weight: how much better/worse than baseline
            # baseline_err = expected error on SUMO data (from training)
            # error < baseline → positive margin → w > 0.5 (good)
            # error > baseline → negative margin → w < 0.5 (bad)
            margin = (self.baseline_err - error) / max(self.temperature, 0.1)
            w = torch.sigmoid(margin)
            return w.clamp(0.01, 1.0)

    def train_step(self, obs, action, next_obs, optimizer) -> float:
        """One gradient step on SUMO data. Updates running stats."""
        pred = self.predict_next_obs(obs, action)
        target = self._extract_dyn_features(next_obs)

        # Per-feature normalized loss
        loss = F.mse_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update running feature stats + baseline error
        with torch.no_grad():
            self._ema_count += 1
            ema = min(0.01, 1.0 / self._ema_count)

            # Feature-level mean/var for normalization
            feat_vals = target.detach()
            self.feat_mean = (1 - ema) * self.feat_mean + ema * feat_vals.mean(dim=0)
            self.feat_var = (1 - ema) * self.feat_var + ema * feat_vals.var(dim=0).clamp(min=1.0)

            # Baseline error (what the model achieves on SUMO data)
            norm_err = ((pred.detach() - target) / (self.feat_var.sqrt() + 1e-6)).pow(2).mean(dim=-1)
            self.baseline_err = (1 - ema) * self.baseline_err + ema * norm_err.mean()

        return loss.item()


# ---------------------------------------------------------------------------
# Factored dynamics discriminator: D_sas - D_sa isolates dynamics from policy
# ---------------------------------------------------------------------------

class FactoredDynamicsDiscriminator(nn.Module):
    """DARC-style factored discriminator that isolates dynamics from policy.

    Uses two classifiers:
      D_sa(s, a) → P(real | s, a)      — absorbs policy distribution differences
      D_sas(s, a, s'_dyn) → logits     — captures dynamics + policy differences

    The dynamics ratio is:
      log P_real(s'|s,a) / P_sim(s'|s,a) = D_sas - D_sa  (Bayes rule identity)

    By subtracting D_sa from D_sas, policy distribution differences cancel out,
    leaving only the dynamics signal.

    Key design: D_sas uses only dynamics-sensitive features of s'
    (fwd_hw[5], bwd_hw[6], gap[11]) to prevent the classifier from using
    "easy" non-dynamics features to discriminate.

    Obs layout (17-dim):
      [0-4]  categorical → D_sa input
      [5]    fwd_headway → D_sas dynamics feature
      [6]    bwd_headway → D_sas dynamics feature
      [11]   gap         → D_sas dynamics feature
    """

    # Absolute indices of dynamics-sensitive features in obs
    DYN_FEATURE_INDICES = [5, 6, 11]  # fwd_hw, bwd_hw, gap

    def __init__(
        self,
        obs_dim: int = 17,
        action_dim: int = 2,
        hidden_dim: int = 256,
        n_hidden: int = 2,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        n_dyn = len(self.DYN_FEATURE_INDICES)

        # D_sa: classifies (s, a) → 2 logits [real, sim]
        sa_layers = [nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            sa_layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        sa_layers.append(nn.Linear(hidden_dim, 2))
        self.d_sa = nn.Sequential(*sa_layers)

        # D_sas: advantage logits on (s, a, s'_dyn) → 2 logits
        # Input: obs + action + dynamics features of next_obs
        sas_layers = [nn.Linear(obs_dim + action_dim + n_dyn, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            sas_layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        sas_layers.append(nn.Linear(hidden_dim, 2))
        self.d_sas = nn.Sequential(*sas_layers)

    def forward(self, obs, action, next_obs=None, *args, **kwargs):
        """Returns log dynamics ratio (higher = more real-like)."""
        if next_obs is not None:
            return self.log_dynamics_ratio(obs, action, next_obs).unsqueeze(-1)
        return self.d_sa(torch.cat([obs, action], dim=-1))

    def _extract_dyn(self, next_obs):
        """Extract dynamics-sensitive features from next_obs."""
        return next_obs[:, self.DYN_FEATURE_INDICES]

    def log_dynamics_ratio(self, obs, action, next_obs):
        """log P_real(s'|s,a) / P_sim(s'|s,a) via factored discriminators."""
        sa_input = torch.cat([obs, action], dim=-1)
        sa_logits = self.d_sa(sa_input)
        sa_prob = F.softmax(sa_logits, dim=1)

        dyn_feats = self._extract_dyn(next_obs)
        sas_input = torch.cat([obs, action, dyn_feats], dim=-1)
        adv_logits = self.d_sas(sas_input)
        sas_prob = F.softmax(adv_logits + sa_logits, dim=1)

        # log P_real(s'|s,a) / P_sim(s'|s,a) = log(sas_real/sas_sim) - log(sa_real/sa_sim)
        log_ratio = (torch.log(sas_prob[:, 0] + 1e-8) - torch.log(sas_prob[:, 1] + 1e-8)
                      - torch.log(sa_prob[:, 0] + 1e-8) + torch.log(sa_prob[:, 1] + 1e-8))
        return torch.clamp(log_ratio, -10, 10)

    def compute_weight(self, obs, action, next_obs):
        """IS weight w ∈ [0, 1] based on dynamics ratio."""
        with torch.no_grad():
            log_r = self.log_dynamics_ratio(obs, action, next_obs)
            return torch.sigmoid(log_r).clamp(0.01, 1.0)

    def train_step(self, real_obs, real_action, real_next_obs,
                    sim_obs, sim_action, sim_next_obs,
                    optimizer, label_smooth=0.1):
        """One gradient step training both D_sa and D_sas."""
        # ── D_sa loss ──
        real_sa = torch.cat([real_obs, real_action], dim=-1)
        sim_sa = torch.cat([sim_obs, sim_action], dim=-1)
        real_sa_logits = self.d_sa(real_sa)
        sim_sa_logits = self.d_sa(sim_sa)

        real_sa_prob = F.softmax(real_sa_logits, dim=1)
        sim_sa_prob = F.softmax(sim_sa_logits, dim=1)
        dsa_loss = (-torch.log(real_sa_prob[:, 0] + 1e-8).mean()
                    - torch.log(sim_sa_prob[:, 1] + 1e-8).mean())

        # ── D_sas loss ──
        real_dyn = self._extract_dyn(real_next_obs)
        sim_dyn = self._extract_dyn(sim_next_obs)

        real_sas_input = torch.cat([real_obs, real_action, real_dyn], dim=-1)
        sim_sas_input = torch.cat([sim_obs, sim_action, sim_dyn], dim=-1)

        real_adv_logits = self.d_sas(real_sas_input)
        real_sas_prob = F.softmax(real_adv_logits + real_sa_logits.detach(), dim=1)
        sim_adv_logits = self.d_sas(sim_sas_input)
        sim_sas_prob = F.softmax(sim_adv_logits + sim_sa_logits.detach(), dim=1)

        dsas_loss = (-torch.log(real_sas_prob[:, 0] + 1e-8).mean()
                     - torch.log(sim_sas_prob[:, 1] + 1e-8).mean())

        total_loss = dsa_loss + dsas_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return total_loss.item(), dsa_loss.item(), dsas_loss.item()


# ---------------------------------------------------------------------------
# Approach A: Contrastive Dynamics Embedding (InfoNCE)
# ---------------------------------------------------------------------------

class ContrastiveDynamicsDiscriminator(nn.Module):
    """Learn a transition embedding where same-dynamics transitions cluster.

    Uses TRANSITION DELTAS on dynamics-sensitive features only:
        input = (action, Δfwd_hw, Δbwd_hw, Δgap, fwd_hw, bwd_hw, gap)
    This avoids encoding differences (line_id, bus_id, direction, etc.)
    that differ between SUMO and SIM but are not dynamics-related.

    Training: InfoNCE contrastive loss.
    IS weight: cosine similarity to SUMO prototype embedding.
    """

    DYN_INDICES = [5, 6, 11]  # fwd_hw, bwd_hw, gap (absolute in 17-dim obs)

    def __init__(self, obs_dim=17, action_dim=2, embed_dim=16, hidden_dim=128, n_hidden=2):
        super().__init__()
        # Input: dyn(3) + delta(3) + relative_delta(3) + cross_terms(3) = 12
        # NO action — dynamics discrimination should be action-invariant
        n_dyn = len(self.DYN_INDICES)
        input_dim = n_dyn * 4  # current + delta + relative_delta + cross_terms
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, embed_dim))
        self.encoder = nn.Sequential(*layers)
        self.embed_dim = embed_dim
        self.register_buffer('sumo_prototype', torch.zeros(embed_dim))
        self._proto_count = 0

    def _build_input(self, obs, action, next_obs):
        """Extract dynamics-only features. NO action input.

        Features (12-dim):
          [0:3]  dyn values: fwd_hw, bwd_hw, gap (normalized by /500)
          [3:6]  delta: Δfwd_hw, Δbwd_hw, Δgap (normalized by /100)
          [6:9]  relative delta: Δ/val (captures % change)
          [9:12] cross terms: Δfwd*Δbwd, Δfwd*Δgap, Δbwd*Δgap (captures correlation)
        """
        dyn = obs[:, self.DYN_INDICES]
        dyn_next = next_obs[:, self.DYN_INDICES]
        delta = dyn_next - dyn

        # Normalize to prevent scale issues
        dyn_norm = dyn / 500.0
        delta_norm = delta / 100.0

        # Relative change (how much did headway change proportionally)
        rel_delta = delta / (dyn.abs() + 10.0)  # +10 avoids div by near-zero

        # Cross terms (interaction between headway changes)
        cross = torch.stack([
            delta_norm[:, 0] * delta_norm[:, 1],  # Δfwd * Δbwd
            delta_norm[:, 0] * delta_norm[:, 2],  # Δfwd * Δgap
            delta_norm[:, 1] * delta_norm[:, 2],  # Δbwd * Δgap
        ], dim=-1)

        return torch.cat([dyn_norm, delta_norm, rel_delta, cross], dim=-1)

    def encode(self, obs, action, next_obs):
        x = self._build_input(obs, action, next_obs)
        z = self.encoder(x)
        return F.normalize(z, dim=-1)

    def forward(self, obs, action, next_obs=None, *args, **kwargs):
        if next_obs is not None:
            return self.encode(obs, action, next_obs)
        return self.encoder(torch.cat([obs, action], dim=-1))

    def compute_weight(self, obs, action, next_obs):
        """Weight = cosine similarity to SUMO prototype."""
        with torch.no_grad():
            z = self.encode(obs, action, next_obs)
            # Cosine similarity to prototype, mapped to [0, 1]
            cos_sim = F.cosine_similarity(z, self.sumo_prototype.unsqueeze(0), dim=-1)
            w = (cos_sim + 1) / 2  # [-1,1] → [0,1]
            return w.clamp(0.01, 1.0)

    def train_step(self, real_obs, real_action, real_next_obs,
                    sim_obs, sim_action, sim_next_obs,
                    optimizer, temperature=0.1):
        """InfoNCE contrastive loss."""
        B = real_obs.shape[0]
        # Split real into anchor + positive (two halves)
        mid = B // 2
        z_anchor = self.encode(real_obs[:mid], real_action[:mid], real_next_obs[:mid])
        z_pos = self.encode(real_obs[mid:2*mid], real_action[mid:2*mid], real_next_obs[mid:2*mid])

        # Negatives from SIM
        n_neg = min(sim_obs.shape[0], mid)
        z_neg = self.encode(sim_obs[:n_neg], sim_action[:n_neg], sim_next_obs[:n_neg])

        # InfoNCE: anchor vs (positive + negatives)
        # sim(anchor, positive) should be high, sim(anchor, negatives) should be low
        pos_sim = (z_anchor * z_pos).sum(dim=-1) / temperature  # (mid,)
        neg_sim = torch.mm(z_anchor, z_neg.T) / temperature     # (mid, n_neg)

        # Concatenate: first column = positive, rest = negatives
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (mid, 1+n_neg)
        labels = torch.zeros(mid, dtype=torch.long, device=logits.device)  # positive at index 0
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update SUMO prototype (EMA of all real embeddings)
        with torch.no_grad():
            z_all_real = self.encode(real_obs, real_action, real_next_obs)
            ema = min(0.05, 1.0 / (self._proto_count + 1))
            self._proto_count += 1
            self.sumo_prototype = (1 - ema) * self.sumo_prototype + ema * z_all_real.mean(dim=0)
            self.sumo_prototype = F.normalize(self.sumo_prototype, dim=0)

        return loss.item()


# ---------------------------------------------------------------------------
# Approach B: Domain Adaptation Encoder
# ---------------------------------------------------------------------------

class DomainAdaptiveDiscriminator(nn.Module):
    """Encoder + domain classifier in learned latent space.

    Architecture:
        encoder: (s, a, s') → z ∈ R^d  (compressed representation)
        classifier: z → P(real)         (domain classification)

    Unlike TransitionDiscriminator which classifies raw 76-dim input,
    this forces a bottleneck: the encoder must extract a compact latent
    that captures dynamics-relevant features. The 8-dim bottleneck acts
    as an information filter, discarding noise and policy-specific patterns.

    IS weight = classifier P(real | z) where z = encoder(s, a, s')
    """

    def __init__(self, obs_dim=17, action_dim=2, latent_dim=8,
                 enc_hidden=128, cls_hidden=64, n_enc_hidden=2):
        super().__init__()
        input_dim = obs_dim + action_dim + obs_dim  # s + a + s'
        self.latent_dim = latent_dim

        # Encoder: (s,a,s') → z
        enc_layers = [nn.Linear(input_dim, enc_hidden), nn.ReLU()]
        for _ in range(n_enc_hidden - 1):
            enc_layers += [nn.Linear(enc_hidden, enc_hidden), nn.ReLU()]
        enc_layers.append(nn.Linear(enc_hidden, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Domain classifier: z → P(real)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, cls_hidden), nn.ReLU(),
            nn.Linear(cls_hidden, 1),  # logit for P(real)
        )

    def encode(self, obs, action, next_obs):
        return self.encoder(torch.cat([obs, action, next_obs], dim=-1))

    def classify(self, obs, action, next_obs):
        z = self.encode(obs, action, next_obs)
        return self.classifier(z)  # (B, 1)

    def forward(self, obs, action, next_obs=None, *args, **kwargs):
        if next_obs is not None:
            return self.classify(obs, action, next_obs)
        return self.encoder(torch.cat([obs, action], dim=-1))

    def compute_weight(self, obs, action, next_obs):
        with torch.no_grad():
            logit = self.classify(obs, action, next_obs).squeeze(-1)
            return torch.sigmoid(logit).clamp(0.01, 1.0)

    def train_step(self, real_obs, real_action, real_next_obs,
                    sim_obs, sim_action, sim_next_obs,
                    optimizer, label_smooth=0.1):
        """Train encoder + classifier jointly."""
        real_logits = self.classify(real_obs, real_action, real_next_obs)
        sim_logits = self.classify(sim_obs, sim_action, sim_next_obs)

        real_labels = torch.full_like(real_logits, 1.0 - label_smooth)
        sim_labels = torch.full_like(sim_logits, label_smooth)

        loss = (F.binary_cross_entropy_with_logits(real_logits, real_labels)
                + F.binary_cross_entropy_with_logits(sim_logits, sim_labels))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            real_acc = (torch.sigmoid(real_logits) > 0.5).float().mean().item()
            sim_acc = (torch.sigmoid(sim_logits) < 0.5).float().mean().item()

        return loss.item(), real_acc, sim_acc


# ---------------------------------------------------------------------------
# Approach C: Transition VAE — anomaly-based dynamics discrimination
# ---------------------------------------------------------------------------

class TransitionVAE(nn.Module):
    """VAE trained on SUMO transitions. SIM transitions have high reconstruction
    error / low ELBO, providing a natural dynamics anomaly score.

    Architecture:
        encoder: (s, a, s'_dyn_norm) → μ, log_σ
        decoder: (z, s, a) → ŝ'_dyn_norm

    All dynamics features are internally normalized (running mean/std)
    to prevent gradient explosion from large raw values (fwd_hw ~400-600).

    IS weight = sigmoid((baseline_nll - sample_nll) / temperature)

    Trained ONLY on SUMO data — no SIM data needed.
    """

    DYN_INDICES = [5, 6, 11]  # fwd_hw, bwd_hw, gap (absolute indices in obs)

    def __init__(self, obs_dim=17, action_dim=2, latent_dim=8,
                 hidden_dim=128, temperature=1.0):
        super().__init__()
        n_dyn = len(self.DYN_INDICES)
        input_dim = obs_dim + action_dim

        # Encoder: (s, a, s'_dyn_norm) → μ, log_σ
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + n_dyn, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # Decoder: (z, s, a) → ŝ'_dyn_norm
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_dyn),
        )

        self.temperature = temperature
        self.latent_dim = latent_dim
        self.register_buffer('baseline_nll', torch.tensor(0.0))
        self._baseline_count = 0

        # Running normalization for dynamics features
        self.register_buffer('dyn_mean', torch.zeros(n_dyn))
        self.register_buffer('dyn_std', torch.ones(n_dyn))
        self._norm_count = 0

    def _extract_dyn(self, next_obs):
        return next_obs[:, self.DYN_INDICES]

    def _normalize_dyn(self, dyn):
        return (dyn - self.dyn_mean) / (self.dyn_std + 1e-6)

    def _update_norm(self, dyn):
        """Update running mean/std from a batch of dynamics features."""
        self._norm_count += 1
        ema = min(0.01, 1.0 / self._norm_count)
        with torch.no_grad():
            batch_mean = dyn.mean(dim=0)
            batch_std = dyn.std(dim=0).clamp(min=1.0)
            self.dyn_mean = (1 - ema) * self.dyn_mean + ema * batch_mean
            self.dyn_std = (1 - ema) * self.dyn_std + ema * batch_std

    def encode(self, obs, action, dyn_norm):
        sa = torch.cat([obs, action], dim=-1)
        h = self.encoder(torch.cat([sa, dyn_norm], dim=-1))
        return self.mu_head(h), self.logvar_head(h).clamp(-10, 2)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + torch.randn_like(std) * std

    def decode(self, z, obs, action):
        sa = torch.cat([obs, action], dim=-1)
        return self.decoder(torch.cat([z, sa], dim=-1))

    def forward(self, obs, action, next_obs=None, *args, **kwargs):
        if next_obs is None:
            return torch.zeros(obs.shape[0], 1, device=obs.device)
        dyn = self._normalize_dyn(self._extract_dyn(next_obs))
        mu, logvar = self.encode(obs, action, dyn)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, obs, action)
        recon_loss = (recon - dyn).pow(2).sum(dim=-1)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)
        return (recon_loss + kl).unsqueeze(-1)

    def compute_nll(self, obs, action, next_obs, n_samples=5):
        dyn = self._normalize_dyn(self._extract_dyn(next_obs))
        mu, logvar = self.encode(obs, action, dyn)
        total_nll = torch.zeros(obs.shape[0], device=obs.device)
        for _ in range(n_samples):
            z = self.reparameterize(mu, logvar)
            recon = self.decode(z, obs, action)
            recon_loss = (recon - dyn).pow(2).sum(dim=-1)
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)
            total_nll += recon_loss + kl
        return total_nll / n_samples

    def compute_weight(self, obs, action, next_obs):
        with torch.no_grad():
            nll = self.compute_nll(obs, action, next_obs)
            margin = (self.baseline_nll - nll) / max(self.temperature, 0.1)
            return torch.sigmoid(margin).clamp(0.01, 1.0)

    def train_step(self, obs, action, next_obs, optimizer, beta=1.0):
        dyn_raw = self._extract_dyn(next_obs)
        self._update_norm(dyn_raw)
        dyn = self._normalize_dyn(dyn_raw)

        mu, logvar = self.encode(obs, action, dyn)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, obs, action)

        recon_loss = F.mse_loss(recon, dyn, reduction='mean')
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        loss = recon_loss + beta * kl

        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            per_sample_nll = (recon.detach() - dyn).pow(2).sum(dim=-1) + kl
            batch_nll = per_sample_nll.mean()
            ema = min(0.01, 1.0 / (self._baseline_count + 1))
            self._baseline_count += 1
            self.baseline_nll = (1 - ema) * self.baseline_nll + ema * batch_nll

        return loss.item(), recon_loss.item(), kl.item()


# ---------------------------------------------------------------------------
# Approach D: VAE + Domain Adaptive (combined)
# ---------------------------------------------------------------------------

class VAEDomainDiscriminator(nn.Module):
    """Combines TransitionVAE anomaly score with DomainAdaptiveDiscriminator.

    VAE provides unsupervised dynamics anomaly (trained on SUMO only).
    Domain classifier provides supervised discrimination (trained on both).
    Combined weight uses both signals for robustness.

    w = α * w_vae + (1-α) * w_domain
    """

    def __init__(self, obs_dim=17, action_dim=2, latent_dim=8,
                 hidden_dim=128, alpha=0.5, temperature=1.0):
        super().__init__()
        self.alpha = alpha

        # VAE component
        self.vae = TransitionVAE(
            obs_dim=obs_dim, action_dim=action_dim, latent_dim=latent_dim,
            hidden_dim=hidden_dim, temperature=temperature,
        )

        # Domain classifier on VAE latent + raw dynamics features
        n_dyn = len(TransitionVAE.DYN_INDICES)
        # Classifier input: VAE latent (μ) + dynamics features
        cls_input = latent_dim + n_dyn
        self.domain_classifier = nn.Sequential(
            nn.Linear(cls_input, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),  # logit for P(real)
        )

    def forward(self, obs, action, next_obs=None, *args, **kwargs):
        if next_obs is None:
            return torch.zeros(obs.shape[0], 1, device=obs.device)
        return self.vae(obs, action, next_obs)

    def _get_classifier_input(self, obs, action, next_obs):
        """Get VAE latent μ + dynamics features for domain classifier."""
        dyn = next_obs[:, TransitionVAE.DYN_INDICES]
        mu, _ = self.vae.encode(obs, action, dyn)
        return torch.cat([mu.detach(), dyn], dim=-1)  # detach: don't backprop to VAE

    def compute_weight(self, obs, action, next_obs):
        with torch.no_grad():
            w_vae = self.vae.compute_weight(obs, action, next_obs)
            cls_input = self._get_classifier_input(obs, action, next_obs)
            w_domain = torch.sigmoid(self.domain_classifier(cls_input)).squeeze(-1)
            w = self.alpha * w_vae + (1 - self.alpha) * w_domain
            return w.clamp(0.01, 1.0)

    def train_step_vae(self, obs, action, next_obs, optimizer, beta=1.0):
        """Train VAE on SUMO data only."""
        return self.vae.train_step(obs, action, next_obs, optimizer, beta)

    def train_step_classifier(self, real_obs, real_action, real_next_obs,
                               sim_obs, sim_action, sim_next_obs,
                               optimizer, label_smooth=0.1):
        """Train domain classifier on both SUMO and SIM."""
        real_input = self._get_classifier_input(real_obs, real_action, real_next_obs)
        sim_input = self._get_classifier_input(sim_obs, sim_action, sim_next_obs)

        real_logits = self.domain_classifier(real_input)
        sim_logits = self.domain_classifier(sim_input)

        real_labels = torch.full_like(real_logits, 1.0 - label_smooth)
        sim_labels = torch.full_like(sim_logits, label_smooth)

        loss = (F.binary_cross_entropy_with_logits(real_logits, real_labels)
                + F.binary_cross_entropy_with_logits(sim_logits, sim_labels))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            real_acc = (torch.sigmoid(real_logits) > 0.5).float().mean().item()
            sim_acc = (torch.sigmoid(sim_logits) < 0.5).float().mean().item()
        return loss.item(), real_acc, sim_acc

    def train_step(self, real_obs, real_action, real_next_obs,
                    sim_obs, sim_action, sim_next_obs,
                    optimizer, **kwargs):
        """Combined training: VAE on real + classifier on both."""
        vae_loss, recon, kl = self.train_step_vae(
            real_obs, real_action, real_next_obs, optimizer)
        cls_loss, racc, sacc = self.train_step_classifier(
            real_obs, real_action, real_next_obs,
            sim_obs, sim_action, sim_next_obs, optimizer)
        return vae_loss + cls_loss, racc, sacc
