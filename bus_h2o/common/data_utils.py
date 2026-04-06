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
    Importance weight: w = sigmoid(D(...)) / (1 - sigmoid(D(...))).
    Supports both ZOnlyDiscriminator and TransitionDiscriminator.
    Returns detached (B, 1) tensor.
    """
    with torch.no_grad():
        if isinstance(discriminator, TransitionDiscriminator):
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
