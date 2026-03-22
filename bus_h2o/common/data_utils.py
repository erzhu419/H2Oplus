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
) -> np.ndarray:
    """
    Convert a full SnapshotDict into a compact 30-dim spatial fingerprint z.

    The route is divided into `num_segments` equal spatial bins.
    For each bin we compute:
        - mean speed of buses in that bin  (normalised by 30 m/s)
        - bus density                      (normalised by 5 buses/bin)
        - total waiting passengers at stops in that bin  (normalised by 20)

    Args:
        snapshot    : SnapshotDict as defined in Phase 0.1 of H2O+.md
                      Must have `all_buses` and `all_stations` keys.
                      Each bus entry needs: pos (float), speed (float), load (int)
                      Each station entry needs: pos (float), waiting_count (int)
        num_segments: Number of spatial bins (default 10 → output dim = 30)

    Returns:
        z : np.ndarray, shape = (num_segments * 3,), dtype float32
            Concatenation of [vec_speed, vec_density, vec_waiting]

    Notes:
        - `pos` in both buses and stations must already be linear absolute
          distance from route origin (m).  Use sumo_pos_to_linear() before
          building the snapshot if coming from SUMO traci.
        - SimBus side: bus.absolute_distance is used directly.
        - ROUTE_LENGTH must be set correctly via set_route_length().
    """
    if ROUTE_LENGTH <= 0:
        raise RuntimeError(
            "ROUTE_LENGTH is not set. Call set_route_length(total_m) at startup."
        )

    segment_len = ROUTE_LENGTH / num_segments

    # Per-segment accumulators
    seg_speeds:  list[list[float]] = [[] for _ in range(num_segments)]
    seg_counts:  list[int]         = [0] * num_segments
    seg_waiting: list[int]         = [0] * num_segments

    # --- Bin buses ---
    for bus in snapshot.get("all_buses", []):
        pos = float(bus.get("pos", 0.0))
        idx = min(int(pos / segment_len), num_segments - 1)
        seg_speeds[idx].append(float(bus.get("speed", 0.0)))
        seg_counts[idx] += 1

    # --- Bin station waiting passengers ---
    for st in snapshot.get("all_stations", []):
        pos = float(st.get("pos", 0.0))
        idx = min(int(pos / segment_len), num_segments - 1)
        seg_waiting[idx] += int(st.get("waiting_count", 0))

    # --- Normalise ---
    # Speed: default to 30 m/s (free-flow) when no bus is present
    vec_speed   = np.array(
        [np.mean(s) if s else 30.0 for s in seg_speeds], dtype=np.float32
    ) / 30.0

    vec_density = np.array(seg_counts, dtype=np.float32) / 5.0  # ~5 buses per bin max
    vec_waiting = np.array(seg_waiting, dtype=np.float32) / 20.0  # ~20 pax per bin max

    z = np.concatenate([vec_speed, vec_density, vec_waiting])
    return z.astype(np.float32)


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
    discriminator: ZOnlyDiscriminator,
    z_t:  torch.Tensor,
    z_t1: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Importance weight: w = sigmoid(D(z_t, z_t1)) / (1 - sigmoid(D(z_t, z_t1))).
    Used to re-weight real (SUMO) transitions in offline RL.
    Returns detached (B, 1) tensor.
    """
    with torch.no_grad():
        logit     = discriminator(z_t, z_t1)
        prob_real = torch.sigmoid(logit)
        w         = prob_real / (1.0 - prob_real + eps)
    return w
