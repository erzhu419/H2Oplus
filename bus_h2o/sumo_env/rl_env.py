import traci
# traci = libsumo
LIBSUMO = False
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Callable, Dict, List, Tuple, Optional, Any

import numpy as np
from gym.spaces import Box


def _median_headway(sorted_times: List[float], fallback: float) -> float:
    if len(sorted_times) < 2:
        return fallback
    diffs = [j - i for i, j in zip(sorted_times[:-1], sorted_times[1:])]
    diffs = [d for d in diffs if d > 0]
    if not diffs:
        return fallback
    return float(np.median(diffs))


@dataclass
class DecisionEvent:
    line_id: str
    bus_id: str
    stop_id: str
    stop_idx: int
    direction: int
    sim_time: float
    forward_headway: float
    backward_headway: float
    waiting_passengers: int
    base_stop_duration: float
    forward_bus_present: bool = True
    backward_bus_present: bool = True
    target_forward_headway: float = 360.0
    target_backward_headway: float = 360.0
    co_line_forward_headway: float = 360.0
    co_line_backward_headway: float = 360.0
    segment_mean_speed: float = 10.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    assigned_action: Optional[float] = None
    fleet_idx: int = -1   # physical-bus index (FIFO reuse); -1 means not set


class SumoBusHoldingEnv:
    """Bridge SUMO multi-line bus simulation with the SAC training loop.

    The environment expects a *decision provider* callable which advances the
    underlying simulation until a new batch of DecisionEvent objects are ready.
    Each event corresponds to one bus that just completed passenger exchange and
    is waiting for a holding decision. After the agent supplies holding times via
    ``step`` we call the *action executor* to push these decisions back to SUMO.

    The class maintains observations and rewards in a nested dictionary structure
    (line -> bus -> list_of_observations) to mirror multi-line settings.
    """

    expects_nested_actions = True

    def __init__(
        self,
        root_dir: str,
        schedule_file: str = "initialize_obj/save_obj_bus.add.xml",
        *,
        decision_provider: Optional[Callable[[], Tuple[List[DecisionEvent], bool]]] = None,
        action_executor: Optional[Callable[[DecisionEvent, float], None]] = None,
        reset_callback: Optional[Callable[[], None]] = None,
        close_callback: Optional[Callable[[], None]] = None,
        time_period_span: int = 3600,
        headway_fallback: float = 360.0,
        debug: bool = False,
        reward_type: str = "positive_step",
    ) -> None:
        self.reward_type = reward_type
        self.debug = debug
        self.root_dir = os.path.abspath(root_dir)
        self.schedule_path = schedule_file if os.path.isabs(schedule_file) else os.path.join(self.root_dir, schedule_file)
        if not os.path.exists(self.schedule_path):
            raise FileNotFoundError(f"Schedule xml not found: {self.schedule_path}")

        self.decision_provider = decision_provider
        self.action_executor = action_executor or (lambda event, action: setattr(event, "assigned_action", action))
        self.reset_callback = reset_callback
        self.close_callback = close_callback
        self.time_period_span = time_period_span
        self.headway_fallback = headway_fallback

        self.line_departures: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
        self._line_headway: Dict[str, float] = {}
        self._line_index: Dict[str, int] = {}
        self._bus_index: Dict[str, int] = {}   # trip_id → trip index (kept for compatibility)
        self._fleet_index: Dict[str, int] = {} # fleet_id string → fleet index (physical bus)
        self._station_index: Dict[Tuple[str, str], int] = {}
        self._time_period_index: Dict[int, int] = {}

        # 5 categorical features: line, bus, station, time_period, direction
        self.cat_cols = ["line_id", "bus_id", "station_id", "time_period", "direction"]
        self.continuous_features = ["forward_headway", "backward_headway", "waiting_passengers", "target_headway", "base_stop_duration", "sim_time", "gap", "co_line_forward_headway", "co_line_backward_headway", "segment_mean_speed"]

        self._state_buffers: Dict[str, Dict[str, List[List[float]]]] = defaultdict(lambda: defaultdict(list))
        self._reward_buffers: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._pending_events: Dict[Tuple[str, str], DecisionEvent] = {}
        self._done = False

        self.action_space = Box(low=np.array([0.0], dtype=np.float32), high=np.array([60.0], dtype=np.float32))

        self._load_schedule()
        self._initialize_indices()

    # region schedule & feature encoding helpers
    def _load_schedule(self) -> None:
        tree = ET.parse(self.schedule_path)
        root = tree.getroot()
        for bus_elem in root.findall(".//bus_obj"):
            line_id = bus_elem.get("belong_line_id_s")
            bus_id = bus_elem.get("bus_id_s")
            start_time = float(bus_elem.get("start_time_n", "0"))
            if line_id and bus_id:
                self.line_departures[line_id].append((start_time, bus_id))
        for line_id, entries in self.line_departures.items():
            entries.sort(key=lambda pair: pair[0])
            times = [item[0] for item in entries]
            self._line_headway[line_id] = _median_headway(times, self.headway_fallback)

    def _initialize_indices(self) -> None:
        for idx, line_id in enumerate(sorted(self.line_departures.keys())):
            self._line_index[line_id] = idx
        bus_counter = 0
        for line_id, departures in self.line_departures.items():
            for _, bus_id in departures:
                if bus_id not in self._bus_index:
                    self._bus_index[bus_id] = bus_counter
                    bus_counter += 1

    def _encode_time_period(self, sim_time: float) -> int:
        period = int(sim_time // self.time_period_span)
        if period not in self._time_period_index:
            self._time_period_index[period] = len(self._time_period_index)
        return self._time_period_index[period]

    def _encode_station(self, line_id: str, stop_id: str, stop_idx: Optional[int]) -> int:
        key = (line_id, stop_id)
        if key in self._station_index:
            return self._station_index[key]
        if stop_idx is not None and stop_idx >= 0:
            self._station_index[key] = stop_idx
            return stop_idx
        new_idx = len(self._station_index)
        self._station_index[key] = new_idx
        return new_idx
    # endregion

    # region public API
    @property
    def expects_nested_actions(self) -> bool:
        return True

    @property
    def line_codes(self) -> List[str]:
        return list(self._line_index.keys())

    @property
    def bus_codes(self) -> List[str]:
        return list(self._bus_index.keys())

    @property
    def max_agent_num(self) -> int:
        return max(len(self._bus_index), 1)

    def get_feature_spec(self) -> Dict[str, Any]:
        cat_sizes = {
            "line_id":     max(len(self._line_index), 1),
            "bus_id":      max(len(self._bus_index), 1),
            "station_id":  max(len(self._station_index), 1) or 1,
            "time_period": max(len(self._time_period_index), 1) or 1,
            "direction":   2,
        }
        return {
            "cat_cols": list(self.cat_cols),
            "cat_sizes": cat_sizes,
            "num_cont_features": len(self.continuous_features),
        }

    def reset(self) -> Tuple[Dict[str, Dict[str, List[List[float]]]], Dict[str, Dict[str, float]], bool]:
        self._state_buffers.clear()
        self._reward_buffers.clear()
        self._pending_events.clear()
        self._done = False
        if self.reset_callback is not None:
            self.reset_callback()
        self._advance_until_state()
        return self._snapshot_state(), self._snapshot_reward(), self._done

    def initialize_state(self) -> Tuple[Dict[str, Dict[str, List[List[float]]]], Dict[str, Dict[str, float]], bool]:
        if not any(self._state_buffers.values()):
            self._advance_until_state()
        return self._snapshot_state(), self._snapshot_reward(), self._done

    def step(self, action_dict: Dict[str, Dict[str, Optional[float]]]) -> Tuple[Dict, Dict, bool, Dict]:
        self._apply_actions(action_dict)
        self._advance_until_state()
        
        obs = self._snapshot_state()
        rewards = self._snapshot_reward()
        # print(f"DEBUG_STEP_EXIT: Done={self._done}", flush=True)
        return obs, rewards, self._done, {}

    def close(self) -> None:
        if self.close_callback is not None:
            self.close_callback()
        
        # Optimization: Do NOT close traci if using libsumo reuse strategy
        if not LIBSUMO:
            try:
                traci.close()
            except:
                pass

    # region core mechanics
    def _apply_actions(self, action_dict: Dict[str, Dict[str, Optional[float]]]) -> None:
        for line_id, bus_actions in action_dict.items():
            if not isinstance(bus_actions, dict):
                continue
            for bus_id, action_value in bus_actions.items():
                if action_value is None:
                    continue
                event = self._pending_events.pop((line_id, bus_id), None)
                if event is None:
                    continue
                if hasattr(action_value, '__len__') and len(action_value) >= 2:
                    raw_hold = float(action_value[0])
                    raw_speed = float(action_value[1])
                    
                    hold_value = max(0.0, raw_hold)
                    speed_value = max(0.1, raw_speed)  # Ensure speed doesn't go below 0.1 to avoid stopping indefinitely
                    
                    action_to_apply = [hold_value, speed_value]
                else:
                    # Fallback for 1D action space
                    hold_value = max(0.0, float(action_value))
                    action_to_apply = hold_value
                
                station_idx = self._encode_station(event.line_id, event.stop_id, event.stop_idx)
                # print(f"Simulation: Bus id: {event.line_id}_{event.bus_id} , station id: {station_idx} , dwelling time is: {hold_value:.4f}")

                # Pass the dynamically filtered action_to_apply
                self.action_executor(event, action_to_apply)

    def _advance_until_state(self) -> None:
        if self.decision_provider is None:
            raise RuntimeError("decision_provider callable is required to advance the environment")
        while not self._done:
            events, terminated, departed = self._pull_events()
            
            # Cleanup departed buses
            for bus_id in departed:
                for line_id in self._state_buffers:
                    if bus_id in self._state_buffers[line_id]:
                        del self._state_buffers[line_id][bus_id]
                    if bus_id in self._reward_buffers[line_id]:
                        del self._reward_buffers[line_id][bus_id]
                # Also clear pending events for this bus
                keys_to_remove = [k for k in self._pending_events if k[1] == bus_id]
                for k in keys_to_remove:
                    del self._pending_events[k]

            if terminated:
                self._done = True
            if not events:
                if self._done:
                    break
                continue
            for event in events:
                self._register_event(event)
            
            if events:
                break

    def _start_simulation(self) -> None:
        # Optimization: If traci is already loaded (e.g. by bridge or previous run), skip start
        try:
            if traci.isLoaded():
                return
        except:
            pass

        sumo_cmd = [
            "sumo-gui" if self.sumo_gui else "sumo",
            "-c", self.sumo_cfg,
            "--no-warnings",
            "--duration-log.disable",
            "--log", "/dev/null"
        ]
        
        if LIBSUMO:
            # libsumo requires argv list
            traci.start(sumo_cmd)
        else:
            traci.start(sumo_cmd)

    def _pull_events(self) -> Tuple[List[DecisionEvent], bool, List[str]]:
        result = self.decision_provider()
        terminated = False
        events: List[DecisionEvent] = []
        departed: List[str] = []
        
        if isinstance(result, tuple):
            if len(result) == 3:
                events, terminated, departed = result
            elif len(result) == 2:
                events, terminated = result
            else:
                # Fallback or error
                events = result[0]
        else:
            events = result
            
        if events is None:
            events = []
        return events, terminated, departed

    def _register_event(self, event: DecisionEvent) -> None:
        line_idx = self._line_index.setdefault(event.line_id, len(self._line_index))
        if event.bus_id not in self._bus_index:
            self._bus_index[event.bus_id] = len(self._bus_index)
        bus_idx = self._bus_index[event.bus_id]
        station_idx = self._encode_station(event.line_id, event.stop_id, event.stop_idx)
        time_period_idx = self._encode_time_period(event.sim_time)
        direction = int(event.direction)

        target_headway = self._line_headway.get(event.line_id, self.headway_fallback)
        gap = (event.target_forward_headway - event.forward_headway) if getattr(event, 'forward_bus_present', True) else 0.0
        
        obs = [
            float(line_idx),          # 0:  line_id        (cat)
            float(bus_idx),           # 1:  bus_id         (cat, trip index)
            float(station_idx),       # 2:  station_id     (cat)
            float(time_period_idx),   # 3:  time_period    (cat)
            float(direction),         # 4:  direction      (cat)
            float(event.forward_headway),
            float(event.backward_headway),
            float(event.waiting_passengers),
            float(target_headway),
            float(event.base_stop_duration),
            float(event.sim_time),
            float(gap),
            float(event.co_line_forward_headway),
            float(event.co_line_backward_headway),
            float(event.segment_mean_speed),
        ]

        reward = self._compute_reward(event, target_headway)
        state_list = self._state_buffers[event.line_id][event.bus_id]
        state_list.append(obs)
        if len(state_list) > 2:
            state_list.pop(0)
        self._reward_buffers[event.line_id][event.bus_id] = reward
        self._pending_events[(event.line_id, event.bus_id)] = event

    def _compute_reward(self, event: DecisionEvent, target_headway: float) -> float:
        """Calculate reward based on the selected reward_type."""
        # Using self.reward_type if available (defaulting to "positive_step" if not set in __init__ yet?)
        # Actually I need to update __init__ as well.
        # But this edit is just for this block.
        # I'll rely on the fact that I will update __init__ in a separate call or same if I can view it.
        # Wait, I cannot invoke multiple edits unless using multi_replace.
        # I will just use hasattr for now to be safe, defaulting to old behavior.
        
        rtype = getattr(self, "reward_type", "positive_step")
        
        if rtype == "linear_penalty":
            return self._compute_reward_linear(event, target_headway)
        else:
            return self._compute_reward_default(event, target_headway)

    def _compute_reward_linear(self, event: DecisionEvent, target_headway: float) -> float:
        """Linear negative penalty reward function (matches sac_v2_bus.py logic)."""
        def headway_reward(headway, target):
            return -abs(headway - target)

        forward_reward = headway_reward(event.forward_headway, event.target_forward_headway) if event.forward_bus_present else None
        backward_reward = headway_reward(event.backward_headway, event.target_backward_headway) if event.backward_bus_present else None

        if forward_reward is not None and backward_reward is not None:
            fwd_dev = abs(event.forward_headway - event.target_forward_headway)
            bwd_dev = abs(event.backward_headway - event.target_backward_headway)
            weight = fwd_dev / (fwd_dev + bwd_dev + 1e-6)
            
            # Use proportional deviation asymmetry for irregular schedules instead of absolute distance
            # Ensures optimal ridge follows the intended ratio, rather than straight 1:1 symmetry.
            R = event.target_forward_headway / max(event.target_backward_headway, 1e-6)
            similarity_bonus = -abs(event.forward_headway - R * event.backward_headway) * 0.5 / ((1 + R) / 2)
            reward = forward_reward * weight + backward_reward * (1 - weight) + similarity_bonus
        elif forward_reward is not None:
            reward = forward_reward
        elif backward_reward is not None:
            reward = backward_reward
        else:
            reward = -50.0

        # Replace hard threshold with smooth Tanh penalty dropping exactly like the original Q-table tests
        # Smoothly penalizes deviations beyond 50% of the target headway
        f_pen = 20.0 * np.tanh((abs(event.forward_headway - event.target_forward_headway) - 0.5 * event.target_forward_headway) / 30.0) if event.forward_bus_present and event.target_forward_headway > 0 else 0.0
        b_pen = 20.0 * np.tanh((abs(event.backward_headway - event.target_backward_headway) - 0.5 * event.target_backward_headway) / 30.0) if event.backward_bus_present and event.target_backward_headway > 0 else 0.0
        penalty_factor = f_pen + b_pen
        reward -= max(0.0, penalty_factor)
        
        # Removed empirical action penalty
        return reward

    def _compute_reward_default(self, event: DecisionEvent, target_headway: float) -> float:
        # Note: target_headway arg is kept for compatibility but we use event-specific targets
        
        def headway_reward(headway, target):
            dev = abs(headway - target)
            if dev <= 10:
                return 100.0  # Perfect match
            elif dev <= 60:
                return 5.0    # Small deviation
            else:
                return np.exp(-dev) * 10.0  # Large deviation, decaying positive reward

        forward_reward = headway_reward(event.forward_headway, event.target_forward_headway) if event.forward_bus_present else None
        backward_reward = headway_reward(event.backward_headway, event.target_backward_headway) if event.backward_bus_present else None

        if forward_reward is not None and backward_reward is not None:
            # Weight emphasizes the larger deviation (worse headway)
            fwd_dev = abs(event.forward_headway - event.target_forward_headway)
            bwd_dev = abs(event.backward_headway - event.target_backward_headway)
            weight = fwd_dev / (fwd_dev + bwd_dev + 1e-6)
            
            # Use proportional deviation asymmetry for irregular schedules instead of absolute distance
            # Ensures optimal ridge follows the intended ratio, rather than straight 1:1 symmetry.
            R = event.target_forward_headway / max(event.target_backward_headway, 1e-6)
            similarity_bonus = -abs(event.forward_headway - R * event.backward_headway) * 0.5 / ((1 + R) / 2)
            reward = forward_reward * weight + backward_reward * (1 - weight) + similarity_bonus
        elif forward_reward is not None:
            reward = forward_reward
        elif backward_reward is not None:
            reward = backward_reward
        else:
            reward = -50.0

        # Replace hard threshold with smooth Tanh penalty dropping exactly like the original Q-table tests
        # Smoothly penalizes deviations beyond 50% of the target headway
        f_pen = 20.0 * np.tanh((abs(event.forward_headway - event.target_forward_headway) - 0.5 * event.target_forward_headway) / 30.0) if event.forward_bus_present and event.target_forward_headway > 0 else 0.0
        b_pen = 20.0 * np.tanh((abs(event.backward_headway - event.target_backward_headway) - 0.5 * event.target_backward_headway) / 30.0) if event.backward_bus_present and event.target_backward_headway > 0 else 0.0
        penalty_factor = f_pen + b_pen
        reward -= max(0.0, penalty_factor)
        
        # Removed empirical action penalty
        pass
        
        # DEBUG: Print simulation-side reward calculation
        # Format: From Simulation , bus id is: ... , station id is: ... , current time: ... , reward: ...
        # Matching temp_lstm_rl/env/bus.py line 270
        # print(f"From Simulation , bus id is: {event.line_id}_{event.bus_id}, station id is: {self._encode_station(event.line_id, event.stop_id, event.stop_idx)} , current time: {event.sim_time:.1f} , reward: {reward:.4f}")
            
        return reward

    def _snapshot_state(self) -> Dict[str, Dict[str, List[List[float]]]]:
        # Return the reference to _state_buffers so the agent can modify/consume it
        # CRITICAL FIX: Only return states for buses that have a PENDING decision event.
        # This matches LSTM-RL logic where only buses in WAITING_ACTION state are returned.
        snapshot = defaultdict(dict)
        for (line_id, bus_id) in self._pending_events.keys():
            if line_id in self._state_buffers and bus_id in self._state_buffers[line_id]:
                snapshot[line_id][bus_id] = self._state_buffers[line_id][bus_id]
        return snapshot

    def _snapshot_reward(self) -> Dict[str, Dict[str, float]]:
        # Similarly, only return rewards for the current pending events
        snapshot = defaultdict(dict)
        for (line_id, bus_id) in self._pending_events.keys():
            if line_id in self._reward_buffers and bus_id in self._reward_buffers[line_id]:
                snapshot[line_id][bus_id] = self._reward_buffers[line_id][bus_id]
        return snapshot
    # endregion


__all__ = [
    "DecisionEvent",
    "SumoBusHoldingEnv",
]
