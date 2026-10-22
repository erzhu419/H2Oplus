from __future__ import annotations

from enum import Enum, auto
import numbers
import numpy as np
import random


# ── Per-line signal configuration (from SUMO network analysis) ──────────
# n_signals: number of traffic lights on the route
# green_frac: average per-direction green fraction (~0.38 means 62% chance of red)
# cycle: signal cycle length in seconds (all 90s in SUMO)
LINE_SIGNAL_CONFIG = {
    '7X':   {'n_signals': 17, 'green_frac': 0.38, 'cycle': 90},
    '7S':   {'n_signals': 17, 'green_frac': 0.39, 'cycle': 90},
    '102X': {'n_signals': 26, 'green_frac': 0.37, 'cycle': 90},
    '102S': {'n_signals': 31, 'green_frac': 0.37, 'cycle': 90},
    '122X': {'n_signals': 26, 'green_frac': 0.39, 'cycle': 90},
    '122S': {'n_signals': 26, 'green_frac': 0.39, 'cycle': 90},
    '311X': {'n_signals': 36, 'green_frac': 0.40, 'cycle': 90},
    '311S': {'n_signals': 36, 'green_frac': 0.40, 'cycle': 90},
    '406X': {'n_signals': 29, 'green_frac': 0.40, 'cycle': 90},
    '406S': {'n_signals': 30, 'green_frac': 0.39, 'cycle': 90},
    '705X': {'n_signals': 34, 'green_frac': 0.40, 'cycle': 90},
    '705S': {'n_signals': 35, 'green_frac': 0.40, 'cycle': 90},
}



class BusState(Enum):
    HOLDING = auto()
    WAITING_ACTION = auto()
    DWELLING = auto()
    TRAVEL = auto()


class Bus(object):
    def __init__(self, bus_id, trip_id, launch_time, direction, routes, stations, one_directional=False):
        self.bus_id = bus_id
        self.trip_id = trip_id
        self.sumo_trip_index = trip_id  # SUMO _bus_index value; overridden by BusSimEnv for alignment
        self.trip_id_list = [trip_id]
        self.launch_time = launch_time
        self.direction = direction
        self.one_directional = one_directional

        self.routes_list = routes
        self.stations_list = stations
        self.in_station = True
        self.passengers = []  # list of Passenger objects on bus
        self.capacity = 50 # upper bound of passengers on bus
        self.current_speed = 0. # current speed of bus

        self.trip_turn = len(self.trip_id_list)
        self.effective_station = self._compute_effective_station()
        self.last_station = self.effective_station[0] # 初始化首站
        self.next_station = self.effective_station[1] # 初始化次站
        self.last_station_dis = 0. # 上一站到当前站的距离
        self.route_index = {(route.start_stop, route.end_stop): route for route in routes} # GPT优化方案1 构建索引字典
        self.next_station_dis = self.current_route.distance # 当前站到下一站的距离
        self.absolute_distance = 0. if self.direction else len(self.stations_list) // 2 * 500 # 在上行时，绝对距离从0开始，下行时从11500开始
        self.trajectory = [] # 轨迹记录
        self.trajectory_dict = {} # 轨迹字典
        self.service_completion_history = {}
        for station in self.effective_station:
            self.trajectory_dict[station.station_name] = []
            self.service_completion_history[station.station_name] = []

        self.obs = [] # 状态值
        self.line_idx = 0  # overridden by multi-line sim; single-line default
        self.forward_bus = None # 前车对象
        self.backward_bus = None  # 后车对象
        self.forward_headway = 360. # 前车车头时距
        self.backward_headway = 360. # 后车车头时距
        self.reward = None # 奖励值

        # SUMO-style fields (rl_bridge.py compatibility)
        self.last_forward_headway: float = 360.0   # cached for backward bus to read
        self.forward_bus_present:  bool  = False
        self.backward_bus_present: bool  = False
        self.target_forward_headway:  float = 360.0
        self.target_backward_headway: float = 360.0

        self.alight_num = 0. # 下车人数
        self.board_num = 0. # 上车人数
        self.back_to_terminal_time = None
        self.terminal_arrival_time = None

        self.acceleration = 1.2  # 加速度 (SUMO bus vClass default: 1.2 m/s²)
        self.deceleration = 4.0  # 刹车加速度 (SUMO bus vClass default: 4.0 m/s²)

        # ── Signal model state ──────────────────────────────────────────
        self._signal_wait_remaining = 0.0     # remaining red-light wait (seconds)
        self._signal_stopped = False          # currently stopped at signal?
        self._segment_effective_speed = 10.0  # dynamic speed (for obs[14] alignment)
        self._backward_headway_observation_speed = 10.0
        self._backward_distance_gap = 0.0
        self._empirical_runtime_calibrated = False
        self._calibrated_travel_steps_remaining = None
        self._segment_start_time = None
        self._current_sim_time = float(launch_time)

        self.state = BusState.HOLDING  # 初始状态：在站内上下客
        self.on_route = True # 是否在路上，如果在路上，为True，否则为False，用于判断是否到达终点站

        self.holding_time = 0. # 停站时间，用于上下乘客
        self.dwelling_time = 0. # 驻站时间，用于执行动作，停车等待

        self.headway_dif = []
        self.is_unhealthy = False # False if the bus is healthy, True if the bus is unhealthy, then terminate env early

        # record of stop intervals [station_name, start_time, end_time]
        self.stop_records = []
        self._stop_start_time = None
        self._stop_station = None

        # per-stop board/alight counts — parallel lists matching stop_records
        # (lists avoid overwrite when bus visits same stop on return trip)
        self.stop_board_l:  list = []   # boardings per stop visit
        self.stop_alight_l: list = []   # alightings per stop visit
        self.stop_strand_l: list = []   # stranded per stop visit
        self.stop_load_l:   list = []   # load on departure per stop visit
        self.completed_passenger_sink = None
        self.scheduled_forward_headway = None
        self.scheduled_backward_headway = None

    @property
    def occupancy(self):
        return str(len(self.passengers)) + '/' + str(self.capacity)

    # decide if the negative or positive of step_length, when direction == 1, step_length > 0, vise versa
    @property
    def direction_int(self):
        return 1 if self.direction else -1

    def _compute_effective_station(self):
        """Return the stations this bus should visit on its current trip."""
        if self.one_directional:
            return list(self.stations_list)  # all stations, single direction
        # Legacy bi-directional: first half = forward, second half = reverse
        half = round(len(self.stations_list) / 2)
        return self.stations_list[:half] if self.direction else self.stations_list[half - 1:]

    # effective_route is effective routes for every bus, same as effective_station
    @property
    def effective_route(self):
        if self.one_directional:
            return list(self.routes_list)  # all routes, single direction
        half = round(len(self.routes_list) / 2)
        return self.routes_list[:half] if self.direction else self.routes_list[half:]

    # searching for next_station when last_station changed
    @property
    def travel_distance(self):
        return self.absolute_distance if self.direction else sum([route.distance for route in self.effective_route]) - self.absolute_distance

    def next_station_func(self):
        try:
            return self.effective_station[self.last_station.station_id + self.direction_int] if self.direction \
                   else self.effective_station[-(self.last_station.station_id + self.direction_int + 1)]
        except IndexError:
            # Bus has reached terminal on a one-directional (SUMO-calibrated) line; retire it.
            self.on_route = False
            return self.last_station   # hold at terminal — will be cleaned up by sim.py done check


    @property
    def station_after_the_next(self):
        # return the station after the next station
        return self.effective_station[self.last_station.station_id + 2 * self.direction_int] if self.direction else self.effective_station[-(self.last_station.station_id + 2 * self.direction_int + 1)]

    @property
    def station_before_the_last(self):
        # return the station before the last station
        return self.effective_station[self.last_station.station_id - 2 * self.direction_int] if self.direction else self.effective_station[-(self.last_station.station_id - 2 * self.direction_int + 1)]
    # searching for current_route when last_station and next_station changed
    # @property
    # def current_route(self):
    #     return list(filter(lambda i: i.start_stop == self.last_station.station_name and i.end_stop == self.next_station.station_name, self.effective_route))[0]

    # GPT优化方案1 构建索引字典
    @property
    def current_route(self):
        # 从字典中查找对应路段
        key = (self.last_station.station_name, self.next_station.station_name)
        if key in self.route_index:
            return self.route_index[key]
        # Fallback: after turnaround/direction flip, route key may not exist;
        # return first route in the current effective direction to avoid crash.
        eff = self.effective_route
        return eff[0] if eff else next(iter(self.route_index.values()))

    # When bus is arrived in a station, passengers have to alight and boarding.
    def exchange_passengers(self, current_time, debug, station=None):
        """Board/alight with O(n) list ops (no numpy rebuild).

        waiting_passengers is now a plain list (per station.py v2).
        self.passengers on the bus is a plain list too.
        """
        station = self.next_station if station is None else station

        # ── Alight ───────────────────────────────────────────────────────────
        keep_on_bus = []
        for pax in self.passengers:
            if pax is None:
                continue
            if pax.destination_station.station_name == station.station_name:
                pax.arrived = True
                pax.arrive_time = current_time
                self.alight_num += 1
                if self.completed_passenger_sink is not None:
                    self.completed_passenger_sink.append(pax)
            else:
                keep_on_bus.append(pax)
        self.passengers = keep_on_bus

        # ── Board ─────────────────────────────────────────────────────────────
        self.board_num += self._board_waiting_passengers(current_time, station)

        # SUMO boarding model (rl_bridge.py): base_duration = max(1.5*alight, 2.5*board) + 4.0
        # This matches the SUMO bus object's just_server_stop_data_d computation.
        self.holding_time = max(1.5 * self.alight_num, 2.5 * self.board_num) + 4.0
        # Save for obs[9] — holding_time will be decremented to 0 before obs emission
        self._base_stop_duration = self.holding_time

        # Per-stop accounting
        self.stop_board_l.append(int(self.board_num))
        self.stop_alight_l.append(int(self.alight_num))
        self.stop_strand_l.append(len(station.waiting_passengers))
        self.stop_load_l.append(len(self.passengers))
        self.alight_num = 0.
        self.board_num  = 0.

    def _board_waiting_passengers(self, current_time, station):
        """Board all waiting passengers up to the vehicle capacity."""

        remaining = []
        boarded = 0
        for pax in station.waiting_passengers:
            if pax is None:
                continue
            if len(self.passengers) >= self.capacity:
                remaining.append(pax)
                continue
            pax.boarded = True
            pax.boarding_time = current_time
            pax.travel_bus = self
            self.passengers.append(pax)
            boarded += 1
        station.waiting_passengers = remaining
        return boarded

    def _board_during_open_stop(self, current_time):
        """Mirror SUMO boarding while a stopped bus keeps its doors open."""

        if not self.in_station or not self.stop_board_l:
            return 0
        station = self.last_station
        boarded = self._board_waiting_passengers(current_time, station)
        self.stop_board_l[-1] += int(boarded)
        self.stop_strand_l[-1] = len(station.waiting_passengers)
        self.stop_load_l[-1] = len(self.passengers)
        return boarded


    def bus_update(self):
        # update the bus state
        self.last_station = self.next_station
        self.next_station = self.next_station_func()
        self.last_station_dis = 0
        self.next_station_dis = self.current_route.distance

    def drive(self, current_time, action, bus_all, debug, co_line_buses=None):
        # absolute_distance & last_station_dis is divided by 1000 as kilometers rather than meters. forward_headway & backward_headway
        # is divided by 60 minutes rather than seconds. passengers on bus, boarding passengers and alighting passengers are divided by self.capacity
        # step_length = 0, which means how long a bus moves in a time step, calculated by speeding up and original velocity.

        # Bug fix 1: bus retired at terminal — skip all processing
        if not self.on_route:
            self.obs = []
            return

        if self.state == BusState.TRAVEL:
            if self.next_station_dis <= self.current_speed:
                self.exchange_passengers(current_time, debug)  # self.holding_time is set in this function

                self.trajectory.append([self.next_station.station_name, current_time, self.absolute_distance, self.direction, self.trip_id])
                self.trajectory_dict.setdefault(self.next_station.station_name, []).append([
                    self.next_station.station_name,
                    current_time + self.holding_time + 0.01,
                    self.absolute_distance,
                    self.direction,
                    self.trip_id
                ])
                self.service_completion_history.setdefault(
                    self.next_station.station_name, []
                ).append(current_time + self.holding_time)

                self.arrive_station(current_time, bus_all, debug)
                self.state = BusState.HOLDING
                self.in_station = True
            else:
                self._advance_on_route()
        elif self.state == BusState.HOLDING:
            self._process_holding(current_time, bus_all, debug, co_line_buses)
        elif self.state == BusState.WAITING_ACTION:
            if self.obs:
                self.obs[10] = float(current_time)           # obs_sim_time
                self.obs[3]  = float(int(current_time) // 3600)  # obs_time_period
                # obs[9] = elapsed dwell so far (not remaining; holding_time is 0 here)
            self._board_during_open_stop(current_time)
            hold_time, speed_ratio = self._normalize_action(action)
            self._start_dwelling(hold_time, speed_ratio)
        elif self.state == BusState.DWELLING:
            self._process_dwelling(current_time)
        else:
            # Recover gracefully if state was not initialised as expected
            self.state = BusState.TRAVEL
            self._advance_on_route()


    def _advance_on_route(self):
        """Advance bus along the route, with signal delay model."""
        route = self.current_route
        if self._empirical_runtime_calibrated and route.runtime_calibration is not None:
            if self._calibrated_travel_steps_remaining is None:
                self._segment_start_time = getattr(
                    self, "_current_sim_time", self.launch_time
                )
                calibrated_runtime = route.sample_runtime_seconds(
                    self._segment_start_time,
                    demand_scale=getattr(self, "runtime_demand_scale", None),
                )
                self._calibrated_travel_steps_remaining = max(
                    int(round(calibrated_runtime))
                    - route.runtime_fixed_overhead_seconds,
                    1,
                )
            remaining = max(self._calibrated_travel_steps_remaining, 1)
            step_length = self.next_station_dis / remaining
            self.current_speed = float(step_length)
            self.last_station_dis += step_length
            self.next_station_dis = max(self.next_station_dis - step_length, 0.0)
            self.absolute_distance += step_length * self.direction_int
            self._calibrated_travel_steps_remaining -= 1
            alpha = 0.1
            self._segment_effective_speed = (
                (1 - alpha) * self._segment_effective_speed
                + alpha * self.current_speed
            )
            return

        # ── Signal delay check ──────────────────────────────────────
        if self._signal_wait_remaining > 0:
            self._signal_wait_remaining -= 1.0
            self.current_speed = 0.0
            self._signal_stopped = True
            return  # don't advance this tick

        if self._signal_stopped:
            self._signal_stopped = False

        # ── Normal movement ─────────────────────────────────────────
        # Apply speed_ratio to modulate the effective speed limit
        effective_speed_limit = self.current_route.speed_limit * getattr(self, '_speed_ratio', 1.0)

        if effective_speed_limit >= self.current_speed:
            if effective_speed_limit - self.current_speed > self.acceleration:
                step_length = (self.current_speed + self.acceleration / 2) * self.direction_int
                self.current_speed += self.acceleration
            else:
                step_length = (self.current_speed + effective_speed_limit) * 0.5 * self.direction_int
                self.current_speed = effective_speed_limit
        else:
            if self.current_speed - effective_speed_limit > self.deceleration:
                step_length = (self.current_speed - self.deceleration / 2) * self.direction_int
                self.current_speed -= self.deceleration
            else:
                step_length = (self.current_speed + effective_speed_limit) * 0.5 * self.direction_int
                self.current_speed = effective_speed_limit

        self.last_station_dis += abs(step_length)
        self.next_station_dis -= abs(step_length)
        self.absolute_distance += step_length

        # ── Signal encounter (stochastic) ───────────────────────────
        line_id = getattr(self, 'line_id_str', None)
        sig_cfg = LINE_SIGNAL_CONFIG.get(line_id) if line_id else None
        if sig_cfg and sig_cfg['n_signals'] > 0:
            route = self.current_route
            n_segments = getattr(self, '_n_route_segments', 24)
            sigs_this_seg = sig_cfg['n_signals'] / max(n_segments, 1)
            seg_len = max(route.distance, 1.0)
            p_encounter = sigs_this_seg * abs(step_length) / seg_len
            if random.random() < p_encounter:
                if random.random() > sig_cfg['green_frac']:
                    red_duration = sig_cfg['cycle'] * (1.0 - sig_cfg['green_frac'])
                    wait_time = random.uniform(0, red_duration)
                    self._signal_wait_remaining = wait_time
                    self.current_speed = 0.0
                    self._signal_stopped = True

        # ── Update effective speed for obs[14] ──────────────────────
        alpha = 0.1
        self._segment_effective_speed = (
            (1 - alpha) * self._segment_effective_speed
            + alpha * self.current_speed
        )

    def _process_holding(self, current_time, bus_all, debug, co_line_buses=None):
        self._board_during_open_stop(current_time)
        if self.holding_time <= 1:
            self.holding_time = 0
            self._prepare_for_action(current_time, bus_all, debug, co_line_buses)
        else:
            self.holding_time -= 1

    def _compute_co_line_headways(self, co_line_buses, seg_speed):
        """Compute co-line forward/backward headways from buses on other lines
        sharing the same station. Mirrors rl_bridge.py co-line logic.

        Parameters
        ----------
        co_line_buses : dict | None
            {station_name: [(abs_distance, speed, line_id), ...]} from all other lines
        seg_speed : float
            Current segment speed limit (m/s), used as fallback for distance→time conversion.

        Returns
        -------
        (co_fwd, co_bwd) : (float, float)
            Co-line forward and backward headways in seconds.
        """
        default_hw = self.target_forward_headway
        if co_line_buses is None:
            return default_hw, default_hw

        current_station = self.last_station
        physical_stop_id = getattr(
            current_station, "physical_stop_id", current_station.station_name
        )
        effective_speed = max(seg_speed, 1.0)
        max_cap = default_hw * 2.0

        fwd_times = []
        bwd_times = []
        if hasattr(co_line_buses, "signed_distances"):
            signed_distances = co_line_buses.signed_distances(
                physical_stop_id, getattr(self, "line_id_str", "")
            )
            for signed_distance, _other_line in signed_distances:
                time_difference = signed_distance / effective_speed
                if time_difference > 0:
                    bwd_times.append(time_difference)
                elif time_difference < 0:
                    fwd_times.append(-time_difference)
            co_fwd = min(min(fwd_times), max_cap) if fwd_times else default_hw
            co_bwd = min(min(bwd_times), max_cap) if bwd_times else default_hw
            return co_fwd, co_bwd

        station_name = current_station.station_name
        if station_name not in co_line_buses:
            return default_hw, default_hw
        my_pos = self.absolute_distance
        for other_pos, other_speed, other_line in co_line_buses[station_name]:
            if other_line == getattr(self, "line_id_str", None):
                continue
            if other_speed == 0.0:
                # Direct-time mode (from VirtualCoLineScheduler):
                # other_pos = time_diff in seconds
                #   positive → co-line bus is behind (backward)
                #   negative → co-line bus is ahead (forward)
                time_diff = other_pos
                if time_diff > 0:
                    bwd_times.append(time_diff)
                elif time_diff < 0:
                    fwd_times.append(-time_diff)
            else:
                # Distance mode (from MultiLineEnv): compute time from spatial gap
                dist_diff = my_pos - other_pos
                div_speed = max(other_speed, effective_speed)
                if dist_diff > 0:
                    bwd_times.append(dist_diff / div_speed)
                elif dist_diff < 0:
                    fwd_times.append(-dist_diff / div_speed)

        co_fwd = min(min(fwd_times), max_cap) if fwd_times else default_hw
        co_bwd = min(min(bwd_times), max_cap) if bwd_times else default_hw
        return co_fwd, co_bwd

    def _find_neighbors(self, bus_all):
        """SUMO-style neighbor detection: sort on-route buses by launch_time,
        then take the bus immediately before (forward) and after (backward) in
        the sequence.  Replaces the legacy trip_id±2 filter which only worked
        for the original single-line bidirectional scenario.

        Returns (forward_bus, backward_bus) — either may be None.
        """
        # Retain completed predecessors so their service-completion records
        # remain observable. Future trips do not yet exist in this object list.
        ordered = sorted(bus_all, key=lambda b: b.launch_time)
        try:
            idx = ordered.index(self)
        except ValueError:
            return None, None
        fwd = ordered[idx - 1] if idx > 0 else None
        bwd = ordered[idx + 1] if idx < len(ordered) - 1 else None
        return fwd, bwd

    def _compute_reward_linear(self) -> float:
        """SUMO-style linear-penalty reward matching rl_env.py _compute_reward_linear."""
        def headway_reward(hw, target):
            return -abs(hw - target)

        fwd_r = headway_reward(self.forward_headway, self.target_forward_headway)  if self.forward_bus_present  else None
        bwd_r = headway_reward(self.backward_headway, self.target_backward_headway) if self.backward_bus_present else None

        if fwd_r is not None and bwd_r is not None:
            fwd_dev = abs(self.forward_headway  - self.target_forward_headway)
            bwd_dev = abs(self.backward_headway - self.target_backward_headway)
            weight  = fwd_dev / (fwd_dev + bwd_dev + 1e-6)
            R = self.target_forward_headway / max(self.target_backward_headway, 1e-6)
            similarity_bonus = -abs(self.forward_headway - R * self.backward_headway) * 0.5 / ((1 + R) / 2)
            reward = fwd_r * weight + bwd_r * (1 - weight) + similarity_bonus
        elif fwd_r is not None:
            reward = fwd_r
        elif bwd_r is not None:
            reward = bwd_r
        else:
            return None  # isolated bus — no obs/reward

        # Smooth large-deviation penalty (matches SUMO rl_env.py)
        f_pen = (20.0 * np.tanh((abs(self.forward_headway  - self.target_forward_headway)  - 0.5 * self.target_forward_headway)  / 30.0)
                 if self.forward_bus_present  and self.target_forward_headway  > 0 else 0.0)
        b_pen = (20.0 * np.tanh((abs(self.backward_headway - self.target_backward_headway) - 0.5 * self.target_backward_headway) / 30.0)
                 if self.backward_bus_present and self.target_backward_headway > 0 else 0.0)
        reward -= max(0.0, f_pen + b_pen)
        return reward

    def _prepare_for_action(self, current_time, bus_all, debug, co_line_buses=None):
        # Neighbors and headways are already computed in arrive_station().
        # Emit at every non-terminal physical stop when a scheduled neighbour
        # is observable. This includes the route origin, matching SUMO stop_idx 0.
        if (
            self.on_route
            and self.last_station in self.effective_station[:-1]
            and (self.forward_bus_present or self.backward_bus_present)
        ):

            _line_idx  = getattr(self, 'line_idx', 0)
            _line_hw   = getattr(self, 'line_headway', 360.0)  # static median (matches SUMO _line_headway)
            # gap uses DYNAMIC per-pair target (matches SUMO rl_env / checkpoint training)
            _gap       = (self.target_forward_headway - self.forward_headway
                          if self.forward_bus_present else 0.0)
            _seg_speed = self.last_station.backward_headway_observation_speed(
                current_time
            )
            _co_fwd, _co_bwd = self._compute_co_line_headways(co_line_buses, _seg_speed)

            self.obs = [
                float(_line_idx),                                    # 0: line_id (categorical)
                float(self.sumo_trip_index),                             # 1: bus_id (SUMO trip index — matches checkpoint embedding)
                float(self.last_station.station_id),                  # 2: station_id (SUMO stop_idx: 7X01=0..7X25=24)
                float(int(current_time) // 3600),                    # 3: time_period (categorical)
                float(0 if self.direction else 1),                  # 4: direction (SUMO: 7X=0, 7S=1; Sim: up=True→0)
                float(self.forward_headway),                         # 5: forward_headway (s)
                float(self.backward_headway),                        # 6: backward_headway (s)
                float(len(self.last_station.waiting_passengers)),    # 7: unserved passengers at current stop
                float(_line_hw),                                     # 8: target_headway (static median, matches SUMO)
                float(getattr(self, '_base_stop_duration', 0.0)),    # 9: base_stop_duration (passenger exchange time)
                float(current_time),                                 # 10: sim_time (s)
                float(_gap),                                         # 11: gap = static_target - fwd_hw
                float(_co_fwd),                                      # 12: co_line_fwd_hw
                float(_co_bwd),                                      # 13: co_line_bwd_hw
                float(_seg_speed),                                    # 14: segment_mean_speed (m/s)
            ]

            reward = self._compute_reward_linear()
            self.reward = reward  # may be None if isolated (shouldn't happen given guard above)

        self.state = BusState.WAITING_ACTION

    def _start_dwelling(self, hold_time, speed_ratio=1.0):
        """Begin dwelling phase with optional speed ratio for inter-stop travel."""
        # Store speed_ratio for _advance_on_route to use
        self._speed_ratio = speed_ratio

        if hold_time is None:
            dwell_time = None
        else:
            dwell_time = hold_time

        if (self.trip_id in [0, 1] and hold_time is None) or dwell_time == 0:
            self.dwelling_time = 0
        else:
            self.dwelling_time = dwell_time

        self.state = BusState.DWELLING

    def _process_dwelling(self, current_time):
        self._board_during_open_stop(current_time)
        if self.dwelling_time is None or self.dwelling_time <= 1:
            self.in_station = False
            if self._stop_start_time is not None:
                self.stop_records.append([
                    self._stop_station,
                    self._stop_start_time,
                    current_time
                ])
                self._stop_start_time = None
                self._stop_station = None
            self.dwelling_time = 0
            self._current_sim_time = float(current_time)
            self._segment_start_time = float(current_time)
            self._calibrated_travel_steps_remaining = None
            self.state = BusState.TRAVEL
        else:
            self.dwelling_time -= 1

    def _normalize_action(self, action):
        """Parse action into (hold_time, speed_ratio).

        Supports:
            - None → (None, 1.0)
            - scalar → (float, 1.0)
            - [hold, speed] → (float, float)
            - np.ndarray shape (2,) → (float, float)
        """
        if action is None:
            return None, 1.0

        # Handle 2D action (list, tuple, or ndarray with >= 2 elements)
        if isinstance(action, np.ndarray):
            if action.size == 0:
                return None, 1.0
            flat = action.reshape(-1)
            hold = float(flat[0])
            speed = float(flat[1]) if flat.size >= 2 else 1.0
            return hold, max(0.1, speed)

        if isinstance(action, (list, tuple)):
            if not action:
                return None, 1.0
            hold = float(action[0]) if action[0] is not None else None
            speed = float(action[1]) if len(action) >= 2 else 1.0
            return hold, max(0.1, speed)

        if isinstance(action, numbers.Number):
            return float(action), 1.0

        if hasattr(action, 'item'):
            try:
                return float(action.item()), 1.0
            except (TypeError, ValueError):
                return None, 1.0

        try:
            return float(action), 1.0
        except (TypeError, ValueError):
            return None, 1.0

    def _update_headways_at_station(
        self, current_time, bus_all, station, service_completion_time
    ):
        """Update causal SUMO-style headways for one physical stop event."""

        station_name = station.station_name
        forward_bus, backward_bus = self._find_neighbors(bus_all)
        self.forward_bus = [forward_bus] if forward_bus else []
        self.backward_bus = [backward_bus] if backward_bus else []

        default_headway = getattr(self, "line_headway", 360.0)
        self.target_forward_headway = float(
            self.scheduled_forward_headway
            if self.scheduled_forward_headway is not None
            else default_headway
        )
        self.target_backward_headway = float(
            self.scheduled_backward_headway
            if self.scheduled_backward_headway is not None
            else default_headway
        )

        forward_cap = max(2400.0, 3.0 * self.target_forward_headway)
        forward_records = (
            forward_bus.service_completion_history.get(station_name, [])
            if forward_bus is not None
            else []
        )
        self.forward_bus_present = bool(forward_records)
        if self.forward_bus_present:
            headway = service_completion_time - forward_records[-1]
            self.forward_headway = max(0.0, min(headway, forward_cap))
        else:
            self.forward_headway = self.target_forward_headway
        self.last_forward_headway = self.forward_headway

        backward_cap = max(2400.0, 3.0 * self.target_backward_headway)
        self.backward_bus_present = bool(
            backward_bus is not None
            and backward_bus.on_route
            and backward_bus.launch_time <= current_time
        )
        effective_speed = station.backward_headway_observation_speed(current_time)
        if self.backward_bus_present:
            distance_gap = max(
                self.travel_distance - backward_bus.travel_distance, 0.0
            )
            self._backward_distance_gap = float(distance_gap)
            self._backward_headway_observation_speed = float(effective_speed)
            self.backward_headway = min(
                distance_gap / effective_speed, backward_cap
            )
        else:
            self._backward_distance_gap = 0.0
            self._backward_headway_observation_speed = float(effective_speed)
            self.backward_headway = self.target_backward_headway

    def initialize_origin_headways(self, current_time, bus_all):
        """Initialize the first-stop event after base passenger service."""

        self._update_headways_at_station(
            current_time,
            bus_all,
            self.last_station,
            current_time + self.holding_time,
        )

    def arrive_station(self, current_time, bus_all, debug):
        """Called when bus arrives at a stop.

        Headways use the same causal contract as the SUMO bridge: the forward
        value compares base-service completion times at this stop, while the
        backward value estimates the active following bus's time gap from its
        observable position. Timetable gaps define the two targets.
        """
        self.current_speed = 0
        self._stop_start_time = current_time
        self._stop_station = self.next_station.station_name

        # time when passenger exchange finishes (matches SUMO service_completion_time)
        service_completion_time = current_time + self.holding_time
        self._update_headways_at_station(
            current_time, bus_all, self.next_station, service_completion_time
        )

        # ── Update position ────────────────────────────────────────────────────
        self.absolute_distance += self.next_station_dis * self.direction_int

        if self.next_station.station_type == 0 and self.on_route:
            # Terminal: retire this bus
            self.on_route = False
            self.terminal_arrival_time = float(current_time)
            self.back_to_terminal_time = current_time
            self.last_station = self.effective_station[-1]
            if self.one_directional:
                # One-directional (SUMO): bus retires permanently, no turnaround
                # Keep back_to_terminal_time = None so launch_bus() never reuses it
                self.back_to_terminal_time = None
            else:
                # Bi-directional: flip direction for return trip
                self.direction = int(not self.direction)
                self.effective_station = self._compute_effective_station()
                self.next_station = self.next_station_func()
        else:
            station_id = self.last_station.station_id + 1 if self.direction else self.last_station.station_id - 1
            self.headway_dif.append([self.forward_headway - self.backward_headway, station_id])
            self.bus_update()

    # When a bus is re-launched from terminal, we have to reset the bus like a new bus we created, which means
    # we have to reset many attribute of the bus, then we add the trip_id to the trip history list. absolute_distance is 0
    # if it begins from terminal up, rather than 11500 if it begins from terminal down.

    def reset_bus(self, trip_num, launch_time):
        self.trip_id = trip_num
        self.sumo_trip_index = trip_num  # default; BusSimEnv can override
        self.trip_id_list.append(trip_num)
        self.launch_time = launch_time
        self.last_station = self.effective_station[0]
        self.next_station = self.effective_station[1]  # sync after turnaround

        self.forward_headway  = 360.0
        self.backward_headway = 360.0
        self.last_forward_headway    = None   # reset cache
        self.forward_bus_present     = False
        self.backward_bus_present    = False
        self.target_forward_headway  = 360.0
        self.target_backward_headway = 360.0
        self._backward_headway_observation_speed = 10.0
        self._backward_distance_gap = 0.0
        self.scheduled_forward_headway = None
        self.scheduled_backward_headway = None

        self.last_station_dis = 0.
        self.next_station_dis = self.current_route.distance
        self._segment_start_time = None
        self._calibrated_travel_steps_remaining = None
        if self.one_directional:
            self.absolute_distance = 0.
        else:
            self.absolute_distance = 0. if self.direction else len(self.stations_list) // 2 * 500

        self.passengers = []
        self.current_speed = 0.
        self.holding_time = 0.
        self._segment_start_time = None
        self._calibrated_travel_steps_remaining = None
        self.back_to_terminal_time = None
        self.terminal_arrival_time = None
        self.board_num = 0.
        self.alight_num = 0.
        self.in_station = False
        self.forward_bus = None
        self.backward_bus = None
        self.reward = None
        self.obs = []

        self.state = BusState.TRAVEL
        self.on_route = True
        self.trip_turn = len(self.trip_id_list)
        self.is_unhealthy = False

        # Reinit trajectory_dict for the new trip's effective stations
        self.trajectory_dict.clear()
        self.service_completion_history.clear()
        for station in self.effective_station:
            self.trajectory_dict[station.station_name] = []
            self.service_completion_history[station.station_name] = []
