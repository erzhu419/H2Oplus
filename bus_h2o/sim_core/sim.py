from __future__ import annotations

import atexit
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
from multiprocessing.util import Finalize

import numpy as np
from sim_core.timetable import Timetable
from sim_core.bus import Bus, BusState
from sim_core.route import Route
from sim_core.station import Station
from sim_core.visualize import visualize
import pandas as pd
from gymnasium.spaces import Box, MultiDiscrete
import copy
import xml.etree.ElementTree as ET


_LINE_RUNTIME_ROOTS: dict[tuple[int, str], str] = {}
_LINE_RUNTIME_FINALIZERS: dict[tuple[int, str], Finalize] = {}
_LINE_RUNTIME_LOCK = threading.RLock()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_owned_runtime_root(root: str, owner_pid: int) -> None:
    if os.getpid() == owner_pid:
        shutil.rmtree(root, ignore_errors=True)


def _line_runtime_root(base_path: str) -> str:
    owner_pid = os.getpid()
    key = (owner_pid, os.path.realpath(base_path))
    with _LINE_RUNTIME_LOCK:
        root = _LINE_RUNTIME_ROOTS.get(key)
        if root is None or not os.path.isdir(root):
            root = tempfile.mkdtemp(prefix=f"h2oplus-line-env-{owner_pid}-")
            _LINE_RUNTIME_ROOTS[key] = root
            _LINE_RUNTIME_FINALIZERS[key] = Finalize(
                None,
                _remove_owned_runtime_root,
                args=(root, owner_pid),
                exitpriority=10,
            )
            atexit.register(_remove_owned_runtime_root, root, owner_pid)
        return root


def _install_read_only_link(source: str, destination: str) -> None:
    source = os.path.realpath(source)
    if os.path.islink(destination) and os.path.realpath(destination) == source:
        return
    if os.path.lexists(destination):
        if os.path.isdir(destination) and not os.path.islink(destination):
            shutil.rmtree(destination)
        else:
            os.unlink(destination)
    os.symlink(source, destination)

try:
    from bus_h2o.passenger_time import (
        PassengerTimeLedger,
        aggregate_passenger_time_snapshots,
    )
except ModuleNotFoundError:
    from passenger_time import (
        PassengerTimeLedger,
        aggregate_passenger_time_snapshots,
    )

try:
    from bus_h2o.sim_core.journey import JourneyPassenger
except ModuleNotFoundError:
    from sim_core.journey import JourneyPassenger


class PhysicalCoLineContext:
    """Snapshot of other-line positions relative to shared physical stops."""

    def __init__(self, stop_positions, buses_by_line):
        self.stop_positions = stop_positions
        self.buses_by_line = buses_by_line

    def signed_distances(self, physical_stop_id, own_line_id):
        distances = []
        for line_id, stop_position in self.stop_positions.get(
            physical_stop_id, {}
        ).items():
            if line_id == own_line_id:
                continue
            for bus_position in self.buses_by_line.get(line_id, ()):
                distances.append((stop_position - bus_position, line_id))
        return distances


class env_bus(object):

    # ── Class-level static data cache ─────────────────────────────────────
    # Keyed by absolute data-dir path → {od, station_set, routes_set,
    # timetable_set, args, effective_station_name, effective_period}.
    # Survives across reset() and across multiple env_bus instances that
    # share the same underlying data directory (e.g. multi-line envs).
    _DATA_CACHE: dict = {}

    def __init__(self, path, debug=False, render=False):
        if render:
            # pygame.init()
            pass

        self.path = path
        pass  # H2Oplus: path managed by BusSimEnv

        data_dir = os.path.abspath(os.path.join(path, 'data'))
        if data_dir not in env_bus._DATA_CACHE:
            # First time: read from disk and cache
            config_path = os.path.join(path, 'config.json')
            with open(config_path, 'r') as f:
                args = json.load(f)

            _od_sumo = os.path.join(path, "data/passenger_OD_sumo.xlsx")
            _od_orig = os.path.join(path, "data/passenger_OD.xlsx")
            _od_file = _od_sumo if os.path.exists(_od_sumo) else _od_orig
            od           = pd.read_excel(_od_file, index_col=[1, 0])
            station_set  = pd.read_excel(os.path.join(path, "data/stop_news.xlsx"))
            routes_set   = pd.read_excel(os.path.join(path, "data/route_news.xlsx"))
            _timetable_sumo = os.path.join(path, "data/time_table_sumo.xlsx")
            _timetable_orig = os.path.join(path, "data/time_table.xlsx")
            timetable_set = pd.read_excel(
                _timetable_sumo if os.path.exists(_timetable_sumo) else _timetable_orig
            )
            timetable_set = timetable_set.sort_values(
                by=['launch_time', 'direction']
            ).reset_index(drop=True)
            timetable_set['launch_turn'] = range(timetable_set.shape[0])

            eff_station = sorted(set(od.index[i][0] for i in range(od.shape[0])))
            eff_period  = sorted(list(set(od.index[i][1] for i in range(od.shape[0]))))

            env_bus._DATA_CACHE[data_dir] = {
                'args':         args,
                'od':           od,
                'station_set':  station_set,
                'routes_set':   routes_set,
                'timetable_set': timetable_set,
                'effective_station_name': eff_station,
                'effective_period':       eff_period,
            }

        # Pull from cache (zero I/O after first load)
        cached = env_bus._DATA_CACHE[data_dir]
        self.args                   = cached['args']
        self.od                     = cached['od']
        self.station_set            = cached['station_set']
        self.routes_set             = cached['routes_set']
        self.timetable_set          = cached['timetable_set']
        self.effective_station_name = cached['effective_station_name']
        self.effective_period       = cached['effective_period']

        self.effective_trip_num = len(self.timetable_set)
        self.time_step = self.args["time_step"]
        self.passenger_update_freq = self.args["passenger_state_update_freq"]
        # Max simulation time — matches SUMO bridge max_steps (default 18000s = 5h)
        self.max_time = self.args.get("max_time", 18000)
        # The calibrated SUMO demand contains departures only in [0, 18000).
        # Later simulation time is a clearing period, not another demand hour.
        self.passenger_arrival_end_time = self.args.get(
            "passenger_arrival_end_time", 18000
        )
        self.demand_scale = 1.0
        self.central_journey_demand = False
        # Bus IDs are allocated monotonically whenever no returned vehicle is
        # available.  A fixed cap of 25 therefore fails on longer service days
        # even when fewer than 25 vehicles are concurrently active.  The number
        # of scheduled trips is the strict finite upper bound for one episode.
        self.max_agent_num = self.effective_trip_num

        self.visualizer = visualize(self)
        self.enable_plot = True

        self.action_space = Box(
            low=np.array([0.0, 0.1], dtype=np.float32),
            high=np.array([60.0, 10.0], dtype=np.float32),
        )

        self.runtime_segment_calibrations = None
        self.runtime_fixed_overhead_seconds = 0

        if debug:
            self.summary_data = pd.DataFrame(columns=['bus_id', 'station_id', 'trip_id', 'abs_dis', 'forward_headway',
                                                  'backward_headway', 'headway_diff', 'time'])
            self.summary_reward = pd.DataFrame(columns=['bus_id', 'station_id', 'trip_id', 'forward_headway',
                                                    'backward_headway', 'reward', 'time'])

        self.stations = self.set_stations()
        self.routes = self.set_routes()
        self.timetables = self.set_timetables()

        self.state_dim = 15  # aligned with SUMO rl_env.py (15-dim obs)
        self.passenger_time_ledger = PassengerTimeLedger()
        self.last_step_passenger_time = self.passenger_time_ledger.snapshot()
        self.completed_passengers = []

    @property
    def bus_in_terminal(self):
        return [bus for bus in self.bus_all if not bus.on_route]

    # @property
    # def bus_on_route(self):
    #     return [bus for bus in self.bus_all if bus.on_route]

    def set_timetables(self):
        return [Timetable(self.timetable_set['launch_time'][i], self.timetable_set['launch_turn'][i], self.timetable_set['direction'][i]) for i in range(self.timetable_set.shape[0])]

    def set_routes(self):
        routes = [
            Route(self.routes_set['route_id'][i], self.routes_set['start_stop'][i], self.routes_set['end_stop'][i],
                  self.routes_set['distance'][i], self.routes_set['V_max'][i], self.routes_set.iloc[i, 5:]) for i in
            range(self.routes_set.shape[0])]
        if self.runtime_segment_calibrations is not None:
            missing = []
            for route in routes:
                key = f"{route.start_stop}->{route.end_stop}"
                calibration = self.runtime_segment_calibrations.get(key)
                if calibration is None:
                    missing.append(key)
                    continue
                route.install_runtime_calibration(
                    calibration, self.runtime_fixed_overhead_seconds
                )
            if missing:
                raise ValueError(
                    f"runtime calibration lacks {self.line_id_str} segments {missing}"
                )
        return routes

    def install_runtime_calibration(self, line_calibration, fixed_overhead_seconds):
        self.runtime_segment_calibrations = line_calibration["segments"]
        self.runtime_fixed_overhead_seconds = int(fixed_overhead_seconds)
        self.routes = self.set_routes()
        self.route_index = {
            (route.start_stop, route.end_stop): route for route in self.routes
        }

    def set_stations(self):
        # Detect one-directional lines (all timetable entries direction==1, SUMO-calibrated lines).
        # For these, do NOT mirror the stop list; buses only travel forward and are retired at terminal.
        all_dirs = set(self.timetable_set['direction'].unique())
        one_directional = (all_dirs == {1})
        self.one_directional = one_directional

        if one_directional:
            # SUMO-style: one-way only — stations listed once, all direction=True
            station_concat = self.station_set.reset_index()
            half = len(station_concat)  # all are forward direction
        else:
            station_concat = pd.concat([self.station_set, self.station_set[::-1][1:]]).reset_index()
            half = station_concat.shape[0] / 2

        total_station = []
        for idx, station in station_concat.iterrows():
            # station type is 0 if Terminal else 1
            terminal_names = ['Terminal_up', 'Terminal_down',
                              station_concat.iloc[0]['stop_name'],    # first stop = upstream terminal
                              station_concat.iloc[-1]['stop_name']]   # last stop = downstream terminal
            station_type = 1 if station['stop_name'] not in terminal_names else 0

            direction = False if idx >= half else True
            od = None
            if station['stop_name'] in self.effective_station_name:
                try:
                    od = self.od.loc[station['stop_name'], station['stop_name']:] if direction \
                         else self.od.loc[station['stop_name'], :station['stop_name']]
                    od.index = od.index.map(str)
                    od = od.to_dict(orient='index')
                except Exception:
                    od = None  # guard against OD slice errors for edge stops

            total_station.append(Station(station_type, station['stop_id'], station['stop_name'], direction, od))

        return total_station

    # return default state and reward
    def reset(self):

        self.current_time = 0
        self.passenger_time_ledger.reset(self.current_time)
        self.last_step_passenger_time = self.passenger_time_ledger.snapshot()
        self.completed_bus_ids_this_step = []
        self.completed_passengers = []
        self.last_episode_passengers = ()
        self.last_episode_bus_summaries = ()

        # initialize station, routes and timetables
        self.stations = self.set_stations()
        self.routes = self.set_routes()
        self.timetables = self.set_timetables()

        # initial list of bus on route
        self.bus_id = 0
        self.bus_all = []
        self.route_state = []

        # self.state is combine with route_state, which contains the route.speed_limit of each route, station_state, which
        # contains the station.waiting_passengers of each station and bus_state, which is bus.obs for each bus.
        self.state = {key: [] for key in range(self.max_agent_num)}
        self.reward = {key: 0 for key in range(self.max_agent_num)}
        self.done = False

        # Compute static median headway from timetable (matches SUMO rl_env._line_headway)
        launch_times = sorted(t.launch_time for t in self.timetables)
        if len(launch_times) >= 2:
            diffs = [b - a for a, b in zip(launch_times[:-1], launch_times[1:]) if b > a]
            self.line_headway = float(np.median(diffs)) if diffs else 360.0
        else:
            self.line_headway = 360.0
        # Passenger arrival pre-computation cache
        self._pax_cache_hour = -1
        self._pax_flat_rates = np.array([], dtype=np.float64)
        self._pax_flat_map = []  # [(station_obj, dest_obj), ...]

        self.action_dict = {key: None for key in list(range(self.max_agent_num))}

    def _rebuild_pax_cache(self, effective_hour):
        """Rebuild flat rate array for batched Poisson across all stations."""
        rates = []
        mapping = []
        for station in self.stations:
            if station.od is None:
                continue
            col_key = f"{effective_hour:02d}:00:00"
            if col_key not in station.od:
                continue
            period_od = station.od[col_key]
            for dest_name, demand in period_od.items():
                if demand <= 0:
                    continue
                dest = next(
                    (x for x in self.stations
                     if x.station_name == dest_name and x.direction == station.direction),
                    None,
                )
                if dest is None:
                    continue
                rates.append(self.demand_scale * demand / 3600.0)
                mapping.append((station, dest))
        self._pax_cache_hour = effective_hour
        self._pax_flat_rates = np.array(rates, dtype=np.float64) if rates else np.array([], dtype=np.float64)
        self._pax_flat_map = mapping

    def _batch_passenger_arrival(self, current_time, update_interval):
        """ONE Poisson call for all stations × destinations."""
        from sim_core.passenger import Passenger

        if self.central_journey_demand:
            return
        if current_time >= self.passenger_arrival_end_time:
            return

        hour_offset = int(current_time) // 3600
        effective_hour = max(6, min(6 + hour_offset, 19))

        if self._pax_cache_hour != effective_hour:
            self._rebuild_pax_cache(effective_hour)

        if len(self._pax_flat_rates) == 0:
            return

        # Single batched Poisson call — replaces N separate calls
        # Single batched Poisson call — replaces N separate calls
        arrivals = np.random.poisson(self._pax_flat_rates * update_interval)

        for n_arrive, (station, dest) in zip(arrivals, self._pax_flat_map):
            if n_arrive == 0:
                continue
            new_pax = [Passenger(current_time, station, dest) for _ in range(int(n_arrive))]
            station.waiting_passengers.extend(new_pax)
            station.total_passenger.extend(new_pax)

    def initialize_state(self, render=False):
        def count_non_empty_sublist(lst):
            return sum(1 for sublist in lst if sublist)

        while count_non_empty_sublist(list(self.state.values())) == 0:
            self.state, self.reward, _ = self.step(self.action_dict, render=render)

        return self.state, self.reward, self.done

    def launch_bus(self, trip):
        # Trip set(self.timetable) contain both direction trips. So we have to make sure the direction and launch time
        # is satisfied before the trip launched.
        # For one-directional SUMO lines, buses retire at terminal (back_to_terminal_time=None); never reuse them.
        candidates = list(filter(
            lambda i: i.direction == trip.direction and i.back_to_terminal_time is not None,
            self.bus_in_terminal
        ))
        if len(candidates) == 0:
            # No reusable bus available — create a new one
            bus = Bus(self.bus_id, trip.launch_turn, trip.launch_time, trip.direction, self.routes, self.stations,
                      one_directional=self.one_directional)
            self.bus_all.append(bus)
            self.bus_id += 1
        else:
            # Reuse the bus that returned to terminal earliest
            bus = sorted(candidates, key=lambda b: b.back_to_terminal_time)[0]
            bus.reset_bus(trip.launch_turn, trip.launch_time)
            bus.on_route = True
        # Inject static line headway (matches SUMO rl_env._line_headway)
        bus.line_headway = getattr(self, 'line_headway', 360.0)
        bus.runtime_demand_scale = float(self.demand_scale)
        ordered_trips = sorted(self.timetables, key=lambda item: item.launch_time)
        schedule_index = ordered_trips.index(trip)
        bus.scheduled_forward_headway = (
            float(trip.launch_time - ordered_trips[schedule_index - 1].launch_time)
            if schedule_index > 0
            else None
        )
        bus.scheduled_backward_headway = (
            float(ordered_trips[schedule_index + 1].launch_time - trip.launch_time)
            if schedule_index + 1 < len(ordered_trips)
            else None
        )
        # Inject SUMO line index (matches rl_env._line_index; 7X=11)
        bus.line_idx = getattr(self, 'line_idx', 0)
        # Inject line_id string for signal model lookup
        bus.line_id_str = getattr(self, 'line_id_str', None)
        # Inject number of route segments for signal density calculation
        bus._n_route_segments = len(self.routes)
        bus._empirical_runtime_calibrated = bool(
            getattr(self, "empirical_runtime_calibrated", False)
        )
        bus.completed_passenger_sink = self.completed_passengers
        bus.state = BusState.HOLDING
        bus.in_station = True
        bus._stop_start_time = self.current_time
        bus._stop_station = bus.last_station.station_name
        bus.exchange_passengers(
            self.current_time, debug=False, station=bus.last_station
        )
        bus.trajectory.append(
            [
                bus.last_station.station_name,
                self.current_time,
                bus.absolute_distance,
                bus.direction,
                bus.trip_id,
            ]
        )
        bus.trajectory_dict.setdefault(bus.last_station.station_name, []).append(
            [
                bus.last_station.station_name,
                self.current_time + bus.holding_time + 0.01,
                bus.absolute_distance,
                bus.direction,
                bus.trip_id,
            ]
        )
        bus.service_completion_history.setdefault(
            bus.last_station.station_name, []
        ).append(self.current_time + bus.holding_time)
        bus.initialize_origin_headways(self.current_time, self.bus_all)


    def restore_full_system_snapshot(self, snapshot: dict) -> None:
        """
        God-mode state injection.  Overwrites the current simulation state
        with the given SnapshotDict.  Must be called after reset() so
        objects are properly initialised before being overwritten.
        """
        self.current_time = float(snapshot["current_time"])

        # Mark timetable entries as launched and pre-launch buses
        launched_set = set(snapshot.get("launched_trips", []))
        for i, t in enumerate(self.timetables):
            t.launched = (i in launched_set)
            if i in launched_set:
                self.launch_bus(t)

        bus_by_id = {b.bus_id: b for b in self.bus_all}
        station_by_id = {s.station_id: s for s in self.stations}

        restored_bus_ids = set()
        for bd in snapshot["all_buses"]:
            bid = bd["bus_id"]
            if bid in bus_by_id:
                bus = bus_by_id[bid]
            else:
                unrestored = [b for b in self.bus_all if b.bus_id not in restored_bus_ids]
                if unrestored:
                    bus = unrestored[0]
                else:
                    continue

            bus.trip_id            = bd["trip_id"]
            bus.trip_id_list       = list(bd.get("trip_id_list", [bd["trip_id"]]))
            # Inject SUMO trip index for embedding alignment (Phase 3 H2O+)
            bus.sumo_trip_index    = bd.get("sumo_trip_index", bd["trip_id"])
            bus.direction          = bool(bd["direction"])
            bus.absolute_distance  = float(bd["absolute_distance"])
            bus.current_speed      = float(bd["current_speed"])
            bus.holding_time       = float(bd["holding_time"])
            bus.forward_headway    = float(bd["forward_headway"])
            bus.backward_headway   = float(bd["backward_headway"])
            bus.next_station_dis   = float(bd["next_station_dis"])
            bus.last_station_dis   = float(bd["last_station_dis"])
            bus.on_route           = bool(bd["on_route"])

            lst_id = bd["last_station_id"]
            nxt_id = bd["next_station_id"]
            if lst_id in station_by_id:
                bus.last_station = station_by_id[lst_id]
            if nxt_id in station_by_id:
                bus.next_station = station_by_id[nxt_id]

            state_name = bd.get("state", "TRAVEL")
            try:
                bus.state = BusState[state_name]
            except KeyError:
                bus.state = BusState.TRAVEL

            target_load = int(bd.get("load", 0))
            current_load = len(bus.passengers)
            if current_load > target_load:
                bus.passengers = bus.passengers[:target_load]
            elif current_load < target_load:
                padding = np.array([None] * (target_load - current_load), dtype=object)
                bus.passengers = np.concatenate([bus.passengers, padding])

            bus.in_station = bus.state in (
                BusState.HOLDING, BusState.WAITING_ACTION, BusState.DWELLING)
            restored_bus_ids.add(bus.bus_id)

        station_snap = {sd["station_id"]: sd for sd in snapshot.get("all_stations", [])}
        for st in self.stations:
            if st.station_id in station_snap:
                count = int(station_snap[st.station_id]["waiting_count"])
                if len(st.waiting_passengers) != count:
                    st.waiting_passengers = [None] * count

    def step(self, action, debug=False, render=False, episode=0, co_line_buses=None):
        was_done = self.done
        self.completed_bus_ids_this_step = []
        # Reset per-step state so only THIS tick's obs are visible (not accumulated history).
        for k in self.state:
            self.state[k] = []

        # route
        route_state = []
        # update route speed limit by freq
        if self.current_time % self.args['route_state_update_freq'] == 0:
            for route in self.routes:
                route.route_update(self.current_time, self.effective_period)
                route_state.append(route.speed_limit)
            self.route_state = route_state
        # update waiting passengers of every station
        # station_state = []
        if self.current_time % self.passenger_update_freq == 0:
            self._batch_passenger_arrival(self.current_time, self.passenger_update_freq)
        # Generate this tick's passengers before buses scheduled at the same
        # instant perform their origin-stop service.
        for trip in self.timetables:
            if trip.launch_time <= self.current_time and not trip.launched:
                trip.launched = True
                self.launch_bus(trip)
        # update bus state
        for bus in self.bus_all:
            # if bus.bus_id == 0:
                # print(bus.last_station.station_name, bus.absolute_distance)
            # 每次开始前，清零状态和奖励
            bus.reward = None
            bus.obs = []
            if bus.in_station and hasattr(bus, 'trajectory'):
                bus.trajectory.append([bus.last_station.station_name, self.current_time, bus.absolute_distance, bus.direction, bus.trip_id])
                # Base-service completion is recorded once at arrival. Added
                # holding changes downstream arrivals but not the same-stop
                # service-completion timestamp.
            was_on_route = bool(bus.on_route)
            if was_on_route:
                # NOTE: in-route trajectory.append removed (was O(n_buses * n_steps), only for drawing)
                bus.drive(self.current_time, action.get(bus.bus_id, 0.0), self.bus_all, debug=debug, co_line_buses=co_line_buses)
            if was_on_route and not bus.on_route:
                self.completed_bus_ids_this_step.append(int(bus.bus_id))

        waiting_passengers = sum(
            len(station.waiting_passengers) for station in self.stations
        )
        in_vehicle_passengers = sum(
            len(bus.passengers) for bus in self.bus_all if bus.on_route
        )
        self.passenger_time_ledger.set_counts(
            self.current_time,
            waiting_passengers=waiting_passengers,
            in_vehicle_passengers=in_vehicle_passengers,
        )
        self.last_step_passenger_time = self.passenger_time_ledger.snapshot()
        self.passenger_time_ledger.advance_to(self.current_time + self.time_step)

        self.state_bus_list = state_bus_list = list(filter(lambda x: len(x.obs) != 0, self.bus_all))
        self.reward_list = reward_list = list(filter(lambda x: x.reward is not None, self.bus_all))

        if len(state_bus_list) != 0:
            # state_bus_list = sorted(state_bus_list, key=lambda x: x.bus_id)
            for i in range(len(state_bus_list)):
                # print('return state is ', state_bus_list[i].obs, ' for bus: ', state_bus_list[i].bus_id, 'at time:', self.current_time)
                # if len(self.state[state_bus_list[i].bus_id]) < 2:
                self.state.setdefault(state_bus_list[i].bus_id, []).append(state_bus_list[i].obs)
                # if state_bus_list[i].last_station.station_id not in [0,1,21,22]:
                #     print(1)
                # else:
                #     self.state[state_bus_list[i].bus_id][0] = self.state[state_bus_list[i].bus_id][1]
                #     self.state[state_bus_list[i].bus_id][1] = state_bus_list[i].obs
                # if state_bus_list[i].bus_id == 0:
                #     print(state_bus_list[i].obs[-1], 'bus_id: ', state_bus_list[i].obs[0], ', station_id: ', state_bus_list[i].obs[1], ', trip_id: ', state_bus_list[i].obs[2])
                #     print('return state is ', state_bus_list[i].obs, ' for bus: ', state_bus_list[i].bus_id,
                #           'at time: ', self.current_time)
                # if len(self.state[state_bus_list[i].bus_id]) > 2:
                #     print(1)
                # if debug:
                #     new_data = [state_bus_list[i].obs[0], state_bus_list[i].obs[1], state_bus_list[i].obs[2],
                #                 state_bus_list[i].obs[4]*1000, state_bus_list[i].obs[6] * 60, state_bus_list[i].obs[7]*60,
                #                 state_bus_list[i].obs[6] * 60 - state_bus_list[i].obs[7] * 60, self.current_time]
                #     self.summary_data.loc[len(self.summary_data)] = new_data
        if len(reward_list) != 0:
            # reward_list = sorted(reward_list, key=lambda x: x.bus_id)
            for i in range(len(reward_list)):
                # if reward_list[i].bus_id == 0:
                #     print('return reward is: ', reward_list[i].reward, ' for bus: ', reward_list[i].bus_id, ' at time:', self.current_time)
                # if (reward_list[i].last_station.station_id != 22 and reward_list[i].direction != 0) and \
                #         (reward_list[i].last_station.station_id != 1 and reward_list[i].direction != 1):
                # if len(self.reward[reward_list[i].bus_id]) > 1:
                #     print(2)
                self.reward[reward_list[i].bus_id] = reward_list[i].reward
                # if debugging:
                #     new_reward = [reward_list[i].bus_id, reward_list[i].last_station.station_id,
                #                   reward_list[i].trip_id, reward_list[i].forward_headway,
                #                   reward_list[i].backward_headway, reward_list[i].reward,
                #                   self.current_time + reward_list[i].holding_time]
                #     self.summary_reward.loc[len(self.summary_reward)] = new_reward

        self.current_time += self.time_step
        unhealthy_all = [bus.is_unhealthy for bus in self.bus_all]
        time_exceeded = (self.current_time >= self.max_time)
        # The target SUMO task has a fixed welfare horizon. Stranded passengers
        # must keep accruing waiting time after the last bus retires, so service
        # completion is not a valid early-termination condition.
        if time_exceeded:
            self.done = True
            if not was_done:
                self.last_episode_passengers = tuple(
                    passenger
                    for station in self.stations
                    for passenger in station.total_passenger
                )
                self.last_episode_bus_summaries = tuple(
                    {
                        "bus_id": int(bus.bus_id),
                        "trip_id": int(bus.trip_id),
                        "launch_time_seconds": float(bus.launch_time),
                        "completed": bool(not bus.on_route),
                        "terminal_arrival_seconds": bus.terminal_arrival_time,
                        "stop_visits": int(len(bus.stop_records)),
                    }
                    for bus in self.bus_all
                )
            if not debug and not was_done:
                for bus in self.bus_all:
                    bus.trajectory.clear()      # 清空轨迹列表 (keep attr alive for next episode)
                    bus.trajectory_dict.clear() # 清空轨迹字典 (keep attr alive for next episode)
                for station in self.stations:
                    station.waiting_passengers = []
                    station.total_passenger.clear()
        else:
            self.done = False

        if self.done and debug:
            self.summary_data = self.summary_data.sort_values(['bus_id', 'time'])

            output_dir = os.path.join(self.path, 'pic')
            os.makedirs(output_dir, exist_ok=True)
            if self.enable_plot:
                self.visualizer.plot(episode)

            self.summary_data.to_csv(os.path.join(output_dir, 'summary_data.csv'))
            self.summary_reward = self.summary_reward.sort_values(['bus_id', 'time'])
            self.summary_reward.to_csv(os.path.join(self.path, 'pic', 'summary_reward.csv'))

        if render and self.current_time % 1 == 0:
            self.visualizer.render()
            time.sleep(0.05)  # Add a delay to slow down the rendering

        return self.state, self.reward, self.done


if __name__ == '__main__':
    debug = True
    render = False
    num_runs = 1
    if render:
        # pygame.init()
        pass

    env = env_bus(os.getcwd(), debug=debug)
    env.enable_plot = True
    actions = {key: 0. for key in list(range(env.max_agent_num))}

    all_events = []
    cumulative_time = 0

    for run_idx in range(1, num_runs + 1):
        env.reset()
        while not env.done:
            state, reward, done = env.step(action=actions, debug=debug,
                                           render=render, episode=run_idx)

        events = env.visualizer.extract_bunching_events()
        cumulative_time += env.current_time
        all_events.extend(events)

#     pygame.quit()

    if all_events:
        df = pd.DataFrame(all_events).sort_values(['time'])
        output_dir = os.path.join(env.path, 'pic')
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, f'all_bunching_records_{num_runs}.csv'), index=False)
        # env.visualizer.plot_bunching_events(all_events, exp=str(num_runs))

    print('Total simulation time:', cumulative_time)


# =============================================================================
# Phase 3: Multi-Line Wrapper
# =============================================================================

class MultiLineEnv:
    """
    Multi-line wrapper around env_bus.

    Loads one env_bus per line directory found under `path/data/` that has
    a complete set of {stop_news, route_news, time_table, passenger_OD}.xlsx.

    Each line's bus IDs are offset so the global state/reward dicts
    use keys (line_id, local_bus_id) — compatible with SUMO rl_env.py
    nested action format: {line_id: {bus_id: action}}.

    The aggregate obs/reward/done dicts use nested structure:
        state:  {line_id: {bus_id: [obs_list]}}
        reward: {line_id: {bus_id: float}}
        done:   bool  (True when ALL lines are done)

    Usage:
        env = MultiLineEnv('calibrated_env', debug=False)
        env.reset()
        actions = {lid: {bid: 0.0 for bid in range(env.line_map[lid].max_agent_num)}
                   for lid in env.line_map}
        state, reward, done = env.step(actions)
    """

    REQUIRED_FILES = {'stop_news.xlsx', 'route_news.xlsx',
                      'time_table.xlsx', 'passenger_OD.xlsx'}
    JOURNEY_SCHEMA = "transit-simulator-journeys-v1"

    @classmethod
    def _load_causal_journeys(cls, path: str):
        artifact = os.path.join(path, "simulator_journeys_v1.json")
        if not os.path.isfile(artifact):
            return None
        with open(artifact) as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != cls.JOURNEY_SCHEMA:
            raise ValueError("simulator journey demand has an invalid schema")
        journeys = payload.get("journeys")
        if not isinstance(journeys, list) or not journeys:
            raise ValueError("simulator journey demand contains no journeys")
        jitter = float(payload.get("departure_jitter_seconds", -1.0))
        if not np.isfinite(jitter) or jitter < 0.0:
            raise ValueError("simulator journey departure jitter is invalid")
        return payload

    def __init__(
        self,
        path: str,
        debug: bool = False,
        render: bool = False,
        runtime_calibration_filename: str = "segment_runtime_calibration_v4.json",
        observation_speed_seed: int | None = None,
    ):
        self.path   = path
        self.debug  = debug
        self.render = render
        self.journey_demand = self._load_causal_journeys(path)

        data_dir = os.path.join(path, 'data')
        self.line_map: dict[str, 'env_bus'] = {}
        if (
            not runtime_calibration_filename
            or os.path.basename(runtime_calibration_filename)
            != runtime_calibration_filename
        ):
            raise ValueError("runtime calibration filename must be a basename")
        runtime_calibration_path = os.path.join(
            path, runtime_calibration_filename
        )
        self.runtime_calibration = None
        if os.path.isfile(runtime_calibration_path):
            with open(runtime_calibration_path) as handle:
                self.runtime_calibration = json.load(handle)
            expected_schema = {
                "segment_runtime_calibration_v4.json": (
                    "transit-simulator-runtime-calibration-v4"
                ),
                "segment_runtime_calibration_v7_v2.json": (
                    "transit-simulator-runtime-calibration-v7-v2"
                ),
                "segment_runtime_calibration_v7_v3.json": (
                    "transit-simulator-runtime-calibration-v7-v3"
                ),
                "segment_runtime_calibration_v7_v4.json": (
                    "transit-simulator-runtime-calibration-v7-v4"
                ),
            }.get(runtime_calibration_filename)
            if expected_schema is None:
                raise ValueError("unsupported runtime calibration filename")
            if self.runtime_calibration.get("schema_version") != expected_schema:
                raise ValueError("simulator runtime calibration has an invalid schema")
            if runtime_calibration_filename in {
                "segment_runtime_calibration_v7_v2.json",
                "segment_runtime_calibration_v7_v3.json",
                "segment_runtime_calibration_v7_v4.json",
            } and not self.runtime_calibration.get("coverage_gate", {}).get("passed"):
                raise ValueError("v7 simulator runtime calibration failed its coverage gate")

        self._observation_speed_seed = (
            None if observation_speed_seed is None else int(observation_speed_seed)
        )
        self._observation_speed_episode_index = -1
        self._observation_speed_profiles_by_demand = None
        self._current_observation_speed_profile = None
        self.observation_speed_profile_artifact = None

        # Discover line sub-directories
        for name in sorted(os.listdir(data_dir)):
            sub = os.path.join(data_dir, name)
            if not os.path.isdir(sub):
                continue
            files = set(os.listdir(sub))
            if not self.REQUIRED_FILES.issubset(files):
                continue
            try:
                # Create an isolated read-only runtime view for this process.
                line_path = self._make_line_path(path, name)
                le = env_bus(line_path, debug=debug, render=render)
                le.line_id  = name
                le.line_idx = len(self.line_map)
                le.line_id_str = name  # for signal model lookup in bus.py
                if self.runtime_calibration is not None:
                    line_calibration = self.runtime_calibration["lines"].get(name)
                    if line_calibration is None:
                        raise ValueError(
                            f"runtime calibration contains no entry for line {name}"
                        )
                    fixed_overhead = self.runtime_calibration[
                        "simulator_fixed_transition_overhead_seconds"
                    ]
                    le.install_runtime_calibration(
                        line_calibration, fixed_overhead
                    )
                    le.empirical_runtime_calibrated = True
                le.central_journey_demand = self.journey_demand is not None
                # Patch line_idx into every bus launched
                self.line_map[name] = le
                print(f"  Loaded line {name}: {le.state_dim}-dim obs, "
                      f"{len(le.routes)//2} segs, {len(le.timetables)} trips")
            except Exception as ex:
                raise RuntimeError(f"failed to load line {name}") from ex

        if not self.line_map:
            raise RuntimeError(
                f"No valid line directories found under {data_dir}. "
                "Run data/extract_sumo_network.py first."
            )

        self._install_route_coordinate_calibration()
        self._physical_stop_positions = self._install_physical_stop_ids()
        self._load_dynamic_observation_speed_profiles(runtime_calibration_path)
        self._install_backward_headway_observation_speeds()
        self._stations_by_line = self._station_lookup()
        self._pending_initial_journeys = []
        self._pending_initial_index = 0
        self._pending_transfers = []
        self._generated_journey_ids = set()
        self._completed_journey_ids = set()
        self._created_ride_legs = 0
        self._completed_ride_legs = 0
        self._transfer_ride_legs = 0
        self._journey_episode_index = -1
        self.last_episode_journey_summary = None

        self.state_dim   = 15
        self.max_agent_num = sum(le.max_agent_num for le in self.line_map.values())
        self._event_passenger_time = {}
        self._completed_agent_passenger_time = {}
        self.passenger_time_snapshot = aggregate_passenger_time_snapshots(
            line_env.passenger_time_ledger.snapshot()
            for line_env in self.line_map.values()
        )
        # action_space: 2D [hold_time, speed_ratio]
        self.action_space = Box(
            low=np.array([0.0, 0.1], dtype=np.float32),
            high=np.array([60.0, 10.0], dtype=np.float32),
        )

    def _install_physical_stop_ids(self):
        """Map line-local stop names back to shared SUMO stopping places."""

        network_xml = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                os.pardir,
                "network_data",
                "a_sorted_busline_edge.xml",
            )
        )
        component_to_physical = {}
        if os.path.isfile(network_xml):
            root = ET.parse(network_xml).getroot()
            for busline in root.findall("busline"):
                for element in busline.findall("element"):
                    if element.get("type") != "s":
                        continue
                    physical = element.get("stop_id", "")
                    for component in physical.split("_"):
                        existing = component_to_physical.setdefault(component, physical)
                        if existing != physical:
                            raise ValueError(
                                f"stop component {component!r} maps to multiple physical stops"
                            )

        stop_positions = {}
        for line_id, line_env in self.line_map.items():
            cumulative = 0.0
            for index, station in enumerate(line_env.stations):
                physical = component_to_physical.get(
                    station.station_name, station.station_name
                )
                station.physical_stop_id = physical
                stop_positions.setdefault(physical, {})[line_id] = cumulative
                if index < len(line_env.routes):
                    cumulative += float(line_env.routes[index].distance)
        return {
            stop_id: positions
            for stop_id, positions in stop_positions.items()
            if len(positions) > 1
        }

    def _install_route_coordinate_calibration(self):
        """Use the frozen SUMO route coordinate for v7 spatial observations."""

        if self.runtime_calibration is None:
            return
        calibration = self.runtime_calibration.get("backward_headway_observation")
        if calibration is None:
            return
        line_calibrations = calibration.get("lines", {})
        if set(line_calibrations) != set(self.line_map):
            raise ValueError("route-coordinate calibration line set changed")
        for line_id, line_env in self.line_map.items():
            descriptors = line_calibrations[line_id].get("segments", [])
            if len(descriptors) != len(line_env.routes):
                raise ValueError(
                    f"route-coordinate segment count changed for {line_id}"
                )
            for segment_index, (route, descriptor) in enumerate(
                zip(line_env.routes, descriptors)
            ):
                distance = float(descriptor.get("distance_m", 0.0))
                if (
                    int(descriptor.get("segment_index", -1)) != segment_index
                    or descriptor.get("start_stop") != route.start_stop
                    or descriptor.get("end_stop") != route.end_stop
                    or not np.isfinite(distance)
                    or distance <= 0.0
                ):
                    raise ValueError(
                        f"invalid route-coordinate segment for {line_id}:"
                        f"{route.start_stop}->{route.end_stop}"
                    )
                route.distance = distance

    def _install_backward_headway_observation_speeds(self):
        """Bind the SUMO stop-lane speed used for distance-to-time observations."""

        if self.runtime_calibration is None:
            return
        calibration = self.runtime_calibration.get("backward_headway_observation")
        if calibration is None:
            return
        if calibration.get("coverage_gate", {}).get("passed") is not True:
            raise ValueError("backward-headway observation-speed calibration failed")
        line_calibrations = calibration.get("lines", {})
        if set(line_calibrations) != set(self.line_map):
            raise ValueError("backward-headway observation-speed line set changed")
        for line_id, line_env in self.line_map.items():
            descriptors = line_calibrations[line_id].get("stops", [])
            if len(descriptors) != len(line_env.stations):
                raise ValueError(
                    f"backward-headway observation-speed stop count changed for {line_id}"
                )
            for station_index, (station, descriptor) in enumerate(
                zip(line_env.stations, descriptors)
            ):
                speed = float(descriptor.get("speed_mps", 0.0))
                if (
                    int(descriptor.get("station_index", -1)) != station_index
                    or descriptor.get("station_name") != station.station_name
                    or descriptor.get("physical_stop_id")
                    != getattr(station, "physical_stop_id", None)
                    or not np.isfinite(speed)
                    or speed < 1.0
                ):
                    raise ValueError(
                        f"invalid backward-headway observation speed for "
                        f"{line_id}:{station.station_name}"
                    )
                station.backward_headway_observation_speed_mps = speed
                current = self._current_observation_speed_profile
                if current is not None:
                    key = (line_id, descriptor["physical_stop_id"])
                    profile = current["contexts"].get(key)
                    is_nonterminal = station_index < len(line_env.stations) - 1
                    if is_nonterminal and profile is None:
                        raise ValueError(
                            f"dynamic observation-speed profile lacks {line_id}:"
                            f"{station.station_name}"
                        )
                    station.requires_dynamic_observation_speed = is_nonterminal
                    if profile is not None:
                        station.backward_headway_observation_speed_profile = profile

    def _load_dynamic_observation_speed_profiles(self, runtime_calibration_path):
        """Load and validate the packed v7-v4 SUMO speed-profile artifact."""

        if self.runtime_calibration is None:
            return
        descriptor = self.runtime_calibration.get(
            "backward_headway_observation", {}
        ).get("dynamic_speed_profiles")
        if descriptor is None:
            return
        if self._observation_speed_seed is None:
            raise ValueError("v7-v4 requires an explicit observation_speed_seed")
        if (
            descriptor.get("schema_version")
            != "transit-simulator-observation-speed-profiles-v7-v1"
            or descriptor.get("coverage_gate", {}).get("passed") is not True
        ):
            raise ValueError("dynamic observation-speed descriptor is invalid")
        artifact_name = descriptor.get("artifact")
        if (
            not isinstance(artifact_name, str)
            or not artifact_name
            or os.path.basename(artifact_name) != artifact_name
        ):
            raise ValueError("dynamic observation-speed artifact name is unsafe")
        artifact = os.path.join(os.path.dirname(runtime_calibration_path), artifact_name)
        if not os.path.isfile(artifact):
            raise FileNotFoundError(
                f"dynamic observation-speed artifact is missing: {artifact}"
            )
        if _sha256_file(artifact) != descriptor.get("sha256"):
            raise ValueError("dynamic observation-speed artifact hash changed")
        with np.load(artifact, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        required = {
            "context_profile_index",
            "context_line_index",
            "context_station_index",
            "context_physical_stop_code",
            "context_offsets",
            "sim_time_seconds",
            "speed_mps",
        }
        if set(arrays) != required:
            raise ValueError("dynamic observation-speed arrays changed")
        context_count = len(arrays["context_profile_index"])
        offsets = arrays["context_offsets"]
        if (
            len(offsets) != context_count + 1
            or offsets[0] != 0
            or offsets[-1] != len(arrays["speed_mps"])
            or len(arrays["sim_time_seconds"]) != len(arrays["speed_mps"])
            or np.any(np.diff(offsets) <= 0)
            or not np.isfinite(arrays["sim_time_seconds"]).all()
            or not np.isfinite(arrays["speed_mps"]).all()
            or np.any(arrays["speed_mps"] < 1.0)
        ):
            raise ValueError("dynamic observation-speed packed arrays are invalid")
        line_ids = descriptor.get("line_index_to_id")
        physical_stops = descriptor.get("physical_stop_code_to_id")
        if line_ids != sorted(self.line_map) or not isinstance(physical_stops, list):
            raise ValueError("dynamic observation-speed vocabularies changed")

        profile_rows = descriptor.get("profiles", [])
        profiles_by_index = {}
        for expected_index, row in enumerate(profile_rows):
            if int(row.get("profile_index", -1)) != expected_index:
                raise ValueError("dynamic observation-speed profile order changed")
            profiles_by_index[expected_index] = {
                "profile_index": expected_index,
                "realization_block_id": int(row["realization_block_id"]),
                "demand_scale": float(row["demand_scale"]),
                "contexts": {},
            }
        if len(profiles_by_index) != 15:
            raise ValueError("dynamic observation-speed profile grid is incomplete")

        for context_index in range(context_count):
            profile_index = int(arrays["context_profile_index"][context_index])
            line_index = int(arrays["context_line_index"][context_index])
            physical_code = int(
                arrays["context_physical_stop_code"][context_index]
            )
            station_index = int(arrays["context_station_index"][context_index])
            if (
                profile_index not in profiles_by_index
                or not 0 <= line_index < len(line_ids)
                or not 0 <= physical_code < len(physical_stops)
                or station_index < 0
            ):
                raise ValueError("dynamic observation-speed context index is invalid")
            start, stop = int(offsets[context_index]), int(offsets[context_index + 1])
            times = arrays["sim_time_seconds"][start:stop]
            speeds = arrays["speed_mps"][start:stop]
            if np.any(np.diff(times) <= 0.0):
                raise ValueError("dynamic observation-speed context times are not unique")
            key = (line_ids[line_index], physical_stops[physical_code])
            contexts = profiles_by_index[profile_index]["contexts"]
            if key in contexts:
                raise ValueError("dynamic observation-speed context is duplicated")
            contexts[key] = (times, speeds)

        profiles_by_demand = {}
        for profile in profiles_by_index.values():
            profiles_by_demand.setdefault(profile["demand_scale"], []).append(profile)
        if sorted(profiles_by_demand) != [0.75, 1.0, 1.25] or any(
            sorted(item["realization_block_id"] for item in profiles) != list(range(5))
            for profiles in profiles_by_demand.values()
        ):
            raise ValueError("dynamic observation-speed 5-by-3 design changed")
        for profiles in profiles_by_demand.values():
            profiles.sort(key=lambda item: item["realization_block_id"])
        self._observation_speed_profiles_by_demand = profiles_by_demand
        self.observation_speed_profile_artifact = {
            "path": os.path.realpath(artifact),
            "sha256": descriptor["sha256"],
            "schema_version": descriptor["schema_version"],
        }

    def _select_dynamic_observation_speed_profile(self):
        if self._observation_speed_profiles_by_demand is None:
            self._current_observation_speed_profile = None
            return
        matching_scales = [
            scale
            for scale in self._observation_speed_profiles_by_demand
            if np.isclose(scale, float(self.demand_scale), rtol=0.0, atol=1e-12)
        ]
        if len(matching_scales) != 1:
            raise ValueError(
                f"no registered observation-speed demand scale for {self.demand_scale}"
            )
        self._observation_speed_episode_index += 1
        profiles = self._observation_speed_profiles_by_demand[matching_scales[0]]
        selection = (
            self._observation_speed_seed + self._observation_speed_episode_index
        ) % len(profiles)
        self._current_observation_speed_profile = profiles[selection]

    @property
    def current_observation_speed_profile(self):
        profile = self._current_observation_speed_profile
        if profile is None:
            return None
        return {
            "profile_index": profile["profile_index"],
            "realization_block_id": profile["realization_block_id"],
            "demand_scale": profile["demand_scale"],
            "episode_index": self._observation_speed_episode_index,
            "seed": self._observation_speed_seed,
        }

    def _station_lookup(self):
        lookup = {}
        for line_id, line_env in self.line_map.items():
            by_name = {}
            for station in line_env.stations:
                if station.station_name in by_name:
                    raise ValueError(
                        f"simulator line {line_id} repeats stop {station.station_name}"
                    )
                by_name[station.station_name] = station
            lookup[line_id] = by_name
        return lookup

    def _sample_causal_journeys(self):
        self._pending_initial_journeys = []
        self._pending_initial_index = 0
        self._pending_transfers = []
        self._generated_journey_ids = set()
        self._completed_journey_ids = set()
        self._created_ride_legs = 0
        self._completed_ride_legs = 0
        self._transfer_ride_legs = 0
        self.last_episode_journey_summary = None
        if self.journey_demand is None:
            return

        scale = float(getattr(self, "demand_scale", 1.0))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("causal journey demand scale must be positive")
        whole_copies = int(math.floor(scale))
        fractional_copy = scale - whole_copies
        jitter = float(self.journey_demand["departure_jitter_seconds"])
        self._journey_episode_index += 1
        for template in self.journey_demand["journeys"]:
            copies = whole_copies
            if fractional_copy > 0.0 and np.random.random() < fractional_copy:
                copies += 1
            for copy_index in range(copies):
                base_departure = float(template["departure_seconds"])
                departure = base_departure
                if jitter > 0.0:
                    departure += float(np.random.uniform(-jitter, jitter))
                departure = float(
                    np.clip(
                        departure,
                        0.0,
                        float(template["departure_upper_bound_seconds"]),
                    )
                )
                journey_id = (
                    f"episode{self._journey_episode_index}:"
                    f"{template['template_id']}:{copy_index}"
                )
                if journey_id in self._generated_journey_ids:
                    raise RuntimeError(f"duplicate simulator journey ID {journey_id}")
                self._generated_journey_ids.add(journey_id)
                self._pending_initial_journeys.append(
                    {
                        "appearance_time": departure,
                        "journey_id": journey_id,
                        "leg_index": 0,
                        "legs": tuple(template["legs"]),
                        "source_departure_time": departure,
                    }
                )
        self._pending_initial_journeys.sort(
            key=lambda row: (row["appearance_time"], row["journey_id"])
        )

    @staticmethod
    def _select_leg_candidate(leg):
        candidates = tuple(leg.get("candidates", ()))
        if not candidates:
            raise ValueError("causal journey leg contains no line candidates")
        weights = np.asarray(
            [float(candidate["selection_weight"]) for candidate in candidates],
            dtype=np.float64,
        )
        if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("causal journey line-selection weights are invalid")
        weights /= weights.sum()
        index = int(np.random.choice(len(candidates), p=weights))
        return candidates[index]

    def _inject_leg(
        self,
        *,
        appearance_time: float,
        journey_id: str,
        leg_index: int,
        legs: tuple[dict, ...],
        source_departure_time: float,
    ):
        if not legs:
            raise ValueError("cannot inject a journey without an active leg")
        candidate = self._select_leg_candidate(legs[0])
        line_id = str(candidate["line_id"])
        if line_id not in self._stations_by_line:
            raise ValueError(f"causal journey references unknown line {line_id}")
        try:
            origin = self._stations_by_line[line_id][candidate["origin_stop"]]
            destination = self._stations_by_line[line_id][
                candidate["destination_stop"]
            ]
        except KeyError as error:
            raise ValueError(
                f"causal journey references an unknown {line_id} stop: {error}"
            ) from error
        passenger = JourneyPassenger(
            appearance_time,
            origin,
            destination,
            journey_id=journey_id,
            leg_index=leg_index,
            remaining_legs=tuple(legs[1:]),
            source_departure_time=source_departure_time,
        )
        origin.waiting_passengers.append(passenger)
        origin.total_passenger.append(passenger)
        self._created_ride_legs += 1
        if leg_index > 0:
            self._transfer_ride_legs += 1

    def _inject_due_journeys(self, current_time: float):
        if self.journey_demand is None:
            return
        arrival_end = min(
            float(line_env.passenger_arrival_end_time)
            for line_env in self.line_map.values()
        )
        while self._pending_initial_index < len(self._pending_initial_journeys):
            request = self._pending_initial_journeys[self._pending_initial_index]
            if float(request["appearance_time"]) > current_time + 1e-9:
                break
            self._pending_initial_index += 1
            if float(request["appearance_time"]) >= arrival_end:
                continue
            self._inject_leg(**request)

        due = []
        remaining = []
        for request in self._pending_transfers:
            if float(request["appearance_time"]) <= current_time + 1e-9:
                due.append(request)
            else:
                remaining.append(request)
        self._pending_transfers = remaining
        for request in sorted(
            due, key=lambda row: (row["appearance_time"], row["journey_id"])
        ):
            self._inject_leg(**request)

    def _consume_completed_passengers(self):
        for line_env in self.line_map.values():
            completed = tuple(line_env.completed_passengers)
            line_env.completed_passengers.clear()
            for passenger in completed:
                if not isinstance(passenger, JourneyPassenger):
                    continue
                self._completed_ride_legs += 1
                if passenger.remaining_legs:
                    self._pending_transfers.append(
                        {
                            "appearance_time": float(passenger.arrive_time)
                            + float(line_env.time_step),
                            "journey_id": passenger.journey_id,
                            "leg_index": passenger.leg_index + 1,
                            "legs": passenger.remaining_legs,
                            "source_departure_time": passenger.source_departure_time,
                        }
                    )
                    continue
                if passenger.journey_id in self._completed_journey_ids:
                    raise RuntimeError(
                        f"simulator journey completed twice: {passenger.journey_id}"
                    )
                self._completed_journey_ids.add(passenger.journey_id)

    def _journey_summary(self):
        waiting_ids = []
        in_vehicle_ids = []
        for line_env in self.line_map.values():
            for passenger in line_env.last_episode_passengers:
                if not isinstance(passenger, JourneyPassenger):
                    continue
                if passenger.arrive_time is not None:
                    continue
                if passenger.boarding_time is None:
                    waiting_ids.append(passenger.journey_id)
                else:
                    in_vehicle_ids.append(passenger.journey_id)
        active_ids = waiting_ids + in_vehicle_ids
        if len(active_ids) != len(set(active_ids)):
            raise RuntimeError("one causal journey is active in multiple simulator states")
        pending_initial_ids = {
            row["journey_id"]
            for row in self._pending_initial_journeys[self._pending_initial_index :]
        }
        pending_transfer_ids = {row["journey_id"] for row in self._pending_transfers}
        classified = (
            self._completed_journey_ids
            | set(active_ids)
            | pending_initial_ids
            | pending_transfer_ids
        )
        return {
            "unit": "causally ordered source-person journeys",
            "generated_journeys": len(self._generated_journey_ids),
            "completed_journeys": len(self._completed_journey_ids),
            "unfinished_journeys": len(self._generated_journey_ids)
            - len(self._completed_journey_ids),
            "completion_rate": (
                len(self._completed_journey_ids) / len(self._generated_journey_ids)
                if self._generated_journey_ids
                else None
            ),
            "created_ride_legs": int(self._created_ride_legs),
            "completed_ride_legs": int(self._completed_ride_legs),
            "created_transfer_ride_legs": int(self._transfer_ride_legs),
            "state_counts_at_horizon": {
                "waiting": len(waiting_ids),
                "in_vehicle": len(in_vehicle_ids),
                "between_transfer_legs": len(pending_transfer_ids),
                "not_yet_departed": len(pending_initial_ids),
                "completed": len(self._completed_journey_ids),
            },
            "journey_state_balance_error": len(self._generated_journey_ids)
            - len(classified),
        }

    def _make_line_path(self, base_path: str, line_id: str) -> str:
        """
        Create a process-private temporary directory view for env_bus:
            <tmp>/data/  ->  base_path/data/<line_id>/
            <tmp>/config.json  ->  base_path/config.json
            <tmp>/pic/  →  created
        Returns the path.
        """
        with _LINE_RUNTIME_LOCK:
            tmp = os.path.join(_line_runtime_root(base_path), line_id)
            os.makedirs(tmp, exist_ok=True)
            _install_read_only_link(
                os.path.join(base_path, 'config.json'),
                os.path.join(tmp, 'config.json'),
            )
            _install_read_only_link(
                os.path.join(base_path, 'data', line_id),
                os.path.join(tmp, 'data'),
            )
            os.makedirs(os.path.join(tmp, 'pic'), exist_ok=True)
            return tmp

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all line environments. Returns (state, reward, done)."""
        self._select_dynamic_observation_speed_profile()
        for le in self.line_map.values():
            le.reset()
            # Patch line_idx into future buses (set default in env)
            le._line_idx_for_bus = le.line_idx
        self._install_route_coordinate_calibration()
        self._physical_stop_positions = self._install_physical_stop_ids()
        self._install_backward_headway_observation_speeds()
        self._stations_by_line = self._station_lookup()
        self._sample_causal_journeys()
        self._event_passenger_time = {}
        self._completed_agent_passenger_time = {}
        self.passenger_time_snapshot = aggregate_passenger_time_snapshots(
            line_env.passenger_time_ledger.snapshot()
            for line_env in self.line_map.values()
        )
        return self._aggregate_state(), self._aggregate_reward(), False

    def initialize_state(self, render=False):
        """Step until at least one line has an observation."""
        actions = self._zero_actions()
        while True:
            state, reward, done = self.step(actions, render=render)
            if done:
                break
            if any(any(v for v in bus_dict.values())
                   for bus_dict in state.values()):
                break
        return state, reward, done

    def step(self, action_dict: dict, debug=False, render=False, episode=0):
        """
        action_dict: {line_id: {bus_id: float}}
          or flat {bus_id: float} (forwarded to all lines)
        Returns nested (state, reward, done).
        """
        self._inject_due_journeys(self.current_time)
        state  = {}
        reward = {}
        done_flags = []

        co_line_context = PhysicalCoLineContext(
            self._physical_stop_positions,
            {
                line_id: tuple(
                    float(bus.travel_distance)
                    for bus in line_env.bus_all
                    if bus.on_route
                )
                for line_id, line_env in self.line_map.items()
            },
        )

        for line_id, le in self.line_map.items():
            # Extract per-line actions
            if line_id in action_dict and isinstance(action_dict[line_id], dict):
                flat_act = action_dict[line_id]
            elif isinstance(action_dict, dict) and all(isinstance(k, int) for k in action_dict):
                flat_act = action_dict   # flat fallback
            else:
                flat_act = {i: 0.0 for i in range(le.max_agent_num)}

            s, r, d = le.step(
                flat_act,
                debug=debug,
                render=render,
                episode=episode,
                co_line_buses=co_line_context,
            )
            # Inject line_idx into any new obs vector (pos 0)
            for bus_id, obs_list in s.items():
                for obs in obs_list:
                    if obs:
                        obs[0] = float(le.line_idx)

            state[line_id]  = s
            reward[line_id] = r
            done_flags.append(d)

        self._consume_completed_passengers()
        self.passenger_time_snapshot = aggregate_passenger_time_snapshots(
            line_env.last_step_passenger_time
            for line_env in self.line_map.values()
        )
        self._event_passenger_time = {
            (line_id, int(bus_id)): self.passenger_time_snapshot
            for line_id, buses in state.items()
            for bus_id, observations in buses.items()
            if observations
        }
        self._completed_agent_passenger_time.update(
            {
                (line_id, int(bus_id)): self.passenger_time_snapshot
                for line_id, line_env in self.line_map.items()
                for bus_id in line_env.completed_bus_ids_this_step
            }
        )

        done = all(done_flags)
        self.done = done
        if done and self.journey_demand is not None:
            self.last_episode_journey_summary = self._journey_summary()
        return state, reward, done

    def consume_completed_agent_passenger_times(self):
        """Return terminal passenger-time snapshots accumulated since last call."""

        completed = dict(self._completed_agent_passenger_time)
        self._completed_agent_passenger_time.clear()
        return completed

    def step_to_event(self, action_dict: dict, debug=False, render=False, episode=0):
        """Step until at least one bus emits obs/reward (decision event), or done.

        Returns the same (state, reward, done) as step(), but internally
        fast-forwards through idle ticks where no decisions occur.
        This is critical for training speed: avoids the Python-level dict
        creation overhead on ~60% of ticks that produce zero decisions.
        """
        while True:
            state, reward, done = self.step(action_dict, debug=debug,
                                            render=render, episode=episode)
            if done:
                return state, reward, done
            # Check if ANY line produced a decision this tick
            has_event = False
            for lid, bus_dict in state.items():
                for bid, obs_list in bus_dict.items():
                    if obs_list:
                        has_event = True
                        break
                if has_event:
                    break
            if has_event:
                return state, reward, done
            # No decisions: clear actions and loop (zero-hold for idle tick)
            for lid in action_dict:
                for k in action_dict[lid]:
                    action_dict[lid][k] = None

    def _aggregate_state(self):
        return {lid: le.state for lid, le in self.line_map.items()}

    def _aggregate_reward(self):
        return {lid: le.reward for lid, le in self.line_map.items()}

    def _zero_actions(self):
        return {lid: {i: 0.0 for i in range(le.max_agent_num)}
                for lid, le in self.line_map.items()}

    @property
    def current_time(self):
        return max((le.current_time for le in self.line_map.values()), default=0)

    def capture_snapshot(self):
        """Capture state snapshot across all lines (for test_divergence compatibility)."""
        return {lid: le for lid, le in self.line_map.items()}
