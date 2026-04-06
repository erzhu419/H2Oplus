import json
import time
import numpy as np
from sim_core.timetable import Timetable
from sim_core.bus import Bus, BusState
from sim_core.route import Route
from sim_core.station import Station
from sim_core.visualize import visualize
import pandas as pd
from gym.spaces.box import Box
from gym.spaces import MultiDiscrete
import copy
import os, sys
# import pygame
import json


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
            timetable_set = pd.read_excel(os.path.join(path, "data/time_table.xlsx"))
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
        # max_agent_num: concurrent buses on route (conservative upper bound)
        self.max_agent_num = min(self.effective_trip_num, 25)

        self.visualizer = visualize(self)
        self.enable_plot = True

        self.action_space = Box(
            low=np.array([0.0, 0.1], dtype=np.float32),
            high=np.array([60.0, 10.0], dtype=np.float32),
        )

        if debug:
            self.summary_data = pd.DataFrame(columns=['bus_id', 'station_id', 'trip_id', 'abs_dis', 'forward_headway',
                                                  'backward_headway', 'headway_diff', 'time'])
            self.summary_reward = pd.DataFrame(columns=['bus_id', 'station_id', 'trip_id', 'forward_headway',
                                                    'backward_headway', 'reward', 'time'])

        self.stations = self.set_stations()
        self.routes = self.set_routes()
        self.timetables = self.set_timetables()

        self.state_dim = 15  # aligned with SUMO rl_env.py (15-dim obs)

    @property
    def bus_in_terminal(self):
        return [bus for bus in self.bus_all if not bus.on_route]

    # @property
    # def bus_on_route(self):
    #     return [bus for bus in self.bus_all if bus.on_route]

    def set_timetables(self):
        return [Timetable(self.timetable_set['launch_time'][i], self.timetable_set['launch_turn'][i], self.timetable_set['direction'][i]) for i in range(self.timetable_set.shape[0])]

    def set_routes(self):
        return [
            Route(self.routes_set['route_id'][i], self.routes_set['start_stop'][i], self.routes_set['end_stop'][i],
                  self.routes_set['distance'][i], self.routes_set['V_max'][i], self.routes_set.iloc[i, 5:]) for i in
            range(self.routes_set.shape[0])]

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
                rates.append(demand / 3600.0)
                mapping.append((station, dest))
        self._pax_cache_hour = effective_hour
        self._pax_flat_rates = np.array(rates, dtype=np.float64) if rates else np.array([], dtype=np.float64)
        self._pax_flat_map = mapping

    def _batch_passenger_arrival(self, current_time, update_interval):
        """ONE Poisson call for all stations × destinations."""
        from sim_core.passenger import Passenger

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
        # Inject SUMO line index (matches rl_env._line_index; 7X=11)
        bus.line_idx = getattr(self, 'line_idx', 0)
        # Inject line_id string for signal model lookup
        bus.line_id_str = getattr(self, 'line_id_str', None)
        # Inject number of route segments for signal density calculation
        bus._n_route_segments = len(self.routes)


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
        # Reset per-step state so only THIS tick's obs are visible (not accumulated history).
        for k in self.state:
            self.state[k] = []

        # Enumerate trips in timetables, if current_time<=launch_time of the trip, then launch it.
        for i, trip in enumerate(self.timetables):
            if trip.launch_time <= self.current_time and not trip.launched:
                trip.launched = True
                self.launch_bus(trip)
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
        # update bus state
        for bus in self.bus_all:
            # if bus.bus_id == 0:
                # print(bus.last_station.station_name, bus.absolute_distance)
            # 每次开始前，清零状态和奖励
            bus.reward = None
            bus.obs = []
            if bus.in_station and hasattr(bus, 'trajectory'):
                bus.trajectory.append([bus.last_station.station_name, self.current_time, bus.absolute_distance, bus.direction, bus.trip_id])
                # trajectory_dict is written ONCE at arrival (bus.py drive() line 253-259)
                # with boarding-completion time. Do NOT append here — that would leak
                # the RL hold time into the departure record, making holding "visible"
                # to subsequent buses' headway calculation. In SUMO, hold time is NOT
                # reflected in trajectory_dict, so holding should be headway-neutral.
            if bus.on_route:
                # NOTE: in-route trajectory.append removed (was O(n_buses * n_steps), only for drawing)
                bus.drive(self.current_time, action.get(bus.bus_id, 0.0), self.bus_all, debug=debug, co_line_buses=co_line_buses)

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
        all_retired = (sum([trip.launched for trip in self.timetables]) == len(self.timetables)
                       and sum([bus.on_route for bus in self.bus_all]) == 0)
        time_exceeded = (self.current_time >= self.max_time)
        if all_retired or time_exceeded:
            self.done = True
            if not debug:
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

    def __init__(self, path: str, debug: bool = False, render: bool = False):
        self.path   = path
        self.debug  = debug
        self.render = render

        data_dir = os.path.join(path, 'data')
        self.line_map: dict[str, 'env_bus'] = {}

        # Discover line sub-directories
        for name in sorted(os.listdir(data_dir)):
            sub = os.path.join(data_dir, name)
            if not os.path.isdir(sub):
                continue
            files = set(os.listdir(sub))
            if not self.REQUIRED_FILES.issubset(files):
                continue
            # Patch a minimal config.json for each sub-env
            cfg_path = os.path.join(path, 'config.json')
            try:
                # Create sub-path with symlink/copy of config + data subdir
                line_path = self._make_line_path(path, name)
                le = env_bus(line_path, debug=debug, render=render)
                le.line_id  = name
                le.line_idx = len(self.line_map)
                le.line_id_str = name  # for signal model lookup in bus.py
                # Patch line_idx into every bus launched
                self.line_map[name] = le
                print(f"  Loaded line {name}: {le.state_dim}-dim obs, "
                      f"{len(le.routes)//2} segs, {len(le.timetables)} trips")
            except Exception as ex:
                print(f"  WARNING: line {name} failed to load: {ex}")

        if not self.line_map:
            raise RuntimeError(
                f"No valid line directories found under {data_dir}. "
                "Run data/extract_sumo_network.py first."
            )

        self.state_dim   = 15
        self.max_agent_num = sum(le.max_agent_num for le in self.line_map.values())
        # action_space: 2D [hold_time, speed_ratio]
        self.action_space = Box(
            low=np.array([0.0, 0.1], dtype=np.float32),
            high=np.array([60.0, 10.0], dtype=np.float32),
        )

    def _make_line_path(self, base_path: str, line_id: str) -> str:
        """
        Create a temporary directory view for env_bus:
            <tmp>/data/  →  symlink or copy from base_path/data/<line_id>/
            <tmp>/config.json  →  copied from base_path/config.json
            <tmp>/pic/  →  created
        Returns the path.
        """
        import tempfile, shutil
        tmp = os.path.join(base_path, '_line_envs', line_id)
        os.makedirs(tmp, exist_ok=True)

        # config
        src_cfg = os.path.join(base_path, 'config.json')
        dst_cfg = os.path.join(tmp, 'config.json')
        if not os.path.exists(dst_cfg):
            shutil.copy2(src_cfg, dst_cfg)

        # data dir: symlink to base_path/data/<line_id>/
        src_data = os.path.join(base_path, 'data', line_id)
        dst_data = os.path.join(tmp, 'data')
        if not os.path.exists(dst_data):
            os.symlink(src_data, dst_data)

        # pic dir
        os.makedirs(os.path.join(tmp, 'pic'), exist_ok=True)

        return tmp

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all line environments. Returns (state, reward, done)."""
        for le in self.line_map.values():
            le.reset()
            # Patch line_idx into future buses (set default in env)
            le._line_idx_for_bus = le.line_idx
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
        state  = {}
        reward = {}
        done_flags = []

        for line_id, le in self.line_map.items():
            # Build co_line_buses: for this line, collect buses from all OTHER lines.
            # Each bus is only registered at its current next_station (O(n_buses) per tick).
            co_line_buses = {}  # {station_name: [(abs_distance, speed, line_id)]}
            for other_lid, other_le in self.line_map.items():
                if other_lid == line_id:
                    continue
                for bus in other_le.bus_all:
                    if not bus.on_route or bus.next_station is None:
                        continue
                    sname = bus.next_station.station_name
                    co_line_buses.setdefault(sname, []).append(
                        (bus.absolute_distance, bus.current_speed, other_lid)
                    )

            # Extract per-line actions
            if line_id in action_dict and isinstance(action_dict[line_id], dict):
                flat_act = action_dict[line_id]
            elif isinstance(action_dict, dict) and all(isinstance(k, int) for k in action_dict):
                flat_act = action_dict   # flat fallback
            else:
                flat_act = {i: 0.0 for i in range(le.max_agent_num)}

            s, r, d = le.step(flat_act, debug=debug, render=render, episode=episode, co_line_buses=co_line_buses)
            # Inject line_idx into any new obs vector (pos 0)
            for bus_id, obs_list in s.items():
                for obs in obs_list:
                    if obs:
                        obs[0] = float(le.line_idx)

            state[line_id]  = s
            reward[line_id] = r
            done_flags.append(d)

        done = all(done_flags)
        self.done = done
        return state, reward, done

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
