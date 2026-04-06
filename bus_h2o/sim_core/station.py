from sim_core.passenger import Passenger
import numpy as np


class Station(object):
    def __init__(self, station_type, station_id, station_name, direction, od):
        # if the station is terminal or not terminal,
        self.station_type = station_type
        # the id of stations
        self.station_id = station_id
        self.station_name = station_name
        # waiting passengers in this station
        self.waiting_passengers = []   # list of Passenger (was np.array — O(n) append)
        self.total_passenger = []
        # the direction is True if upstream, else False
        self.direction = direction
        # od is the passengers demand of every hour
        self.od = od

    # def station_update(self, current_time, stations):
    #     # 自己写的
    #     # if self.od is not None:
    #     #     # effective_period_str = effective_period[current_time//3600].strftime("%H:%M:%S")
    #     #     effective_period_str = '0'+str(6+current_time//3600)+':00:00' if 6+current_time//3600 < 10 else str(6+current_time//3600)+':00:00'
    #     #     period_od = self.od[effective_period_str]
    #     #     for destination_name, demand in period_od.items():
    #     #     # for destination_name in effective_station_name:
    #     #     # 对如果period_od[destination_name] == 0,则不计算泊松分布，因为太慢，且太多
    #     #         destination_demand_num = 0 if demand == 0 else np.random.poisson(demand/3600)
    #     #         for _ in range(destination_demand_num):
    #     #             destination = list(filter(lambda x: x.station_name == destination_name and x.direction == self.direction, stations))[0]
    #     #             passenger = Passenger(current_time, self, destination)
    #     #             self.waiting_passengers = np.append(self.waiting_passengers, passenger)
    #     #             self.total_passenger.append(passenger)
    #     #     sorted(self.waiting_passengers, key=lambda i: i.appear_time)
    #
    #     if self.od is not None: # GPT优化的，减少不必要的操作
    #
    #         effective_period_str = f"{6 + current_time // 3600:02}:00:00"
    #         period_od = self.od[effective_period_str]
    #
    #         for destination_name, demand in period_od.items():
    #             if demand > 0:  # 直接过滤掉不需要计算的需求
    #                 destination_demand_num = np.random.poisson(demand / 3600)
    #                 if destination_demand_num > 0:
    #                     destination = next(x for x in stations if x.station_name == destination_name and x.direction == self.direction)
    #                     new_passengers = [Passenger(current_time, self, destination) for _ in range(destination_demand_num)]
    #                     self.waiting_passengers = np.append(self.waiting_passengers, new_passengers)
    #                     self.total_passenger.extend(new_passengers)


    def station_update(self, current_time, stations, passenger_update_interval=1):
        """Vectorised Poisson passenger arrival with cached destination refs.

        v3 optimisations:
          - _dest_cache: pre-built list of (demand_per_s, dest_obj) computed
            once per hour period — eliminates O(N) next() search every call.
          - Single np.random.poisson(rates_array) call per station per update.
          - waiting_passengers is a plain Python list (O(1) extend).
        
        OD data: passenger_OD_sumo.xlsx extracted directly from SUMO rou.xml,
        so demand values are exact per-hour passenger counts (no scaling needed).
        """
        if self.od is None:
            return

        hour_offset   = int(current_time) // 3600
        effective_hour = max(6, min(6 + hour_offset, 19))

        # ── Lazy cache: rebuild when hour period changes ──────────────────────
        if not hasattr(self, '_dest_cache_hour') or self._dest_cache_hour != effective_hour:
            period_od = self.od[f"{effective_hour:02d}:00:00"]
            cache = []
            for name, demand in period_od.items():
                if demand <= 0:
                    continue
                dest = next(
                    (x for x in stations if x.station_name == name and x.direction == self.direction),
                    None,
                )
                if dest is None:
                    continue
                cache.append((demand / 3600.0, dest))      # (rate_per_s, dest_obj)
            # Pre-build the rates array for batch Poisson
            self._dest_cache_hour  = effective_hour
            self._dest_cache       = cache
            self._dest_rates       = np.array([c[0] for c in cache], dtype=np.float64) if cache else np.array([], dtype=np.float64)

        if len(self._dest_cache) == 0:
            return

        # ── Single batch Poisson call ─────────────────────────────────────────
        arrivals = np.random.poisson(self._dest_rates * passenger_update_interval)

        for n_arrive, (_, dest) in zip(arrivals, self._dest_cache):
            if n_arrive == 0:
                continue
            new_pax = [Passenger(current_time, self, dest) for _ in range(int(n_arrive))]
            self.waiting_passengers.extend(new_pax)
            self.total_passenger.extend(new_pax)
