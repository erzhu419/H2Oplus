import math
import random
import numpy as np

class Route(object):
    def __init__(self, route_id, start_stop, end_stop, route_length, max_speed, route_speed_history):
        self.route = []
        self.maximum_velocity = 0
        self.variant_velocity = 0

        self.sigma = 1.5
        self.route_id = route_id
        self.route_max_speed = max_speed
        self.speed_history = route_speed_history
        self.speed_limit = 15

        self.start_stop = start_stop
        self.end_stop = end_stop
        self.distance = route_length
        self.runtime_calibration = None
        self.runtime_fixed_overhead_seconds = 0

    def install_runtime_calibration(self, calibration, fixed_overhead_seconds):
        if calibration.get("all_periods", {}).get("n", 0) <= 0:
            raise ValueError("segment runtime calibration contains no samples")
        self.runtime_calibration = calibration
        self.runtime_fixed_overhead_seconds = int(fixed_overhead_seconds)

    @staticmethod
    def _demand_scale_key(value):
        return format(float(value), ".12g")

    def sample_runtime_seconds(self, current_time, demand_scale=None):
        if self.runtime_calibration is None:
            return None
        selected = self.runtime_calibration
        if demand_scale is not None:
            selected = self.runtime_calibration.get("by_demand_scale", {}).get(
                self._demand_scale_key(demand_scale), selected
            )
        service_hour = str(int(max(float(current_time), 0.0) // 3600))
        period = selected.get("by_service_hour", {}).get(
            service_hour
        )
        if period is None or int(period.get("n", 0)) < 20:
            period = selected["all_periods"]
        samples = period["samples_seconds"]
        if not samples:
            raise RuntimeError("segment runtime calibration has an empty sample set")
        return float(random.choice(samples))

    def route_update(self, current_time, effective_period):
        current_hour = effective_period[int(min(current_time//3600, len(effective_period) -1))]
        v = np.clip(math.log(random.lognormvariate(self.speed_history.loc[current_hour], self.sigma)), 2, 15)
        # v = math.log(random.lognormvariate(self.speed_history.loc[current_hour], self.sigma))
        self.speed_limit = min(self.route_max_speed, max(int(v), 0))
