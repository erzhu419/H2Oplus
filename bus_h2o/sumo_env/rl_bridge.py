import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
import time
import random
import argparse
import math
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Set, Any

if "SUMO_HOME" in os.environ:
    tools_path = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools_path not in sys.path:
        sys.path.append(tools_path)
else:
    raise EnvironmentError("Please declare environment variable 'SUMO_HOME'!")

import traci
LIBSUMO_AVAILABLE = False
try:
    import libsumo
    LIBSUMO_AVAILABLE = True
except ImportError:
    pass

from sumolib import checkBinary
import traci.constants as tc

from .case import f_8_create_obj
from .case import e_8_gurobi_test_considerbusnum_V3
from .case import d_8_compute_running_time

from .rl_env import DecisionEvent


from dataclasses import dataclass, field


@dataclass
class FleetBus:
    """Represents one physical bus in the fleet across its full service day.

    A FleetBus can be reused for multiple SUMO trips in FIFO order, exactly
    mirroring LSTM-RL env/sim.py's `launch_bus()` logic.
    """
    fleet_id: str                    # Permanent identifier, e.g. '7S_fleet_0'
    direction: int                   # 0=northbound / 1=southbound (matches line suffix X/S)
    line_id: str                     # Line this bus operates on
    back_to_terminal_time: float = float('inf')   # Time bus returned to terminal
    on_route: bool = False           # True while assigned to an active SUMO trip
    current_trip_id: str = ''        # Currently assigned SUMO vehicle ID
    trip_ids: list = field(default_factory=list)  # All SUMO trip IDs served so far
    cumulative_trajectory: list = field(default_factory=list)  # Cross-trip stop history

    def reset_trip(self, new_trip_id: str, departure_time: float) -> None:
        """Reassign this fleet bus to a new SUMO trip (mirrors Bus.reset_bus)."""
        self.current_trip_id = new_trip_id
        self.trip_ids.append(new_trip_id)
        self.back_to_terminal_time = float('inf')
        self.on_route = True



class SumoRLBridge:
    def __init__(
        self,
        root_dir: str,
        sumo_cfg: str = "control_sim_traci_period.sumocfg",
        *,
        gui: bool = False,
        max_steps: int = 18000,
        time_step: float = 1.0,
        default_headway: float = 360.0,
        update_freq: int = 10,
        scale: float = 1.0,
    ) -> None:
        self.root_dir = os.path.abspath(root_dir)
        self.sumo_cfg = sumo_cfg if os.path.isabs(sumo_cfg) else os.path.join(self.root_dir, sumo_cfg)
        self.gui = gui
        self.max_steps = max_steps
        self.time_step = time_step
        self.default_headway = default_headway
        self.update_freq = update_freq
        self.scale = scale

        self.initialized = False
        self.first_run = True
        self.done = False
        self.current_time = 0.0
        self.steps = 0

        self.decision_queue: deque[DecisionEvent] = deque()
        self.pending_events: Dict[Tuple[str, str, str], DecisionEvent] = {}
        self.active_events: Dict[str, DecisionEvent] = {}

        self.arrival_history: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.depart_history: Dict[str, Dict[str, float]] = defaultdict(dict)

        self.stop_indices: Dict[str, Dict[str, int]] = defaultdict(dict)

        self.sumo_binary = None

        self.lane_obj_dic = {}
        self.stop_obj_dic = {}
        self.signal_obj_dic = {}
        self.line_obj_dic = {}
        self.bus_obj_dic = {}
        self.passenger_obj_dic = {}

        self.BusCap = 50
        self.AveAlightingTime = 1.5
        self.AveBoardingTime = 2.5

        self.busline_edge_order = {}
        self.involved_tl_ID_l = []
        self.line_headways = {}  # Store median headway per line
        self.line_stop_distances = defaultdict(dict)  # Absolute distance of each stop along each line

        # ── Fleet reuse (FIFO, mirrors LSTM-RL sim.py launch_bus) ────────────
        # fleet_obj_dic: sumo_trip_id → FleetBus (current assignment)
        self.fleet_obj_dic: Dict[str, FleetBus] = {}
        # terminal_pool: direction(int) → FIFO deque of FleetBus objects
        self.terminal_pool: Dict[int, deque] = {0: deque(), 1: deque()}
        self._fleet_counter: Dict[str, int] = {}  # line_id → next fleet index
    # region SUMO lifecycle
    def reset(self) -> None:
        # Pre-cleanup: break references in old bus/fleet objects to help GC
        # reclaim memory between episodes (trajectory_dict etc. accumulate)
        if self.bus_obj_dic:
            for _bo in self.bus_obj_dic.values():
                if hasattr(_bo, 'trajectory_dict'):
                    _bo.trajectory_dict.clear()
                if hasattr(_bo, 'just_server_stop_data_d'):
                    _bo.just_server_stop_data_d.clear()

        # Optimization: Load network once and reuse state
        if self.first_run:
            if self.initialized:
                 self.close()
            self._ensure_sumo_home()
            self._start_traci()
            self._load_objects()
            # Save state for future resets
            if self.sumo_binary == "libsumo":
                traci.simulation.saveState("sumo_start_state.sbx")
            self.first_run = False
        else:
            # Soft reset using saved state
            if self.sumo_binary == "libsumo":
                # Suppress "Loading state..." output
                with open(os.devnull, 'w') as devnull:
                    try:
                        # Redirect stdout (fd 1) and stderr (fd 2) at the OS level
                        # This is necessary because libsumo prints from C++
                        old_stdout = os.dup(1)
                        old_stderr = os.dup(2)
                        os.dup2(devnull.fileno(), 1)
                        os.dup2(devnull.fileno(), 2)
                        try:
                            traci.simulation.loadState("sumo_start_state.sbx")
                        finally:
                            os.dup2(old_stdout, 1)
                            os.dup2(old_stderr, 2)
                            os.close(old_stdout)
                            os.close(old_stderr)
                    except Exception:
                        # Fallback if redirection fails
                        traci.simulation.loadState("sumo_start_state.sbx")
            else:
                # Fallback for standard traci (if needed, though we focus on libsumo)
                self.close()
                self._ensure_sumo_home()
                self._start_traci()
            
            # Always reload objects to ensure fresh python state matching the simulation
            self._load_objects()

        self.initialized = True
        self.done = False
        self.current_time = traci.simulation.getTime()
        self.steps = 0
        self.decision_queue.clear()
        self.pending_events.clear()
        self.active_events.clear()
        self.arrival_history.clear()
        self.depart_history.clear()
        
        # Optimization: Track active agents locally to avoid expensive getIDList() calls
        self.active_bus_ids = set()
        self.active_passenger_ids = set()
        self.just_departed_buses = []

        # Reset fleet terminal pool (keep FleetBus objects, just clear queues)
        for q in self.terminal_pool.values():
            q.clear()
        # Mark all fleet buses as not on route
        for fb in self.fleet_obj_dic.values():
            fb.on_route = False
            fb.back_to_terminal_time = float('inf')

    def close(self) -> None:
        # Optimization: Do NOT close traci if using libsumo reuse strategy
        if self.sumo_binary != "libsumo":
            if traci.isLoaded():
                traci.close()
        
        self.initialized = False
        self.active_bus_ids = set()
        self.active_passenger_ids = set()

    def _ensure_sumo_home(self) -> None:
        if "SUMO_HOME" not in os.environ:
            raise EnvironmentError("Please declare environment variable 'SUMO_HOME'!")
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools not in sys.path:
            sys.path.append(tools)

    def _start_traci(self) -> None:
        global traci
        if self.gui:
            # If GUI is requested, we MUST use standard traci
            binary = checkBinary('sumo-gui')
            traci.start([binary, "-c", self.sumo_cfg, "--no-warnings", "--duration-log.disable", "--log", "/dev/null", "--scale", str(self.scale)])
            self.sumo_binary = binary
        elif LIBSUMO_AVAILABLE:
            # Headless and libsumo available, patch traci globally for speed
            import libsumo
            sys.modules['traci'] = libsumo
            traci = libsumo
            traci.start(["sumo", "-c", self.sumo_cfg, "--no-warnings", "--duration-log.disable", "--log", "/dev/null", "--scale", str(self.scale)])
            self.sumo_binary = "libsumo"
        else:
            # Headless fallback to standard traci
            binary = checkBinary('sumo')
            traci.start([binary, "-c", self.sumo_cfg, "--no-warnings", "--duration-log.disable", "--log", "/dev/null", "--scale", str(self.scale)])
            self.sumo_binary = binary

    def _load_objects(self) -> None:
        result = f_8_create_obj.create_obj_fun()
        self.lane_obj_dic = result[0]
        self.stop_obj_dic = result[1]
        self.signal_obj_dic = result[2]
        self.line_obj_dic = result[3]
        self.bus_obj_dic = result[4]
        self.passenger_obj_dic = result[5]
        for bus in self.bus_obj_dic.values():
            bus.last_forward_headway = None  # Store the last calculated forward headway (for backward bus logic)
            if not hasattr(bus, 'trajectory_dict'):
                bus.trajectory_dict = defaultdict(list)

        (
            _,
            _,
            _,
            _,
            line_station_od_otd_dict,
            bus_arrstation_od_otd_dict,
            _,
            self.BusCap,
            self.AveAlightingTime,
            self.AveBoardingTime,
        ) = e_8_gurobi_test_considerbusnum_V3.get_static_info(self.line_obj_dic)
        self.bus_arrstation_od_otd_dict = bus_arrstation_od_otd_dict

        edge_file_path = os.path.join(self.root_dir, "intersection_delay", "a_sorted_busline_edge.xml")
        edge_file = ET.parse(edge_file_path)
        sorted_busline_edge_d, involved_tl_ID_l, _, _ = d_8_compute_running_time.get_sorted_busline_edge(edge_file)
        self.busline_edge_order = sorted_busline_edge_d
        self.involved_tl_ID_l = involved_tl_ID_l

        trigger_time = 900
        for stop_id, stop_obj in self.stop_obj_dic.items():
            stop_obj.get_accessible_stop(self.line_obj_dic)
            stop_obj.get_passenger_arriver_rate(self.passenger_obj_dic)
            stop_obj.get_initial_just_leave_data(line_station_od_otd_dict, trigger_time)
        
        # Optimization: Skip signal object initialization as it is unused for Holding Control
        for signal_id, signal_obj in self.signal_obj_dic.items():
            signal_obj.get_attribute_by_traci()
            signal_obj.get_pass_line(self.line_obj_dic)
            
        for bus_id, bus_obj in self.bus_obj_dic.items():
            line = self.line_obj_dic[bus_obj.belong_line_id_s]
            bus_obj.get_arriver_timetable(line)

        for line_id, line in self.line_obj_dic.items():
            self.stop_indices[line_id] = {stop_id: idx for idx, stop_id in enumerate(line.stop_id_l)}

        # Compute median headway for each line
        for line_id, line in self.line_obj_dic.items():
            # Find all buses for this line
            buses = [b for b in self.bus_obj_dic.values() if b.belong_line_id_s == line_id]
            if len(buses) > 1:
                buses.sort(key=lambda b: b.start_time_n)
                start_times = [b.start_time_n for b in buses]
                diffs = [j - i for i, j in zip(start_times[:-1], start_times[1:])]
                if diffs:
                    self.line_headways[line_id] = float(np.median(diffs))
                else:
                    self.line_headways[line_id] = self.default_headway
                
        # Parse physical capacity for all stations (assuming ~15m per bus slot)
        self.station_capacities = {}
        for stop_id in self.stop_obj_dic:
            try:
                start_pos = traci.busstop.getStartPos(stop_id)
                end_pos = traci.busstop.getEndPos(stop_id)
                length = max(end_pos - start_pos, 15.0)
                self.station_capacities[stop_id] = int(length // 15.0)
            except Exception:
                self.station_capacities[stop_id] = 2  # Safe default if TraCI fails

        # Precalculate absolute distances of each stop from the start of its respective line
        for line_id, line in self.line_obj_dic.items():
            dist = 0.0
            for stop_id in line.stop_id_l:
                dist += line.distance_between_stop_d.get(stop_id, 0.0)
                self.line_stop_distances[line_id][stop_id] = dist
        # print(f"Computed Line Headways: {self.line_headways}")

        # ── Initialise fleet objects from bus_obj_dic ─────────────────────
        # One FleetBus per unique (line_id, direction) slot.
        # At episode start all buses are pending departure, so pool is empty.
        self.fleet_obj_dic.clear()
        self._fleet_counter.clear()
        for q in self.terminal_pool.values():
            q.clear()
        for bus_id, bus_obj in self.bus_obj_dic.items():
            line_id = bus_obj.belong_line_id_s
            direction = 1 if line_id.endswith('S') else 0
            idx = self._fleet_counter.get(line_id, 0)
            fleet_id = f"{line_id}_fleet_{idx}"
            self._fleet_counter[line_id] = idx + 1
            fb = FleetBus(
                fleet_id=fleet_id,
                direction=direction,
                line_id=line_id,
                current_trip_id=bus_id,
                on_route=False,  # not yet departed
            )
            fb.trip_ids.append(bus_id)
            self.fleet_obj_dic[bus_id] = fb

    # endregion

    # region RL hooks
    def fetch_events(self) -> Tuple[List[DecisionEvent], bool, List[str]]:
        if self.done:
            return [], True, []
        if not self.initialized:
            self.reset()
        if not self.decision_queue:
            self._advance_until_event()
        if not self.decision_queue and self.done:
            return [], True, []
        events = list(self.decision_queue)
        self.decision_queue.clear()
        
        departed = list(self.just_departed_buses)
        self.just_departed_buses.clear()
        
        return events, self.done, departed

    def apply_action(self, event: DecisionEvent, action_value: Any) -> None:
        if not self.initialized or self.done:
            return
            
        # Parse 2D Action: [Holding Time, Speed Ratio]
        if hasattr(action_value, '__len__') and len(action_value) >= 2:
            hold_value = float(action_value[0])
            speed_ratio = float(action_value[1])
        else:
            hold_value = float(action_value)
            speed_ratio = 1.0  # Fallback
            
        duration = max(event.base_stop_duration + hold_value, 0.0)
        bus_id = event.bus_id
        stop_id = event.stop_id
        stopping_place = event.metadata.get('stopping_place', stop_id)
        
        try:
            # Physics Override Fix: setMaxSpeed only caps the vehicle mechanically,
            # but remains bounded by the lane's physical speed limit.
            # Using setSpeedFactor() instructs the driver to treat the lane speed limit multiplied by the factor.
            traci.vehicle.setSpeedFactor(bus_id, speed_ratio)

            # Since we applied pre-emptive holding (duration=3600) in _collect_new_events,
            # the stop should still be active and at index 0.
            # We simply update it to the actual desired duration.
            traci.vehicle.setStopParameter(bus_id, 0, "duration", str(duration))
        except traci.TraCIException as e:
            # If it still fails, log it but don't crash.
            # This might happen if the bus somehow left the stop despite our best efforts.
            print(f"WARNING: Failed to apply holding to {bus_id} at {stop_id}. Error: {e}")
            pass

        key = (event.line_id, bus_id, stop_id)
        if key not in self.pending_events:
            self.pending_events[key] = event
        self.active_events[bus_id] = event
        event.assigned_action = hold_value
        bus_obj = self.bus_obj_dic.get(bus_id)
        if bus_obj and stop_id in bus_obj.just_server_stop_data_d:
            arrive_time = bus_obj.just_server_stop_data_d[stop_id][0]
            bus_obj.just_server_stop_data_d[stop_id][1] = arrive_time + duration
        event.metadata['applied_duration'] = duration

    # endregion

    # region simulation stepping
    def _advance_until_event(self) -> None:
        while not self.done and not self.decision_queue:
            self._advance_one_step()

    def _advance_one_step(self) -> None:
        if self.done:
            return
        simulation_current_time = traci.simulation.getTime()
        
        # Update active buses
        if self.steps == 0:
            # On the first step, check ALL existing vehicles to catch those that departed at time 0
            initial_ids = traci.vehicle.getIDList()
            for veh_id in initial_ids:
                if veh_id in self.bus_obj_dic:
                    self.active_bus_ids.add(veh_id)
    
        
        departed_ids = traci.simulation.getDepartedIDList()
        for veh_id in departed_ids:
            if veh_id in self.bus_obj_dic:
                self.active_bus_ids.add(veh_id)
                self._launch_from_terminal(veh_id, simulation_current_time)

        arrived_ids = traci.simulation.getArrivedIDList()
        for veh_id in arrived_ids:
            if veh_id in self.active_bus_ids:
                self.active_bus_ids.discard(veh_id)
                self.just_departed_buses.append(veh_id)
                self._return_to_terminal(veh_id, simulation_current_time)
                
        # Update active passengers
        departed_persons = traci.simulation.getDepartedPersonIDList()
        for person_id in departed_persons:
            if person_id in self.passenger_obj_dic:
                self.active_passenger_ids.add(person_id)
                # CRITICAL: Activate immediately so they are visible to buses (passable_line_l populated)
                self.passenger_obj_dic[person_id].passenger_activate(simulation_current_time, self.line_obj_dic)
                
        arrived_persons = traci.simulation.getArrivedPersonIDList()
        for person_id in arrived_persons:
            if person_id in self.active_passenger_ids:
                self.active_passenger_ids.discard(person_id)

        # Optimization: Update stops and passengers only every update_freq steps
        if self.steps % self.update_freq == 0:
            for stop in self.stop_obj_dic.values():
                stop.update_stop_state()
            
            # Iterate only active passengers
            for passenger_id in list(self.active_passenger_ids):
                passenger_obj = self.passenger_obj_dic[passenger_id]
                # Note: passenger_activate is now handled immediately on departure
                if passenger_obj.passenger_state_s != "No":
                    passenger_obj.passenger_run(simulation_current_time, self.line_obj_dic)

        # CRITICAL: Bus logic must run EVERY step to detect arrivals/departures correctly
        # Otherwise, short stops (duration=0) are missed if they fall between update intervals
        for vehicle_id in list(self.active_bus_ids):
            bus_obj = self.bus_obj_dic[vehicle_id]
            line_obj = self.line_obj_dic[bus_obj.belong_line_id_s]
            if bus_obj.bus_state_s == "No":
                bus_obj.bus_activate(line_obj, self.stop_obj_dic, self.signal_obj_dic, simulation_current_time)
            else:
                bus_obj.bus_running(
                    line_obj,
                    self.stop_obj_dic,
                    self.signal_obj_dic,
                    self.passenger_obj_dic,
                    simulation_current_time,
                    self.BusCap,
                    self.AveAlightingTime,
                    self.AveBoardingTime,
                    self.bus_arrstation_od_otd_dict,
                    self.bus_obj_dic,
                    self.involved_tl_ID_l,
                    self.busline_edge_order,
                )

        self._collect_new_events(simulation_current_time)

        traci.simulationStep()
        self.current_time = traci.simulation.getTime()
        self.steps += 1

        self._check_station_capacity()

        self._cleanup_departed_buses()
        self._check_termination()

    def _check_station_capacity(self):
        """
        Silent Sentinel: Enforce station capacity limits.
        If a station is full and another bus is waiting to enter, forcibly evict the
        oldest parked bus by truncating its holding duration to 0.
        """
        for stop_id, capacity in self.station_capacities.items():
            try:
                # Get buses currently parked at this stop
                parked_buses = traci.busstop.getVehicleIDs(stop_id)
                if len(parked_buses) >= capacity:
                    stop_obj = self.stop_obj_dic.get(stop_id)
                    if not stop_obj or not stop_obj.at_lane_s:
                        continue
                    
                    # Check if there are buses waiting to enter
                    vehicles_on_lane = traci.lane.getLastStepVehicleIDs(stop_obj.at_lane_s)
                    waiting_buses = []
                    for vid in vehicles_on_lane:
                        if vid in parked_buses:
                            continue
                        if vid in self.bus_obj_dic:
                            # If the bus is slowing down/queued
                            speed = traci.vehicle.getSpeed(vid)
                            if speed < 1.0:
                                stops = traci.vehicle.getStops(vid, 1)
                                if stops and (stops[0].stoppingPlaceID == stop_id or stops[0].lane == stop_obj.at_lane_s):
                                    waiting_buses.append(vid)
                                    
                    # If full and someone is waiting, evict the first parked bus (longest waiter)
                    if waiting_buses:
                        oldest_bus_id = parked_buses[0]
                        traci.vehicle.setStopParameter(oldest_bus_id, 0, "duration", "0")
            except Exception:
                pass

    def _find_neighbors(self, line_id: str, current_bus_id: str) -> Tuple[Optional[Any], Optional[Any]]:
        # Find all buses on this line
        buses = [b for b in self.bus_obj_dic.values() if b.belong_line_id_s == line_id]
        # Sort by start_time_n (earlier start time = forward bus)
        buses.sort(key=lambda b: b.start_time_n)
        
        try:
            current_idx = [b.bus_id_s for b in buses].index(current_bus_id)
        except ValueError:
            return None, None
            
        forward_bus = buses[current_idx - 1] if current_idx > 0 else None
        backward_bus = buses[current_idx + 1] if current_idx < len(buses) - 1 else None
        
        return forward_bus, backward_bus

    def _compute_robust_headway(self, subject_bus, front_bus, stop_id, current_time, target_headway=360.0, force_distance=False) -> float:
        if not front_bus or not subject_bus:
            return target_headway
            
        # 1. Try trajectory-based calculation (if front bus has arrived at this stop)
        if not force_distance and stop_id in front_bus.trajectory_dict and front_bus.trajectory_dict[stop_id]:
            front_arrive_time = front_bus.trajectory_dict[stop_id][-1]
            return current_time - front_arrive_time
            
        # 2. Fallback to distance-based calculation
        dist_diff = subject_bus.distance_n - front_bus.distance_n
        
        # Calculate subject average speed
        duration = current_time - subject_bus.start_time_n
        if duration <= 1.0:
             return target_headway
             
        avg_speed = subject_bus.distance_n / duration
        if avg_speed < 0.1:
            return target_headway
            
        headway = -dist_diff / avg_speed
        return max(headway, 0.0) # Ensure positive

    def _collect_new_events(self, simulation_current_time: float) -> None:
        for bus_id, bus_obj in self.bus_obj_dic.items():
            if bus_obj.bus_state_s != "Stop" or not bus_obj.just_server_stop_data_d:
                continue
            stop_id = next(iter(bus_obj.just_server_stop_data_d.keys()))
            key = (bus_obj.belong_line_id_s, bus_id, stop_id)
            if key in self.pending_events:
                continue
            arrive_time, predicted_depart_time = bus_obj.just_server_stop_data_d[stop_id]
            base_duration = max(predicted_depart_time - arrive_time, 0.0)
            
            # Timing Alignment: Only process if passenger exchange is done
            # This ensures we observe the state AFTER passengers have boarded/alighted, matching original logic
            if simulation_current_time < arrive_time + base_duration:
                continue

            # Prevent duplicate events for the same stop visit
            if getattr(bus_obj, 'last_action_time', -1.0) == arrive_time:
                continue
            bus_obj.last_action_time = arrive_time

            line_id = bus_obj.belong_line_id_s
            
            # Find neighbors for robust headway calculation
            forward_bus, backward_bus = self._find_neighbors(line_id, bus_id)
            
            # "No Control" Logic: Skip isolated buses
            if not forward_bus and not backward_bus:
                continue
                
            # Dynamic Target Headway Calculation
            default_line_headway = self.line_headways.get(line_id, self.default_headway)
            
            target_forward = default_line_headway
            if forward_bus:
                target_forward = abs(bus_obj.start_time_n - forward_bus.start_time_n)
                
            target_backward = default_line_headway
            if backward_bus:
                target_backward = abs(backward_bus.start_time_n - bus_obj.start_time_n)
                
            # Service Completion Time: When passenger exchange finishes (Arrive + Duration)
            # This is the exact time point used for Headway calculation in LSTM-RL
            service_completion_time = arrive_time + base_duration

            # Robust Headway Calculation
            # Forward Headway: Me - Front
            # Use service_completion_time instead of simulation_current_time for precision
            forward_headway = self._compute_robust_headway(bus_obj, forward_bus, stop_id, service_completion_time, target_headway=target_forward)
            
            # Store this forward headway for the backward bus to use later
            bus_obj.last_forward_headway = forward_headway

            # Backward Headway: Back - Me
            # Note: We pass backward_bus as subject, bus_obj as front
            # CRITICAL UPDATE: Use the backward bus's *stored* forward headway (stale but exact), matching LSTM-RL logic.
            # If backward bus hasn't recorded a headway yet (e.g. just started), fallback to target.
            if backward_bus and backward_bus.last_forward_headway is not None:
                backward_headway = backward_bus.last_forward_headway
            else:
                backward_headway = target_backward

            stop_idx = self.stop_indices[line_id].get(stop_id, -1)
            direction = 1 if line_id.endswith('S') else 0
            waiting_passengers = len(traci.busstop.getPersonIDs(stop_id))
            
            # 1. Segment Mean Speed
            stop_internal_obj = self.stop_obj_dic.get(stop_id)
            segment_mean_speed = 10.0
            if stop_internal_obj and stop_internal_obj.at_lane_s:
                try:
                    segment_mean_speed = traci.lane.getLastStepMeanSpeed(stop_internal_obj.at_lane_s)
                except Exception:
                    pass
            
            # 2. Co-Line Headways
            # Set a baseline value matching the scheduled headway of the current line
            co_line_fwd = default_line_headway
            co_line_bwd = default_line_headway
            if stop_internal_obj:
                my_pos = self.line_stop_distances[line_id].get(stop_id, bus_obj.distance_n)
                # Find other lines sharing this stop
                shared_lines = [l for l in stop_internal_obj.service_line_l if l != line_id]
                fwd_dists = []
                bwd_dists = []
                for other_line in shared_lines:
                    if stop_id not in self.line_stop_distances[other_line]:
                        continue
                    stop_pos_on_other_line = self.line_stop_distances[other_line][stop_id]
                    # Find active buses on this other line
                    for other_bus_id, other_bus in self.bus_obj_dic.items():
                        if other_bus.belong_line_id_s == other_line and other_bus.bus_state_s != "No":
                            # Relative distance
                            dist_diff = stop_pos_on_other_line - other_bus.distance_n
                            if dist_diff > 0: # other bus is before the stop (backward bus)
                                bwd_dists.append(dist_diff)
                            elif dist_diff < 0: # other bus has passed the stop (forward bus)
                                fwd_dists.append(-dist_diff)
                
                # Convert spatial distance to temporal headway using current segment speed
                # Limit effective speed to >= 1.0 m/s to prevent division by near-zero causing exploding values
                effective_speed = max(segment_mean_speed, 1.0)
                
                # Cap the maximum returned headway to 2x the default headway (e.g. 720s) to prevent outliers
                max_headway_cap = default_line_headway * 2.0
                
                if fwd_dists:
                    co_line_fwd = min(min(fwd_dists) / effective_speed, max_headway_cap)
                if bwd_dists:
                    co_line_bwd = min(min(bwd_dists) / effective_speed, max_headway_cap)

            stop_data = traci.vehicle.getStops(bus_id, 1)
            stopping_place = stop_data[0].stoppingPlaceID if stop_data else stop_id
            
            # Resolve physical fleet index for obs[2]
            _fleet_idx = -1
            _fb = self.fleet_obj_dic.get(bus_id)
            if _fb is not None:
                # fleet_id is e.g. '7S_fleet_3' — extract the numeric suffix
                try:
                    _fleet_idx = int(_fb.fleet_id.rsplit('_', 1)[-1])
                except (ValueError, IndexError):
                    _fleet_idx = -1

            event = DecisionEvent(
                line_id=line_id,
                bus_id=bus_id,
                stop_id=stop_id,
                stop_idx=stop_idx,
                direction=direction,
                sim_time=simulation_current_time,
                forward_headway=forward_headway,
                backward_headway=backward_headway,
                waiting_passengers=waiting_passengers,
                base_stop_duration=base_duration,
                forward_bus_present=bool(forward_bus),
                backward_bus_present=bool(backward_bus),
                target_forward_headway=target_forward,
                target_backward_headway=target_backward,
                co_line_forward_headway=co_line_fwd,
                co_line_backward_headway=co_line_bwd,
                segment_mean_speed=segment_mean_speed,
                metadata={'arrive_time': arrive_time, 'stopping_place': stopping_place},
                fleet_idx=_fleet_idx,
            )
            
            # Pre-emptive Holding:
            # Immediately set a large duration to keep the stop active in SUMO while the agent decides.
            # This prevents the "not downstream" error caused by SUMO finishing the stop prematurely.
            try:
                # Use setStopParameter to modify the current stop (index 0)
                traci.vehicle.setStopParameter(bus_id, 0, "duration", "3600.0")
            except traci.TraCIException:
                pass

            self.decision_queue.append(event)
            self.pending_events[key] = event
            self.arrival_history[line_id][stop_id].append(arrive_time)
            
            # Populate trajectory_dict with SERVICE COMPLETION time
            # This matches LSTM-RL logic: current_time + holding_time
            bus_obj.trajectory_dict[stop_id].append(service_completion_time)

    def _cleanup_departed_buses(self) -> None:
        to_remove = []
        for bus_id, event in self.active_events.items():
            bus_obj = self.bus_obj_dic.get(bus_id)
            if not bus_obj or bus_obj.bus_state_s != "Stop":
                stop_id = event.stop_id
                line_id = event.line_id
                if bus_obj and stop_id in bus_obj.depart_stop_time_d:
                    depart_time = bus_obj.depart_stop_time_d[stop_id]
                    self.depart_history[line_id][stop_id] = depart_time
                key = (line_id, bus_id, stop_id)
                self.pending_events.pop(key, None)
                to_remove.append(bus_id)
        for bus_id in to_remove:
            self.active_events.pop(bus_id, None)

    def _check_termination(self) -> None:
        if self.steps >= self.max_steps:
            self.done = True
        elif traci.simulation.getMinExpectedNumber() <= 0:
            self.done = True
        if self.done:
            self._cleanup_departed_buses()
            self.close()

    def _compute_forward_headway(self, line_id: str, stop_id: str, arrive_time: float) -> float:
        history = self.arrival_history[line_id][stop_id]
        if history:
            last_time = history[-1]
            gap = arrive_time - last_time
            return gap if gap > 0 else self.default_headway
        return self.default_headway

    def _compute_backward_headway(self, line_id: str, stop_id: str) -> float:
        depart_time = self.depart_history[line_id].get(stop_id)
        arrive_times = self.arrival_history[line_id][stop_id]
        if depart_time is not None and arrive_times:
            gap = depart_time - arrive_times[-1]
            if gap > 0:
                return gap
        return self.default_headway

    # endregion

    # region Fleet reuse (FIFO) — mirrors LSTM-RL sim.py launch_bus()

    @property
    def bus_in_terminal(self) -> List[FleetBus]:
        """All FleetBus objects currently resting at the terminal (not on route)."""
        return [fb for fb in self.fleet_obj_dic.values() if not fb.on_route]

    def _return_to_terminal(self, bus_id: str, current_time: float) -> None:
        """Called when a SUMO bus finishes its route and is removed by SUMO.

        Marks the corresponding FleetBus as in-terminal and enqueues it
        in the FIFO pool for the correct direction.
        """
        fb = self.fleet_obj_dic.get(bus_id)
        if fb is None:
            return
        fb.on_route = False
        fb.back_to_terminal_time = current_time
        # Enqueue FIFO — pool ordered by insertion (= arrival) time
        self.terminal_pool[fb.direction].append(fb)

    def _launch_from_terminal(
        self, new_bus_id: str, departure_time: float
    ) -> FleetBus:
        """Called when a new SUMO bus departs.

        Reuse the earliest-arrived FleetBus from the same direction/line
        (FIFO, exactly like LSTM-RL sim.py), or create a new one if the
        pool is empty.
        """
        bus_obj = self.bus_obj_dic.get(new_bus_id)
        if bus_obj is None:
            return None  # type: ignore[return-value]
        line_id  = bus_obj.belong_line_id_s
        direction = 1 if line_id.endswith('S') else 0

        pool = self.terminal_pool[direction]
        # FIFO: pick the fleet bus that returned to terminal earliest
        # (deque is ordered by arrival time — leftmost = earliest)
        candidate = None
        for fb in pool:
            if fb.line_id == line_id:
                candidate = fb
                pool.remove(fb)   # O(n) but pool is small (≤ fleet size)
                break

        if candidate is not None:
            # Reuse existing fleet bus — update its trip assignment
            candidate.reset_trip(new_bus_id, departure_time)
            # Remap: old trip_id entry stays in fleet_obj_dic for history;
            # also register under new_bus_id for fast lookup
            self.fleet_obj_dic[new_bus_id] = candidate
        else:
            # No matching bus in terminal — allocate a new FleetBus
            idx = self._fleet_counter.get(line_id, 0)
            fleet_id = f"{line_id}_fleet_{idx}"
            self._fleet_counter[line_id] = idx + 1
            candidate = FleetBus(
                fleet_id=fleet_id,
                direction=direction,
                line_id=line_id,
                current_trip_id=new_bus_id,
                on_route=True,
            )
            candidate.trip_ids.append(new_bus_id)
            self.fleet_obj_dic[new_bus_id] = candidate

        return candidate

    # endregion



def build_bridge(*, root_dir: str, sumo_cfg: str = "control_sim_traci_period.sumocfg", gui: bool = False, update_freq: int = 10, scale: float = 1.0) -> Dict[str, Callable]:
    bridge = SumoRLBridge(root_dir=root_dir, sumo_cfg=sumo_cfg, gui=gui, update_freq=update_freq, scale=scale)

    def decision_provider() -> Tuple[List[DecisionEvent], bool]:
        return bridge.fetch_events()

    def action_executor(event: DecisionEvent, action_value: float) -> None:
        bridge.apply_action(event, action_value)

    def reset_callback() -> None:
        bridge.reset()

    def close_callback() -> None:
        bridge.close()

    return {
        'decision_provider': decision_provider,
        'action_executor': action_executor,
        'reset_callback': reset_callback,
        'close_callback': close_callback,
    }

