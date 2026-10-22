import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
import time
import random
import argparse
import math
import tempfile
import uuid
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Set, Any

if "SUMO_HOME" in os.environ:
    tools_path = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools_path not in sys.path:
        sys.path.append(tools_path)
else:
    raise EnvironmentError("Please declare environment variable 'SUMO_HOME'!")

import traci

from sumolib import checkBinary
import traci.constants as tc

from .case import f_8_create_obj
from .case import e_8_gurobi_test_considerbusnum_V3
from .case import d_8_compute_running_time

from .rl_env import DecisionEvent

try:
    from bus_h2o.passenger_time import PassengerTimeLedger, PassengerTimeSnapshot
except ModuleNotFoundError:
    from passenger_time import PassengerTimeLedger, PassengerTimeSnapshot


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
    ALIGHTING_SERVICE_SECONDS_PER_PASSENGER = 1.5
    BOARDING_SERVICE_SECONDS_PER_PASSENGER = 2.5

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
        seed: Optional[int] = None,
        passenger_obj_xml: Optional[str] = None,
        edge_xml: Optional[str] = None,
        sumo_backend: str = "libsumo",
    ) -> None:
        self.root_dir = os.path.abspath(root_dir)
        self.sumo_cfg = sumo_cfg if os.path.isabs(sumo_cfg) else os.path.join(self.root_dir, sumo_cfg)
        self.gui = gui
        self.max_steps = max_steps
        self.time_step = time_step
        self.default_headway = default_headway
        self.update_freq = update_freq
        self.scale = scale
        self.max_hold_seconds = 60.0
        self.seed = None if seed is None else int(seed)
        self.passenger_obj_xml = (
            None
            if passenger_obj_xml is None
            else os.path.abspath(passenger_obj_xml)
        )
        self.edge_xml = (
            os.path.abspath(edge_xml)
            if edge_xml is not None
            else os.path.join(
                self.root_dir,
                "intersection_delay",
                "a_sorted_busline_edge.xml",
            )
        )
        if sumo_backend not in {"libsumo", "traci"}:
            raise ValueError("sumo_backend must be 'libsumo' or 'traci'")
        if gui and sumo_backend != "traci":
            raise ValueError("SUMO GUI requires the explicit traci backend")
        self.requested_sumo_backend = sumo_backend
        self.sumo_backend: Optional[str] = None
        self._start_state_path = os.path.join(
            tempfile.gettempdir(),
            f"transit_sumo_start_{os.getpid()}_{uuid.uuid4().hex}.sbx",
        )
        self._runtime_sumo_cfg_path: Optional[str] = None

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
        self.AveAlightingTime = (
            self.ALIGHTING_SERVICE_SECONDS_PER_PASSENGER
        )
        self.AveBoardingTime = self.BOARDING_SERVICE_SECONDS_PER_PASSENGER

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
        self.departed_passenger_ids: Set[str] = set()
        self.departed_passenger_times: Dict[str, float] = {}
        self.completed_passenger_times: Dict[str, float] = {}
        self.departed_trip_ids: Set[str] = set()
        self.completed_trip_times: Dict[str, float] = {}
        self.passenger_time_ledger = PassengerTimeLedger()
        self._completed_trip_passenger_times: Dict[
            str, PassengerTimeSnapshot
        ] = {}
        self._passenger_time_subscribed_bus_ids: Set[str] = set()
        self._passenger_time_count_syncs = 0
        self._passenger_time_subscription_fallbacks = 0
        self._passenger_time_direct_audits = 0
        self._passenger_time_max_waiting_count_error = 0
        self._passenger_time_max_in_vehicle_count_error = 0
        self._station_capacity_subscription_fallbacks = 0
        self._station_capacity_observation_errors = 0
        self._station_capacity_saturation_station_steps = 0
        self._station_capacity_max_occupancy: Dict[str, int] = {}
        self._station_capacity_forced_evictions = 0
        self._protected_base_service_visits: Set[Tuple[str, str, str]] = set()
        self._sumo_bus_type_boarding_durations: Dict[str, float] = {}
        self._bus_onboard_passenger_ids: Dict[str, Set[str]] = {}
    # region SUMO lifecycle
    def reset(self) -> None:
        # Pre-cleanup: break references in old bus/fleet objects to help GC
        # reclaim memory between episodes (trajectory_dict etc. accumulate)
        if self.bus_obj_dic:
            for _bo in self.bus_obj_dic.values():
                if hasattr(_bo, 'trajectory_dict'):
                    _bo.trajectory_dict.clear()
                if hasattr(_bo, 'service_completion_history'):
                    _bo.service_completion_history.clear()
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
            if self.sumo_backend == "libsumo":
                traci.simulation.saveState(self._start_state_path)
            self.first_run = False
        else:
            # Soft reset using saved state
            if self.sumo_backend == "libsumo":
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
                            traci.simulation.loadState(self._start_state_path)
                        finally:
                            os.dup2(old_stdout, 1)
                            os.dup2(old_stderr, 2)
                            os.close(old_stdout)
                            os.close(old_stderr)
                    except Exception:
                        # Fallback if redirection fails
                        traci.simulation.loadState(self._start_state_path)
                self._configure_sumo_bus_service_durations()
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
        self.passenger_time_ledger.reset(self.current_time)
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
        self.departed_passenger_ids.clear()
        self.departed_passenger_times.clear()
        self.completed_passenger_times.clear()
        self.departed_trip_ids.clear()
        self.completed_trip_times.clear()
        self._completed_trip_passenger_times.clear()
        self._passenger_time_count_syncs = 0
        self._passenger_time_subscription_fallbacks = 0
        self._passenger_time_direct_audits = 0
        self._passenger_time_max_waiting_count_error = 0
        self._passenger_time_max_in_vehicle_count_error = 0
        self._station_capacity_subscription_fallbacks = 0
        self._station_capacity_observation_errors = 0
        self._station_capacity_saturation_station_steps = 0
        self._station_capacity_max_occupancy.clear()
        self._station_capacity_forced_evictions = 0
        self._protected_base_service_visits.clear()
        self._bus_onboard_passenger_ids.clear()

        # Reset fleet terminal pool (keep FleetBus objects, just clear queues)
        for q in self.terminal_pool.values():
            q.clear()
        # Mark all fleet buses as not on route
        for fb in self.fleet_obj_dic.values():
            fb.on_route = False
            fb.back_to_terminal_time = float('inf')

    def close(self) -> None:
        # Optimization: Do NOT close traci if using libsumo reuse strategy
        if self.sumo_backend != "libsumo":
            if traci.isLoaded():
                traci.close()
        
        self.initialized = False
        self.active_bus_ids = set()
        self.active_passenger_ids = set()

    def shutdown(self) -> None:
        """Close the backend and remove this bridge's private runtime files."""

        try:
            if traci.isLoaded():
                traci.close()
        except Exception:
            pass
        self.initialized = False
        try:
            os.remove(self._start_state_path)
        except FileNotFoundError:
            pass
        if self._runtime_sumo_cfg_path is not None:
            try:
                os.remove(self._runtime_sumo_cfg_path)
            except FileNotFoundError:
                pass
            self._runtime_sumo_cfg_path = None

    def _ensure_sumo_home(self) -> None:
        if "SUMO_HOME" not in os.environ:
            raise EnvironmentError("Please declare environment variable 'SUMO_HOME'!")
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools not in sys.path:
            sys.path.append(tools)

    def _start_traci(self) -> None:
        global traci
        runtime_sumo_cfg = self._prepare_runtime_sumo_config()
        common_options = [
            "-c", runtime_sumo_cfg,
            "--no-warnings",
            "--no-step-log",
            "--duration-log.disable",
            "--log", "/dev/null",
            "--scale", str(self.scale),
        ]
        if self.seed is not None:
            common_options.extend(("--seed", str(self.seed)))
        if self.requested_sumo_backend == "traci":
            if bool(getattr(traci, "isLibsumo", lambda: False)()):
                raise RuntimeError(
                    "requested traci but the process imported traci in libsumo mode"
                )
            binary = checkBinary('sumo-gui')
            if not self.gui:
                binary = checkBinary('sumo')
            traci.start([binary, *common_options])
            self.sumo_binary = binary
            self.sumo_backend = "traci"
        else:
            if not bool(getattr(traci, "isLibsumo", lambda: False)()):
                raise RuntimeError(
                    "requested libsumo but traci was not imported in libsumo mode"
                )
            try:
                import libsumo
            except ImportError as error:
                raise RuntimeError("requested libsumo is not installed") from error
            sys.modules['traci'] = libsumo
            traci = libsumo
            traci.start(["sumo", *common_options])
            self.sumo_binary = "libsumo"
            self.sumo_backend = "libsumo"
        if self.sumo_backend != self.requested_sumo_backend:
            raise RuntimeError(
                f"SUMO backend drifted from {self.requested_sumo_backend} "
                f"to {self.sumo_backend}"
            )
        self._configure_sumo_bus_service_durations()

    def _configure_sumo_bus_service_durations(self) -> None:
        configured = {}
        for type_id in traci.vehicletype.getIDList():
            if traci.vehicletype.getVehicleClass(type_id) != "bus":
                continue
            traci.vehicletype.setBoardingDuration(
                type_id, self.BOARDING_SERVICE_SECONDS_PER_PASSENGER
            )
            observed = float(traci.vehicletype.getBoardingDuration(type_id))
            if not np.isclose(
                observed,
                self.BOARDING_SERVICE_SECONDS_PER_PASSENGER,
                rtol=0.0,
                atol=1e-9,
            ):
                raise RuntimeError(
                    f"SUMO bus type {type_id} retained boarding duration {observed}"
                )
            configured[str(type_id)] = observed
        if not configured:
            raise RuntimeError("SUMO configuration contains no bus vehicle type")
        self._sumo_bus_type_boarding_durations = configured

    def _prepare_runtime_sumo_config(self) -> str:
        """Strip diagnostic outputs without changing the simulation inputs.

        Supplying ``--fcd-output /dev/null`` still makes SUMO construct an FCD
        record for every selected vehicle at every step.  Canonical rollouts do
        not consume those records, so remove output-only configuration before
        startup.  The temporary file lives beside the source config so relative
        network and route paths keep exactly the same resolution semantics.
        """

        if self._runtime_sumo_cfg_path is not None:
            return self._runtime_sumo_cfg_path

        tree = ET.parse(self.sumo_cfg)
        root = tree.getroot()
        changed = False
        for section_name in ("output", "fcd_device"):
            section = root.find(section_name)
            if section is not None:
                root.remove(section)
                changed = True

        report = root.find("report")
        if report is not None:
            for option in list(report):
                if option.tag in {"log", "duration-log.statistics"}:
                    report.remove(option)
                    changed = True

        if not changed:
            return self.sumo_cfg

        file_descriptor, runtime_path = tempfile.mkstemp(
            prefix=".transit_runtime_",
            suffix=".sumocfg",
            dir=os.path.dirname(self.sumo_cfg),
        )
        try:
            with os.fdopen(file_descriptor, "wb") as handle:
                tree.write(handle, encoding="utf-8", xml_declaration=True)
        except Exception:
            try:
                os.remove(runtime_path)
            except FileNotFoundError:
                pass
            raise
        self._runtime_sumo_cfg_path = runtime_path
        return runtime_path

    def _load_objects(self) -> None:
        result = f_8_create_obj.create_obj_fun(
            self.passenger_obj_xml,
            initialize_obj_dir=os.path.join(self.root_dir, "initialize_obj"),
        )
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
            bus.service_completion_history = defaultdict(list)

        (
            _,
            _,
            _,
            _,
            line_station_od_otd_dict,
            bus_arrstation_od_otd_dict,
            _,
            self.BusCap,
            _legacy_alighting_time,
            _legacy_boarding_time,
        ) = e_8_gurobi_test_considerbusnum_V3.get_static_info(self.line_obj_dic)
        self.bus_arrstation_od_otd_dict = bus_arrstation_od_otd_dict
        # The canonical control simulator overrides the legacy optimization
        # constants with 1.5 s/alighting and 2.5 s/boarding in g_8_SUMO.py.
        self.AveAlightingTime = (
            self.ALIGHTING_SERVICE_SECONDS_PER_PASSENGER
        )
        self.AveBoardingTime = self.BOARDING_SERVICE_SECONDS_PER_PASSENGER

        edge_file = ET.parse(self.edge_xml)
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

        self._initialize_passenger_time_subscriptions()

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

    def consume_completed_trip_passenger_times(
        self,
    ) -> Dict[str, PassengerTimeSnapshot]:
        """Return exact terminal contexts accumulated since the previous call."""

        completed = dict(self._completed_trip_passenger_times)
        self._completed_trip_passenger_times.clear()
        return completed

    def _initialize_passenger_time_subscriptions(self) -> None:
        """Subscribe to aggregate welfare and station-capacity variables."""

        self._passenger_time_subscribed_bus_ids.clear()
        for stop_id in self.stop_obj_dic:
            traci.busstop.subscribe(
                stop_id,
                (
                    tc.VAR_BUS_STOP_WAITING,
                    tc.VAR_STOP_STARTING_VEHICLES_IDS,
                ),
            )

    @staticmethod
    def _next_passable_lines(passenger: Any, line_obj_dic: Dict[str, Any]) -> List[str]:
        next_stop = passenger.transfer_stop_id_s or passenger.end_stop_id_s
        return [
            line_id
            for line_id, line in line_obj_dic.items()
            if passenger.start_stop_id_s in line.stop_id_l
            and next_stop in line.stop_id_l
        ]

    def _record_passenger_alighting(
        self, passenger: Any, stop_id: str, simulation_current_time: float
    ) -> None:
        passenger.passenger_state_s = "Stop"
        passenger.alighting_time_n = simulation_current_time
        passenger.travel_data_l.append(
            [
                passenger.arriver_time_n,
                passenger.boarding_time_n,
                passenger.alighting_time_n,
                passenger.boarding_time_n - passenger.arriver_time_n,
                passenger.alighting_time_n - passenger.arriver_time_n,
                passenger.take_bus_id_s,
            ]
        )
        passenger.arriver_time_n = simulation_current_time
        if passenger.transfer_stop_id_s == stop_id:
            passenger.passable_line_l = [
                line_id
                for line_id, line in self.line_obj_dic.items()
                if stop_id in line.stop_id_l
                and passenger.end_stop_id_s in line.stop_id_l
            ]
        passenger.last_appearance_n = simulation_current_time

    def _synchronize_bus_passenger_transition(
        self,
        bus_id: str,
        previous_state: str,
        current_state: str,
        bus_obj: Any,
        simulation_current_time: float,
    ) -> None:
        """Update legacy person records only at bus stop boundaries."""

        previous_onboard = self._bus_onboard_passenger_ids.setdefault(
            bus_id, set()
        )
        if previous_state != "Stop" and current_state == "Stop":
            stop_id = next(iter(bus_obj.just_server_stop_data_d), None)
            if stop_id is None:
                return
            for passenger_id in previous_onboard:
                passenger = self.passenger_obj_dic.get(passenger_id)
                if passenger is None:
                    continue
                destination = (
                    passenger.transfer_stop_id_s or passenger.end_stop_id_s
                )
                if destination == stop_id:
                    self._record_passenger_alighting(
                        passenger, stop_id, simulation_current_time
                    )
            return

        if previous_state == "Stop" and current_state != "Stop":
            current_onboard = set(traci.vehicle.getPersonIDList(bus_id))
            for passenger_id in current_onboard:
                passenger = self.passenger_obj_dic.get(passenger_id)
                if passenger is None:
                    continue
                if (
                    passenger_id not in previous_onboard
                    or passenger.passenger_state_s != "Bus"
                ):
                    passenger.passenger_state_s = "Bus"
                    passenger.take_bus_id_s = bus_id
                    passenger.boarding_time_n = simulation_current_time
                    passenger.last_appearance_n = simulation_current_time
            self._bus_onboard_passenger_ids[bus_id] = current_onboard

    def _subscribe_bus_passenger_count(self, bus_id: str) -> None:
        if bus_id in self._passenger_time_subscribed_bus_ids:
            return
        traci.vehicle.subscribe(bus_id, (tc.VAR_PERSON_NUMBER,))
        self._passenger_time_subscribed_bus_ids.add(bus_id)

    def _subscription_passenger_counts(
        self, bus_ids: Set[str]
    ) -> Tuple[int, int]:
        stop_results = traci.busstop.getAllSubscriptionResults() or {}
        waiting = 0
        for stop_id in self.stop_obj_dic:
            values = stop_results.get(stop_id) or {}
            value = values.get(tc.VAR_BUS_STOP_WAITING)
            if value is None:
                value = traci.busstop.getPersonCount(stop_id)
                self._passenger_time_subscription_fallbacks += 1
            waiting += int(value)

        vehicle_results = traci.vehicle.getAllSubscriptionResults() or {}
        in_vehicle = 0
        for bus_id in bus_ids:
            self._subscribe_bus_passenger_count(bus_id)
            values = vehicle_results.get(bus_id) or {}
            value = values.get(tc.VAR_PERSON_NUMBER)
            if value is None:
                value = traci.vehicle.getPersonNumber(bus_id)
                self._passenger_time_subscription_fallbacks += 1
            in_vehicle += int(value)
        return waiting, in_vehicle

    def _direct_passenger_counts(self, bus_ids: Set[str]) -> Tuple[int, int]:
        waiting = sum(
            int(traci.busstop.getPersonCount(stop_id))
            for stop_id in self.stop_obj_dic
        )
        in_vehicle = sum(
            int(traci.vehicle.getPersonNumber(bus_id)) for bus_id in bus_ids
        )
        return waiting, in_vehicle

    def _synchronize_passenger_time_counts(
        self, simulation_current_time: float
    ) -> None:
        waiting, in_vehicle = self._subscription_passenger_counts(
            set(self.active_bus_ids)
        )
        self.passenger_time_ledger.set_counts(
            simulation_current_time,
            waiting_passengers=waiting,
            in_vehicle_passengers=in_vehicle,
        )
        self._passenger_time_count_syncs += 1

        # Periodic direct queries independently detect stale subscription values.
        if self.steps % 600 == 0:
            direct_waiting, direct_in_vehicle = self._direct_passenger_counts(
                set(self.active_bus_ids)
            )
            wait_error = abs(waiting - direct_waiting)
            in_vehicle_error = abs(in_vehicle - direct_in_vehicle)
            self._passenger_time_direct_audits += 1
            self._passenger_time_max_waiting_count_error = max(
                self._passenger_time_max_waiting_count_error, wait_error
            )
            self._passenger_time_max_in_vehicle_count_error = max(
                self._passenger_time_max_in_vehicle_count_error,
                in_vehicle_error,
            )
            if wait_error or in_vehicle_error:
                raise RuntimeError(
                    "SUMO passenger-count subscriptions disagree with direct "
                    f"queries: waiting={wait_error}, in_vehicle={in_vehicle_error}"
                )

    def audit_passenger_time_counts(self) -> dict:
        """Synchronize final counts and return an independently checked audit."""

        loaded_bus_ids = {
            bus_id
            for bus_id in traci.vehicle.getIDList()
            if bus_id in self.bus_obj_dic
        }
        subscribed = self._subscription_passenger_counts(loaded_bus_ids)
        direct = self._direct_passenger_counts(loaded_bus_ids)
        wait_error = int(subscribed[0] - direct[0])
        in_vehicle_error = int(subscribed[1] - direct[1])
        self._passenger_time_direct_audits += 1
        self._passenger_time_max_waiting_count_error = max(
            self._passenger_time_max_waiting_count_error, abs(wait_error)
        )
        self._passenger_time_max_in_vehicle_count_error = max(
            self._passenger_time_max_in_vehicle_count_error,
            abs(in_vehicle_error),
        )
        if wait_error or in_vehicle_error:
            raise RuntimeError(
                "final SUMO passenger-count audit failed: "
                f"waiting={wait_error}, in_vehicle={in_vehicle_error}"
            )
        self.passenger_time_ledger.set_counts(
            self.current_time,
            waiting_passengers=direct[0],
            in_vehicle_passengers=direct[1],
        )
        return {
            "source": (
                "SUMO bus-stop waiting-count and transit-vehicle person-count "
                "subscriptions at every simulation step"
            ),
            "simulation_step_seconds": float(self.time_step),
            "count_synchronizations": self._passenger_time_count_syncs,
            "direct_count_audits": self._passenger_time_direct_audits,
            "subscription_fallback_queries": (
                self._passenger_time_subscription_fallbacks
            ),
            "final_subscription_waiting_passengers": subscribed[0],
            "final_direct_waiting_passengers": direct[0],
            "final_subscription_in_vehicle_passengers": subscribed[1],
            "final_direct_in_vehicle_passengers": direct[1],
            "max_waiting_count_error": (
                self._passenger_time_max_waiting_count_error
            ),
            "max_in_vehicle_count_error": (
                self._passenger_time_max_in_vehicle_count_error
            ),
        }

    def audit_bus_service_semantics(
        self, *, require_all_departed_trips_completed: bool = True
    ) -> dict:
        """Verify complete base-service timing for every completed bus trip."""

        missing_arrivals = []
        missing_service_completions = []
        missing_departures = []
        duplicate_service_completions = []
        unprotected_base_service_visits = []
        completion_before_arrival = []
        departure_before_service_completion = []
        expected_stop_visits = 0
        for bus_id in sorted(self.completed_trip_times):
            bus = self.bus_obj_dic[bus_id]
            line_id = str(bus.belong_line_id_s)
            stops = [str(value) for value in self.line_obj_dic[line_id].stop_id_l]
            expected_stop_visits += len(stops)
            for stop_id in stops:
                label = f"{bus_id}:{stop_id}"
                arrival = bus.arriver_stop_time_d.get(stop_id)
                completions = bus.service_completion_history.get(stop_id, ())
                departure = bus.depart_stop_time_d.get(stop_id)
                if arrival is None:
                    missing_arrivals.append(label)
                if len(completions) != 1:
                    missing_service_completions.append(label)
                    if len(completions) > 1:
                        duplicate_service_completions.append(label)
                if departure is None:
                    missing_departures.append(label)
                if (line_id, bus_id, stop_id) not in self._protected_base_service_visits:
                    unprotected_base_service_visits.append(label)
                if arrival is not None and len(completions) == 1:
                    completion = float(completions[0])
                    if completion + 1e-9 < float(arrival):
                        completion_before_arrival.append(label)
                    if (
                        departure is not None
                        and float(departure) + 1e-9 < completion
                    ):
                        departure_before_service_completion.append(label)

        observed_boarding_durations = {}
        for type_id, configured in sorted(
            self._sumo_bus_type_boarding_durations.items()
        ):
            observed = float(traci.vehicletype.getBoardingDuration(type_id))
            observed_boarding_durations[type_id] = observed
            if not np.isclose(observed, configured, rtol=0.0, atol=1e-9):
                raise RuntimeError(
                    f"SUMO bus type {type_id} boarding duration changed mid-episode"
                )
        boarding_duration_aligned = bool(observed_boarding_durations) and all(
            np.isclose(
                value,
                self.BOARDING_SERVICE_SECONDS_PER_PASSENGER,
                rtol=0.0,
                atol=1e-9,
            )
            for value in observed_boarding_durations.values()
        )
        checks = {
            "all_departed_trips_completed": (
                not require_all_departed_trips_completed
                or set(self.departed_trip_ids) == set(self.completed_trip_times)
            ),
            "all_stop_arrivals_recorded": not missing_arrivals,
            "all_base_service_completions_recorded": not missing_service_completions,
            "one_base_service_completion_per_stop": not duplicate_service_completions,
            "all_stop_departures_recorded": not missing_departures,
            "all_base_service_intervals_protected": not unprotected_base_service_visits,
            "service_completion_not_before_arrival": not completion_before_arrival,
            "departure_not_before_service_completion": not departure_before_service_completion,
            "sumo_and_bridge_boarding_durations_aligned": boarding_duration_aligned,
            "station_capacity_never_forces_departure": self._station_capacity_forced_evictions
            == 0,
            "station_capacity_observation_has_no_errors": self._station_capacity_observation_errors
            == 0,
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "completed_trips": len(self.completed_trip_times),
            "departed_trips": len(self.departed_trip_ids),
            "require_all_departed_trips_completed": require_all_departed_trips_completed,
            "expected_stop_visits": expected_stop_visits,
            "protected_base_service_visits": len(
                self._protected_base_service_visits
            ),
            "service_seconds_per_passenger": {
                "boarding_bridge": self.BOARDING_SERVICE_SECONDS_PER_PASSENGER,
                "alighting_bridge": self.ALIGHTING_SERVICE_SECONDS_PER_PASSENGER,
                "sumo_bus_type_boarding": observed_boarding_durations,
            },
            "station_capacity": {
                "semantics": "observe physical saturation; never truncate base service or policy holding",
                "forced_evictions": self._station_capacity_forced_evictions,
                "saturation_station_steps": self._station_capacity_saturation_station_steps,
                "maximum_occupancy_by_stop": dict(
                    sorted(self._station_capacity_max_occupancy.items())
                ),
                "subscription_fallback_queries": self._station_capacity_subscription_fallbacks,
                "observation_errors": self._station_capacity_observation_errors,
            },
            "violations": {
                "missing_arrivals": missing_arrivals,
                "missing_service_completions": missing_service_completions,
                "duplicate_service_completions": duplicate_service_completions,
                "missing_departures": missing_departures,
                "unprotected_base_service_visits": unprotected_base_service_visits,
                "completion_before_arrival": completion_before_arrival,
                "departure_before_service_completion": departure_before_service_completion,
            },
        }

    def apply_action(self, event: DecisionEvent, action_value: Any) -> None:
        if not self.initialized or self.done:
            return

        # The registered control problem has one physical degree of freedom:
        # additional holding.  Legacy callers may still supply the historical
        # [hold, speed] command shape, but speed is a fixed executor parameter.
        command = np.asarray(action_value, dtype=np.float64)
        if command.ndim == 0:
            hold_value = float(command)
        else:
            command = command.reshape(-1)
            if command.size not in (1, 2):
                raise ValueError(
                    "holding command must be a scalar or [hold_seconds, 1.0]"
                )
            hold_value = float(command[0])
            if command.size == 2 and not np.isclose(
                command[1], 1.0, rtol=0.0, atol=1e-9
            ):
                raise ValueError(
                    "speed ratio is fixed at 1.0 in the holding-only contract"
                )
        if not np.isfinite(hold_value):
            raise ValueError("additional hold must be finite")
        if hold_value < -1e-9 or hold_value > self.max_hold_seconds + 1e-9:
            raise ValueError(
                "additional hold must lie in "
                f"[0, {self.max_hold_seconds}] seconds"
            )
        hold_value = float(np.clip(hold_value, 0.0, self.max_hold_seconds))
        bus_id = event.bus_id
        stop_id = event.stop_id
        service_completion_time = float(
            event.metadata.get(
                'service_completion_time',
                float(event.metadata['arrive_time'])
                + float(event.base_stop_duration),
            )
        )
        
        try:
            # Physics Override Fix: setMaxSpeed only caps the vehicle mechanically,
            # but remains bounded by the lane's physical speed limit.
            # Using setSpeedFactor() instructs the driver to treat the lane speed limit multiplied by the factor.
            traci.vehicle.setSpeedFactor(bus_id, 1.0)

            # The event is emitted after base passenger service. SUMO treats a
            # changed duration as time remaining from this modification instant;
            # using base+hold here would execute the base dwell twice.
            traci.vehicle.setStopParameter(bus_id, 0, "duration", str(hold_value))
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
            bus_obj.just_server_stop_data_d[stop_id][1] = (
                service_completion_time + hold_value
            )
        event.metadata['applied_additional_hold_seconds'] = hold_value
        event.metadata['predicted_departure_time_seconds'] = (
            service_completion_time + hold_value
        )

    # endregion

    # region simulation stepping
    def _advance_until_event(self) -> None:
        while not self.done and not self.decision_queue:
            self._advance_one_step()

    def _advance_one_step(self) -> None:
        if self.done:
            return
        simulation_current_time = traci.simulation.getTime()
        self.passenger_time_ledger.advance_to(simulation_current_time)
        completed_bus_ids = []
        
        # Update active buses
        if self.steps == 0:
            # On the first step, check ALL existing vehicles to catch those that departed at time 0
            initial_ids = traci.vehicle.getIDList()
            for veh_id in initial_ids:
                if veh_id in self.bus_obj_dic:
                    self.active_bus_ids.add(veh_id)
                    self._subscribe_bus_passenger_count(veh_id)
                    self._bus_onboard_passenger_ids[veh_id] = set(
                        traci.vehicle.getPersonIDList(veh_id)
                    )
    
        
        departed_ids = traci.simulation.getDepartedIDList()
        for veh_id in departed_ids:
            if veh_id in self.bus_obj_dic:
                self.active_bus_ids.add(veh_id)
                self.departed_trip_ids.add(veh_id)
                self._subscribe_bus_passenger_count(veh_id)
                self._bus_onboard_passenger_ids[veh_id] = set(
                    traci.vehicle.getPersonIDList(veh_id)
                )
                self._launch_from_terminal(veh_id, simulation_current_time)

        arrived_ids = traci.simulation.getArrivedIDList()
        for veh_id in arrived_ids:
            if veh_id in self.active_bus_ids:
                self.active_bus_ids.discard(veh_id)
                self.just_departed_buses.append(veh_id)
                self.completed_trip_times[veh_id] = simulation_current_time
                self._return_to_terminal(veh_id, simulation_current_time)
                completed_bus_ids.append(veh_id)
                self._bus_onboard_passenger_ids.pop(veh_id, None)
                
        # Update active passengers
        departed_persons = traci.simulation.getDepartedPersonIDList()
        for person_id in departed_persons:
            if person_id in self.passenger_obj_dic:
                self.active_passenger_ids.add(person_id)
                self.departed_passenger_ids.add(person_id)
                self.departed_passenger_times[person_id] = float(
                    simulation_current_time
                )
                # CRITICAL: Activate immediately so they are visible to buses (passable_line_l populated)
                passenger = self.passenger_obj_dic[person_id]
                passenger.passenger_activate(simulation_current_time, self.line_obj_dic)
                
        arrived_persons = traci.simulation.getArrivedPersonIDList()
        for person_id in arrived_persons:
            if person_id in self.active_passenger_ids:
                passenger = self.passenger_obj_dic[person_id]
                if passenger.passenger_state_s == "Bus":
                    alighting_time = float(simulation_current_time)
                    passenger.alighting_time_n = alighting_time
                    passenger.travel_data_l.append(
                        [
                            passenger.arriver_time_n,
                            passenger.boarding_time_n,
                            alighting_time,
                            passenger.boarding_time_n - passenger.arriver_time_n,
                            alighting_time - passenger.arriver_time_n,
                            passenger.take_bus_id_s,
                        ]
                    )
                self.completed_passenger_times[person_id] = float(
                    simulation_current_time
                )
                self.active_passenger_ids.discard(person_id)
                for onboard_ids in self._bus_onboard_passenger_ids.values():
                    onboard_ids.discard(person_id)
                passenger.passenger_state_s = "Arrived"

        # Welfare accounting is sourced directly from SUMO every simulation
        # step. Legacy passenger records are synchronized at bus stop boundaries
        # below and do not determine the reward signal.
        self._synchronize_passenger_time_counts(simulation_current_time)
        completion_snapshot = self.passenger_time_ledger.snapshot()
        for bus_id in completed_bus_ids:
            self._completed_trip_passenger_times[bus_id] = completion_snapshot

        # CRITICAL: Bus logic must run EVERY step to detect arrivals/departures correctly
        # Otherwise, short stops (duration=0) are missed if they fall between update intervals
        for vehicle_id in list(self.active_bus_ids):
            bus_obj = self.bus_obj_dic[vehicle_id]
            line_obj = self.line_obj_dic[bus_obj.belong_line_id_s]
            previous_bus_state = bus_obj.bus_state_s
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
            self._synchronize_bus_passenger_transition(
                vehicle_id,
                previous_bus_state,
                bus_obj.bus_state_s,
                bus_obj,
                simulation_current_time,
            )

        self._collect_new_events(simulation_current_time)

        traci.simulationStep()
        self.current_time = traci.simulation.getTime()
        self.passenger_time_ledger.advance_to(self.current_time)
        self.steps += 1

        self._check_station_capacity()

        self._cleanup_departed_buses()
        self._check_termination()

    def _check_station_capacity(self):
        """Observe physical stop saturation without changing bus service."""
        subscription_results = traci.busstop.getAllSubscriptionResults() or {}
        for stop_id, capacity in self.station_capacities.items():
            try:
                values = subscription_results.get(stop_id) or {}
                parked_buses = values.get(tc.VAR_STOP_STARTING_VEHICLES_IDS)
                if parked_buses is None:
                    parked_buses = traci.busstop.getVehicleIDs(stop_id)
                    self._station_capacity_subscription_fallbacks += 1
                occupancy = len(parked_buses)
                self._station_capacity_max_occupancy[stop_id] = max(
                    occupancy,
                    self._station_capacity_max_occupancy.get(stop_id, 0),
                )
                if occupancy >= capacity:
                    self._station_capacity_saturation_station_steps += 1
            except Exception:
                self._station_capacity_observation_errors += 1

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
            
        # 1. Compare like-for-like base-service completion timestamps. The
        # legacy trajectory_dict stores arrivals and must not mix both meanings.
        completion_history = getattr(front_bus, 'service_completion_history', {})
        if not force_distance and completion_history.get(stop_id):
            front_completion_time = completion_history[stop_id][-1]
            return max(current_time - front_completion_time, 0.0)
            
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

    def _eligible_arrival_rate(
        self, stop_internal_obj, line_id: str, simulation_current_time: float
    ) -> float:
        """Return scheduled passenger arrivals allocated to one eligible line.

        Shared-stop OD demand is divided among acceptable lines in proportion
        to their scheduled frequency.  This is the same allocation principle
        used to build the calibrated low-fidelity simulator demand.
        """

        if stop_internal_obj is None:
            return 0.0
        period = int(max(simulation_current_time, 0.0) // 3600)
        total_rate = 0.0
        for destination, rates in getattr(
            stop_internal_obj, "passenger_arriver_rate_d", {}
        ).items():
            if period < 0 or period >= len(rates):
                continue
            rate = float(rates[period])
            if not np.isfinite(rate) or rate <= 0.0:
                continue
            eligible_lines = tuple(
                getattr(stop_internal_obj, "accessible_stop_d", {}).get(
                    destination, ()
                )
            )
            if line_id not in eligible_lines:
                continue
            frequencies = {
                eligible_line: 1.0
                / max(float(self.line_headways.get(eligible_line, self.default_headway)), 1.0)
                for eligible_line in eligible_lines
            }
            denominator = sum(frequencies.values())
            if denominator > 0.0:
                total_rate += rate * frequencies[line_id] / denominator
        return float(total_rate)

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
            service_completion_time = arrive_time + base_duration

            # Protect the full passenger-exchange interval immediately on
            # arrival. Waiting until the decision time lets a one-slot station
            # or the route's shorter default stop release the bus early.
            if key not in self._protected_base_service_visits:
                try:
                    traci.vehicle.setStopParameter(
                        bus_id,
                        0,
                        "duration",
                        "3600.0",
                    )
                except traci.TraCIException as error:
                    raise RuntimeError(
                        f"failed to protect base service for {bus_id} at {stop_id}"
                    ) from error
                self._protected_base_service_visits.add(key)
            
            # Timing Alignment: Only process if passenger exchange is done
            # This ensures we observe the state AFTER passengers have boarded/alighted, matching original logic
            if simulation_current_time < service_completion_time:
                continue

            # Prevent duplicate events for the same stop visit
            if getattr(bus_obj, 'last_action_time', -1.0) == arrive_time:
                continue
            bus_obj.last_action_time = arrive_time

            line_id = bus_obj.belong_line_id_s

            # Schedule adjacency determines target headways. Presence flags
            # below describe physically observable vehicles, not future trips.
            scheduled_forward_bus, scheduled_backward_bus = self._find_neighbors(
                line_id, bus_id
            )

            # Dynamic Target Headway Calculation
            default_line_headway = self.line_headways.get(line_id, self.default_headway)
            
            target_forward = default_line_headway
            if scheduled_forward_bus:
                target_forward = abs(
                    bus_obj.start_time_n - scheduled_forward_bus.start_time_n
                )
                
            target_backward = default_line_headway
            if scheduled_backward_bus:
                target_backward = abs(
                    scheduled_backward_bus.start_time_n - bus_obj.start_time_n
                )
                
            # Service Completion Time: When passenger exchange finishes (Arrive + Duration)
            # This is the exact time point used for Headway calculation in LSTM-RL
            bus_obj.service_completion_history[stop_id].append(
                service_completion_time
            )
            forward_history = (
                getattr(scheduled_forward_bus, 'service_completion_history', {})
                if scheduled_forward_bus is not None
                else {}
            )
            forward_bus_present = bool(forward_history.get(stop_id))
            backward_bus_present = bool(
                scheduled_backward_bus is not None
                and scheduled_backward_bus.bus_id_s in self.active_bus_ids
            )

            # Retain the service-completion record for later buses even when
            # this first isolated event has no meaningful control decision.
            if not forward_bus_present and not backward_bus_present:
                try:
                    traci.vehicle.setStopParameter(bus_id, 0, "duration", "0")
                except traci.TraCIException as error:
                    raise RuntimeError(
                        f"failed to release isolated bus {bus_id} at {stop_id}"
                    ) from error
                continue

            stop_internal_obj = self.stop_obj_dic.get(stop_id)
            segment_mean_speed = 10.0
            if stop_internal_obj and stop_internal_obj.at_lane_s:
                try:
                    measured_speed = float(
                        traci.lane.getLastStepMeanSpeed(stop_internal_obj.at_lane_s)
                    )
                    if np.isfinite(measured_speed) and measured_speed >= 1.0:
                        segment_mean_speed = measured_speed
                except Exception:
                    pass

            # Robust Headway Calculation
            # Forward Headway: Me - Front
            # Use service_completion_time instead of simulation_current_time for precision
            forward_headway = self._compute_robust_headway(
                bus_obj,
                scheduled_forward_bus if forward_bus_present else None,
                stop_id,
                service_completion_time,
                target_headway=target_forward,
            )
            
            # Store this forward headway for the backward bus to use later
            bus_obj.last_forward_headway = forward_headway

            # Estimate the following bus's current gap from its observable
            # position. A previous-stop cached headway is not a backward
            # headway at this decision location.
            if backward_bus_present:
                distance_gap = max(
                    float(bus_obj.distance_n)
                    - float(scheduled_backward_bus.distance_n),
                    0.0,
                )
                backward_cap = max(2400.0, 3.0 * target_backward)
                backward_headway = min(
                    distance_gap / max(segment_mean_speed, 1.0),
                    backward_cap,
                )
            else:
                backward_headway = target_backward

            stop_idx = self.stop_indices[line_id].get(stop_id, -1)
            direction = 1 if line_id.endswith('S') else 0
            waiting_person_ids = tuple(traci.busstop.getPersonIDs(stop_id))
            waiting_passengers = len(waiting_person_ids)
            eligible_stranded_passengers = sum(
                1
                for passenger_id in waiting_person_ids
                if passenger_id in self.passenger_obj_dic
                and line_id
                in getattr(
                    self.passenger_obj_dic[passenger_id], "passable_line_l", ()
                )
            )
            try:
                onboard_passengers = int(traci.vehicle.getPersonNumber(bus_id))
            except Exception:
                onboard_passengers = 0
            try:
                bus_capacity = float(traci.vehicle.getPersonCapacity(bus_id))
            except Exception:
                bus_capacity = 50.0
            if not np.isfinite(bus_capacity) or bus_capacity <= 0.0:
                bus_capacity = 50.0

            eligible_arrival_rate = self._eligible_arrival_rate(
                stop_internal_obj=self.stop_obj_dic.get(stop_id),
                line_id=line_id,
                simulation_current_time=simulation_current_time,
            )
            
            # Co-line headways
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
                forward_bus_present=forward_bus_present,
                backward_bus_present=backward_bus_present,
                target_forward_headway=target_forward,
                target_backward_headway=target_backward,
                co_line_forward_headway=co_line_fwd,
                co_line_backward_headway=co_line_bwd,
                segment_mean_speed=segment_mean_speed,
                eligible_stranded_passengers=eligible_stranded_passengers,
                onboard_passengers=onboard_passengers,
                eligible_arrival_rate_per_second=eligible_arrival_rate,
                bus_capacity=bus_capacity,
                metadata={
                    'arrive_time': arrive_time,
                    'service_completion_time': service_completion_time,
                    'stopping_place': stopping_place,
                    'passenger_time_at_decision': (
                        self.passenger_time_ledger.snapshot().to_dict()
                    ),
                },
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
