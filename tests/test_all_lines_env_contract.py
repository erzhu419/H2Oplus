from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from bus_h2o.envs.all_lines_sim_env import AllLinesSimEnv
from bus_h2o.envs.all_lines_sumo_env import AllLinesSumoEnv
from bus_h2o.mdp_contract_v4 import RAW_EVENT_STATE_DIM, RAW_EVENT_STATE_INDEX
from bus_h2o.sim_core.passenger import Passenger
from bus_h2o.sim_core.journey import JourneyPassenger
from bus_h2o.sim_core.sim import MultiLineEnv, PhysicalCoLineContext


ROOT = Path(__file__).resolve().parents[1]


def _create_runtime_view_in_worker(arguments):
    base, line_id = arguments
    return MultiLineEnv._make_line_path(object(), base, line_id)


def test_sim_adapter_enforces_canonical_episode_horizon():
    env = AllLinesSimEnv(
        str(ROOT / "bus_h2o" / "calibrated_env"), episode_seconds=321.0
    )
    assert env.episode_seconds == 321.0
    assert len(env.line_map) == 12
    assert all(line.max_time == 321.0 for line in env.line_map.values())
    assert all(
        line.passenger_arrival_end_time == 18_000.0
        for line in env.line_map.values()
    )
    assert all(
        line.max_agent_num == line.effective_trip_num
        for line in env.line_map.values()
    )
    assert all(
        route.runtime_calibration is not None
        for line in env.line_map.values()
        for route in line.routes
    )
    env.reset()
    assert all(
        route.runtime_calibration is not None
        for line in env.line_map.values()
        for route in line.routes
    )
    largest_line = max(env.line_map, key=lambda line_id: env.line_map[line_id].max_agent_num)
    largest_bus_id = env.line_map[largest_line].max_agent_num - 1
    nested = env._nested_actions({(largest_line, largest_bus_id): [0.0, 1.0]})
    assert nested[largest_line][largest_bus_id] == [0.0, 1.0]
    with pytest.raises(ValueError, match="speed ratio is fixed"):
        env._nested_actions({(largest_line, largest_bus_id): [0.0, 1.01]})
    shared_7x = env.line_map["7X"].stations[4].physical_stop_id
    shared_102x = env.line_map["102X"].stations[17].physical_stop_id
    assert shared_7x == shared_102x == "7X05_102X18"
    assert set(env._physical_stop_positions[shared_7x]) == {"7X", "102X"}


def test_sim_adapter_does_not_end_when_all_buses_retire_before_horizon():
    env = AllLinesSimEnv(
        str(ROOT / "bus_h2o" / "calibrated_env"), episode_seconds=3.0
    )
    env.reset()
    for line in env.line_map.values():
        line.timetables = []
        line.bus_all = []
        line.passenger_arrival_end_time = 0.0

    for expected_time in (1, 2):
        _, _, done = env.step({})
        assert not done
        assert env.current_time == expected_time
    _, _, done = env.step({})
    assert done
    assert env.current_time == 3
    assert all(line.current_time == 3 for line in env.line_map.values())


def test_physical_coline_context_uses_each_lines_route_coordinate():
    context = PhysicalCoLineContext(
        {"shared": {"A": 1_000.0, "B": 800.0}},
        {"A": (900.0,), "B": (700.0, 900.0)},
    )
    assert context.signed_distances("shared", "A") == [
        (100.0, "B"),
        (-100.0, "B"),
    ]
    assert context.signed_distances("missing", "A") == []


def test_line_runtime_view_replaces_stale_copied_data(tmp_path):
    base = tmp_path / "sim"
    source_data = base / "data" / "line"
    source_data.mkdir(parents=True)
    (source_data / "canonical.txt").write_text("canonical")
    (base / "config.json").write_text("{}")
    stale_data = base / "_line_envs" / "line" / "data"
    stale_data.mkdir(parents=True)
    (stale_data / "obsolete.txt").write_text("obsolete")

    runtime = MultiLineEnv._make_line_path(object(), str(base), "line")

    assert base / "_line_envs" not in Path(runtime).parents
    linked_config = Path(runtime) / "config.json"
    assert linked_config.is_symlink()
    assert linked_config.resolve() == (base / "config.json").resolve()
    linked_data = Path(runtime) / "data"
    assert linked_data.is_symlink()
    assert linked_data.resolve() == source_data.resolve()
    assert (linked_data / "canonical.txt").read_text() == "canonical"
    assert not (linked_data / "obsolete.txt").exists()


def test_line_runtime_views_are_removed_when_process_pool_exits(tmp_path):
    base = tmp_path / "sim"
    source_data = base / "data" / "line"
    source_data.mkdir(parents=True)
    (base / "config.json").write_text("{}")

    arguments = [(str(base), "line")] * 4
    with ProcessPoolExecutor(max_workers=2) as executor:
        runtime_paths = list(executor.map(_create_runtime_view_in_worker, arguments))
        assert all(Path(path).is_dir() for path in runtime_paths)

    assert all(not Path(path).exists() for path in runtime_paths)


def test_simulator_bus_serves_passengers_at_origin():
    env = AllLinesSimEnv(str(ROOT / "bus_h2o" / "calibrated_env"))
    line = env.line_map["7X"]
    line.reset()
    origin = line.stations[0]
    passenger = Passenger(0.0, origin, line.stations[2])
    origin.waiting_passengers.append(passenger)
    origin.total_passenger.append(passenger)

    line.launch_bus(line.timetables[0])
    bus = line.bus_all[-1]

    assert passenger.boarded
    assert passenger.boarding_time == 0.0
    assert passenger in bus.passengers
    assert origin.waiting_passengers == []
    assert bus.holding_time == 6.5


def test_simulator_bus_boards_arrivals_while_stop_remains_open():
    env = AllLinesSimEnv(str(ROOT / "bus_h2o" / "calibrated_env"))
    line = env.line_map["7X"]
    line.reset()
    origin = line.stations[0]
    destination = line.stations[2]
    line.launch_bus(line.timetables[0])
    bus = line.bus_all[-1]

    assert bus.holding_time == 4.0
    during_base = Passenger(1.0, origin, destination)
    origin.waiting_passengers.append(during_base)
    origin.total_passenger.append(during_base)
    bus.drive(1.0, 0.0, line.bus_all, debug=False)

    assert during_base.boarding_time == 1.0
    assert during_base in bus.passengers
    assert bus.stop_board_l[-1] == 1
    assert bus._base_stop_duration == 4.0

    bus.state = type(bus.state).WAITING_ACTION
    at_action = Passenger(2.0, origin, destination)
    origin.waiting_passengers.append(at_action)
    origin.total_passenger.append(at_action)
    bus.drive(2.0, [2.0, 1.0], line.bus_all, debug=False)
    assert at_action.boarding_time == 2.0

    during_hold = Passenger(3.0, origin, destination)
    origin.waiting_passengers.append(during_hold)
    origin.total_passenger.append(during_hold)
    bus.drive(3.0, 0.0, line.bus_all, debug=False)

    assert during_hold.boarding_time == 3.0
    assert bus.stop_board_l[-1] == 3
    assert bus.stop_load_l[-1] == 3
    assert origin.waiting_passengers == []


def test_sumo_action_duration_is_only_the_additional_hold(monkeypatch):
    sumo_root = ROOT.parent / "SUMO_ruiguang" / "online_control"
    env = AllLinesSumoEnv(
        str(sumo_root),
        str(ROOT / "bus_h2o" / "network_data" / "a_sorted_busline_edge.xml"),
    )
    env._ensure_bridge()
    bridge = env._bridge
    calls = []

    class FakeVehicle:
        @staticmethod
        def setSpeedFactor(bus_id, value):
            calls.append(("speed", bus_id, value))

        @staticmethod
        def setStopParameter(bus_id, index, name, value):
            calls.append(("stop", bus_id, index, name, value))

    bridge_module = sys.modules[bridge.__class__.__module__]
    monkeypatch.setattr(
        bridge_module,
        "traci",
        SimpleNamespace(vehicle=FakeVehicle, TraCIException=RuntimeError),
    )
    bridge.initialized = True
    bridge.done = False
    bridge.pending_events = {}
    bridge.active_events = {}
    bridge.bus_obj_dic = {
        "bus": SimpleNamespace(just_server_stop_data_d={"stop": [100.0, 112.0]})
    }
    event = SimpleNamespace(
        line_id="line",
        bus_id="bus",
        stop_id="stop",
        base_stop_duration=12.0,
        metadata={"arrive_time": 100.0, "service_completion_time": 112.0},
        assigned_action=None,
    )

    bridge.apply_action(event, [20.0, 1.0])

    assert ("stop", "bus", 0, "duration", "20.0") in calls
    assert ("speed", "bus", 1.0) in calls
    assert bridge.bus_obj_dic["bus"].just_server_stop_data_d["stop"] == [
        100.0,
        132.0,
    ]
    assert event.metadata["applied_additional_hold_seconds"] == 20.0
    assert event.metadata["predicted_departure_time_seconds"] == 132.0

    with pytest.raises(ValueError, match="speed ratio is fixed"):
        bridge.apply_action(event, [20.0, 1.25])


def test_sumo_headway_uses_explicit_service_completion_history():
    sumo_root = ROOT.parent / "SUMO_ruiguang" / "online_control"
    env = AllLinesSumoEnv(
        str(sumo_root),
        str(ROOT / "bus_h2o" / "network_data" / "a_sorted_busline_edge.xml"),
    )
    env._ensure_bridge()
    bridge = env._bridge
    subject = SimpleNamespace(start_time_n=0.0, distance_n=1000.0)
    front = SimpleNamespace(
        distance_n=1200.0,
        trajectory_dict={"stop": [90.0, 95.0]},
        service_completion_history={"stop": [100.0]},
    )

    assert bridge._compute_robust_headway(
        subject, front, "stop", 150.0, target_headway=360.0
    ) == 50.0


def test_sumo_base_service_is_protected_immediately_on_arrival(monkeypatch):
    sumo_root = ROOT.parent / "SUMO_ruiguang" / "online_control"
    env = AllLinesSumoEnv(
        str(sumo_root),
        str(ROOT / "bus_h2o" / "network_data" / "a_sorted_busline_edge.xml"),
    )
    env._ensure_bridge()
    bridge = env._bridge
    calls = []

    class FakeVehicle:
        @staticmethod
        def setStopParameter(bus_id, index, name, value):
            calls.append((bus_id, index, name, value))

    bridge_module = sys.modules[bridge.__class__.__module__]
    monkeypatch.setattr(
        bridge_module,
        "traci",
        SimpleNamespace(vehicle=FakeVehicle, TraCIException=RuntimeError),
    )
    bus = SimpleNamespace(
        bus_state_s="Stop",
        bus_id_s="bus",
        start_time_n=0.0,
        just_server_stop_data_d={"stop": [100.0, 106.5]},
        belong_line_id_s="line",
        service_completion_history=defaultdict(list),
    )
    bridge.bus_obj_dic = {"bus": bus}
    bridge.line_obj_dic = {"line": SimpleNamespace(stop_id_l=["stop"])}
    bridge.pending_events = {}
    bridge._protected_base_service_visits.clear()

    bridge._collect_new_events(100.0)
    bridge._collect_new_events(101.0)
    bridge._collect_new_events(107.0)

    assert calls == [
        ("bus", 0, "duration", "3600.0"),
        ("bus", 0, "duration", "0"),
    ]
    assert ("line", "bus", "stop") in bridge._protected_base_service_visits
    assert bus.service_completion_history["stop"] == [106.5]


def test_sumo_station_capacity_observation_never_truncates_service(monkeypatch):
    sumo_root = ROOT.parent / "SUMO_ruiguang" / "online_control"
    env = AllLinesSumoEnv(
        str(sumo_root),
        str(ROOT / "bus_h2o" / "network_data" / "a_sorted_busline_edge.xml"),
    )
    env._ensure_bridge()
    bridge = env._bridge
    bridge.station_capacities = {"stop": 1}
    bridge._station_capacity_max_occupancy = {}
    bridge._station_capacity_saturation_station_steps = 0
    bridge._station_capacity_subscription_fallbacks = 0
    bridge._station_capacity_observation_errors = 0
    bridge._station_capacity_forced_evictions = 0
    bridge_module = sys.modules[bridge.__class__.__module__]
    variable = bridge_module.tc.VAR_STOP_STARTING_VEHICLES_IDS
    monkeypatch.setattr(
        bridge_module,
        "traci",
        SimpleNamespace(
            busstop=SimpleNamespace(
                getAllSubscriptionResults=lambda: {"stop": {variable: ["bus"]}},
                getVehicleIDs=lambda stop: ["bus"],
            )
        ),
    )

    bridge._check_station_capacity()

    assert bridge._station_capacity_saturation_station_steps == 1
    assert bridge._station_capacity_max_occupancy == {"stop": 1}
    assert bridge._station_capacity_forced_evictions == 0


def test_sumo_service_semantics_audit_rejects_early_departure(monkeypatch):
    sumo_root = ROOT.parent / "SUMO_ruiguang" / "online_control"
    env = AllLinesSumoEnv(
        str(sumo_root),
        str(ROOT / "bus_h2o" / "network_data" / "a_sorted_busline_edge.xml"),
    )
    env._ensure_bridge()
    bridge = env._bridge
    bus = SimpleNamespace(
        belong_line_id_s="line",
        arriver_stop_time_d={"stop": 10.0},
        service_completion_history={"stop": [15.0]},
        depart_stop_time_d={"stop": 16.0},
    )
    bridge.bus_obj_dic = {"bus": bus}
    bridge.line_obj_dic = {"line": SimpleNamespace(stop_id_l=["stop"])}
    bridge.departed_trip_ids = {"bus"}
    bridge.completed_trip_times = {"bus": 20.0}
    bridge._protected_base_service_visits = {("line", "bus", "stop")}
    bridge._sumo_bus_type_boarding_durations = {"Bus": 2.5}
    bridge._station_capacity_forced_evictions = 0
    bridge._station_capacity_observation_errors = 0
    bridge._station_capacity_saturation_station_steps = 0
    bridge._station_capacity_max_occupancy = {"stop": 1}
    bridge._station_capacity_subscription_fallbacks = 0
    bridge_module = sys.modules[bridge.__class__.__module__]
    monkeypatch.setattr(
        bridge_module,
        "traci",
        SimpleNamespace(
            vehicletype=SimpleNamespace(getBoardingDuration=lambda type_id: 2.5)
        ),
    )

    assert bridge.audit_bus_service_semantics()["passed"] is True
    bus.depart_stop_time_d["stop"] = 14.0
    audit = bridge.audit_bus_service_semantics()
    assert audit["passed"] is False
    assert audit["violations"]["departure_before_service_completion"] == [
        "bus:stop"
    ]


def test_transfer_leg_appears_only_after_previous_leg_alights():
    station_a0 = SimpleNamespace(
        station_name="A01", waiting_passengers=[], total_passenger=[]
    )
    station_a1 = SimpleNamespace(
        station_name="A02", waiting_passengers=[], total_passenger=[]
    )
    station_b0 = SimpleNamespace(
        station_name="B01", waiting_passengers=[], total_passenger=[]
    )
    station_b1 = SimpleNamespace(
        station_name="B02", waiting_passengers=[], total_passenger=[]
    )
    line_a = SimpleNamespace(
        stations=[station_a0, station_a1],
        completed_passengers=[],
        time_step=1.0,
        passenger_arrival_end_time=18_000.0,
    )
    line_b = SimpleNamespace(
        stations=[station_b0, station_b1],
        completed_passengers=[],
        time_step=1.0,
        passenger_arrival_end_time=18_000.0,
    )
    env = MultiLineEnv.__new__(MultiLineEnv)
    env.line_map = {"A": line_a, "B": line_b}
    env._stations_by_line = {
        "A": {station.station_name: station for station in line_a.stations},
        "B": {station.station_name: station for station in line_b.stations},
    }
    env.journey_demand = {"enabled": True}
    env._pending_initial_journeys = []
    env._pending_initial_index = 0
    env._pending_transfers = []
    env._generated_journey_ids = {"j"}
    env._completed_journey_ids = set()
    env._created_ride_legs = 0
    env._completed_ride_legs = 0
    env._transfer_ride_legs = 0
    first_leg = {
        "candidates": [
            {
                "line_id": "A",
                "origin_stop": "A01",
                "destination_stop": "A02",
                "selection_weight": 1.0,
            }
        ]
    }
    second_leg = {
        "candidates": [
            {
                "line_id": "B",
                "origin_stop": "B01",
                "destination_stop": "B02",
                "selection_weight": 1.0,
            }
        ]
    }
    env._inject_leg(
        appearance_time=0.0,
        journey_id="j",
        leg_index=0,
        legs=(first_leg, second_leg),
        source_departure_time=0.0,
    )
    passenger = station_a0.waiting_passengers.pop()
    assert isinstance(passenger, JourneyPassenger)
    assert station_b0.waiting_passengers == []

    passenger.boarding_time = 1.0
    passenger.arrive_time = 10.0
    line_a.completed_passengers.append(passenger)
    env._consume_completed_passengers()
    env._inject_due_journeys(10.0)
    assert station_b0.waiting_passengers == []

    env._inject_due_journeys(11.0)
    assert len(station_b0.waiting_passengers) == 1
    transfer = station_b0.waiting_passengers[0]
    assert transfer.journey_id == "j"
    assert transfer.leg_index == 1
    assert transfer.appear_time == 11.0


def test_sumo_adapter_disables_snapshot_pool_without_starting_sumo():
    sumo_root = ROOT.parent / "SUMO_ruiguang" / "online_control"
    env = AllLinesSumoEnv(
        str(sumo_root),
        str(ROOT / "bus_h2o" / "network_data" / "a_sorted_busline_edge.xml"),
        sumo_cfg="scenario.sumocfg",
        passenger_obj_xml="scenario_passengers.xml",
    )
    assert env._bridge is None
    assert env._snapshot_pool_max == 0
    assert env.sumo_seed == 42
    assert env._sumo_cfg == "scenario.sumocfg"
    assert env._passenger_obj_xml == "scenario_passengers.xml"
    with pytest.raises(NotImplementedError):
        env.reset(snapshot="not-allowed.xml")


def test_sumo_adapter_does_not_prepend_sim_obj_to_import_path(monkeypatch):
    sumo_root = ROOT.parent / "SUMO_ruiguang" / "online_control"
    monkeypatch.setattr(sys, "path", list(sys.path))
    env = AllLinesSumoEnv(
        str(sumo_root),
        str(ROOT / "bus_h2o" / "network_data" / "a_sorted_busline_edge.xml"),
    )

    env._ensure_bridge()

    assert str(sumo_root) in sys.path
    assert str(sumo_root / "sim_obj") not in sys.path
    assert env._bridge is not None


def test_sumo_adapter_emits_physical_v4_event_fields_without_starting_sumo():
    sumo_root = ROOT.parent / "SUMO_ruiguang" / "online_control"
    env = AllLinesSumoEnv(
        str(sumo_root),
        str(ROOT / "bus_h2o" / "network_data" / "a_sorted_busline_edge.xml"),
        observation_schema="v4",
    )
    event = SimpleNamespace(
        line_id="7X",
        bus_id="7X_3",
        stop_id="7X05_102X18",
        stop_idx=4,
        direction=0,
        sim_time=1234.0,
        forward_headway=410.0,
        backward_headway=290.0,
        waiting_passengers=99,
        base_stop_duration=14.0,
        target_forward_headway=360.0,
        target_backward_headway=300.0,
        eligible_stranded_passengers=7,
        onboard_passengers=31,
        eligible_arrival_rate_per_second=0.025,
        bus_capacity=50.0,
    )
    observation = env._event_to_obs(event)
    assert observation.shape == (RAW_EVENT_STATE_DIM,)
    assert observation[RAW_EVENT_STATE_INDEX["eligible_stranded_passengers"]] == 7
    assert observation[RAW_EVENT_STATE_INDEX["onboard_passengers"]] == 31
    assert observation[
        RAW_EVENT_STATE_INDEX["eligible_arrival_rate_per_second"]
    ] == pytest.approx(0.025)
    assert observation[RAW_EVENT_STATE_INDEX["target_backward_headway_seconds"]] == 300
    assert 99 not in observation


def test_sim_adapter_v4_fields_and_demand_cutoff_match_sumo_contract():
    env = AllLinesSimEnv(
        str(ROOT / "bus_h2o" / "calibrated_env"),
        demand_scale=1.5,
        observation_schema="v4",
    )
    line_id = "7X"
    line_env = env.line_map[line_id]
    station = SimpleNamespace(
        waiting_passengers=[object(), object(), object()],
        od={"07:00:00": {"downstream": 72.0}},
    )
    bus = SimpleNamespace(
        bus_id=9,
        last_station=station,
        passengers=[object()] * 17,
        target_forward_headway=360.0,
        target_backward_headway=420.0,
        forward_bus_present=True,
        backward_bus_present=False,
        capacity=50,
    )
    line_env.bus_all = [bus]
    legacy = [
        line_env.line_idx,
        9,
        4,
        1,
        0,
        400,
        420,
        999,
        360,
        11,
        3600,
        -40,
        360,
        360,
        10,
    ]
    observation = env._v4_observation(line_id, 9, legacy)
    assert len(observation) == RAW_EVENT_STATE_DIM
    assert observation[RAW_EVENT_STATE_INDEX["eligible_stranded_passengers"]] == 3
    assert observation[RAW_EVENT_STATE_INDEX["onboard_passengers"]] == 17
    assert observation[
        RAW_EVENT_STATE_INDEX["eligible_arrival_rate_per_second"]
    ] == pytest.approx(0.03)
    assert all(line.demand_scale == 1.5 for line in env.line_map.values())
    assert env._eligible_arrival_rate(line_env, station, 18_000.0) == 0.0


def test_sim_adapter_rejects_nonpositive_demand_scale():
    with pytest.raises(ValueError, match="demand_scale"):
        AllLinesSimEnv(
            str(ROOT / "bus_h2o" / "calibrated_env"), demand_scale=0.0
        )
