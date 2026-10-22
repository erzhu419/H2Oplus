import importlib.util
from pathlib import Path
import sys
from types import ModuleType

from bus_h2o.envs.sumo_gym_env import SumoGymEnv

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "bus_h2o" / "sumo_env" / "case" / "f_8_create_obj.py"


class _EmptyDocument:
    def getroot(self):
        return []


def test_create_objects_uses_explicit_static_object_directory(tmp_path, monkeypatch):
    sim_obj = ModuleType("sim_obj")
    monkeypatch.setitem(sys.modules, "sim_obj", sim_obj)
    for name in ("passenger", "signal", "lane", "stop", "line", "bus"):
        module = ModuleType(f"sim_obj.{name}")
        setattr(sim_obj, name, module)
        monkeypatch.setitem(sys.modules, f"sim_obj.{name}", module)
    spec = importlib.util.spec_from_file_location("f_8_create_obj_path_test", SOURCE)
    assert spec is not None and spec.loader is not None
    f_8_create_obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f_8_create_obj)

    initialize_obj_dir = tmp_path / "canonical-online-control" / "initialize_obj"
    passenger_xml = tmp_path / "scenario" / "passengers.xml"
    parsed = []

    def parse(path):
        parsed.append(Path(path))
        return _EmptyDocument()

    monkeypatch.setattr(f_8_create_obj.ET, "parse", parse)
    result = f_8_create_obj.create_obj_fun(
        passenger_xml,
        initialize_obj_dir=initialize_obj_dir,
    )

    assert result == ({}, {}, {}, {}, {}, {})
    assert parsed == [
        initialize_obj_dir / "save_obj_lane.add.xml",
        initialize_obj_dir / "save_obj_stop.add.xml",
        initialize_obj_dir / "save_obj_signal.add.xml",
        initialize_obj_dir / "save_obj_line.add.xml",
        initialize_obj_dir / "save_obj_bus.add.xml",
        passenger_xml.resolve(),
    ]


def test_sumo_gym_forwards_explicit_edge_xml_to_bridge(tmp_path, monkeypatch):
    captured = {}

    class Bridge:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    bridge_module = ModuleType("sumo_env.rl_bridge")
    bridge_module.SumoRLBridge = Bridge
    monkeypatch.setitem(sys.modules, "sumo_env.rl_bridge", bridge_module)

    edge_xml = tmp_path / "network" / "canonical-edges.xml"
    env = SumoGymEnv(
        sumo_dir=str(tmp_path / "online_control"),
        edge_xml=str(edge_xml),
        sumo_backend="libsumo",
    )
    env._ensure_bridge()

    assert captured["edge_xml"] == str(edge_xml)
