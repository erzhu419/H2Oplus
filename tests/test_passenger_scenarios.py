import xml.etree.ElementTree as ET
import json

from reproducibility.build_passenger_scenarios import (
    build_scenario,
    scaled_person_elements,
    sha256,
)


def test_passenger_scaling_is_deterministic_and_passenger_only(tmp_path):
    root = ET.Element("routes")
    ET.SubElement(root, "vType", {"id": "Passenger"})
    for index in range(20):
        person = ET.SubElement(
            root,
            "person",
            {"id": f"p{index}", "depart": str(index)},
        )
        ET.SubElement(person, "ride", {"lines": "7X"})
    source = tmp_path / "people.xml"
    ET.ElementTree(root).write(source)

    low_a, low_records_a = scaled_person_elements(source, 0.75, 11)
    low_b, low_records_b = scaled_person_elements(source, 0.75, 11)
    assert [element.get("id") for element in low_a] == [
        element.get("id") for element in low_b
    ]
    assert low_records_a == low_records_b
    assert 0 < len(low_a) < 20

    high, high_records = scaled_person_elements(source, 1.25, 11)
    assert len(high) > 20
    assert len({record[0] for record in high_records}) == len(high_records)
    assert all(element.tag == "person" for element in high)


def test_departure_jitter_is_bounded_seeded_and_changes_unit_scale(tmp_path):
    root = ET.Element("routes")
    for index in range(20):
        person = ET.SubElement(
            root,
            "person",
            {"id": f"p{index}", "depart": str(300 + 10 * index)},
        )
        ET.SubElement(person, "ride", {"lines": "7X"})
    source = tmp_path / "people.xml"
    ET.ElementTree(root).write(source)

    jitter_a, records_a = scaled_person_elements(
        source, 1.0, 101, depart_jitter_seconds=60.0
    )
    jitter_a_repeat, records_a_repeat = scaled_person_elements(
        source, 1.0, 101, depart_jitter_seconds=60.0
    )
    jitter_b, records_b = scaled_person_elements(
        source, 1.0, 102, depart_jitter_seconds=60.0
    )

    assert records_a == records_a_repeat
    assert [element.get("depart") for element in jitter_a] == [
        element.get("depart") for element in jitter_a_repeat
    ]
    assert len(records_a) == len(records_b) == 20
    assert {row[0] for row in records_a} == {row[0] for row in records_b}
    assert {row[2] for row in records_a} != {row[2] for row in records_b}
    original_depart = {f"p{index}": 300 + 10 * index for index in range(20)}
    assert all(
        abs(depart - original_depart[source_id]) <= 60.0
        for _, source_id, depart in records_a
    )
    assert max(row[2] for row in records_a) <= max(original_depart.values())


def test_identical_scenario_rebuild_preserves_manifest_identity(tmp_path):
    routes = []
    for index, name in enumerate(("direct.xml", "transfer.xml")):
        root = ET.Element("routes")
        person = ET.SubElement(
            root,
            "person",
            {"id": f"p{index}", "depart": str(300 + index)},
        )
        ET.SubElement(person, "ride", {"lines": "7X"})
        path = tmp_path / name
        ET.ElementTree(root).write(path)
        routes.append(path)

    objects_root = ET.Element("additional")
    for index in range(2):
        ET.SubElement(
            objects_root,
            "passenger_obj",
            {"passenger_id_s": f"p{index}", "start_time_n": str(300 + index)},
        )
    objects = tmp_path / "passengers.add.xml"
    ET.ElementTree(objects_root).write(objects)
    (tmp_path / "net.xml").write_text("network")
    (tmp_path / "additional.xml").write_text("additional")
    config_root = ET.Element("configuration")
    inputs = ET.SubElement(config_root, "input")
    ET.SubElement(inputs, "net-file", {"value": "net.xml"})
    ET.SubElement(inputs, "additional-files", {"value": "additional.xml"})
    ET.SubElement(
        inputs,
        "route-files",
        {"value": "direct.xml,transfer.xml"},
    )
    config = tmp_path / "base.sumocfg"
    ET.ElementTree(config_root).write(config)

    arguments = {
        "base_config": config,
        "passenger_routes": tuple(routes),
        "passenger_objects": objects,
        "output_root": tmp_path / "scenarios",
        "scale": 1.0,
        "seed": 17,
        "depart_jitter_seconds": 10.0,
    }
    first = build_scenario(**arguments)
    first_hash = sha256(first)
    first_created = json.loads(first.read_text())["created_utc"]
    second = build_scenario(**arguments)

    assert second == first
    assert sha256(second) == first_hash
    assert json.loads(second.read_text())["created_utc"] == first_created
