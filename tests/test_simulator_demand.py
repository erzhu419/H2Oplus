import json
from pathlib import Path

from reproducibility.build_simulator_demand import (
    allocate_route_legs,
    extract_journey_templates,
    line_stop_component,
    migrate_report_paths,
    sha256,
)


def test_line_stop_component_and_frequency_weighted_allocation(tmp_path: Path):
    route = tmp_path / "routes.xml"
    route.write_text(
        """<routes><person id="p" depart="3600">
        <walk busStop="7X05_102X18"/>
        <ride busStop="7X08_102X21" lines="7X 102X"/>
        </person></routes>"""
    )
    demand, diagnostics = allocate_route_legs(
        [route], {"7X": 50, "102X": 25}
    )
    assert line_stop_component("7X05_102X18", "102X") == "102X18"
    assert demand[("7X", 7, "7X05", "7X08")] == 2 / 3
    assert demand[("102X", 7, "102X18", "102X21")] == 1 / 3
    assert diagnostics["persons"] == 1
    assert diagnostics["ride_legs"] == 1
    assert diagnostics["allocated_line_legs"] == 1.0


def test_transfer_legs_remain_in_one_ordered_journey(tmp_path: Path):
    route = tmp_path / "transfer.xml"
    route.write_text(
        """<routes><person id="p" depart="100">
        <walk busStop="A01"/>
        <ride busStop="A02_B01" lines="A"/>
        <ride busStop="B02" lines="B"/>
        </person></routes>"""
    )

    templates, diagnostics = extract_journey_templates(
        [route], {"A": 10, "B": 10}
    )

    assert diagnostics["persons"] == 1
    assert diagnostics["ride_legs"] == 2
    assert diagnostics["transfer_journeys"] == 1
    assert len(templates) == 1
    assert templates[0]["departure_seconds"] == 100.0
    assert [
        leg["candidates"][0]["line_id"] for leg in templates[0]["legs"]
    ] == ["A", "B"]


def test_path_migration_does_not_claim_causal_v3_semantics(tmp_path: Path):
    artifact = tmp_path / "data" / "calibrated.xlsx"
    artifact.parent.mkdir()
    artifact.write_bytes(b"calibrated")
    report_path = tmp_path / "simulator_demand_calibration.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "transit-simulator-demand-calibration-v1",
                "output_sha256": {str(artifact.resolve()): sha256(artifact)},
            }
        )
    )

    migrated = migrate_report_paths(report_path)

    assert migrated["schema_version"] == "transit-simulator-demand-calibration-v2"
    assert migrated["output_path_base"] == "report_directory"
    assert migrated["output_sha256"] == {
        "data/calibrated.xlsx": sha256(artifact)
    }
