import xml.etree.ElementTree as ET

import h5py
import numpy as np

from bus_h2o.mdp_contract_v6 import (
    CANONICAL_PREFIX_NAMES,
    CANONICAL_SLOT_NAMES,
    CANONICAL_SLOT_WIDTH,
    CANONICAL_STATE_DIM,
    RAW_PREFIX_NAMES,
    RAW_SLOT_NAMES,
    RAW_SLOT_WIDTH,
    RAW_STATE_DIM,
    SCHEMA_VERSION,
)
from reproducibility.audit_control_scope_v6 import audit_control_scope


def test_control_scope_audit_accepts_all_scheduled_lines(tmp_path):
    line_ids = [f"line-{index:02d}" for index in range(12)]
    schedule = tmp_path / "schedule.xml"
    root = ET.Element("additional")
    for index, line_id in enumerate(line_ids):
        ET.SubElement(
            root,
            "bus_obj",
            belong_line_id_s=line_id,
            bus_id_s=f"bus-{index:02d}",
        )
    ET.ElementTree(root).write(schedule, encoding="utf-8", xml_declaration=True)

    raw = np.zeros((len(line_ids), RAW_STATE_DIM), dtype=np.float32)
    canonical = np.zeros((len(line_ids), CANONICAL_STATE_DIM), dtype=np.float32)
    raw_base = len(RAW_PREFIX_NAMES)
    canonical_base = len(CANONICAL_PREFIX_NAMES)
    raw[:, raw_base] = 1.0
    canonical[:, canonical_base] = 1.0
    raw[:, raw_base + RAW_SLOT_NAMES.index("line_id")] = np.arange(12)
    canonical[
        :, canonical_base + CANONICAL_SLOT_NAMES.index("line_id")
    ] = np.arange(12)
    assert RAW_SLOT_WIDTH > RAW_SLOT_NAMES.index("line_id")
    assert CANONICAL_SLOT_WIDTH > CANONICAL_SLOT_NAMES.index("line_id")

    dataset = tmp_path / "dataset.h5"
    with h5py.File(dataset, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.create_dataset("raw_observations", data=raw)
        handle.create_dataset("observations", data=canonical)

    audit = audit_control_scope(dataset, schedule, chunk_size=5)

    assert audit["passed"] is True
    assert audit["errors"] == []
    assert audit["dataset_rows"] == 12
    assert audit["schedule_scope"]["unique_bus_ids"] == 12
    assert set(audit["action_bearing_transitions_by_line"].values()) == {1}
    assert all(value == 0 for value in audit["encoding_checks"].values())
