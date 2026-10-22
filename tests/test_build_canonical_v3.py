import json

import h5py
import numpy as np

from bus_h2o.mdp_contract import CANONICAL_STATE_DIM, SCHEMA_VERSION, TransitMDPConfig
from collect_policy.build_canonical_v3 import convert


def _row(bus, station, time, fwd=360.0, bwd=360.0):
    return [
        0,
        bus,
        station,
        0,
        0,
        fwd,
        bwd,
        1,
        360,
        5,
        time,
        360 - fwd,
        360,
        360,
        13.89,
        -1,
        0,
    ]


def test_converter_recomputes_reward_and_writes_provenance(tmp_path):
    source = tmp_path / "v2.h5"
    target = tmp_path / "v3.h5"
    observations = np.asarray(
        [_row(1, 1, 100), _row(2, 2, 200, 300, 420)], dtype=np.float32
    )
    next_observations = np.asarray(
        [_row(1, 2, 180), _row(2, 3, 300, 180, 540)], dtype=np.float32
    )
    with h5py.File(source, "w") as handle:
        handle.create_dataset("observations", data=observations)
        handle.create_dataset("next_observations", data=next_observations)
        handle.create_dataset("actions", data=np.asarray([[-1, 0.8], [1, -0.4]], np.float32))
        handle.create_dataset("rewards", data=np.asarray([-999, -999], np.float32))
        handle.create_dataset("seed", data=np.asarray([42, 1024], np.int32))
        handle.create_dataset("policy_id", data=np.asarray([4, 2], np.int8))
        handle.attrs["policy_names"] = json.dumps({"random": 2, "zero": 4})

    summary = convert(source, target, TransitMDPConfig(), chunk_rows=1)
    assert summary["rows"] == 2
    with h5py.File(target, "r") as handle:
        assert handle.attrs["schema_version"] == SCHEMA_VERSION
        assert handle["observations"].shape == (2, CANONICAL_STATE_DIM)
        assert handle["actions"].shape == (2, 1)
        np.testing.assert_allclose(handle["actions"][:, 0], [-1, 1])
        assert handle["rewards"][0] == 0.0
        assert handle["rewards"][1] < 0.0
        assert handle["discounts"][1] < handle["discounts"][0]
        np.testing.assert_array_equal(handle["split_id"][:], [0, 0])
        np.testing.assert_array_equal(handle["policy_seed"][:], [42, 1024])
        np.testing.assert_array_equal(handle["sumo_seed"][:], [-1, -1])
        np.testing.assert_array_equal(handle["source_index"][:], [0, 1])
        np.testing.assert_array_equal(handle["source_file_id"][:], [0, 1])
        np.testing.assert_array_equal(handle["episode_id"][:], [0, 1])
        assert len(np.unique(handle["trajectory_id"][:])) == 2
        assert "discarded" in handle.attrs["legacy_second_action"]
