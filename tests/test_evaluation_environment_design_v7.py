from argparse import Namespace
import json
from pathlib import Path

from reproducibility.build_evaluation_environment_design_v7 import build_design
from reproducibility.run_evaluation_matrix_v4 import load_environment_design


def test_v7_evaluation_design_is_hash_disjoint_and_runner_compatible(tmp_path):
    output = tmp_path / "evaluation-v7.json"
    args = Namespace(
        source_design=(
            Path(__file__).resolve().parents[1]
            / "reproducibility"
            / "evaluation_environment_design_v4_formal_v1.json"
        ),
        training_design=(
            Path(__file__).resolve().parents[1]
            / "reproducibility"
            / "training_environment_design_v7_formal_v1.json"
        ),
        output=output,
        force=False,
    )
    payload = build_design(args)
    output.write_text(json.dumps(payload))

    assert payload["mdp_version"] == "v7"
    assert payload["environment_cells"] == 30
    assert payload["independent_realization_blocks"] == 10
    assert payload["training_disjointness"]["verified_disjoint"] is True
    assert payload["training_disjointness"]["scenario_manifest_hash_overlap"] == []
    assert payload["training_disjointness"]["scenario_runtime_hash_overlap"] == []
    cells, evidence = load_environment_design(output, expected_mdp_version="v7")
    assert len(cells) == 30
    assert evidence["mode"] == "explicit-blocks"
    assert evidence["mdp_version"] == "v7"


def test_v7_runner_rejects_unwrapped_v4_environment_design():
    source = (
        Path(__file__).resolve().parents[1]
        / "reproducibility"
        / "evaluation_environment_design_v4_formal_v1.json"
    )
    import pytest

    with pytest.raises(ValueError, match="hash-disjoint v7"):
        load_environment_design(source, expected_mdp_version="v7")
