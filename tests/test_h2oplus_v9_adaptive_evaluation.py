import json
import hashlib
from pathlib import Path

from SimpleSAC import evaluate_transit_v9_adaptive as evaluator
from reproducibility import run_evaluation_matrix_v9_adaptive as matrix
from reproducibility.analyze_h2oplus_v9_adaptive import expand_aliases


ROOT = evaluator.ROOT


def test_methods_manifest_loads_exact_physical_actor_grid(monkeypatch):
    monkeypatch.delenv(matrix.V8_PROTOCOL_SHA_ENV, raising=False)
    monkeypatch.delenv(matrix.V9_PROTOCOL_SHA_ENV, raising=False)
    path = ROOT / "reproducibility/h2oplus_v9_adaptive_evaluation_methods_v1.json"

    methods = matrix.load_development_methods(path, matrix.METHODS_SCHEMA)

    assert len(methods) == 42
    assert len({item["actor_tensor_sha256"] for item in methods}) == 42
    assert matrix.os.environ[matrix.V8_PROTOCOL_SHA_ENV] == matrix.V8_PROTOCOL_SHA256
    assert matrix.os.environ[matrix.V9_PROTOCOL_SHA_ENV] == (
        "e0652db511c5ca46e0e14801895bdfdc57761223842a4e30673075b9163d3c7d"
    )


def test_evaluator_and_matrix_source_contracts_are_equal(monkeypatch):
    base_source = Path("/tmp/base-v7-evaluator.py")
    monkeypatch.setattr(
        evaluator,
        "_base_evaluation_source_paths",
        lambda _sumo_dir, _edge_xml: (base_source,),
    )
    monkeypatch.setattr(
        matrix,
        "original_sources",
        lambda _sumo_dir, _edge_xml: (base_source,),
    )

    child = set(evaluator.evaluation_source_paths(Path("/tmp/sumo"), Path("/tmp/edge")))
    parent = set(matrix.evaluation_sources(Path("/tmp/sumo"), Path("/tmp/edge")))

    assert child == parent
    assert Path(evaluator.__file__).resolve() in child
    assert Path(matrix.__file__).resolve() in child


def test_job_command_propagates_actor_and_step_contract(monkeypatch):
    monkeypatch.setattr(matrix, "original_job_command", lambda _job: ["python", "eval.py"])
    method = {
        "type": "checkpoint",
        "actor_tensor_sha256": "a" * 64,
        "learner_update_steps": 54_000,
        "ratio_optimizer_steps": 2_000,
    }

    command = matrix.job_command({"method": method})

    assert command[-6:] == [
        "--expected-actor-tensor-sha256",
        "a" * 64,
        "--expected-learner-update-steps",
        "54000",
        "--expected-ratio-optimizer-steps",
        "2000",
    ]


def test_alias_manifest_expands_252_physical_rows_to_360_logical_rows():
    payload = json.loads(
        (ROOT / "reproducibility/h2oplus_v9_adaptive_evaluation_methods_v1.json").read_text()
    )
    rows = []
    for method in payload["methods"]:
        for block in ("block01", "block02"):
            for scale in (0.85, 1.0, 1.15):
                rows.append(
                    {
                        "physical_method": method["name"],
                        "block": block,
                        "scale": scale,
                        "primary": 1.0,
                        "completion": 1.0,
                        "output": "synthetic",
                        "sha256": "b" * 64,
                    }
                )

    expanded = expand_aliases(rows, payload["aliases"])

    assert len(rows) == 252
    assert len(expanded) == 360
    assert len({(row["algorithm"], row["seed"], row["events"]) for row in expanded}) == 60


def test_v2_methods_manifest_is_relocatable():
    path = ROOT / "reproducibility/h2oplus_v9_adaptive_evaluation_methods_v2.json"
    payload = json.loads(path.read_text())

    for field in ("count_correction", "portability_correction"):
        descriptor = payload[field]
        assert not Path(descriptor["path"]).is_absolute()
        target = (path.parent / descriptor["path"]).resolve(strict=True)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == descriptor["sha256"]

    methods = matrix.load_development_methods(path, matrix.METHODS_SCHEMA)
    assert len(methods) == 42


def test_v9_evaluation_sources_are_python38_compatible():
    for path in (
        ROOT / "SimpleSAC/evaluate_transit_v9_adaptive.py",
        ROOT / "reproducibility/run_evaluation_matrix_v9_adaptive.py",
    ):
        source = path.read_text()
        assert ".removeprefix(" not in source
        assert ".removesuffix(" not in source
