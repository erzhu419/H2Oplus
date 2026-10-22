#!/usr/bin/env python3
"""Run the frozen v20 endpoint-only formal confirmation matrix."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import sha256  # noqa: E402
from reproducibility import (  # noqa: E402
    run_evaluation_matrix_v14_terminal_taper as v14,
)
from reproducibility import (  # noqa: E402
    run_evaluation_matrix_v19_completion_safety as v19,
)


METHODS_SCHEMA = "transit-evaluation-methods-v20-formal-confirmation-v1"
SELECTION_RULE = (
    "v19 quarter-load-sixty fixed as the sole formal candidate after the "
    "verified v20 development closeout; exact "
    "seed-matched v19 direct parents, raw v14 terminal-quarter parents and "
    "six frozen v14 lineage controls at 25k fixed without behavioral outcome "
    "selection; endpoint-only physical behaviors are evaluated on thirty fresh "
    "formal confirmation cells"
)
V20_PROTOCOL_SHA_ENV = "H2OPLUS_V20_FORMAL_CONFIRMATION_PROTOCOL_SHA256"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SEEDS = tuple(range(6201, 6206))
CANDIDATE = "h2oplus-v19-completion-safety-quarter-load-sixty"
DIRECT_PARENT = "h2oplus-v19-completion-safety-baseline"
RAW_PARENT = "h2oplus-v14-terminal-quarter"
LINEAGE_CONTROLS = (
    "h2oplus",
    "sac-online",
    "h2oplus-v10-delayed-actor",
    "h2oplus-v11-actor-hold",
    "h2oplus-v12-actor-ramp",
    "h2oplus-v13-proximal-strong",
)
ALGORITHMS = (CANDIDATE, DIRECT_PARENT, RAW_PARENT, *LINEAGE_CONTROLS)
EXPECTED_ROLES = {
    CANDIDATE: "predesignated-candidate",
    DIRECT_PARENT: "direct-behavior-parent",
    RAW_PARENT: "raw-actor-parent",
    **{algorithm: "independent-lineage-control" for algorithm in LINEAGE_CONTROLS},
}


def _resolve(owner: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve(strict=True)


def _bind_protocol(payload: dict) -> None:
    value = payload.get("v20_protocol_sha256")
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError("v20 methods manifest lacks a protocol identity")
    existing = os.environ.get(V20_PROTOCOL_SHA_ENV)
    if existing not in (None, value):
        raise ValueError("v20 protocol environment conflicts with methods manifest")
    os.environ[V20_PROTOCOL_SHA_ENV] = value


def _source_methods(manifest: Path, source: str) -> dict[str, dict]:
    if source == "v19":
        rows = v19.load_development_methods(manifest, v19.METHODS_SCHEMA)
    elif source == "v14":
        rows = v14.load_development_methods(manifest, v14.METHODS_SCHEMA)
    else:
        raise ValueError(f"unsupported v20 source manifest {source!r}")
    return {row["name"]: row for row in rows}


def load_formal_methods(path: Path, schema: str) -> list[dict]:
    manifest = path.expanduser().resolve(strict=True)
    payload = json.loads(manifest.read_text())
    selections = payload.get("methods", [])
    parents = payload.get("parent_method_manifests", {})
    if (
        schema != METHODS_SCHEMA
        or payload.get("schema_version") != METHODS_SCHEMA
        or payload.get("study_stage") != "formal-confirmation"
        or payload.get("matrix_type") != "endpoint-only-formal-confirmation"
        or payload.get("selection_rule") != SELECTION_RULE
        or payload.get("formal_held_out_outcomes_read") is not False
        or payload.get("formal_confirmation_outcomes_read") is not False
        or payload.get("physical_method_count") != 45
        or payload.get("logical_method_count") != 45
        or payload.get("expected_new_remote_sumo_episodes") != 1350
        or payload.get("predesignated_candidate") != CANDIDATE
        or payload.get("algorithms") != list(ALGORITHMS)
        or len(selections) != 45
        or set(parents) != {"v14", "v19"}
    ):
        raise ValueError("v20 formal-confirmation methods manifest scope drifted")
    _bind_protocol(payload)

    source_rows: dict[str, dict[str, dict]] = {}
    for source, descriptor in parents.items():
        parent_path = _resolve(manifest, descriptor["path"])
        if sha256(parent_path) != descriptor.get("sha256"):
            raise ValueError(f"v20 parent methods identity drifted: {source}")
        source_rows[source] = _source_methods(parent_path, source)

    candidate_defaults = {
        row["training_seed"]: row
        for row in source_rows["v19"].values()
        if row["algorithm"] == CANDIDATE
    }
    if set(candidate_defaults) != set(SEEDS):
        raise ValueError("v20 candidate seed grid is incomplete")

    methods = []
    names = set()
    algorithm_seed_pairs = set()
    for selected in selections:
        source = selected.get("source_manifest")
        name = selected.get("name")
        parent = source_rows.get(source, {}).get(name)
        if parent is None:
            raise ValueError(f"v20 selected method is absent from its parent: {name}")
        algorithm = parent["algorithm"]
        seed = parent["training_seed"]
        pair = (algorithm, seed)
        if (
            name in names
            or pair in algorithm_seed_pairs
            or algorithm not in ALGORITHMS
            or seed not in SEEDS
            or parent.get("online_events") != 25_000
            or selected.get("algorithm") != algorithm
            or selected.get("training_seed") != seed
            or selected.get("role") != EXPECTED_ROLES[algorithm]
            or selected.get("checkpoint_sha256") != parent["checkpoint_sha256"]
            or selected.get("actor_tensor_sha256") != parent["actor_tensor_sha256"]
            or selected.get("predesignated_candidate") is not (algorithm == CANDIDATE)
        ):
            raise ValueError(f"malformed or duplicate v20 selection: {selected!r}")
        names.add(name)
        algorithm_seed_pairs.add(pair)
        method = dict(parent)
        method.update(
            {
                "v20_role": selected["role"],
                "v20_source_manifest": source,
                "v20_protocol_sha256": payload["v20_protocol_sha256"],
                "formal_confirmation_protocol_sha256": payload[
                    "v20_protocol_sha256"
                ],
            }
        )
        # Later evaluators require their frozen identity arguments even when a
        # legacy method immediately delegates to an older behavior loader.
        defaults = candidate_defaults[seed]
        for key in (
            "temperature_tensor_sha256",
            "parent_method",
            "v17_parent_method",
            "parent_base_hold_scale",
            "parent_headway_balance_gain",
            "parent_reserve_horizon_fraction",
            "tail_service_floor_seconds",
            "activation_start_fraction",
            "activation_end_fraction",
            "max_onboard_capacity_fraction",
            "v19_reserve_enabled",
            "v19_reserve_floor_seconds",
            "v19_activation_start_fraction",
            "v19_activation_end_fraction",
            "v19_max_onboard_fraction",
        ):
            method.setdefault(key, defaults[key])
        methods.append(method)
    expected = {(algorithm, seed) for algorithm in ALGORITHMS for seed in SEEDS}
    if algorithm_seed_pairs != expected or len(methods) != 45:
        raise ValueError("v20 matrix is not the exact 9-by-5 physical grid")
    return sorted(methods, key=lambda item: item["name"])


def evaluation_sources(sumo_dir: Path, edge_xml: Path) -> tuple[Path, ...]:
    paths = set(v19.evaluation_sources(sumo_dir, edge_xml))
    paths.update(
        {
            ROOT / "SimpleSAC/evaluate_transit_v20_formal_confirmation.py",
            Path(__file__).resolve(),
        }
    )
    return tuple(sorted(path.resolve() for path in paths))


def job_command(job: dict) -> list[str]:
    return v19.job_command(job)


base = v19.base
base.EVALUATORS["v7"] = ROOT / "SimpleSAC/evaluate_transit_v20_formal_confirmation.py"
base.METHODS_SCHEMAS["v7"] = METHODS_SCHEMA
base.load_methods = load_formal_methods
base.evaluation_source_paths_v7 = evaluation_sources
base._job_command = job_command
base.__file__ = str(Path(__file__).resolve())


if __name__ == "__main__":
    base.main()
