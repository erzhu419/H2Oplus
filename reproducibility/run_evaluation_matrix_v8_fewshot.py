#!/usr/bin/env python3
"""Run the v7 matrix engine with explicit v8 few-shot provenance rules."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import sha256
from reproducibility import run_evaluation_matrix_v4 as base


METHODS_SCHEMA = "transit-evaluation-methods-v8-fewshot-v1"
SELECTION_RULE = (
    "all preregistered online-event milestones from every registered training "
    "seed; no evaluation outcome selects a checkpoint"
)
METHOD_NAME = re.compile(
    r"^(?P<algorithm>h2oplus|sac-online)-seed(?P<seed>62\d{2})-events(?P<events>\d{9})$"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
PROTOCOL_SHA_ENV = "H2OPLUS_V8_FEWSHOT_PROTOCOL_SHA256"


def bind_protocol_environment(protocol_hash: str) -> None:
    """Bind the verified protocol identity for evaluator child processes."""
    existing = os.environ.get(PROTOCOL_SHA_ENV)
    if existing not in (None, protocol_hash):
        raise ValueError(
            f"few-shot protocol environment conflicts with methods manifest: {existing}"
        )
    os.environ[PROTOCOL_SHA_ENV] = protocol_hash


def load_development_methods(path: Path, schema: str) -> list[dict]:
    manifest = path.expanduser().resolve(strict=True)
    payload = json.loads(manifest.read_text())
    if (
        schema != METHODS_SCHEMA
        or payload.get("schema_version") != METHODS_SCHEMA
        or payload.get("study_stage") != "development"
        or payload.get("selection_rule") != SELECTION_RULE
        or payload.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("few-shot methods manifest scope drifted")
    protocol_hash = payload.get("fewshot_protocol_sha256")
    if not isinstance(protocol_hash, str) or not SHA256.fullmatch(protocol_hash):
        raise ValueError("few-shot methods manifest lacks its protocol hash")
    bind_protocol_environment(protocol_hash)
    methods = []
    names = set()
    for raw in payload.get("methods", []):
        name = raw.get("name")
        match = METHOD_NAME.fullmatch(name or "")
        checkpoint_value = raw.get("checkpoint")
        if match is None or name in names or not isinstance(checkpoint_value, str):
            raise ValueError(f"malformed few-shot method: {raw!r}")
        names.add(name)
        algorithm = match.group("algorithm")
        seed = int(match.group("seed"))
        events = int(match.group("events"))
        checkpoint = Path(checkpoint_value).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = manifest.parent / checkpoint
        checkpoint = checkpoint.resolve(strict=True)
        values = {
            key: raw.get(key)
            for key in (
                "training_run_signature",
                "training_admission_sha256",
                "training_checkpoint_sha256",
            )
        }
        if any(not isinstance(value, str) or not SHA256.fullmatch(value) for value in values.values()):
            raise ValueError(f"few-shot method has malformed provenance: {name}")
        artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
        extra = artifact.get("extra", {})
        expected = {
            "method": algorithm,
            "training_seed": seed,
            "fewshot_online_events_nominal": events,
            "run_signature": values["training_run_signature"],
            "training_checkpoint_sha256": values["training_checkpoint_sha256"],
        }
        mismatches = {
            key: (extra.get(key), value)
            for key, value in expected.items()
            if extra.get(key) != value
        }
        if (
            mismatches
            or extra.get("training_admission", {}).get("sha256")
            != values["training_admission_sha256"]
            or extra.get("fewshot_protocol", {}).get("sha256") != protocol_hash
            or raw.get("learner_update_steps")
            != extra.get("fewshot_learner_update_steps")
            or raw.get("ratio_optimizer_steps")
            != extra.get("fewshot_ratio_optimizer_steps")
        ):
            raise ValueError(f"few-shot policy artifact is misbound: {name}: {mismatches}")
        methods.append(
            {
                "name": name,
                "algorithm": algorithm,
                "training_seed": seed,
                "type": "checkpoint",
                "checkpoint": checkpoint,
                "checkpoint_sha256": sha256(checkpoint),
                "training_run_signature": values["training_run_signature"],
                "training_admission_sha256": values["training_admission_sha256"],
                "training_checkpoint_sha256": values["training_checkpoint_sha256"],
                "training_plan_sha256": protocol_hash,
                "online_events": events,
                "actual_online_events": raw.get("actual_online_events"),
                "learner_update_steps": raw.get("learner_update_steps"),
                "ratio_optimizer_steps": raw.get("ratio_optimizer_steps"),
            }
        )
    if len(methods) != 70:
        raise ValueError(f"few-shot matrix requires exactly 70 unique policies, got {len(methods)}")
    return sorted(methods, key=lambda item: item["name"])


original_sources = base.evaluation_source_paths_v7


def evaluation_sources(sumo_dir: Path, edge_xml: Path) -> tuple[Path, ...]:
    paths = set(original_sources(sumo_dir, edge_xml))
    paths.update(
        {
            Path(__file__).resolve(),
            ROOT / "SimpleSAC/evaluate_transit_v8_fewshot.py",
        }
    )
    return tuple(sorted(path.resolve() for path in paths))


base.EVALUATORS["v7"] = ROOT / "SimpleSAC/evaluate_transit_v8_fewshot.py"
base.METHODS_SCHEMAS["v7"] = METHODS_SCHEMA
base.load_methods = load_development_methods
base.evaluation_source_paths_v7 = evaluation_sources
base.__file__ = str(Path(__file__).resolve())


if __name__ == "__main__":
    base.main()
