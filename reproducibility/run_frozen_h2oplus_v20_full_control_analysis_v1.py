#!/usr/bin/env python3
"""Run the unchanged frozen v20 analyzer through its local fail-closed path."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
RESULTS = REPRO / "h2oplus_v20_full_control_results"
PARTITION = RESULTS / "evaluation_partition_v1/single_jtl110gpu"
ANALYZER = REPRO / "analyze_h2oplus_v20_full_control.py"
REGISTRATION = REPRO / "h2oplus_v20_full_control_registration_v1.json"
PROTOCOL = REPRO / "h2oplus_v20_full_control_protocol_v1.json"
METHODS = REPRO / "h2oplus_v20_full_control_evaluation_methods_v1.json"
ENVIRONMENT = REPRO / "h2oplus_v20_full_control_environment_v1.json"
SOURCE_FREEZE = REPRO / "h2oplus_v20_full_control_source_freeze_v1.json"
V19_RESULTS = REPRO / "h2oplus_v19_completion_safety_results"
V19_AUDIT = V19_RESULTS / "evaluation_audit_v2.json"
V19_CLOSEOUT = V19_RESULTS / "evaluation_closeout_v2.json"
V19_VERIFICATION = V19_RESULTS / "evaluation_closeout_verification_v2.json"
DATA_AUDIT = REPRO / "h2oplus_v17_data_environment_convergence_audit_v1.json"
DATA_VERIFICATION = (
    REPRO / "h2oplus_v17_data_environment_convergence_verification_v1.json"
)
EVALUATION_AUDIT = RESULTS / "evaluation_audit_v1.json"
FILE_MANIFEST = RESULTS / "evaluation_file_manifest_v1.json"
FILE_MANIFEST_VERIFICATION = RESULTS / "evaluation_file_manifest_verification_v1.json"
OUTPUT = RESULTS / "full_control_analysis_v1.json"
EVIDENCE = RESULTS / "analysis_execution_evidence_v1.json"
MATRIX_ROOTS = tuple(
    PARTITION / f"v20-full-control-development-{index:02d}" for index in range(1, 4)
)
BLOCK_MAP = {
    "v20-full-control-development-01": ("sumo-seed-7501", 7501),
    "v20-full-control-development-02": ("sumo-seed-7502", 7502),
    "v20-full-control-development-03": ("sumo-seed-7503", 7503),
}

EXPECTED = {
    ANALYZER: "19815b30eaab6278e737ae15b1bea86a91c2621581041fef6c48f9f0ccdae5ce",
    REGISTRATION: "f930a3612720097d4f5e5b635501405245f04ffd54445f389a23800d06f7f7d8",
    PROTOCOL: "7211d32aeb1416840de442ca43fa997a918dfa282afcec5009091f78834ffa5d",
    METHODS: "77162e22cc3b748fc2ad65a229ebca5c9212fd2af172b37ba81cdaded2d2e2e9",
    ENVIRONMENT: "7210826548c94aaa8fbbbb73b62eec5450fec131f2ae3dd1d1dbc93922dff2ba",
    SOURCE_FREEZE: "b30de4416829a0bfa683184712a842dcff64732d27c3c51a5f8a3b107da22159",
    V19_AUDIT: "267370a31bed2bc8053842dc28e9223ae76c517ef3877603b3c9c81588b8636b",
    V19_CLOSEOUT: "3e95e4cf964983257fd3fabc5cab2987e1a249b5b9a25d3ae455f081932b3f74",
    V19_VERIFICATION: "ec44263eebc8ba61049973f34aadf5154d0b07f5ccc096d3659662002b95f9ab",
    DATA_AUDIT: "ff5210e82913c456314ba1e47a1f3400e0e5bdb58fce2f14d8e9bc1cdd0c95e9",
    DATA_VERIFICATION: "925fa9ace1af907bd50042718ba9d7492a892a8e8a2a863b567091540af62390",
    EVALUATION_AUDIT: "2e516ab3ea0cff10dc88607d6fb8adf3e3888a822e32a14f84a86953a466f583",
    FILE_MANIFEST: "ea41597ee6b05b136ba16d5efeaaa50b9368542419913e18c79cb9ebc7bcb51b",
    FILE_MANIFEST_VERIFICATION: "6f50f7de8b87519349d48805e2ac7f98a6ac7d9ed2df989378fd2448a95c5cb0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved.relative_to(ROOT)),
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def load_analyzer():
    spec = importlib.util.spec_from_file_location("_frozen_h2oplus_v20_analysis", ANALYZER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen v20 analyzer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_registered_output(root: Path, raw: str) -> Path:
    resolved_root = root.resolve(strict=True)
    parts = Path(raw).parts
    indices = [index for index, part in enumerate(parts) if part == resolved_root.name]
    if len(indices) != 1 or indices[0] + 1 >= len(parts):
        raise FileNotFoundError(f"v20 output lacks its registered block: {raw}")
    output = resolved_root.joinpath(*parts[indices[0] + 1 :]).resolve(strict=True)
    output.relative_to(resolved_root)
    return output


def validate_controls() -> None:
    for path, expected in EXPECTED.items():
        if sha256(path.resolve(strict=True)) != expected:
            raise ValueError(f"v20 analysis control drifted: {path}")
    audit = json.loads(EVALUATION_AUDIT.read_text())
    manifest = json.loads(FILE_MANIFEST.read_text())
    verification = json.loads(FILE_MANIFEST_VERIFICATION.read_text())
    freeze = json.loads(SOURCE_FREEZE.read_text())
    if (
        audit.get("passed") is not True
        or audit.get("formal_held_out_outcomes_read") is not False
        or audit.get("local_sumo_executed") is not False
        or audit.get("artifact_integrity", {}).get("physical_result_count") != 405
        or manifest.get("passed") is not True
        or manifest.get("physical_result_count") != 405
        or verification.get("passed") is not True
        or verification.get("physical_result_count") != 405
        or freeze.get("formal_held_out_outcomes_read") is not False
        or freeze.get("v20_development_evaluation_outcomes_read") is not False
        or len(MATRIX_ROOTS) != 3
        or any(not root.is_dir() for root in MATRIX_ROOTS)
    ):
        raise ValueError("v20 fail-closed analysis contract failed")
    mapped = []
    for root in MATRIX_ROOTS:
        expected = BLOCK_MAP.get(root.name)
        manifest = json.loads((root / "matrix_manifest.json").read_text())
        jobs = manifest.get("jobs", [])
        if (
            expected is None
            or len(jobs) != 135
            or {str(job.get("environment_block")) for job in jobs} != {expected[0]}
            or {int(job.get("sumo_seed", -1)) for job in jobs} != {expected[1]}
        ):
            raise ValueError(f"v20 registered-to-matrix block binding failed: {root}")
        mapped.extend(resolve_registered_output(root, job["output"]) for job in jobs)
    if len(mapped) != 405 or len(set(mapped)) != 405:
        raise ValueError("v20 local output mapping is not exactly 405 unique files")


def main() -> int:
    if OUTPUT.exists() or EVIDENCE.exists():
        raise FileExistsError("refusing to overwrite v20 analysis evidence")
    validate_controls()
    command = [
        str(ANALYZER),
        "--registration",
        str(REGISTRATION),
        "--protocol",
        str(PROTOCOL),
        "--methods",
        str(METHODS),
        "--environment",
        str(ENVIRONMENT),
        "--source-freeze",
        str(SOURCE_FREEZE),
        "--v19-audit",
        str(V19_AUDIT),
        "--v19-closeout",
        str(V19_CLOSEOUT),
        "--v19-verification",
        str(V19_VERIFICATION),
        "--data-environment-audit",
        str(DATA_AUDIT),
        "--data-environment-verification",
        str(DATA_VERIFICATION),
        "--matrix-root",
        *(str(root) for root in MATRIX_ROOTS),
        "--output",
        str(OUTPUT),
    ]
    module = load_analyzer()
    registered_blocks = tuple(module.BLOCKS)
    if registered_blocks != tuple(BLOCK_MAP):
        raise RuntimeError("frozen v20 analyzer block contract drifted")
    original_argv = sys.argv
    original_resolver = module.resolve_output
    module.BLOCKS = tuple(BLOCK_MAP[name][0] for name in registered_blocks)
    module.resolve_output = resolve_registered_output
    sys.argv = command
    try:
        result = int(module.main())
    finally:
        sys.argv = original_argv
        module.BLOCKS = registered_blocks
        module.resolve_output = original_resolver
    if result != 0 or not OUTPUT.is_file():
        raise RuntimeError("frozen v20 analyzer did not produce its registered output")
    if sha256(ANALYZER) != EXPECTED[ANALYZER]:
        raise RuntimeError("frozen v20 analyzer changed during execution")
    analysis = json.loads(OUTPUT.read_text())
    scientific = analysis.get("development_scientific_result", {})
    if (
        analysis.get("schema_version")
        != "transit-h2oplus-v20-full-control-analysis-v1"
        or analysis.get("formal_held_out_outcomes_read") is not False
        or analysis.get("artifact_integrity", {}).get("result_count") != 405
        or len(analysis.get("result_files", [])) != 405
        or analysis.get("formal_confirmation_authorized") is not False
        or analysis.get("formal_confirmation_submitted") is not False
        or scientific.get("full_development_control_matrix_passed")
        is not analysis.get("formal_confirmation_eligible")
    ):
        raise ValueError("v20 analyzer output failed its execution boundary")
    payload = {
        "schema_version": "transit-h2oplus-v20-full-control-analysis-execution-evidence-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "frozen_analyzer": descriptor(ANALYZER),
        "frozen_analyzer_exit_code": result,
        "analysis_output": descriptor(OUTPUT),
        "evaluation_audit": descriptor(EVALUATION_AUDIT),
        "file_manifest": descriptor(FILE_MANIFEST),
        "file_manifest_verification": descriptor(FILE_MANIFEST_VERIFICATION),
        "matrix_manifests": [
            descriptor(root / "matrix_manifest.json") for root in MATRIX_ROOTS
        ],
        "registered_to_matrix_block_mapping": {
            name: {"environment_block": value[0], "sumo_seed": value[1]}
            for name, value in BLOCK_MAP.items()
        },
        "local_output_path_mapping": "registered-block-relative",
        "physical_result_count": 405,
        "development_gate_passed": scientific[
            "full_development_control_matrix_passed"
        ],
        "no_remote_sync_performed": True,
        "checkpoint_files_synchronized": False,
        "local_sumo_executed": False,
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_authorized_or_submitted": False,
    }
    partial = EVIDENCE.with_suffix(EVIDENCE.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, EVIDENCE)
    print(
        json.dumps(
            {
                "analysis": descriptor(OUTPUT),
                "execution_evidence": descriptor(EVIDENCE),
                "development_gate_passed": payload["development_gate_passed"],
                "formal_confirmation_authorized_or_submitted": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
