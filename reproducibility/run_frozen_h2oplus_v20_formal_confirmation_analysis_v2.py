#!/usr/bin/env python3
"""Run frozen v20 formal analysis with exact label and local-path adapters."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproducibility import (  # noqa: E402
    run_frozen_h2oplus_v20_formal_confirmation_analysis_v1 as base,
)


REPRO = ROOT / "reproducibility"
RESULTS = REPRO / "h2oplus_v20_formal_confirmation_results"
REGISTRATION = REPRO / "h2oplus_v20_formal_confirmation_analysis_adapter_registration_v2.json"
EVIDENCE = RESULTS / "analysis_boundary_adapter_evidence_v2.json"


def load_registration() -> dict:
    payload = json.loads(REGISTRATION.read_text())
    if (
        payload.get("schema_version")
        != "transit-h2oplus-v20-formal-confirmation-analysis-adapter-registration-v2"
        or payload.get("status")
        != "registered-after-two-zero-output-analysis-boundary-incidents"
        or payload.get("frozen_analyzer_sha256") != base.EXPECTED_ANALYZER_SHA256
        or payload.get("adapter_sha256") != base.sha256(Path(__file__).resolve())
        or payload.get("registered_to_matrix_block")
        != base.REGISTERED_TO_MATRIX_BLOCK
        or payload.get("result_files_or_manifests_may_be_modified") is not False
        or payload.get("performance_values_or_gate_formulas_may_change") is not False
        or payload.get("formal_outcome_access_authorized") is not True
    ):
        raise ValueError("formal analysis adapter v2 registration drifted")
    for item in payload.get("bindings", {}).values():
        path = ROOT / item["path"]
        if base.sha256(path.resolve(strict=True)) != item.get("sha256"):
            raise ValueError(f"analysis adapter v2 binding drifted: {item.get('path')}")
    return payload


def strict_local_output(root: Path, raw: str) -> Path:
    value = Path(raw).expanduser()
    if len(value.parts) < 3:
        raise ValueError(f"formal output path lacks its registered suffix: {raw!r}")
    suffix = value.parts[-3:]
    local = root.joinpath(*suffix)
    if not local.is_file():
        raise FileNotFoundError(f"formal output suffix is absent: {suffix!r}")
    resolved = local.resolve(strict=True)
    if root.resolve(strict=True) not in resolved.parents:
        raise ValueError("formal output resolved outside its matrix root")
    return resolved


def main() -> int:
    if base.sha256(base.ANALYZER) != base.EXPECTED_ANALYZER_SHA256:
        raise RuntimeError("frozen v20 formal analyzer identity drifted")
    if EVIDENCE.exists():
        raise FileExistsError(f"refusing to overwrite adapter evidence: {EVIDENCE}")
    registration = load_registration()
    roots = base.matrix_roots(sys.argv)
    actual_blocks, matrices = base.validate_block_labels(roots)
    output = base.output_path(sys.argv)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite analysis output: {output}")

    module = base.load_analyzer()
    if tuple(module.BLOCKS) != tuple(base.REGISTERED_TO_MATRIX_BLOCK):
        raise RuntimeError("frozen analyzer registered block contract drifted")
    module.BLOCKS = actual_blocks
    module.resolve_output = strict_local_output
    try:
        result = int(module.main())
    finally:
        if base.sha256(base.ANALYZER) != base.EXPECTED_ANALYZER_SHA256:
            raise RuntimeError("frozen analyzer changed during execution")
    if result != 0 or not output.is_file():
        raise RuntimeError("frozen analyzer did not complete and create its output")

    base.atomic_json(
        EVIDENCE,
        {
            "schema_version": (
                "transit-h2oplus-v20-formal-confirmation-analysis-boundary-"
                "adapter-evidence-v2"
            ),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "passed": True,
            "registration": {
                "path": str(REGISTRATION.relative_to(ROOT)),
                "sha256": base.sha256(REGISTRATION),
            },
            "frozen_analyzer": {
                "path": str(base.ANALYZER.relative_to(ROOT)),
                "sha256": base.EXPECTED_ANALYZER_SHA256,
            },
            "frozen_analyzer_exit_code": result,
            "analysis_output": {
                "path": str(output.relative_to(ROOT)),
                "sha256": base.sha256(output),
                "bytes": output.stat().st_size,
            },
            "registered_to_matrix_block": base.REGISTERED_TO_MATRIX_BLOCK,
            "matrices": matrices,
            "adapter_scope": (
                "in-memory frozen analyzer BLOCKS constant and exact final-three-"
                "path-components local result resolver only"
            ),
            "result_files_or_manifests_modified": False,
            "performance_values_or_preregistered_gate_formulas_changed": False,
            "scientific_sources_changed": False,
            "prior_zero_output_analysis_incidents_bound": 2,
            "formal_held_out_outcome_access_authorized": True,
            "formal_held_out_outcomes_read": True,
            "registration_status": registration["status"],
        },
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
