#!/usr/bin/env python3
"""Run the frozen v12 analyzer with a JSON-only NumPy scalar adapter."""

from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
ANALYZER = REPO / "reproducibility/analyze_h2oplus_v12_staged_release.py"
EXPECTED_ANALYZER_SHA256 = (
    "a77ffa04f3e2f2bfc5c6e3495505cb70cace2326df54e6906f6fcca34a9aa4e4"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    actual = sha256(ANALYZER)
    if actual != EXPECTED_ANALYZER_SHA256:
        raise RuntimeError(f"frozen v12 analyzer identity drifted: {actual}")

    original_default = json.JSONEncoder.default

    def numpy_scalar_default(encoder: json.JSONEncoder, value: object) -> object:
        if isinstance(value, np.generic):
            return value.item()
        return original_default(encoder, value)

    json.JSONEncoder.default = numpy_scalar_default
    try:
        runpy.run_path(str(ANALYZER), run_name="__main__")
    finally:
        json.JSONEncoder.default = original_default
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
