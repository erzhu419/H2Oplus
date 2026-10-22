#!/usr/bin/env python3
"""Run the frozen v20 formal dispatch with JSON-safe evidence paths."""

from __future__ import annotations

import json
from pathlib import Path

from reproducibility import run_h2oplus_v20_formal_confirmation_dispatch as frozen


_FROZEN_VERIFY_INPUTS = frozen.verify_inputs


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    return value


def verify_inputs(args):
    evidence = json_ready(_FROZEN_VERIFY_INPUTS(args))
    json.dumps(evidence, sort_keys=True)
    return evidence


def main() -> int:
    frozen.verify_inputs = verify_inputs
    return frozen.main()


if __name__ == "__main__":
    raise SystemExit(main())
