#!/usr/bin/env python3
"""Validate and execute a capacity-audited v7 training dispatch manifest."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproducibility.run_training_dispatch_v6 import main


if __name__ == "__main__":
    raise SystemExit(main())
