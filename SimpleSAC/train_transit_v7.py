#!/usr/bin/env python3
"""Train transit-smdp-v7 policies with exact resume and physical rewards."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SimpleSAC.train_transit_v3 import main


if __name__ == "__main__":
    main("v7")
