#!/usr/bin/env python3
"""Canonical server entry point for the dependency-ordered v7 trainer gates."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproducibility.run_local_trainer_gates_v7 import main


if __name__ == "__main__":
    main()
