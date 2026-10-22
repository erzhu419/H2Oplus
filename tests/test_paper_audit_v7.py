from __future__ import annotations

import sys
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parents[1] / "paper"
if str(PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_DIR))

from audit_manuscript import math_atoms  # noqa: E402
from audit_manuscript_v7 import intervention_fidelity_findings  # noqa: E402


def test_math_atoms_ignores_tex_spacing_but_keeps_relations() -> None:
    source = (
        r"s_k\\[-0.25em] &\\hspace{5.2em} x_{m,j}\\in\\mathbb{R}^{13}"
    )

    atoms = math_atoms(source)

    assert "hspace" not in atoms
    assert "em" not in atoms
    assert "in" in atoms
    assert {"s", "k", "x", "m", "j", "R"}.issubset(atoms)


def test_intervention_fidelity_table_matches_frozen_artifact() -> None:
    summary, findings = intervention_fidelity_findings()

    assert findings == []
    assert summary["matrix_jobs"] == 240
    assert summary["sumo_rows"] == 240
    assert summary["simulator_episodes"] == 240
    assert summary["strata_per_metric"] == 21
    assert len(summary["metrics"]) == 8
