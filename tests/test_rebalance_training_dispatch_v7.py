from __future__ import annotations

from reproducibility.rebalance_training_dispatch_v7 import schedule_jobs


def _profile() -> dict:
    return {
        "hosts": {
            "fast": {"methods": {"long": 6.0, "short": 1.0}},
            "slow": {"methods": {"long": 9.0, "short": 1.5}},
        }
    }


def _jobs() -> list[dict]:
    return [
        {"name": "long-1", "method": "long"},
        {"name": "long-2", "method": "long"},
        {"name": "long-3", "method": "long"},
        {"name": "short-1", "method": "short"},
        {"name": "short-2", "method": "short"},
        {"name": "short-3", "method": "short"},
    ]


def test_host_preserving_balance_does_not_move_jobs_between_hosts() -> None:
    lanes = [("fast", "0"), ("fast", "1"), ("slow", "0"), ("slow", "1")]
    original_host = {
        "long-1": "fast",
        "long-2": "fast",
        "long-3": "fast",
        "short-1": "slow",
        "short-2": "slow",
        "short-3": "slow",
    }

    assignments, loads = schedule_jobs(
        jobs=_jobs(),
        lanes=lanes,
        profile=_profile(),
        original_host=original_host,
        allow_cross_host=False,
    )

    assert {name for lane in lanes[:2] for name in assignments[lane]} == {
        "long-1",
        "long-2",
        "long-3",
    }
    assert {name for lane in lanes[2:] for name in assignments[lane]} == {
        "short-1",
        "short-2",
        "short-3",
    }
    assert max(loads[("fast", "0")], loads[("fast", "1")]) == 12.0
    assert max(loads[("slow", "0")], loads[("slow", "1")]) == 3.0


def test_cross_host_balance_is_deterministic_and_uses_all_lanes() -> None:
    lanes = [("fast", "0"), ("fast", "1"), ("slow", "0"), ("slow", "1")]
    original_host = {job["name"]: "fast" for job in _jobs()}

    first, first_loads = schedule_jobs(
        jobs=_jobs(),
        lanes=lanes,
        profile=_profile(),
        original_host=original_host,
        allow_cross_host=True,
    )
    second, second_loads = schedule_jobs(
        jobs=reversed(_jobs()),
        lanes=lanes,
        profile=_profile(),
        original_host=original_host,
        allow_cross_host=True,
    )

    assert first == second
    assert first_loads == second_loads
    assert all(first[lane] for lane in lanes)
    assert {name for names in first.values() for name in names} == {
        job["name"] for job in _jobs()
    }
