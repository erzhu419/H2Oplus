import hashlib

import pytest

from reproducibility.build_h2oplus_v8_fewshot_evaluation_manifest import (
    _method_identity,
    finite as build_finite,
    records_aggregate,
)
from reproducibility.verify_h2oplus_v8_fewshot_evaluation_manifest import (
    aggregate,
    finite as verify_finite,
)


def test_manifest_aggregate_matches_independent_verifier():
    records = [
        {"path": "b", "size_bytes": 2, "sha256": hashlib.sha256(b"b").hexdigest()},
        {"path": "a", "size_bytes": 1, "sha256": hashlib.sha256(b"a").hexdigest()},
    ]
    assert records_aggregate(records) == records_aggregate(list(reversed(records)))
    assert records_aggregate(records) == aggregate(records)


@pytest.mark.parametrize("checker", [build_finite, verify_finite])
def test_recursive_finite_check_rejects_nonfinite(checker):
    assert checker({"a": [1, 2.5, {"b": None}]})
    assert not checker({"a": [float("nan")]})
    assert not checker({"a": [float("inf")]})


def test_method_identity_accepts_only_the_frozen_grid():
    method = {
        "name": "h2oplus-seed6201-events000025000",
        "algorithm": "h2oplus",
        "training_seed": 6201,
        "online_events": 25_000,
        "training_plan_sha256": (
            "26d4f5c4ca9f337c11886855658127ef93a35954ca58eb43710aa70350610294"
        ),
    }
    assert _method_identity(method) == ("h2oplus", 6201, 25_000)
    method["online_events"] = 5_000
    with pytest.raises(ValueError, match="contract drifted"):
        _method_identity(method)
