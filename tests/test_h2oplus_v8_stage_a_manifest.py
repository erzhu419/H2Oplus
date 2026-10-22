import hashlib

import pytest

from reproducibility.build_h2oplus_v8_stage_a_manifest import (
    all_numbers_finite,
    records_aggregate,
)
from reproducibility.verify_h2oplus_v8_stage_a_manifest import aggregate, finite


def test_manifest_aggregate_is_order_independent_and_matches_verifier():
    records = [
        {"path": "b", "size_bytes": 2, "sha256": hashlib.sha256(b"b").hexdigest()},
        {"path": "a", "size_bytes": 1, "sha256": hashlib.sha256(b"a").hexdigest()},
    ]
    assert records_aggregate(records) == records_aggregate(list(reversed(records)))
    assert records_aggregate(records) == aggregate(records)


@pytest.mark.parametrize("checker", [all_numbers_finite, finite])
def test_recursive_finite_check_rejects_nonfinite(checker):
    assert checker({"a": [1, 2.5, {"b": None}]})
    assert not checker({"a": [float("nan")]})
    assert not checker({"a": [float("inf")]})
