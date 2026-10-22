"""Canonical provenance hashing for v10 branchable checkpoints."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _hash_bytes(digest: Any, tag: bytes, value: bytes = b"") -> None:
    digest.update(struct.pack("!I", len(tag)))
    digest.update(tag)
    digest.update(struct.pack("!Q", len(value)))
    digest.update(value)


def _hash_value(digest: Any, value: Any) -> None:
    if value is None:
        _hash_bytes(digest, b"none")
    elif isinstance(value, bool):
        _hash_bytes(digest, b"bool", b"1" if value else b"0")
    elif isinstance(value, int):
        _hash_bytes(digest, b"int", str(value).encode("ascii"))
    elif isinstance(value, float):
        _hash_bytes(digest, b"float64", struct.pack("!d", value))
    elif isinstance(value, str):
        _hash_bytes(digest, b"str", value.encode("utf-8"))
    elif isinstance(value, bytes):
        _hash_bytes(digest, b"bytes", value)
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        _hash_bytes(digest, b"torch.dtype", str(tensor.dtype).encode("ascii"))
        _hash_bytes(
            digest,
            b"torch.shape",
            json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"),
        )
        try:
            raw = tensor.numpy().tobytes(order="C")
        except TypeError:
            raw = bytes(tensor.reshape(-1).view(torch.uint8).untyped_storage())
        _hash_bytes(digest, b"torch.data", raw)
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        _hash_bytes(digest, b"numpy.dtype", array.dtype.str.encode("ascii"))
        _hash_bytes(
            digest,
            b"numpy.shape",
            json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"),
        )
        _hash_bytes(digest, b"numpy.data", array.tobytes(order="C"))
    elif isinstance(value, np.generic):
        _hash_value(digest, value.item())
    elif isinstance(value, dict):
        _hash_bytes(digest, b"dict.length", str(len(value)).encode("ascii"))
        keys = sorted(
            value,
            key=lambda key: (
                type(key).__module__,
                type(key).__qualname__,
                repr(key),
            ),
        )
        for key in keys:
            _hash_value(digest, key)
            _hash_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        _hash_bytes(
            digest,
            b"list" if isinstance(value, list) else b"tuple",
            str(len(value)).encode("ascii"),
        )
        for item in value:
            _hash_value(digest, item)
    else:
        raise TypeError(
            "unsupported v10 checkpoint value for canonical hashing: "
            f"{type(value).__module__}.{type(value).__qualname__}"
        )


def checkpoint_state_sha256(path: Path) -> str:
    """Hash every checkpoint field except its mutable provenance envelope."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = {key: value for key, value in payload.items() if key != "extra"}
    digest = hashlib.sha256()
    _hash_value(digest, state)
    return digest.hexdigest()
