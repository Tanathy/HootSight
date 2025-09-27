"""Minimal dice: a single roll(min, max) function using aggressive local entropy.

This implementation tries to harvest as much local entropy as practical without
reaching the network: high-resolution timing jitter (time.perf_counter_ns),
small chunks of OS-provided entropy (os.urandom), and process-specific values
are mixed and hashed with SHA-512 to produce a large digest. The digest is
used for unbiased integer sampling (rejection sampling) or to map to a float
in [min, max].

Important: this is still not a hardware TRNG. It is a pragmatic, offline
approach that provides high-quality randomness for games and many other
use-cases. For absolute hardware-backed true randomness you need a dedicated
TRNG device or an online quantum RNG.
"""
from __future__ import annotations

import os
import time
import hashlib
from typing import Union

Number = Union[int, float]


def _gather_local_entropy(samples: int = 64) -> bytes:
    """Mix local entropy sources and return a SHA-512 digest.

    - samples: number of timing/os samples to mix. Higher => more CPU work
      and (slightly) larger mixing surface.
    """
    if samples < 1:
        samples = 1

    h = hashlib.sha512()
    # mix stable-ish process info
    try:
        h.update(os.getpid().to_bytes(4, "little", signed=False))
    except Exception:
        h.update(b"pid")

    # try to mix a small amount of OS entropy up-front
    try:
        h.update(os.urandom(16))
    except Exception:
        h.update(b"urandom-fail")

    for i in range(samples):
        # high-resolution timing jitter
        t = time.perf_counter_ns()
        h.update(t.to_bytes(8, "little", signed=False))
        # small os.urandom chunk if available
        try:
            h.update(os.urandom(4))
        except Exception:
            # fall back to mixing a derived value
            h.update(((t ^ i) & 0xFFFFFFFF).to_bytes(4, "little", signed=False))

    return h.digest()


def _digest_to_int(digest: bytes) -> int:
    return int.from_bytes(digest, "big")


def roll(min_value: Number = 1, max_value: Number = 20) -> Number:
    """Roll between min_value and max_value using local entropy mixing.

    - If both inputs are int instances, returns an int in [min, max] inclusive.
    - Otherwise returns a float in [min, max].

    Raises ValueError for invalid bounds.
    """
    try:
        fmin = float(min_value)
        fmax = float(max_value)
    except Exception:
        raise ValueError("min and max must be numbers")

    if fmin > fmax:
        raise ValueError("min must be <= max")

    # Harvest entropy
    digest = _gather_local_entropy(samples=64)
    bits = len(digest) * 8
    cur = _digest_to_int(digest)

    # Integer path: unbiased sampling via rejection sampling
    if isinstance(min_value, int) and isinstance(max_value, int):
        imin = int(fmin)
        imax = int(fmax)
        rng = imax - imin + 1
        if rng <= 0:
            raise ValueError("invalid integer range")

        max_val = 1 << bits
        threshold = (max_val // rng) * rng
        # rehash until within threshold
        while cur >= threshold:
            digest = hashlib.sha512(digest).digest()
            cur = _digest_to_int(digest)
        return imin + (cur % rng)

    # Float path: map the digest to [0.0, 1.0]
    max_val = (1 << bits) - 1
    unit = cur / float(max_val)
    return fmin + unit * (fmax - fmin)


__all__ = ["roll"]

