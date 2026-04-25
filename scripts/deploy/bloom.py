"""Compact Bloom filter for adapter-name membership (stable hashes, no extra deps)."""

from __future__ import annotations

import hashlib
import math
import struct


class BloomFilter:
    """Classic Bloom filter: no false negatives; false positives bounded by m, k, n."""

    __slots__ = ("_m", "_k", "_bits")

    def __init__(self, num_bits: int, num_hashes: int) -> None:
        self._m = max(8, int(num_bits))
        self._k = max(1, min(int(num_hashes), 32))
        nbytes = (self._m + 7) // 8
        self._bits = bytearray(nbytes)

    @classmethod
    def for_capacity(cls, n: int, false_positive_rate: float = 0.001) -> BloomFilter:
        """Size filter for up to ``n`` inserts with target upper bound on false positive rate."""
        n = max(1, int(n))
        p = min(max(false_positive_rate, 1e-9), 0.25)
        m = int(-n * math.log(p) / (math.log(2) ** 2))
        m = max(m, 64)
        k = max(1, min(int(round(m / n * math.log(2))), 16))
        return cls(m, k)

    def clear(self) -> None:
        self._bits[:] = b"\x00" * len(self._bits)

    def _positions(self, item: str) -> list[int]:
        digest = hashlib.blake2b(item.encode("utf-8"), digest_size=16).digest()
        h1, h2 = struct.unpack("<QQ", digest)
        h1 = h1 % self._m
        h2 = h2 % self._m
        if h2 == 0:
            h2 = 1
        return [(h1 + i * h2) % self._m for i in range(self._k)]

    def add(self, item: str) -> None:
        for i in self._positions(item):
            self._bits[i // 8] |= 1 << (i % 8)

    def might_contain(self, item: str) -> bool:
        for i in self._positions(item):
            if (self._bits[i // 8] >> (i % 8)) & 1 == 0:
                return False
        return True
