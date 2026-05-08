"""Compact Bloom filter for adapter-name membership (stable hashes, no extra deps)."""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
import struct
from collections.abc import Iterable

# Reject oversized wire blobs (bits length == ceil(m/8)).
MAX_BLOOM_PACKED_BYTES = 262_144


class BloomFilter:
    """Classic Bloom filter: no false negatives; false positives bounded by m, k, n."""

    __slots__ = ("_m", "_k", "_bits", "_sized_for_n")

    def __init__(self, num_bits: int, num_hashes: int) -> None:
        self._m = max(8, int(num_bits))
        self._k = max(1, min(int(num_hashes), 32))
        nbytes = (self._m + 7) // 8
        self._bits = bytearray(nbytes)
        self._sized_for_n: int | None = None

    @classmethod
    def for_capacity(cls, n: int, false_positive_rate: float = 0.001) -> BloomFilter:
        """Size filter for up to ``n`` inserts with target upper bound on false positive rate."""
        n = max(1, int(n))
        p = min(max(false_positive_rate, 1e-9), 0.25)
        m = int(-n * math.log(p) / (math.log(2) ** 2))
        m = max(m, 64)
        k = max(1, min(int(round(m / n * math.log(2))), 16))
        inst = cls(m, k)
        inst._sized_for_n = n  # used to reuse same (m, k) across refills
        return inst

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

    def refill_from_adapter_names(self, names: Iterable[str]) -> None:
        """Clear and insert ``names`` (same ``m``/``k`` as construction; exact vs this set)."""
        self.clear()
        for name in names:
            self.add(name)

    def wire_shape(self) -> dict[str, int]:
        """Small summary for debug/metrics (no bit payload)."""
        return {
            "m": self._m,
            "k": self._k,
            "n": int(self._sized_for_n or 0),
            "packed_bytes": len(self._bits),
        }

    def pack_json(self) -> dict:
        """JSON-serializable payload for gossip (``v`` for future format changes)."""
        return {
            "v": 1,
            "m": self._m,
            "k": self._k,
            "n": int(self._sized_for_n or 0),
            "bits_b64": base64.b64encode(bytes(self._bits)).decode("ascii"),
        }

    @classmethod
    def unpack_json(cls, d: dict) -> BloomFilter | None:
        """Rebuild from :meth:`pack_json` output; returns ``None`` if invalid or too large."""
        try:
            if d.get("v") != 1:
                return None
            m = int(d["m"])
            k = int(d["k"])
            n = int(d.get("n") or 0)
            raw = base64.b64decode(d["bits_b64"], validate=True)
        except (KeyError, TypeError, ValueError, binascii.Error):
            return None

        if m < 8 or k < 1 or k > 32:
            return None
        need = (m + 7) // 8
        if len(raw) != need or len(raw) > MAX_BLOOM_PACKED_BYTES:
            return None
        inst = cls(m, k)
        inst._bits[:] = raw
        inst._sized_for_n = n if n > 0 else None
        return inst
