"""Near-duplicate detection across tiers.

The same guidance reaches this corpus by several routes: the ERAS
gynecologic/oncology recommendations exist as Tier 1 society PDFs, as the Tier 4
Elsevier copy of the same article, and again via OpenAlex and PubMed. That is a
problem for a system whose central claim is that provenance tier drives ranking
— if identical text sits at three tiers, tier weighting partly measures which
copy the retriever happened to find rather than authority.

Duplicates are detected with MinHash + LSH over character-shingle sets, which
finds reformatted copies (different line breaks, different extractor) that exact
hashing misses, without the O(n^2) cost of comparing every pair.

Nothing is deleted. Each cluster elects a canonical chunk and the rest are
marked, so the duplication rate stays reportable and the retrieval ablation can
switch it on and off.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

import numpy as _np

# Word-level shingles rather than character-level: a 1,700-character chunk has
# ~1,700 char shingles but only ~300 word shingles, and word n-grams survive the
# formatting differences between two extractors of the same passage just as well.
SHINGLE = 5           # words per shingle
NUM_PERM = 96         # MinHash permutations
BANDS = 24            # LSH bands; rows = NUM_PERM // BANDS
JACCARD_THRESHOLD = 0.80
MIN_CHARS = 200       # shorter texts collide by chance; compare them exactly

_MASK = (1 << 61) - 1
_NORMALISE = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Strip formatting noise so two extractions of one passage look identical."""
    text = _NORMALISE.sub(" ", text.lower())
    return _SPACES.sub(" ", text).strip()


def shingles(text: str, size: int = SHINGLE) -> set[int]:
    """Hashed word n-grams. Stable across runs — Python's hash() is salted."""
    words = normalise(text).split()
    if not words:
        return set()
    if len(words) <= size:
        return {_stable_hash(" ".join(words))}
    return {_stable_hash(" ".join(words[i:i + size]))
            for i in range(len(words) - size + 1)}


def _stable_hash(text: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(text.encode(), digest_size=8).digest(), "big") & _MASK


def _hash_params(num_perm: int) -> list[tuple[int, int]]:
    """Deterministic (a, b) pairs for the universal hash family."""
    rng = []
    seed = 0x9E3779B97F4A7C15
    for i in range(num_perm):
        digest = hashlib.blake2b(f"{seed}:{i}".encode(), digest_size=16).digest()
        a = int.from_bytes(digest[:8], "big") | 1        # odd multiplier
        b = int.from_bytes(digest[8:], "big")
        rng.append((a & _MASK, b & _MASK))
    return rng


_PARAMS = _hash_params(NUM_PERM)


_A = _np.array([a for a, _ in _PARAMS], dtype=_np.uint64)
_B = _np.array([b for _, b in _PARAMS], dtype=_np.uint64)


def minhash(text: str) -> tuple[int, ...]:
    """Vectorised MinHash.

    The scalar version is billions of modular multiplications over this corpus;
    computing all permutations at once turns the whole pass into seconds.
    """
    shingle_set = shingles(text)
    if not shingle_set:
        return tuple([0] * NUM_PERM)
    values = _np.fromiter(shingle_set, dtype=_np.uint64, count=len(shingle_set))
    # (num_perm, n_shingles) -> min along the shingle axis
    hashed = (_A[:, None] * values[None, :] + _B[:, None]) & _np.uint64(_MASK)
    return tuple(int(x) for x in hashed.min(axis=1))


def jaccard(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
    """Estimated Jaccard similarity from two MinHash signatures."""
    matches = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return matches / len(sig_a)


def candidate_pairs(signatures: dict[str, tuple[int, ...]]) -> set[tuple[str, str]]:
    """LSH banding: chunks sharing a whole band are worth comparing."""
    rows = NUM_PERM // BANDS
    pairs: set[tuple[str, str]] = set()

    for band in range(BANDS):
        buckets: dict[tuple, list[str]] = defaultdict(list)
        start = band * rows
        for chunk_id, signature in signatures.items():
            buckets[signature[start:start + rows]].append(chunk_id)
        for members in buckets.values():
            if len(members) < 2 or len(members) > 200:
                continue   # huge buckets are boilerplate, not real duplicates
            members.sort()
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    pairs.add((a, b))
    return pairs


def cluster(rows: list[dict], threshold: float = JACCARD_THRESHOLD) -> dict[str, str]:
    """Return {chunk_id: cluster_root} via union-find over similar pairs."""
    by_id = {r["chunk_id"]: r for r in rows}
    signatures = {r["chunk_id"]: minhash(r["text"]) for r in rows
                  if len(r["text"]) >= MIN_CHARS}

    parent: dict[str, str] = {cid: cid for cid in by_id}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    # Exact duplicates of any length, including short ones.
    exact: dict[str, str] = {}
    for chunk_id, row in by_id.items():
        key = hashlib.sha1(normalise(row["text"]).encode()).hexdigest()
        if key in exact:
            union(exact[key], chunk_id)
        else:
            exact[key] = chunk_id

    for a, b in candidate_pairs(signatures):
        if find(a) == find(b):
            continue
        if jaccard(signatures[a], signatures[b]) >= threshold:
            union(a, b)

    return {cid: find(cid) for cid in by_id}


def _rank(row: dict) -> tuple:
    """Canonical preference: highest authority, then newest, then fullest text."""
    try:
        year = int(str(row.get("publication_year", "") or 0)[:4])
    except ValueError:
        year = 0
    return (
        row.get("tier", 4),        # lower tier number wins
        -year,                     # newer wins
        -len(row.get("text", "")), # more complete wins
        row["chunk_id"],           # deterministic tie-break
    )


def elect(rows: list[dict], assignment: dict[str, str]) -> dict[str, dict]:
    """Pick one canonical chunk per cluster.

    Returns {chunk_id: {"is_canonical": bool, "duplicate_of": str|None,
                        "cluster_size": int, "cluster_tiers": [int, ...]}}
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[assignment[row["chunk_id"]]].append(row)

    verdict: dict[str, dict] = {}
    for members in groups.values():
        members.sort(key=_rank)
        winner = members[0]
        tiers = sorted({m.get("tier", 4) for m in members})
        for member in members:
            is_canonical = member["chunk_id"] == winner["chunk_id"]
            verdict[member["chunk_id"]] = {
                "is_canonical": is_canonical,
                "duplicate_of": None if is_canonical else winner["chunk_id"],
                "cluster_size": len(members),
                "cluster_tiers": tiers,
            }
    return verdict
