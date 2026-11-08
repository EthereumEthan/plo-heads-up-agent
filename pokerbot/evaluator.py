"""Short-deck (36-card) 5-card hand evaluator with the Omaha 2-hole + 3-board rule.

Hand ranks are ints: ``(category << 20) | kickers``, where ``kickers`` packs five 4-bit
rank indices (0='6' .. 8='A'), most significant first. Higher int = better hand.
"""

from __future__ import annotations

import itertools
import math
import os
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numba
import numpy as np

from pokerbot.types import Card

N_CARDS = 36
N_COMBOS = math.comb(N_CARDS, 5)  # 376992
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Category numbers for the default ranking (straight beats trips).
HIGH, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FULL_HOUSE, FLUSH, QUADS, STRAIGHT_FLUSH = range(9)

_NAMES_DEFAULT = (
    "high",
    "pair",
    "two_pair",
    "trips",
    "straight",
    "full_house",
    "flush",
    "quads",
    "straight_flush",
)
_NAMES_TBS = (
    "high",
    "pair",
    "two_pair",
    "straight",
    "trips",
    "full_house",
    "flush",
    "quads",
    "straight_flush",
)

_WHEEL = frozenset({8, 0, 1, 2, 3})  # A-6-7-8-9

# BINOM[n, k] = C(n, k) for n in 0..36, k in 0..5.
BINOM = np.array([[math.comb(n, k) for k in range(6)] for n in range(N_CARDS + 1)], dtype=np.int64)


def _pack(category: int, kick: Sequence[int]) -> int:
    v = 0
    for i in range(5):
        v = (v << 4) | (kick[i] if i < len(kick) else 0)
    return (category << 20) | v


def rank5(cards: Sequence[Card], trips_beat_straight: bool = False) -> int:
    """Rank exactly five cards. Higher is better."""
    if len(cards) != 5:
        raise ValueError(f"rank5 needs 5 cards, got {len(cards)}")
    ranks = [c // 4 for c in cards]
    is_flush = len({c % 4 for c in cards}) == 1
    counts = Counter(ranks)
    trips_cat, straight_cat = (STRAIGHT, TRIPS) if trips_beat_straight else (TRIPS, STRAIGHT)

    if len(counts) == 5:
        rs = sorted(ranks, reverse=True)
        top = None
        if rs[0] - rs[4] == 4:
            top = rs[0]
        elif set(rs) == _WHEEL:
            top = 3  # the 9
        if top is not None:
            return _pack(STRAIGHT_FLUSH if is_flush else straight_cat, [top])
        return _pack(FLUSH if is_flush else HIGH, rs)

    # Group by (count desc, rank desc).
    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    shape = tuple(n for _, n in groups)
    order = [r for r, _ in groups]
    if shape == (4, 1):
        cat = QUADS
    elif shape == (3, 2):
        cat = FULL_HOUSE
    elif shape == (3, 1, 1):
        cat = trips_cat
    elif shape == (2, 2, 1):
        cat = TWO_PAIR
    else:  # (2, 1, 1, 1)
        cat = PAIR
    return _pack(cat, order)


def hand_category_name(rank: int, trips_beat_straight: bool = False) -> str:
    names = _NAMES_TBS if trips_beat_straight else _NAMES_DEFAULT
    return names[rank >> 20]


def combo_index(sorted_cards: Sequence[Card]) -> int:
    """Colex index of an ascending 5-card combination, in 0..376991."""
    return sum(math.comb(c, i + 1) for i, c in enumerate(sorted_cards))


def _build_table(trips_beat_straight: bool) -> np.ndarray:
    table = np.empty(N_COMBOS, dtype=np.int32)
    for combo in itertools.combinations(range(N_CARDS), 5):
        table[combo_index(combo)] = rank5(combo, trips_beat_straight)
    return table


def _cache_path(trips_beat_straight: bool) -> Path:
    return DATA_DIR / f"rank5_tbs{int(trips_beat_straight)}.npy"


def _load_cached(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        arr = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    if arr.shape != (N_COMBOS,) or arr.dtype != np.int32:
        return None
    return arr


def _save_atomic(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, arr)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


_TABLES: dict[bool, np.ndarray] = {}


def get_rank_table(trips_beat_straight: bool = False) -> np.ndarray:
    """int32[376992] of rank5 values indexed by combo_index. Cached in memory and on disk."""
    key = bool(trips_beat_straight)
    table = _TABLES.get(key)
    if table is not None:
        return table
    path = _cache_path(key)
    table = _load_cached(path)
    if table is None:
        table = _build_table(key)
        try:
            _save_atomic(path, table)
        except OSError:
            pass  # read-only or missing data dir (e.g. a packaged submission): keep it in memory only
    _TABLES[key] = table
    return table


def best_omaha(hole: Sequence[Card], board: Sequence[Card], trips_beat_straight: bool = False) -> int:
    """Best rank using exactly 2 hole cards and exactly 3 board cards."""
    if len(hole) != 5:
        raise ValueError(f"hole must have 5 cards, got {len(hole)}")
    if not 3 <= len(board) <= 5:
        raise ValueError(f"board must have 3-5 cards, got {len(board)}")
    best = -1
    for h in itertools.combinations(hole, 2):
        for b in itertools.combinations(board, 3):
            best = max(best, rank5(h + b, trips_beat_straight))
    return best


@numba.njit(cache=True)
def best_omaha_nb(hole: np.ndarray, board: np.ndarray, table: np.ndarray) -> int:
    """Numba version of best_omaha using a table from get_rank_table()."""
    nh = hole.shape[0]
    nb = board.shape[0]
    buf = np.empty(5, dtype=np.int64)
    best = -1
    for h0 in range(nh - 1):
        for h1 in range(h0 + 1, nh):
            for b0 in range(nb - 2):
                for b1 in range(b0 + 1, nb - 1):
                    for b2 in range(b1 + 1, nb):
                        buf[0] = hole[h0]
                        buf[1] = hole[h1]
                        buf[2] = board[b0]
                        buf[3] = board[b1]
                        buf[4] = board[b2]
                        for i in range(1, 5):
                            v = buf[i]
                            j = i - 1
                            while j >= 0 and buf[j] > v:
                                buf[j + 1] = buf[j]
                                j -= 1
                            buf[j + 1] = v
                        idx = 0
                        for i in range(5):
                            idx += BINOM[buf[i], i + 1]
                        best = max(best, np.int64(table[idx]))
    return int(best)
