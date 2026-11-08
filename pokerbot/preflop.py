"""Preflop equity table (5-card hand vs a random hand) and top-X% hand sampling."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from pokerbot.equity import estimate_equity
from pokerbot.evaluator import DATA_DIR, N_CARDS, N_COMBOS, combo_index
from pokerbot.types import Card

DEFAULT_TABLE_PATH = DATA_DIR / "preflop_equity.npy"
TABLE_TRIPS_BEAT_STRAIGHT = True  # ranking data/preflop_equity.npy was built with; must match scripts/build_preflop_table.py

_STATE: dict[str, np.ndarray | None] = {}


def _build_all_combos() -> np.ndarray:
    combos = np.empty((N_COMBOS, 5), dtype=np.int64)
    for combo in itertools.combinations(range(N_CARDS), 5):
        combos[combo_index(combo)] = combo
    return combos


def _all_combos() -> np.ndarray:
    arr = _STATE.get("combos")
    if arr is None:
        arr = _build_all_combos()
        _STATE["combos"] = arr
    return arr


def __getattr__(name: str) -> np.ndarray:
    if name == "ALL_COMBOS":
        return _all_combos()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load_table(path: Path | None = None) -> np.ndarray | None:
    """float32[376992] of preflop equities indexed by combo_index, or None if missing."""
    path = DEFAULT_TABLE_PATH if path is None else Path(path)
    if not path.exists():
        return None
    try:
        arr = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    if arr.shape != (N_COMBOS,):
        return None
    return arr


def _cached_table() -> np.ndarray | None:
    if "table" not in _STATE:
        _STATE["table"] = load_table()
    return _STATE["table"]


def preflop_equity(hole: Sequence[Card], rng: np.random.Generator | None = None) -> float:
    """Equity of a 5-card hand vs a random hand: table lookup, else Monte Carlo."""
    table = _cached_table()
    if table is None:
        table = load_table()
        _STATE["table"] = table
    if table is not None:
        return float(table[combo_index(sorted(hole))])
    if rng is None:
        rng = np.random.default_rng()
    return estimate_equity(
        hole, (), rng, n_samples=300, trips_beat_straight=TABLE_TRIPS_BEAT_STRAIGHT
    )


def _order() -> np.ndarray | None:
    order = _STATE.get("order")
    if order is None:
        table = _cached_table()
        if table is None:
            return None
        order = np.argsort(table, kind="stable")
        _STATE["order"] = order
    return order


def warmup() -> None:
    """Load the table and build the sampling arrays now, so the first act() doesn't pay ~0.2 s."""
    if _order() is not None:
        _all_combos()


def top_fraction_hands(
    frac: float, rng: np.random.Generator, k: int = 400, exclude: Sequence[Card] = ()
) -> np.ndarray:
    """k rows sampled uniformly from the top ``frac`` of hands by preflop equity.

    Rows sharing a card with ``exclude`` are dropped. Returns int64[k, 5].
    """
    if not 0 < frac <= 1:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    order = _order()
    if order is None:
        raise FileNotFoundError(f"preflop table not found at {DEFAULT_TABLE_PATH}")
    n = max(1, math.ceil(frac * N_COMBOS))
    top = order[N_COMBOS - n :]
    combos = _all_combos()
    if len(exclude) == 0:
        return combos[top[rng.integers(0, n, size=k)]]
    dead = np.zeros(N_CARDS, dtype=np.bool_)
    dead[np.asarray(exclude, dtype=np.int64)] = True
    # Rejection sampling: uniform over top-n rows that avoid `exclude`, without gathering all of them.
    parts: list[np.ndarray] = []
    have = 0
    for _ in range(20):
        rows = combos[top[rng.integers(0, n, size=2 * k)]]
        rows = rows[~dead[rows].any(axis=1)]
        parts.append(rows)
        have += rows.shape[0]
        if have >= k:
            return np.concatenate(parts)[:k]
    if have == 0:
        return combos[top[rng.integers(0, n, size=k)]]
    surv = np.concatenate(parts)
    return surv[rng.integers(0, have, size=k)]
