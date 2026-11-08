"""Monte Carlo equity of a 5-card short-deck PLO hand against one opponent."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

import numba
import numpy as np

from pokerbot.cards import cards_from_str
from pokerbot.evaluator import N_CARDS, best_omaha_nb, get_rank_table
from pokerbot.types import Card

CHUNK = 32


@numba.njit(cache=True)
def _mc_equity(
    hole: np.ndarray, board: np.ndarray, opp_pool: np.ndarray, n: int, seed: int, table: np.ndarray
) -> tuple[float, int]:
    """Run n samples. Returns (sum of scores, samples run); win = 1, tie = 0.5."""
    np.random.seed(seed)
    dead = np.zeros(N_CARDS, dtype=np.bool_)
    for c in hole:
        dead[c] = True
    for c in board:
        dead[c] = True
    base = np.empty(N_CARDS, dtype=np.int64)
    n_base = 0
    for c in range(N_CARDS):
        if not dead[c]:
            base[n_base] = c
            n_base += 1

    nb = board.shape[0]
    need = 5 - nb
    use_pool = opp_pool.shape[0] > 0
    deck = np.empty(N_CARDS, dtype=np.int64)
    full_board = np.empty(5, dtype=np.int64)
    for i in range(nb):
        full_board[i] = board[i]
    opp = np.empty(5, dtype=np.int64)

    total = 0.0
    for _ in range(n):
        if use_pool:
            row = np.random.randint(0, opp_pool.shape[0])
            for i in range(5):
                opp[i] = opp_pool[row, i]
            m = 0
            for j in range(n_base):
                c = base[j]
                if c != opp[0] and c != opp[1] and c != opp[2] and c != opp[3] and c != opp[4]:
                    deck[m] = c
                    m += 1
            k = need
        else:
            for j in range(n_base):
                deck[j] = base[j]
            m = n_base
            k = need + 5
        # Partial Fisher-Yates: first k entries become a uniform sample without replacement.
        for i in range(k):
            j = np.random.randint(i, m)
            t = deck[i]
            deck[i] = deck[j]
            deck[j] = t
        p = 0
        if not use_pool:
            for i in range(5):
                opp[i] = deck[i]
            p = 5
        for i in range(need):
            full_board[nb + i] = deck[p + i]
        mine = best_omaha_nb(hole, full_board, table)
        theirs = best_omaha_nb(opp, full_board, table)
        if mine > theirs:
            total += 1.0
        elif mine == theirs:
            total += 0.5
    return total, n


@numba.njit(cache=True)
def _sample_scored_hands(
    unseen: np.ndarray, board: np.ndarray, n: int, seed: int, table: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Draw n random 5-card hands from unseen; score each on board. Returns (int64[n,5], int64[n])."""
    np.random.seed(seed)
    m = unseen.shape[0]
    deck = unseen.copy()
    hands = np.empty((n, 5), dtype=np.int64)
    scores = np.empty(n, dtype=np.int64)
    row = np.empty(5, dtype=np.int64)
    for r in range(n):
        # Partial Fisher-Yates: first 5 entries become a uniform sample without replacement.
        for i in range(5):
            j = np.random.randint(i, m)
            t = deck[i]
            deck[i] = deck[j]
            deck[j] = t
            row[i] = deck[i]
            hands[r, i] = deck[i]
        scores[r] = best_omaha_nb(row, board, table)
    return hands, scores


def sample_top_hands(
    hole: Sequence[Card],
    board: Sequence[Card],
    frac: float,
    rng: np.random.Generator,
    n: int = 400,
    trips_beat_straight: bool = True,
) -> np.ndarray:
    """Top ceil(frac * n) of n random opponent hands by score on the current board (int64[k, 5]).

    Rows are ordered by score, highest last.
    """
    board_a = np.ascontiguousarray(np.asarray(board, dtype=np.int64).reshape(-1))
    if board_a.shape[0] not in (3, 4, 5):
        raise ValueError(f"board must have 3, 4 or 5 cards, got {board_a.shape[0]}")
    if not 0 < frac <= 1:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    seen = {int(c) for c in hole} | {int(c) for c in board_a}
    unseen = np.array([c for c in range(N_CARDS) if c not in seen], dtype=np.int64)
    table = get_rank_table(trips_beat_straight)
    seed = int(rng.integers(2**31))
    hands, scores = _sample_scored_hands(unseen, board_a, n, seed, table)
    keep = max(1, math.ceil(frac * n))
    return hands[np.argsort(scores, kind="stable")[n - keep :]]


def _estimate(
    hole: Sequence[Card],
    board: Sequence[Card],
    rng: np.random.Generator,
    n_samples: int = 400,
    opp_hands: np.ndarray | None = None,
    time_limit_s: float | None = None,
    trips_beat_straight: bool = False,
) -> tuple[float, int]:
    """Same as estimate_equity, but also returns the number of samples run."""
    start = time.perf_counter()
    hole_a = np.ascontiguousarray(np.asarray(hole, dtype=np.int64))
    board_a = np.ascontiguousarray(np.asarray(board, dtype=np.int64).reshape(-1))
    if hole_a.shape != (5,):
        raise ValueError(f"hole must have 5 cards, got {hole_a.shape[0]}")
    if board_a.shape[0] not in (0, 3, 4, 5):
        raise ValueError(f"board must have 0, 3, 4 or 5 cards, got {board_a.shape[0]}")

    pool = np.empty((0, 5), dtype=np.int64)
    if opp_hands is not None:
        opp_a = np.asarray(opp_hands, dtype=np.int64).reshape(-1, 5)
        used = np.concatenate((hole_a, board_a))
        keep = ~np.isin(opp_a, used).any(axis=1)
        if keep.any():
            pool = np.ascontiguousarray(opp_a[keep])

    table = get_rank_table(trips_beat_straight)
    total_score = 0.0
    total_n = 0
    while total_n < n_samples:
        if total_n > 0 and time_limit_s is not None and time.perf_counter() - start >= time_limit_s:
            break
        chunk = min(CHUNK, n_samples - total_n)
        seed = int(rng.integers(2**31))
        s, k = _mc_equity(hole_a, board_a, pool, chunk, seed, table)
        total_score += s
        total_n += k
    if total_n == 0:
        raise ValueError("n_samples must be positive")
    return total_score / total_n, total_n


def estimate_equity(
    hole: Sequence[Card],
    board: Sequence[Card],
    rng: np.random.Generator,
    n_samples: int = 400,
    opp_hands: np.ndarray | None = None,
    time_limit_s: float | None = None,
    trips_beat_straight: bool = False,
) -> float:
    """P(win) + 0.5 * P(tie) vs one opponent.

    Opponent hand: uniform random from remaining cards, or a uniformly chosen row of opp_hands
    (int array shape (k, 5)); rows overlapping hole/board are dropped (if none remain, fall back
    to uniform random). Remaining board is completed at random. Stops early once time_limit_s
    elapses (at least 32 samples always run).
    """
    eq, _ = _estimate(hole, board, rng, n_samples, opp_hands, time_limit_s, trips_beat_straight)
    return eq


def warmup() -> None:
    """Force JIT compilation (or cache load) of the Monte Carlo kernel."""
    hole = cards_from_str("As Ks Qh Jh Td")
    board = cards_from_str("6c 7c 8d")
    estimate_equity(hole, board, np.random.default_rng(0), n_samples=CHUNK)
    sample_top_hands(hole, board, 0.5, np.random.default_rng(0), n=CHUNK)
