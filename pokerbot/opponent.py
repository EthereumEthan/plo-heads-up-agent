"""Per-opponent tendency model, built from finished hands.

Rates are shrunk toward fixed priors: (hits + prior * w) / (trials + w).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pokerbot.types import PREFLOP, RIVER, HandResult

N_STREETS = RIVER + 1


@dataclass(frozen=True)
class OpponentView:
    n_hands: int
    vpip: float  # P(opp voluntarily puts chips in preflop)
    pfr: float  # P(opp raises preflop)
    fold_to_bet: float  # P(opp folds | we bet or raised), on the street asked for
    aggression: float  # opp raises / opp actions, on the street asked for
    fold_to_half: float = 0.4  # P(opp folds | we bet under 3/4 pot), on the street asked for
    fold_to_pot: float = 0.4  # P(opp folds | we bet 3/4 pot or more), on the street asked for


@dataclass
class _Tally:
    hits: int = 0
    trials: int = 0

    def add(self, hit: bool) -> None:
        self.trials += 1
        if hit:
            self.hits += 1

    def rate(self, prior: float, weight: float) -> float:
        return (self.hits + prior * weight) / (self.trials + weight)


def raise_fractions(history, small_blind: int = 1, big_blind: int = 2) -> dict[int, float]:
    """Map history index -> raise size as a fraction of the pot after the call (0.5 = RAISE_HALF, 1.0 = RAISE_POT)."""
    bets = [small_blind, big_blind]  # seat 0 is the button / small blind
    pot = small_blind + big_blind
    street = 0
    out: dict[int, float] = {}
    for i, r in enumerate(history):
        if r.street != street:
            street = r.street
            bets = [0, 0]
        me, other = r.seat, 1 - r.seat
        if r.action.kind == "call":
            pot += bets[other] - bets[me]
            bets[me] = bets[other]
        elif r.action.kind == "raise":
            to_call = max(bets[other] - bets[me], 0)
            after_call = pot + to_call
            out[i] = (r.action.amount - bets[other]) / after_call if after_call > 0 else 1.0
            pot += r.action.amount - bets[me]
            bets[me] = r.action.amount
    return out


class OpponentModel:
    PRIORS: ClassVar[dict[str, float]] = {"vpip": 0.6, "pfr": 0.3, "fold_to_bet": 0.4, "aggression": 0.3}

    def __init__(self, prior_weight: float = 10.0) -> None:
        self.prior_weight = prior_weight
        self.n_hands = 0
        self._vpip = _Tally()
        self._pfr = _Tally()
        self._fold_to_bet = [_Tally() for _ in range(N_STREETS)]
        self._fold_to_half = [_Tally() for _ in range(N_STREETS)]
        self._fold_to_pot = [_Tally() for _ in range(N_STREETS)]
        self._aggression = [_Tally() for _ in range(N_STREETS)]

    def observe_hand(self, result: HandResult) -> None:
        us = result.seat
        opp = 1 - us
        self.n_hands += 1

        opp_preflop = [r.action.kind for r in result.history if r.seat == opp and r.street == PREFLOP]
        self._vpip.add(any(k in ("call", "raise") for k in opp_preflop))
        self._pfr.add("raise" in opp_preflop)

        for rec in result.history:
            if rec.seat == opp:
                self._aggression[rec.street].add(rec.action.kind == "raise")

        fracs = raise_fractions(result.history)
        for i, rec in enumerate(result.history):
            if rec.seat != us or rec.action.kind != "raise":
                continue
            next_opp = next(
                (r for r in result.history[i + 1 :] if r.seat == opp and r.street == rec.street),
                None,
            )
            if next_opp is not None:
                folded = next_opp.action.kind == "fold"
                self._fold_to_bet[rec.street].add(folded)
                small = fracs.get(i, 1.0) < 0.75
                (self._fold_to_half if small else self._fold_to_pot)[rec.street].add(folded)

    def view(self, street: int) -> OpponentView:
        if not 0 <= street < N_STREETS:
            raise ValueError(f"street must be 0..{N_STREETS - 1}, got {street}")
        w = self.prior_weight
        return OpponentView(
            n_hands=self.n_hands,
            vpip=self._vpip.rate(self.PRIORS["vpip"], w),
            pfr=self._pfr.rate(self.PRIORS["pfr"], w),
            fold_to_bet=self._fold_to_bet[street].rate(self.PRIORS["fold_to_bet"], w),
            aggression=self._aggression[street].rate(self.PRIORS["aggression"], w),
            fold_to_half=self._fold_to_half[street].rate(self.PRIORS["fold_to_bet"], w),
            fold_to_pot=self._fold_to_pot[street].rate(self.PRIORS["fold_to_bet"], w),
        )
