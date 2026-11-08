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


class OpponentModel:
    PRIORS: ClassVar[dict[str, float]] = {"vpip": 0.6, "pfr": 0.3, "fold_to_bet": 0.4, "aggression": 0.3}

    def __init__(self, prior_weight: float = 10.0) -> None:
        self.prior_weight = prior_weight
        self.n_hands = 0
        self._vpip = _Tally()
        self._pfr = _Tally()
        self._fold_to_bet = [_Tally() for _ in range(N_STREETS)]
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

        for i, rec in enumerate(result.history):
            if rec.seat != us or rec.action.kind != "raise":
                continue
            next_opp = next(
                (r for r in result.history[i + 1 :] if r.seat == opp and r.street == rec.street),
                None,
            )
            if next_opp is not None:
                self._fold_to_bet[rec.street].add(next_opp.action.kind == "fold")

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
        )
