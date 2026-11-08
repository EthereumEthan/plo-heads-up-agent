from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pokerbot.opponent import OpponentView, raise_fractions
from pokerbot.types import Action, Observation


@dataclass(frozen=True)
class Decision:
    action: Action
    equity: float
    reason: str  # "open_fold" | "value" | "trap" | "bluff" | "check" | "call" | "fold"


def choose_raise_to(obs: Observation, frac: float, rng: np.random.Generator) -> int:
    """Pick an official raise size whose expected pot fraction is `frac`.

    Mixes pot-size and half-pot raises so the expected pot share equals `frac`
    (frac 1.0 always pot, 0.5 always half pot).
    """
    if not obs.legal.can_raise:
        raise ValueError("raise not allowed")
    p_pot = min(max((frac - 0.5) / 0.5, 0.0), 1.0)
    if rng.random() < p_pot:
        return int(obs.legal.raise_pot_to)
    return int(obs.legal.raise_half_to)


def bet_fraction(obs: Observation) -> float:
    """Size of the bet we face as a fraction of the pot before it (0.5 = half pot, 1.0 = pot), clipped to [0, 1]."""
    to_call = obs.legal.call_amount
    before = obs.pot - to_call
    if to_call <= 0 or before <= 0:
        return 1.0
    return float(min(max(to_call / before, 0.0), 1.0))


def opp_raise_fractions(obs: Observation) -> list[float]:
    """For each opponent raise in this hand, its size as a fraction of the pot after the call (0.5 = official RAISE_HALF, 1.0 = RAISE_POT)."""
    fr = raise_fractions(obs.history, obs.config.small_blind, obs.config.big_blind)
    return [fr[i] for i, r in enumerate(obs.history) if i in fr and r.seat != obs.seat]


def ev_raise_to(obs: Observation, equity: float, opp: OpponentView, ex: float, bluff: bool) -> int:
    """Pick half-pot or pot by one-street expected value against the opponent's size-specific fold rates.

    EV(b) = P(fold) * pot + (1 - P(fold)) * (eq_call * (pot + 2b) - b), with eq_call the equity
    once the opponent continues (a bluff's equity shrinks, a value hand's a little). Fold rates are
    shrunk toward the street's overall fold-to-bet by (1 - ex).
    """
    legal = obs.legal
    pot_after_call = obs.pot + legal.call_amount
    best_to, best_ev = int(legal.raise_half_to), -1e9
    for to in legal.raise_targets():
        b = to - obs.opp_street_bet  # chips we add beyond the call
        if b <= 0:
            continue
        frac = b / pot_after_call if pot_after_call > 0 else 1.0
        size_rate = opp.fold_to_half if frac < 0.75 else opp.fold_to_pot
        p_fold = (1 - ex) * opp.fold_to_bet + ex * size_rate
        eq_call = equity * (0.75 if bluff else 0.92)
        ev = p_fold * pot_after_call + (1 - p_fold) * (eq_call * (pot_after_call + 2 * b) - b)
        if ev > best_ev:
            best_ev, best_to = ev, int(to)
    return best_to


def decide(
    obs: Observation,
    equity: float,
    params: dict[str, float],
    opp: OpponentView,
    rng: np.random.Generator,
) -> Decision:
    s = obs.street
    legal = obs.legal
    to_call = legal.call_amount
    pot_odds = to_call / (obs.pot + to_call) if to_call > 0 else 0.0
    my_raises = sum(
        1
        for r in obs.history
        if r.seat == obs.seat and r.street == s and r.action.kind == "raise"
    )
    ex = params["exploit_strength"] * min(1.0, opp.n_hands / 200)
    in_position = s >= 1 and obs.seat == 0  # button acts last postflop
    if s == 0:
        bluff_freq = 0.0
    else:
        bluff_freq = float(
            np.clip(
                params[f"bluff_freq_{s}"]
                * (1 + ex * (opp.fold_to_bet - 0.4) / 0.4)
                * (params["ip_bluff_mult"] if in_position else 1.0),
                0.0,
                0.8,
            )
        )
    call_margin = params[f"call_margin_{s}"] - ex * 0.1 * (opp.aggression - 0.3)
    if to_call > 0:
        call_margin -= params["size_disc"] * (1.0 - bet_fraction(obs))
    raise_thr = params[f"raise_thr_{s}"] + (params["ip_raise_adj"] if in_position else 0.0)

    # Open fold
    if s == 0 and obs.seat == 0 and not obs.history and equity < params["open_thr"]:
        return Decision(Action("fold"), equity, "open_fold")

    # Value raise (or trap)
    if (
        legal.can_raise
        and equity >= raise_thr
        and (my_raises < round(params["max_raises"]) or equity >= params["reraise_thr"])
    ):
        if s < 3 and rng.random() < params["trap_freq"]:
            if to_call == 0:
                return Decision(Action("check"), equity, "trap")
            return Decision(Action("call"), equity, "trap")
        if equity >= params["pot_size_thr"]:
            amount = int(legal.raise_pot_to)
        elif params["ev_sizing"] >= 0.5:
            amount = ev_raise_to(obs, equity, opp, ex, bluff=False)
        else:
            amount = choose_raise_to(obs, params["bet_frac_value"], rng)
        return Decision(Action("raise", amount), equity, "value")

    # Free check or bluff
    if to_call == 0:
        if s >= 1 and legal.can_raise and rng.random() < bluff_freq:
            if params["ev_sizing"] >= 0.5:
                amount = ev_raise_to(obs, equity, opp, ex, bluff=True)
            else:
                amount = choose_raise_to(obs, params["bet_frac_bluff"], rng)
            return Decision(Action("raise", amount), equity, "bluff")
        return Decision(Action("check"), equity, "check")

    # Call or fold
    if equity >= pot_odds + call_margin:
        return Decision(Action("call"), equity, "call")
    return Decision(Action("fold"), equity, "fold")
