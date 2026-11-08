"""Adapter between the official judge API (`game.py` objects) and our `pokerbot` BaseBot interface.

The official objects are used duck-typed (attribute access only), so the tests can build them from
the bundle's `game.py` without this module importing it at import time. `game` is imported lazily,
only to build the returned `game.Action`.

Known gap: the judge's HandResult carries no action history, so the history passed to
`bot.hand_over` is the last history we saw in that hand (the last `/act` observation) plus our own
reply, plus (best effort) the opponent's final action inferred from the result:
  * we won without showdown while our last action ended the hand -> opponent folded
  * showdown after our last action was a raise -> opponent called; after a check we made first on
    the street -> opponent checked
Opponent actions that happened after our last decision and are not covered by those two rules
(e.g. a check-back on a street we never saw because of an all-in run-out) are missing.
"""

from __future__ import annotations

import dataclasses

from pokerbot.cards import from_official_card
from pokerbot.types import (
    Action,
    ActionRecord,
    BaseBot,
    GameConfig,
    HandResult,
    LegalActions,
    Observation,
)

# Official Action ids (game.py:62-68). IntEnum members compare/hash equal to these ints.
_FOLD, _CHECK_CALL, _RAISE_HALF, _RAISE_POT = 0, 1, 2, 3

_MAX_TRACKED_HANDS = 8  # results arrive within a hand or two, never more than a few hands late

_OFFICIAL_ACTION = None


def _official_action_cls():
    global _OFFICIAL_ACTION
    if _OFFICIAL_ACTION is None:
        from game import Action as OfficialAction

        _OFFICIAL_ACTION = OfficialAction
    return _OFFICIAL_ACTION


def _default_reply(obs):
    """What the judge would play for us on an error: check if free, else fold."""
    cls = _official_action_cls()
    return cls.CHECK_CALL if obs.to_call == 0 else cls.FOLD


class _HandTrack:
    """What we know about one hand: our seat and the history as of our last decision."""

    __slots__ = ("history", "seat")

    def __init__(self, seat: int, history: tuple[ActionRecord, ...]) -> None:
        self.seat = seat
        self.history = history


class OfficialAdapter:
    def __init__(self, bot: BaseBot) -> None:
        self.bot = bot
        self.config = GameConfig()
        self._hands: dict[int, _HandTrack] = {}
        self._ref: tuple[int, int] | None = None  # (hand_index, seat) of the latest known hand

    # ------------------------------------------------------------------ hooks
    def new_half(self, config, memory=None) -> None:
        """Reset adapter state and tell the bot a new match starts. Safe to call repeatedly."""
        self._hands = {}
        self._ref = None
        cfg = GameConfig()
        try:
            hands = int(getattr(config, "num_hands", cfg.hands_per_half))
            bank = float(getattr(config, "time_bank_seconds", cfg.time_bank_s))
            if hands != cfg.hands_per_half or bank != cfg.time_bank_s:
                cfg = dataclasses.replace(cfg, hands_per_half=hands, time_bank_s=bank)
        except Exception:  # noqa: BLE001 - config is informational only
            cfg = GameConfig()
        self.config = cfg
        self.bot.new_match("unknown")

    def choose_action(self, obs, memory=None):
        try:
            return self._choose(obs)
        except Exception:  # noqa: BLE001 - never crash a hand; a bad reply would cost an error
            return _default_reply(obs)

    def on_hand_end(self, result, memory=None) -> None:
        try:
            self._hand_end(result)
        except Exception:  # noqa: BLE001, S110 - an exception here counts as an official error
            pass

    # ------------------------------------------------------------- decisions
    def _choose(self, obs):
        seat = 0 if obs.is_button else 1
        hand_id = obs.hand_index
        self._ref = (hand_id, seat)

        to_call = obs.to_call
        raise_to = obs.raise_to
        half_to = raise_to.get(_RAISE_HALF)
        pot_to = raise_to.get(_RAISE_POT)
        legal = LegalActions(
            can_fold=_FOLD in obs.legal_actions,
            can_check=to_call == 0,
            call_amount=to_call,
            raise_half_to=half_to,
            raise_pot_to=pot_to,
        )

        history = []
        for oa in obs.action_history:
            a = int(oa.action)
            rec_seat = seat if oa.is_me else 1 - seat
            if a == _FOLD:
                act = Action("fold")
            elif a == _CHECK_CALL:
                act = Action("check") if oa.amount == 0 else Action("call")
            else:
                act = Action("raise", oa.street_total)
            history.append(ActionRecord(rec_seat, int(oa.street), act))

        ours = Observation(
            hand_id=hand_id,
            seat=seat,
            hole=tuple(from_official_card(c) for c in obs.hole_cards),
            board=tuple(from_official_card(c) for c in obs.board),
            street=int(obs.street),
            pot=obs.pot,
            my_stack=obs.my_stack,
            opp_stack=obs.opp_stack,
            my_street_bet=obs.my_bet,
            opp_street_bet=obs.opp_bet,
            history=tuple(history),
            legal=legal,
            config=self.config,
        )

        reply = self.bot.act(ours)
        official, record = self._convert(reply, obs, legal)

        history.append(ActionRecord(seat, ours.street, record))
        self._hands[hand_id] = _HandTrack(seat, tuple(history))
        if len(self._hands) > _MAX_TRACKED_HANDS:
            for k in sorted(self._hands)[: len(self._hands) - _MAX_TRACKED_HANDS]:
                del self._hands[k]
        return official

    def _convert(self, reply, obs, legal: LegalActions):
        """Our Action -> (official Action, the Action record that this reply produces in history)."""
        cls = _official_action_cls()
        to_call = obs.to_call
        kind = getattr(reply, "kind", None)

        if kind == "raise" and legal.can_raise:
            targets = legal.raise_targets()
            amount = reply.amount
            if amount == legal.raise_half_to:
                name, target = cls.RAISE_HALF, legal.raise_half_to
            elif amount == legal.raise_pot_to:
                name, target = cls.RAISE_POT, legal.raise_pot_to
            else:
                target = min(targets, key=lambda t: abs(t - amount))
                name = cls.RAISE_HALF if target == legal.raise_half_to else cls.RAISE_POT
            if name in obs.legal_actions:
                return name, Action("raise", target)
        elif kind == "fold" and _FOLD in obs.legal_actions:
            return cls.FOLD, Action("fold")

        # check / call, a fold with a free check, a raise with none legal, or anything unrecognized
        return cls.CHECK_CALL, Action("check") if to_call == 0 else Action("call")

    # ---------------------------------------------------------------- results
    def _seat_for(self, hand_id: int, result=None) -> int:
        track = self._hands.get(hand_id)
        if track is not None:
            return track.seat
        # No decision seen yet at all. A hand we never acted in that we won without showdown for
        # exactly the small blind can only be the button folding preflop to us: we are the big blind.
        if (
            self._ref is None
            and result is not None
            and not result.went_to_showdown
            and result.winner == "me"
            and result.my_net == self.config.small_blind
        ):
            return 1
        if self._ref is not None:
            ref_hand, ref_seat = self._ref
            # The button alternates every deal (game.py:103-105), so equal parity = same seat.
            return ref_seat if (hand_id - ref_hand) % 2 == 0 else 1 - ref_seat
        # Nothing to go on (e.g. we lost, or the hand ended oddly): assume deal 0 had seat 0 on the
        # button, which is true for agent1 in half 1. Wrong only for agent2 in half 1 / agent1 in half 2.
        return hand_id % 2

    def _hand_end(self, result) -> None:
        hand_id = result.hand_index
        seat = self._seat_for(hand_id, result)
        if self._ref is None:
            self._ref = (hand_id, seat)
        track = self._hands.pop(hand_id, None)
        history = list(track.history) if track is not None else []

        showdown = bool(result.went_to_showdown)
        self._infer_final_opp_action(history, seat, result, showdown)

        net = int(result.my_net)
        deltas = (net, -net) if seat == 0 else (-net, net)
        opp_hole = result.opp_hole
        self.bot.hand_over(
            HandResult(
                hand_id=hand_id,
                seat=seat,
                deltas=deltas,
                board=tuple(from_official_card(c) for c in result.board),
                opp_hole=None if opp_hole is None else tuple(from_official_card(c) for c in opp_hole),
                history=tuple(history),
                showdown=showdown,
            )
        )

    @staticmethod
    def _infer_final_opp_action(history: list[ActionRecord], seat: int, result, showdown: bool) -> None:
        opp = 1 - seat
        if not history:
            # We never acted: the only way is an opponent fold before our first decision.
            if not showdown and result.winner == "me":
                history.append(ActionRecord(opp, 0, Action("fold")))
            return
        last = history[-1]
        if last.seat != seat:
            return
        if not showdown:
            if result.winner == "me":
                history.append(ActionRecord(opp, last.street, Action("fold")))
        elif last.action.kind == "raise":
            history.append(ActionRecord(opp, last.street, Action("call")))
        elif last.action.kind == "check":
            prev = history[-2] if len(history) > 1 else None
            if not (prev is not None and prev.seat == opp and prev.street == last.street):
                # we checked first on this street, so the opponent's closing check is missing
                history.append(ActionRecord(opp, last.street, Action("check")))
