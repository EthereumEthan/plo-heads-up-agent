"""Pot-limit heads-up engine for short-deck 5-card PLO.

Seat 0 is the button (small blind, acts first preflop, last postflop). Seat 1 is the big blind.
Deck layout: seat0 hole = deck[0:5], seat1 hole = deck[5:10], flop = deck[10:13],
turn = deck[13], river = deck[14].
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from pokerbot.evaluator import best_omaha_nb, get_rank_table
from pokerbot.types import (
    PREFLOP,
    RIVER,
    Action,
    ActionRecord,
    BaseBot,
    Card,
    GameConfig,
    HandResult,
    IllegalActionError,
    LegalActions,
    Observation,
)

_VALID_KINDS = frozenset({"fold", "check", "call", "raise"})
_BOARD_SIZES = (0, 3, 4, 5)  # board cards visible on PREFLOP..RIVER


@dataclass(frozen=True)
class HandOutcome:
    deltas: tuple[int, int]
    illegal: tuple[int, int]  # illegal action count per seat (non-strict mode)
    max_act_time_s: tuple[float, float]
    act_time_s: tuple[float, float]  # total seconds per seat in act() + hand_over() this hand


@dataclass
class _HandState:
    stacks: list[int]
    committed: list[int]  # whole hand, blinds included
    street_bet: list[int]  # current street, blinds included preflop
    street: int
    history: list[ActionRecord]
    last_raise_increment: int
    acted_since_raise: list[bool]
    folded: int | None
    big_blind: int = 2
    raises_this_street: int = 0
    config: GameConfig = field(default_factory=GameConfig)


def legal_actions(state: _HandState, seat: int) -> LegalActions:
    opp = 1 - seat
    my_bet = state.street_bet[seat]
    opp_bet = state.street_bet[opp]
    my_stack = state.stacks[seat]
    opp_stack = state.stacks[opp]
    to_call = max(opp_bet - my_bet, 0)
    pot = state.committed[0] + state.committed[1]

    can_check = to_call == 0
    call_amount = 0 if can_check else min(to_call, my_stack)

    raise_half_to: int | None
    raise_pot_to: int | None
    cfg = state.config
    can_raise = state.raises_this_street < cfg.raise_cap and my_stack > to_call and opp_stack > 0
    if can_raise:
        all_in_to = my_bet + my_stack
        pot_to = opp_bet + pot + to_call
        min_to = opp_bet + max(state.last_raise_increment, state.big_blind)

        def clamp(t: int) -> int:
            return min(max(t, min_to), pot_to, all_in_to)

        raise_half_to = clamp(opp_bet + (pot + to_call) // 2)
        raise_pot_to = clamp(pot_to)
    else:
        raise_half_to = raise_pot_to = None
    return LegalActions(
        can_fold=to_call > 0,
        can_check=can_check,
        call_amount=call_amount,
        raise_half_to=raise_half_to,
        raise_pot_to=raise_pot_to,
    )


def _is_int(x: object) -> bool:
    return not isinstance(x, (bool, np.bool_)) and isinstance(x, (int, np.integer))


def _resolve(action: object, legal: LegalActions) -> Action | None:
    """Return the normalized legal action, or None if the action is illegal."""
    if not isinstance(action, Action) or action.kind not in _VALID_KINDS:
        return None
    kind = action.kind
    if kind == "fold":
        return Action("fold") if legal.can_fold else None
    if kind == "call" and legal.can_check:
        return Action("check")
    if kind == "check":
        return Action("check") if legal.can_check else None
    if kind == "call":
        return Action("call")
    # raise
    amount = action.amount
    if not _is_int(amount) or int(amount) not in legal.raise_targets():
        return None
    return Action("raise", int(amount))


def _move_chips(state: _HandState, seat: int, chips: int) -> None:
    state.stacks[seat] -= chips
    state.committed[seat] += chips
    state.street_bet[seat] += chips


def play_hand(
    bots: tuple[BaseBot, BaseBot], deck: Sequence[Card], hand_id: int, config: GameConfig
) -> HandOutcome:
    cards = [int(c) for c in deck[:15]]
    holes = (tuple(cards[0:5]), tuple(cards[5:10]))
    full_board = tuple(cards[10:15])

    sb, bb = config.small_blind, config.big_blind
    stack = config.starting_stack
    state = _HandState(
        stacks=[stack - sb, stack - bb],
        committed=[sb, bb],
        street_bet=[sb, bb],
        street=PREFLOP,
        history=[],
        last_raise_increment=bb,
        acted_since_raise=[False, False],
        folded=None,
        big_blind=bb,
        config=config,
    )
    illegal = [0, 0]
    max_time = [0.0, 0.0]
    act_time = [0.0, 0.0]

    actor = 0
    hand_done = False
    while not hand_done:
        legal = legal_actions(state, actor)
        opp = 1 - actor
        obs = Observation(
            hand_id=hand_id,
            seat=actor,
            hole=holes[actor],
            board=full_board[: _BOARD_SIZES[state.street]],
            street=state.street,
            pot=state.committed[0] + state.committed[1],
            my_stack=state.stacks[actor],
            opp_stack=state.stacks[opp],
            my_street_bet=state.street_bet[actor],
            opp_street_bet=state.street_bet[opp],
            history=tuple(state.history),
            legal=legal,
            config=config,
        )

        t0 = time.perf_counter()
        try:
            raw = bots[actor].act(obs)
        except Exception:
            if config.strict:
                raise
            raw = None
        finally:
            elapsed = time.perf_counter() - t0
            max_time[actor] = max(max_time[actor], elapsed)
            act_time[actor] += elapsed

        action = _resolve(raw, legal)
        if action is None:
            if config.strict:
                raise IllegalActionError(f"seat {actor} illegal action {raw!r} with legal {legal}")
            illegal[actor] += 1
            action = Action("check") if legal.can_check else Action("fold")

        state.history.append(ActionRecord(seat=actor, street=state.street, action=action))

        if action.kind == "fold":
            state.folded = actor
            break
        if action.kind == "call":
            _move_chips(state, actor, legal.call_amount)
            state.acted_since_raise[actor] = True
        elif action.kind == "raise":
            prev_high = max(state.street_bet)
            _move_chips(state, actor, action.amount - state.street_bet[actor])
            state.last_raise_increment = action.amount - prev_high
            state.raises_this_street += 1
            state.acted_since_raise = [False, False]
            state.acted_since_raise[actor] = True
        else:  # check
            state.acted_since_raise[actor] = True

        street_over = all(state.acted_since_raise) and state.street_bet[0] == state.street_bet[1]
        if not street_over:
            actor = opp
            continue
        if state.street == RIVER or 0 in state.stacks:
            hand_done = True
            state.street = RIVER  # run out any remaining board
            continue
        state.street += 1
        state.street_bet = [0, 0]
        state.raises_this_street = 0
        state.last_raise_increment = 0
        state.acted_since_raise = [False, False]
        actor = 1  # postflop, seat 1 acts first

    showdown = state.folded is None
    if showdown:
        board = full_board
        table = get_rank_table(config.trips_beat_straight)
        board_arr = np.array(board, dtype=np.int64)
        r0 = best_omaha_nb(np.array(holes[0], dtype=np.int64), board_arr, table)
        r1 = best_omaha_nb(np.array(holes[1], dtype=np.int64), board_arr, table)
        if r0 > r1:
            won = state.committed[1]
            deltas = (won, -won)
        elif r1 > r0:
            won = state.committed[0]
            deltas = (-won, won)
        else:
            # Tie. Commitments are equal at showdown (equal stacks, no side pots), so no odd chip.
            deltas = (0, 0)
    else:
        board = full_board[: _BOARD_SIZES[state.street]]
        folder = state.folded
        lost = state.committed[folder]
        deltas = (-lost, lost) if folder == 0 else (lost, -lost)

    history = tuple(state.history)
    for seat in (0, 1):
        t0 = time.perf_counter()
        bots[seat].hand_over(
            HandResult(
                hand_id=hand_id,
                seat=seat,
                deltas=deltas,
                board=board,
                opp_hole=holes[1 - seat] if showdown else None,
                history=history,
                showdown=showdown,
            )
        )
        act_time[seat] += time.perf_counter() - t0

    return HandOutcome(
        deltas=deltas,
        illegal=(illegal[0], illegal[1]),
        max_act_time_s=(max_time[0], max_time[1]),
        act_time_s=(act_time[0], act_time[1]),
    )
