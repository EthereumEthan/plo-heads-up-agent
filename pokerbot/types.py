from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Card = int  # 0..35. rank_index = card // 4 (0='6',1='7',2='8',3='9',4='T',5='J',6='Q',7='K',8='A'); suit = card % 4 (0='s',1='h',2='d',3='c')
ActionKind = Literal["fold", "check", "call", "raise"]
PREFLOP, FLOP, TURN, RIVER = 0, 1, 2, 3


@dataclass(frozen=True)
class GameConfig:
    small_blind: int = 1
    big_blind: int = 2
    starting_stack: int = 80  # official game.py:32. Both stacks reset to this every hand.
    trips_beat_straight: bool = True  # official evaluator.py:93-104. Flush > full house always.
    raise_cap: int = 4  # official game.py:33. Max voluntary bets/raises per street; blinds don't count.
    time_budget_s: float = 1.0  # hard cap for one act() call (official ACTION_TIMEOUT_SECONDS, game.py:37)
    time_bank_s: float = 8.0  # thinking time per bot per half (official TIME_BANK_SECONDS, game.py:35)
    hands_per_half: int = 1000  # hands one bot plays per half (official NUM_DEALS, game.py:34)
    strict: bool = True  # True: illegal action raises IllegalActionError. False: treated as check/fold and counted.


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    amount: int = 0  # "raise" only: the TOTAL street bet the actor raises TO; must be one of legal.raise_targets(). Ignored for other kinds.


@dataclass(frozen=True)
class LegalActions:
    can_fold: bool  # True only when to_call > 0. Folding when a check is free is ILLEGAL.
    can_check: bool  # True only when to_call == 0
    call_amount: int  # chips to call, 0 when can_check
    raise_half_to: int | None  # total street bet after official RAISE_HALF; None when no raise is allowed
    raise_pot_to: int | None  # total street bet after official RAISE_POT; None when no raise is allowed

    @property
    def can_raise(self) -> bool:
        return self.raise_pot_to is not None

    def raise_targets(self) -> tuple[int, ...]:
        """Legal raise-to amounts, ascending, duplicates removed. Empty when no raise is allowed."""
        if self.raise_half_to is None or self.raise_pot_to is None:
            return ()
        return tuple(sorted({self.raise_half_to, self.raise_pot_to}))


@dataclass(frozen=True)
class ActionRecord:
    seat: int
    street: int
    action: Action


@dataclass(frozen=True)
class Observation:
    hand_id: int
    seat: int  # 0 = button/small blind (acts first preflop, last postflop), 1 = big blind
    hole: tuple[Card, ...]  # 5 cards
    board: tuple[Card, ...]  # 0, 3, 4 or 5 cards
    street: int  # PREFLOP..RIVER
    pot: int  # all chips both players committed this hand, current street included
    my_stack: int  # chips behind, not yet committed
    opp_stack: int
    my_street_bet: int
    opp_street_bet: int
    history: tuple[ActionRecord, ...]  # this hand only; blinds are NOT recorded as actions
    legal: LegalActions
    config: GameConfig


@dataclass(frozen=True)
class HandResult:
    hand_id: int
    seat: int  # the receiving bot's seat this hand
    deltas: tuple[int, int]  # chip change for seat 0 and seat 1; sums to 0
    board: tuple[Card, ...]  # board cards dealt by the end of the hand
    opp_hole: tuple[Card, ...] | None  # only set when the hand reached showdown
    history: tuple[ActionRecord, ...]
    showdown: bool


@dataclass(frozen=True)
class MatchResult:
    bot_a: str
    bot_b: str
    n_deals: int
    per_deal: tuple[int, ...]  # bot A's net chips per deal, summed over both seat orders
    illegal_actions: dict[str, int]  # bot name -> count (non-strict mode)
    max_act_time_s: dict[str, float]  # bot name -> slowest act() call
    act_time_s: dict[str, float]  # bot name -> total seconds in act() + hand_over() over the match

    @property
    def total(self) -> int:
        return sum(self.per_deal)


class IllegalActionError(Exception):
    pass


class BaseBot:
    """All bots subclass this. Subclasses override act(); the others are optional hooks."""

    name: str = "base"

    def __init__(self, params: dict[str, float] | None = None, seed: int = 0) -> None:
        self.params = params or {}
        self.seed = seed

    def new_match(self, opponent_name: str) -> None:
        pass

    def act(self, obs: Observation) -> Action:
        raise NotImplementedError

    def hand_over(self, result: HandResult) -> None:
        pass
