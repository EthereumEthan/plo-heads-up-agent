"""Duplicate match runner: each deal is played twice with the same deck and seats swapped."""

from __future__ import annotations

import numpy as np

from pokerbot.engine import play_hand
from pokerbot.types import BaseBot, Card, GameConfig, MatchResult


def deal_deck(match_seed: int, deal_index: int) -> tuple[Card, ...]:
    perm = np.random.default_rng([match_seed, deal_index]).permutation(36)
    return tuple(int(c) for c in perm)


def play_duplicate_match(
    bot_a: BaseBot,
    bot_b: BaseBot,
    n_deals: int,
    match_seed: int,
    config: GameConfig = GameConfig(),  # noqa: B008 - frozen dataclass
    deal_offset: int = 0,
) -> MatchResult:
    name_a = bot_a.name
    name_b = bot_b.name if bot_b.name != bot_a.name else bot_b.name + "#2"

    bot_a.new_match(bot_b.name)
    bot_b.new_match(bot_a.name)

    illegal = {name_a: 0, name_b: 0}
    max_time = {name_a: 0.0, name_b: 0.0}
    act_time = {name_a: 0.0, name_b: 0.0}
    per_deal: list[int] = []

    for i in range(deal_offset, deal_offset + n_deals):
        deck = deal_deck(match_seed, i)
        hand_id = 2 * i
        out1 = play_hand((bot_a, bot_b), deck, hand_id, config)
        out2 = play_hand((bot_b, bot_a), deck, hand_id + 1, config)
        per_deal.append(out1.deltas[0] + out2.deltas[1])
        for out, seat_names in ((out1, (name_a, name_b)), (out2, (name_b, name_a))):
            for seat, name in enumerate(seat_names):
                illegal[name] += out.illegal[seat]
                max_time[name] = max(max_time[name], out.max_act_time_s[seat])
                act_time[name] += out.act_time_s[seat]

    return MatchResult(
        bot_a=name_a,
        bot_b=name_b,
        n_deals=n_deals,
        per_deal=tuple(per_deal),
        illegal_actions=illegal,
        max_act_time_s=max_time,
        act_time_s=act_time,
    )
