"""Builds the bot once at import and exposes the three official hooks.

Imports only `pokerbot` (no `harness/`) so the submission stays light. Importing this module does
the slow setup (numba kernels, preflop tables, rank tables) so the first `/act` is fast.
"""

from __future__ import annotations

import json
from pathlib import Path

from pokerbot import equity, preflop
from pokerbot.evaluator import get_rank_table
from pokerbot.params import clamp_params
from submission.adapter import OfficialAdapter

PARAMS_PATH = Path(__file__).resolve().parent / "frozen_params.json"
SEED = 12345


def load_frozen(path: Path = PARAMS_PATH) -> tuple[str, dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("params", {})
    params = clamp_params({str(k): float(v) for k, v in raw.items()})
    return str(data.get("bot", "main")), params


def build_bot():
    kind, params = load_frozen()
    if kind == "deepcfr":
        from cfr.deepcfr_bot import DeepCFRBot  # lazy: needs torch, not used by the current submission

        bot = DeepCFRBot(params, seed=SEED)
    else:
        from pokerbot.bot import MainBot

        bot = MainBot(params, seed=SEED)
    return bot


BOT = build_bot()
equity.warmup()
preflop.warmup()
get_rank_table(True)

ADAPTER = OfficialAdapter(BOT)


def new_half(config, memory):
    ADAPTER.new_half(config, memory)


def choose_action(obs, memory):
    return ADAPTER.choose_action(obs, memory)


def on_hand_end(result, memory):
    ADAPTER.on_hand_end(result, memory)
