"""Builds the bot once per process and exposes the three official hooks.

Imports only `pokerbot` (no `harness/`) so the submission stays light. The slow setup (numba
kernels, preflop tables, rank tables) runs in a background thread started at import, so the server
answers `GET /` right away even on a slow host (the organizers' 15 s startup gate). Until the setup
is done, decisions fall back to the judge's default (check if free, else fold); on the dev machine
the setup takes about 2 s, well before the first hand.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from pokerbot import equity, preflop
from pokerbot.evaluator import get_rank_table
from pokerbot.params import clamp_params
from submission.adapter import OfficialAdapter, _default_reply

PARAMS_PATH = Path(__file__).resolve().parent / "frozen_params.json"
SEED = 12345
COLD_WAIT_S = 0.25  # how long a decision waits for the setup before answering with the default


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


_READY = threading.Event()
_LOCK = threading.Lock()
_BOT = None
_ADAPTER: OfficialAdapter | None = None
_PENDING_HALF: tuple | None = None  # (config, memory) of a new_half that arrived before setup finished
_SETUP_ERROR: BaseException | None = None


def _setup() -> None:
    global _BOT, _ADAPTER, _SETUP_ERROR
    try:
        bot = build_bot()
        equity.warmup()
        preflop.warmup()
        get_rank_table(True)
        adapter = OfficialAdapter(bot)
        with _LOCK:
            _BOT, _ADAPTER = bot, adapter
            if _PENDING_HALF is not None:
                adapter.new_half(*_PENDING_HALF)
    except BaseException as e:  # noqa: BLE001 - keep serving default replies rather than crash
        _SETUP_ERROR = e
    finally:
        _READY.set()


_THREAD = threading.Thread(target=_setup, name="bot-setup", daemon=True)
_THREAD.start()


def wait_ready(timeout: float | None = None) -> bool:
    """Block until the setup is done (or `timeout` s pass). Returns True when the adapter is usable."""
    _READY.wait(timeout)
    return _ADAPTER is not None


def __getattr__(name: str):
    # `entry.BOT` / `entry.ADAPTER` wait for the setup, so tests and tools see the built objects.
    if name in ("BOT", "ADAPTER"):
        wait_ready()
        if _SETUP_ERROR is not None:
            raise _SETUP_ERROR
        return _BOT if name == "BOT" else _ADAPTER
    raise AttributeError(name)


def new_half(config, memory=None):
    global _PENDING_HALF
    with _LOCK:
        if _ADAPTER is None:
            _PENDING_HALF = (config, memory)
            return
    _ADAPTER.new_half(config, memory)


def choose_action(obs, memory=None):
    if _ADAPTER is None and not wait_ready(COLD_WAIT_S):
        return _default_reply(obs)
    return _ADAPTER.choose_action(obs, memory)


def on_hand_end(result, memory=None):
    if _ADAPTER is None:
        return
    _ADAPTER.on_hand_end(result, memory)
