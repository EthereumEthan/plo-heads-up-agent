"""MainBot: Monte Carlo equity + tunable thresholds + opponent model + Exp3 style bandit."""

from __future__ import annotations

import time

import numpy as np

from pokerbot import preflop
from pokerbot.equity import estimate_equity, sample_top_hands, warmup
from pokerbot.evaluator import get_rank_table
from pokerbot.opponent import OpponentModel
from pokerbot.params import clamp_params
from pokerbot.preflop import preflop_equity, top_fraction_hands
from pokerbot.strategy import Decision, decide, opp_raise_fractions
from pokerbot.styles import STYLE_NAMES, Exp3, apply_style, reward01
from pokerbot.timebank import TimeBank
from pokerbot.types import Action, BaseBot, GameConfig, HandResult, Observation

N_SAMPLES = 200  # fixed per decision: ~1.8 s/1000h worst case on the dev Mac (300 hit the 4.4 s ceiling)
N_OPP_SAMPLE = 400
N_CARDS = 36
BANK_SPEND_FRAC = 0.3  # safety ceiling (2.4 s of the 8 s bank, ~2.6 s with pace slack) for a slower tournament CPU


class MainBot(BaseBot):
    name = "main"

    def __init__(self, params: dict[str, float] | None = None, seed: int = 0) -> None:
        super().__init__(params, seed)
        self.base = clamp_params(params)
        self.rng = np.random.default_rng(seed)
        self.last_decision: Decision | None = None
        self._tables: dict[bool, np.ndarray] = {}
        self._last_config: GameConfig | None = None
        self.bank: TimeBank | None = None
        warmup()
        preflop.warmup()
        # Build or load both rank tables up front; the trips-beat-straight one is otherwise
        # built inside the first act() under those rules (~0.6 s with no disk cache).
        self._table(False)
        self._table(True)
        self.new_match("unknown")

    def new_match(self, opponent_name: str) -> None:
        self.opp = OpponentModel()
        self.exp3 = Exp3(len(STYLE_NAMES), self.base["exp3_gamma"], self.rng)
        self.cur = dict(self.base)
        self._hand_id: int | None = None
        self._style_idx: int | None = None
        self.bank = None

    # ------------------------------------------------------------------
    def _table(self, trips_beat_straight: bool) -> np.ndarray:
        t = self._tables.get(trips_beat_straight)
        if t is None:
            t = get_rank_table(trips_beat_straight)
            self._tables[trips_beat_straight] = t
        return t

    def _equity(self, obs: Observation, f: float, limit_s: float) -> float:
        cfg = obs.config
        tbs = cfg.trips_beat_straight
        hole = np.asarray(obs.hole, dtype=np.int64)
        board = np.asarray(obs.board, dtype=np.int64)
        opp_hands = None
        if obs.street == 0:
            if f >= 1.0:
                if tbs == preflop.TABLE_TRIPS_BEAT_STRAIGHT:
                    return preflop_equity(obs.hole)
            else:
                opp_hands = top_fraction_hands(f, self.rng, N_OPP_SAMPLE, exclude=obs.hole)
        elif f < 1.0:
            opp_hands = sample_top_hands(obs.hole, obs.board, f, self.rng, N_OPP_SAMPLE, tbs)
        return estimate_equity(
            hole,
            board,
            self.rng,
            n_samples=N_SAMPLES,
            opp_hands=opp_hands,
            time_limit_s=limit_s,
            trips_beat_straight=tbs,
        )

    def _fallback(self, obs: Observation) -> Action:
        return Action("check") if obs.legal.can_check else Action("fold")

    def act(self, obs: Observation) -> Action:
        t0 = time.perf_counter()
        try:
            self._last_config = obs.config
            if self.bank is None:
                self.bank = TimeBank.from_config(obs.config, BANK_SPEND_FRAC)
            limit = self.bank.decision_budget_s()
            if obs.hand_id != self._hand_id:
                self._hand_id = obs.hand_id
                self._style_idx = self.exp3.choose()
                self.cur = apply_style(self.base, STYLE_NAMES[self._style_idx])
            w_small = self.cur["small_raise_w"]
            n = sum(w_small if frac < 0.75 else 1.0 for frac in opp_raise_fractions(obs))
            v = self.opp.view(obs.street)
            f = max(0.15, 1 - self.cur["range_tighten_k"] * (1 - v.aggression)) ** n
            equity = self._equity(obs, f, limit)
            dec = decide(obs, equity, self.cur, v, self.rng)
            self.last_decision = dec
            return self._sanitize(obs, dec.action)
        except Exception:  # noqa: BLE001 - safety net: never crash a hand
            return self._fallback(obs)
        finally:
            if self.bank is not None:
                self.bank.charge(time.perf_counter() - t0)

    def _sanitize(self, obs: Observation, a: Action) -> Action:
        if a.kind == "raise":
            legal = obs.legal
            if not legal.can_raise:
                return Action("check") if legal.can_check else Action("call")
            return Action("raise", min(legal.raise_targets(), key=lambda t: abs(t - a.amount)))
        return a

    def hand_over(self, result: HandResult) -> None:
        t0 = time.perf_counter()
        self._update_hand(result)
        if self.bank is not None:
            self.bank.charge(time.perf_counter() - t0)
            self.bank.hand_finished()

    def _update_hand(self, result: HandResult) -> None:
        self.opp.observe_hand(result)
        if self._style_idx is None or self._hand_id != result.hand_id:
            self._style_idx = None
            return
        cfg = self._last_config or GameConfig()
        self.exp3.update(self._style_idx, reward01(result.deltas[result.seat], cfg))
        self._style_idx = None
