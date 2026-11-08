"""Play styles (parameter presets) and the Exp3 bandit that picks one per hand."""

from __future__ import annotations

import numpy as np

from pokerbot.params import PARAM_SPACE, clamp_params
from pokerbot.types import GameConfig

STYLE_NAMES = ("balanced", "tight", "aggressive", "trappy")

_RAISE_KEYS = tuple(f"raise_thr_{s}" for s in range(4))
_CALL_KEYS = tuple(f"call_margin_{s}" for s in range(4))
_BLUFF_KEYS = tuple(f"bluff_freq_{s}" for s in range(1, 4))


def _add(out: dict[str, float], key: str, delta: float) -> None:
    out[key] = out.get(key, PARAM_SPACE[key][0]) + delta


def _mul(out: dict[str, float], key: str, factor: float) -> None:
    out[key] = out.get(key, PARAM_SPACE[key][0]) * factor


def apply_style(params: dict[str, float], style: str) -> dict[str, float]:
    """Return a new, clamped param dict with the style's adjustments applied. The input is not mutated."""
    if style not in STYLE_NAMES:
        raise ValueError(f"unknown style: {style!r}")
    out = dict(params)
    if style == "tight":
        _add(out, "open_thr", 0.05)
        for k in _RAISE_KEYS:
            _add(out, k, 0.05)
        for k in _CALL_KEYS:
            _add(out, k, 0.04)
        for k in _BLUFF_KEYS:
            _mul(out, k, 0.5)
    elif style == "aggressive":
        _add(out, "open_thr", -0.07)
        for k in _RAISE_KEYS:
            _add(out, k, -0.05)
        for k in _CALL_KEYS:
            _add(out, k, -0.02)
        for k in _BLUFF_KEYS:
            _mul(out, k, 1.8)
        _add(out, "bet_frac_value", 0.15)
    elif style == "trappy":
        out["trap_freq"] = max(0.15, 3 * out.get("trap_freq", PARAM_SPACE["trap_freq"][0]))
        for k in _BLUFF_KEYS:
            _mul(out, k, 0.7)
        _add(out, "bet_frac_value", -0.15)
    return clamp_params(out)


class Exp3:
    """Exp3 adversarial bandit over n_arms arms with exploration rate gamma."""

    def __init__(self, n_arms: int, gamma: float, rng: np.random.Generator) -> None:
        if n_arms < 1:
            raise ValueError("n_arms must be >= 1")
        self.n_arms = n_arms
        self.gamma = gamma
        self.rng = rng
        self.log_w = np.zeros(n_arms, dtype=float)

    def probs(self) -> np.ndarray:
        z = np.exp(self.log_w - self.log_w.max())
        soft = z / z.sum()
        return (1.0 - self.gamma) * soft + self.gamma / self.n_arms

    def choose(self) -> int:
        p = self.probs()
        return int(self.rng.choice(self.n_arms, p=p / p.sum()))

    def update(self, arm: int, reward01: float) -> None:
        p = self.probs()
        self.log_w[arm] += self.gamma * (reward01 / p[arm]) / self.n_arms
        self.log_w -= self.log_w.max()


def reward01(delta: int, config: GameConfig) -> float:
    """Map a chip delta for one hand to [0, 1]: 0 chips -> 0.5, +/- starting stack -> 1 / 0."""
    return float(np.clip(0.5 + delta / (2 * config.starting_stack), 0.0, 1.0))
