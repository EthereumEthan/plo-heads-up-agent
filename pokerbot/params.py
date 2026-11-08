# name: (default, low, high). Street suffix: 0 preflop, 1 flop, 2 turn, 3 river.
PARAM_SPACE: dict[str, tuple[float, float, float]] = {
    "open_thr":         (0.45, 0.30, 0.60),  # button preflop: fold first-in below this preflop equity
    "raise_thr_0":      (0.58, 0.48, 0.80),  # equity needed to raise for value
    "raise_thr_1":      (0.65, 0.50, 0.90),
    "raise_thr_2":      (0.70, 0.50, 0.90),
    "raise_thr_3":      (0.75, 0.50, 0.95),
    "call_margin_0":    (0.00, -0.10, 0.15), # call if equity >= pot_odds + margin
    "call_margin_1":    (0.02, -0.10, 0.15),
    "call_margin_2":    (0.03, -0.10, 0.15),
    "call_margin_3":    (0.04, -0.10, 0.15),
    "bluff_freq_1":     (0.15, 0.00, 0.50),  # bet as a bluff when checked to and below call line
    "bluff_freq_2":     (0.10, 0.00, 0.50),
    "bluff_freq_3":     (0.08, 0.00, 0.50),
    "bet_frac_value":   (0.75, 0.50, 1.00),  # expected pot fraction: pot-size bet with prob (f-0.5)/0.5, else half-pot
    "bet_frac_bluff":   (0.66, 0.50, 1.00),  # expected pot fraction: pot-size bet with prob (f-0.5)/0.5, else half-pot
    "trap_freq":        (0.10, 0.00, 0.40),  # with a raise-worthy hand, just call/check this often
    "max_raises":       (2.00, 1.00, 4.00),  # our raises per street before we only call; round() to int
    "reraise_thr":      (0.80, 0.60, 0.95),  # once we have raised max_raises times this street, raise again only at or above this equity
    "exploit_strength": (0.50, 0.00, 1.00),  # 0 = ignore opponent model, 1 = full adjustment
    "range_tighten_k":  (0.50, 0.00, 1.00),  # how hard opponent bets narrow their range
    "exp3_gamma":       (0.10, 0.01, 0.30),  # style bandit exploration
    "size_disc":        (0.00, 0.00, 0.30),  # facing a bet of fraction r of the pot, call margin -= size_disc * (1 - r): defend more vs small bets
    "small_raise_w":    (1.00, 0.30, 1.00),  # an opponent raise under 3/4 pot narrows their range this much of a full raise
    "pot_size_thr":     (1.00, 0.70, 1.00),  # value bets at or above this equity are always pot sized (1.0 = off)
    "ip_bluff_mult":    (1.00, 0.50, 2.00),  # multiply bluff frequency when in position (button, postflop)
    "ip_raise_adj":     (0.00, -0.10, 0.05), # add to raise threshold when in position (negative = bet more in position)
}


def default_params() -> dict[str, float]:
    return {k: v[0] for k, v in PARAM_SPACE.items()}


def clamp_params(p: dict[str, float] | None) -> dict[str, float]:
    """Fill missing keys with defaults, clamp each value to [low, high]. Raise KeyError on unknown keys."""
    out = default_params()
    if not p:
        return out
    for k, v in p.items():
        if k not in PARAM_SPACE:
            raise KeyError(k)
        _, low, high = PARAM_SPACE[k]
        out[k] = min(max(float(v), low), high)
    return out
