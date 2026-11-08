"""Per-half thinking-time bank: spreads a share of the bank over the hands of one half."""

from __future__ import annotations

from pokerbot.types import GameConfig

DECISIONS_PER_HAND = 2.5  # average act() calls per hand for one bot, used to spread the bank
PACE_SLACK_FRAC = 0.1  # spend may run this share of the target ahead of an even pace before throttling


class TimeBank:
    def __init__(self, bank_s: float, hands_per_half: int, spend_frac: float = 0.35, cap_s: float = 0.4) -> None:
        self.bank_s = bank_s
        self.hands_per_half = hands_per_half
        self.spend_frac = spend_frac
        self.cap_s = cap_s
        self.spent_s = 0.0  # thinking time charged in the current half
        self.hands_done = 0  # hands finished in the current half

    @classmethod
    def from_config(cls, cfg: GameConfig, spend_frac: float = 0.35) -> TimeBank:
        return cls(cfg.time_bank_s, cfg.hands_per_half, spend_frac, 0.4 * cfg.time_budget_s)

    def charge(self, seconds: float) -> None:
        self.spent_s += max(0.0, seconds)

    def hand_finished(self) -> None:
        self.hands_done += 1
        if self.hands_done >= self.hands_per_half:
            self.spent_s = 0.0
            self.hands_done = 0

    def decision_budget_s(self) -> float:
        """Time limit for one decision.

        On pace, returns cap_s: a safety cap that should never bind, so the bot's fixed sample
        count (and RNG use) doesn't depend on wall-clock speed. Only when spending runs ahead of an
        even pace (plus slack) does it throttle to an even share of what's left of the target.
        """
        target = self.spend_frac * self.bank_s
        pace = target * (self.hands_done / self.hands_per_half + PACE_SLACK_FRAC)
        if self.spent_s <= pace:
            return self.cap_s
        hands_left = max(1, self.hands_per_half - self.hands_done)
        per = (target - self.spent_s) / (hands_left * DECISIONS_PER_HAND)
        return min(max(per, 0.0), self.cap_s)
