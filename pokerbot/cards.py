from __future__ import annotations

import operator

from pokerbot.types import Card

RANKS = "6789TJQKA"
SUITS = "shdc"
FULL_DECK: tuple[Card, ...] = tuple(range(36))


def card_from_str(s: str) -> Card:
    """Parse a two-char card like "As" into its index (rank_index * 4 + suit_index)."""
    if len(s) != 2 or s[0] not in RANKS or s[1] not in SUITS:
        raise ValueError(f"invalid card: {s!r}")
    return RANKS.index(s[0]) * 4 + SUITS.index(s[1])


def card_to_str(c: Card) -> str:
    try:
        c = operator.index(c)
    except TypeError:
        raise ValueError(f"invalid card index: {c!r}") from None
    if not 0 <= c < 36:
        raise ValueError(f"invalid card index: {c!r}")
    return RANKS[c // 4] + SUITS[c % 4]


def cards_from_str(s: str) -> tuple[Card, ...]:
    """Parse cards written space-separated, concatenated, or mixed ("AsKd 6c")."""
    joined = "".join(s.split())
    if len(joined) % 2:
        raise ValueError(f"invalid card string: {s!r}")
    return tuple(card_from_str(joined[i : i + 2]) for i in range(0, len(joined), 2))


def _flip_suit(c: int) -> int:
    rank, suit = divmod(c, 4)
    return rank * 4 + (3 - suit)


def to_official_card(c: Card) -> int:
    """Our card index -> official engine encoding (same rank, suit 0=c,1=d,2=h,3=s)."""
    return _flip_suit(c)


def from_official_card(c: int) -> Card:
    """Official engine card index -> our encoding. The map is its own inverse."""
    return _flip_suit(c)
