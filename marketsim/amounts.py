from __future__ import annotations

import math

# Smallest position / order in *units* (fractional shares / coins)
MIN_LOT = 1e-8
# Upper sanity bound per order
MAX_LOT = 1e15
# Minimum *positive* price for quotes / limit levels (CLOB reprice, listed mid; matches Market.MIN_MID).
MIN_QUOTE_PX = 1e-4

# Notional (USD) for cash-based orders: invest down to "every penny" (float-safe floor)
MIN_NOTIONAL_USD = 1e-6
MAX_NOTIONAL_USD = 1e20

# Book / engine dust threshold (slightly looser than MIN_LOT for float safety)
QTY_EPS = 1e-10


def is_valid_order_size(x: float) -> bool:
    return math.isfinite(x) and MIN_LOT <= x <= MAX_LOT


def is_valid_cash_notional(x: float) -> bool:
    return math.isfinite(x) and MIN_NOTIONAL_USD <= x <= MAX_NOTIONAL_USD


def round_pos(x: float) -> float:
    """For display / stable storage: trim float dust."""

    if abs(x) < MIN_LOT:
        return 0.0
    return float(x)
