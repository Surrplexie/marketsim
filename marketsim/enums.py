from __future__ import annotations

from enum import Enum


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"
