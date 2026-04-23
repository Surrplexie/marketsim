from __future__ import annotations

import math
from typing import Any


def sim_minute_of_day(tick: int, sim_minutes_per_tick: float) -> float:
    """Minutes 0..1440-ε within one *sim* day (cyclic)."""

    m = (float(tick) * float(sim_minutes_per_tick)) % (24.0 * 60.0)
    return m


def tod_drift_shape(minute: float) -> float:
    """
    U-shaped *signature* in [-1, 1] for open / midday / close style variation.
    Used as a weight on TOD bps, not a literal price.
    """

    u = 2.0 * math.pi * (minute / 1440.0)
    return 0.42 * math.sin(u) + 0.28 * math.sin(2.0 * u) + 0.18 * math.sin(3.0 * u)


def round_magnet_delta(price: float, magnet_bps: float) -> float:
    """Nudge *price* a fraction of the way toward the nearest **round dollar** (testing magnet)."""

    if not math.isfinite(price) or price < 0.2:
        return 0.0
    r = float(round(float(price), 0))
    if r <= 0.0:
        return 0.0
    gap = r - float(price)
    if abs(gap) < 1e-8:
        return 0.0
    return gap * (magnet_bps / 10_000.0) * 0.4


def slippage_bps_total(notional: float, bps_base: float, bps_per_million: float) -> float:
    return float(bps_base) + float(bps_per_million) * (max(0.0, notional) / 1_000_000.0)


def taker_notional_cash(
    fills: list[Any], remainder_qty: float, rest_price: float
) -> float:
    """Sum price×qty (lit + synthetic remainder) for notional and fees."""

    t = 0.0
    for f in fills:
        t += float(f.price) * float(f.qty)
    r = max(0.0, float(remainder_qty))
    if r > 1e-12:
        t += float(rest_price) * r
    return t


def scale_fill_prices(
    fills: list[Any], rest_price: list[float] | None, *, mult: float
) -> None:
    """In-place: multiply all taker fill prices and synthetic *rest* by *mult* (worse: buy>1, sell<1)."""

    m = max(0.5, min(1.5, mult))
    for f in fills:
        f.price = float(f.price) * m
    if rest_price is not None and len(rest_price) > 0:
        rest_price[0] = float(rest_price[0]) * m


def wick_ohlc(o: float, h: float, lo: float, c: float) -> dict[str, float]:
    r = h - lo
    if r < 1e-15 or not (math.isfinite(r) and r > 0.0):
        return {"upper": 0.0, "lower": 0.0, "body": 0.0}
    body = abs(c - o) / r
    up = (h - max(o, c)) / r
    dn = (min(o, c) - lo) / r
    return {"upper": float(up), "lower": float(dn), "body": float(body)}
