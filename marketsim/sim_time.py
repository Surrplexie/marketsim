from __future__ import annotations

import math
import re
from dataclasses import dataclass


# canonical keys for _MINUTES; spelling aliases
_UNIT_ALIASES: dict[str, str] = {
    "s": "second",
    "sec": "second",
    "second": "second",
    "seconds": "second",
    "m": "minute",  # use "mon" for month
    "min": "minute",
    "minute": "minute",
    "minutes": "minute",
    "h": "hour",
    "hr": "hour",
    "hour": "hour",
    "hours": "hour",
    "d": "day",
    "day": "day",
    "days": "day",
    "w": "week",
    "week": "week",
    "weeks": "week",
    "mon": "month",
    "month": "month",
    "months": "month",
    "y": "year",
    "yr": "year",
    "year": "year",
    "years": "year",
}


def _norm_unit(s: str) -> str:
    t = s.strip().lower()
    if t in _UNIT_ALIASES:
        return _UNIT_ALIASES[t]
    return t


# minutes of *sim* time each canonical unit represents (one count)
_MINUTES: dict[str, float] = {
    "second": 1.0 / 60.0,
    "minute": 1.0,
    "hour": 60.0,
    "day": 24.0 * 60.0,
    "week": 7.0 * 24.0 * 60.0,
    "month": 30.0 * 24.0 * 60.0,  # 30d month
    "year": 365.0 * 24.0 * 60.0,
}


def unit_minutes(name: str) -> float:
    u = _norm_unit(name)
    u = _UNIT_ALIASES.get(u, u)
    if u not in _MINUTES:
        raise ValueError(f"unknown time unit: {name!r}")
    return _MINUTES[u]


def sim_minutes_in_interval(count: int, unit: str) -> float:
    if count < 0:
        raise ValueError("count must be non-negative")
    return float(count) * unit_minutes(unit)


@dataclass(frozen=True, slots=True)
class TimeAdvance:
    """Result of mapping wall-clock interval → engine ticks (discrete *step()* calls)."""

    ticks: int
    sim_minutes: float
    sim_minutes_per_tick: float


def interval_to_ticks(
    count: int,
    unit: str,
    *,
    sim_minutes_per_tick: float,
) -> TimeAdvance:
    """
    *sim_minutes_per_tick* = how many **simulation** minutes one call to *step()*
    represents (drives only the *run* / calendar commands, not GBM *dt*).
    """

    if count < 0:
        raise ValueError("count must be non-negative")
    mpt = max(float(sim_minutes_per_tick), 1e-9)
    if count == 0:
        return TimeAdvance(0, 0.0, mpt)
    sm = sim_minutes_in_interval(count, unit)
    raw = sm / mpt
    t = int(math.ceil(float(raw) - 1e-12))
    if t < 0:
        t = 0
    if sm > 0 and t == 0:
        t = 1
    return TimeAdvance(t, sm, mpt)


def parse_run_tokens(a: str, b: str) -> tuple[int, str]:
    """
    * *run day 2*  →  (2, "day")
    * *run 2 day*  →  (2, "day")
    * *run 1 hr*  →  (1, "hr")
    """
    a0 = a.strip()
    b0 = b.strip()
    m_n1 = re.fullmatch(r"(\d+)", a0)
    m_n2 = re.fullmatch(r"(\d+)", b0)
    if m_n1 and not m_n2:
        return int(m_n1.group(1)), b0  # run 2 day
    if m_n2 and not m_n1:
        return int(m_n2.group(1)), a0  # run day 2
    if m_n1 and m_n2:
        raise ValueError("ambiguous: two numbers")
    raise ValueError("need a number and a time unit, e.g.  run day 2  or  run 2 day")


def parse_run_line(parts: list[str]) -> tuple[int, str]:
    if len(parts) < 2:
        raise ValueError("usage:  run <unit> <n>  or  run <n> <unit>")
    a, b = parts[0], parts[1]
    return parse_run_tokens(a, b)
