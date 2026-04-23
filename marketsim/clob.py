from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.random import Generator as RNG

from .amounts import MIN_QUOTE_PX, QTY_EPS
from .enums import Side


def _pkey(p: float) -> float:
    return round(float(p), 6)


@dataclass(slots=True)
class BookOrder:
    order_id: int
    size_remaining: float
    is_npc: bool = True


@dataclass
class BookFill:
    price: float
    qty: float
    taker_side: Side
    maker_is_npc: bool = True


@dataclass
class OrderBook:
    """Per-symbol central limit order book. Prices *absolute*; FIFO; fractional *qty*."""

    _next_id: int = field(default=0, init=False, repr=False)
    bids: dict[float, deque[BookOrder]] = field(default_factory=dict)
    asks: dict[float, deque[BookOrder]] = field(default_factory=dict)

    def _bump_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def best_bid(self) -> float | None:
        if not self.bids:
            return None
        return max(self.bids)

    def best_ask(self) -> float | None:
        if not self.asks:
            return None
        return min(self.asks)

    def depth_bids(self, levels: int = 5) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for px in sorted(self.bids, reverse=True)[:levels]:
            q = float(sum(o.size_remaining for o in self.bids[px]))
            out.append((px, q))
        return out

    def depth_asks(self, levels: int = 5) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for px in sorted(self.asks)[:levels]:
            q = float(sum(o.size_remaining for o in self.asks[px]))
            out.append((px, q))
        return out

    def total_resting_qty_asks(self) -> float:
        """Sum of *size_remaining* on all ask levels (player + NPC)."""

        t = 0.0
        for dq in self.asks.values():
            for o in dq:
                t += float(o.size_remaining)
        return t

    def total_resting_qty_bids(self) -> float:
        """Sum of *size_remaining* on all bid levels (player + NPC)."""

        t = 0.0
        for dq in self.bids.values():
            for o in dq:
                t += float(o.size_remaining)
        return t

    def _remove_if_empty(self, d: dict[float, deque[BookOrder]], p: float) -> None:
        if p in d and not d[p]:
            del d[p]

    def reprice_npc_sides(self, bid_delta: float, ask_delta: float) -> None:
        """Widen or narrow the NPC half-spread: bids *bid_delta*, asks *ask_delta* (typically bid −, ask +)."""

        if abs(bid_delta) < 1e-15 and abs(ask_delta) < 1e-15:
            return

        def rebuild(
            side: dict[float, deque[BookOrder]], d: float
        ) -> dict[float, deque[BookOrder]]:
            out: dict[float, deque[BookOrder]] = {}
            for p, dq in list(side.items()):
                for o in list(dq):
                    if o.is_npc:
                        k = _pkey(
                            max(float(MIN_QUOTE_PX), float(p) + float(d))
                        )
                    else:
                        k = _pkey(float(p))
                    out.setdefault(k, deque()).append(o)
            return out

        self.bids = rebuild(self.bids, bid_delta)
        self.asks = rebuild(self.asks, ask_delta)

    def reprice_npcs(self, price_delta: float) -> None:
        if abs(price_delta) < 1e-15:
            return

        def rebuild(side: dict[float, deque[BookOrder]]) -> dict[float, deque[BookOrder]]:
            out: dict[float, deque[BookOrder]] = {}
            for p, dq in list(side.items()):
                for o in list(dq):
                    if o.is_npc:
                        k = _pkey(
                            max(float(MIN_QUOTE_PX), float(p) + float(price_delta))
                        )
                    else:
                        k = _pkey(float(p))
                    out.setdefault(k, deque()).append(o)
            return out

        self.bids = rebuild(self.bids)
        self.asks = rebuild(self.asks)

    def apply_forward_split(self, ratio: float) -> None:
        """
        Forward stock split: each limit price is divided by *ratio*; each resting size
        in shares is multiplied by *ratio* (notional at each level unchanged).
        """
        r = float(ratio)
        if not math.isfinite(r) or r <= 1.0 + 1e-9:
            return

        def rebuild(side: dict[float, deque[BookOrder]]) -> dict[float, deque[BookOrder]]:
            out: dict[float, deque[BookOrder]] = {}
            for p, dq in list(side.items()):
                for o in list(dq):
                    np_ = _pkey(max(float(MIN_QUOTE_PX), float(p) / r))
                    nrem = float(o.size_remaining) * r
                    co = BookOrder(
                        o.order_id, nrem, is_npc=bool(o.is_npc)
                    )
                    out.setdefault(np_, deque()).append(co)
            return out

        self.bids = rebuild(self.bids)
        self.asks = rebuild(self.asks)

    def shift_price_levels(self, delta: float) -> None:
        """Ex-div / corporate: move every resting limit *price* by *delta* (e.g. negative for cash dividend)."""

        d = float(delta)
        if not math.isfinite(d) or abs(d) < 1e-15:
            return

        def rebuild(side: dict[float, deque[BookOrder]]) -> dict[float, deque[BookOrder]]:
            out: dict[float, deque[BookOrder]] = {}
            for p, dq in list(side.items()):
                for o in list(dq):
                    np_ = _pkey(max(float(MIN_QUOTE_PX), float(p) + d))
                    co = BookOrder(
                        o.order_id, float(o.size_remaining), is_npc=bool(o.is_npc)
                    )
                    out.setdefault(np_, deque()).append(co)
            return out

        self.bids = rebuild(self.bids)
        self.asks = rebuild(self.asks)

    def scale_resting_shares(self, mult: float) -> None:
        """
        Proportional share reduction (e.g. buyback): every resting *size_remaining* is multiplied
        by *mult*; tiny lots are removed.
        """

        m = float(mult)
        if not math.isfinite(m) or m <= 0.0 or m > 1.0 + 1e-6:
            return

        def rebuild(side: dict[float, deque[BookOrder]]) -> dict[float, deque[BookOrder]]:
            out: dict[float, deque[BookOrder]] = {}
            for p, dq in list(side.items()):
                for o in list(dq):
                    nr = float(o.size_remaining) * m
                    if nr <= QTY_EPS:
                        continue
                    co = BookOrder(o.order_id, nr, is_npc=bool(o.is_npc))
                    out.setdefault(p, deque()).append(co)
            return out

        self.bids = rebuild(self.bids)
        self.asks = rebuild(self.asks)

    def _take_asks(
        self,
        price_limit: float,
        size: float,
        taker: Side,
        *,
        taker_is_player: bool,
    ) -> tuple[list[BookFill], float]:
        fills: list[BookFill] = []
        rem = float(size)
        while rem > QTY_EPS and self.asks:
            p = _pkey(self.best_ask() or 0.0)
            if p > price_limit + 1e-9:
                break
            dq = self.asks[p]
            while rem > QTY_EPS and dq:
                o = dq[0]
                if taker_is_player and not o.is_npc:
                    return fills, rem
                t = min(rem, o.size_remaining)
                o.size_remaining -= t
                if o.size_remaining <= QTY_EPS:
                    dq.popleft()
                rem -= t
                fills.append(
                    BookFill(
                        price=p, qty=t, taker_side=taker, maker_is_npc=bool(o.is_npc)
                    )
                )
            self._remove_if_empty(self.asks, p)
        return fills, rem

    def _take_bids(
        self,
        price_limit: float,
        size: float,
        taker: Side,
        *,
        taker_is_player: bool,
    ) -> tuple[list[BookFill], float]:
        fills: list[BookFill] = []
        rem = float(size)
        while rem > QTY_EPS and self.bids:
            p = _pkey(self.best_bid() or 0.0)
            if p < price_limit - 1e-9:
                break
            dq = self.bids[p]
            while rem > QTY_EPS and dq:
                o = dq[0]
                if taker_is_player and not o.is_npc:
                    return fills, rem
                t = min(rem, o.size_remaining)
                o.size_remaining -= t
                if o.size_remaining <= QTY_EPS:
                    dq.popleft()
                rem -= t
                fills.append(
                    BookFill(
                        price=p, qty=t, taker_side=taker, maker_is_npc=bool(o.is_npc)
                    )
                )
            self._remove_if_empty(self.bids, p)
        return fills, rem

    def match_market_buy(
        self, size: float, *, taker_is_player: bool = True
    ) -> tuple[list[BookFill], float]:
        if size <= 0:
            return [], 0.0
        pmax = self.best_ask() if self.best_ask() is not None else 1e30
        return self._take_asks(pmax, size, Side.BUY, taker_is_player=taker_is_player)

    def match_market_sell(
        self, size: float, *, taker_is_player: bool = True
    ) -> tuple[list[BookFill], float]:
        if size <= 0:
            return [], 0.0
        pmin = self.best_bid() if self.best_bid() is not None else -1.0
        return self._take_bids(pmin, size, Side.SELL, taker_is_player=taker_is_player)

    def _rest_bid(self, p: float, rem: float, *, is_npc: bool) -> BookOrder:
        k = _pkey(p)
        o = BookOrder(self._bump_id(), rem, is_npc=is_npc)
        self.bids.setdefault(k, deque()).append(o)
        return o

    def _rest_ask(self, p: float, rem: float, *, is_npc: bool) -> BookOrder:
        k = _pkey(p)
        o = BookOrder(self._bump_id(), rem, is_npc=is_npc)
        self.asks.setdefault(k, deque()).append(o)
        return o

    def add_limit_buy(
        self, price: float, size: float, *, taker_is_player: bool = True
    ) -> tuple[list[BookFill], float, BookOrder | None]:
        fills, rem = self._take_asks(
            float(price), size, Side.BUY, taker_is_player=taker_is_player
        )
        if rem <= QTY_EPS:
            return fills, 0.0, None
        o = self._rest_bid(float(price), rem, is_npc=False)
        return fills, rem, o

    def add_limit_sell(
        self, price: float, size: float, *, taker_is_player: bool = True
    ) -> tuple[list[BookFill], float, BookOrder | None]:
        fills, rem = self._take_bids(
            float(price), size, Side.SELL, taker_is_player=taker_is_player
        )
        if rem <= QTY_EPS:
            return fills, 0.0, None
        o = self._rest_ask(float(price), rem, is_npc=False)
        return fills, rem, o


def seed_liquidity(
    book: OrderBook,
    mid: float,
    spread_bps: float,
    rng: "RNG",
    extra_levels: int = 2,
) -> None:
    m0 = max(float(MIN_QUOTE_PX), float(mid))
    half = m0 * 0.5 * (spread_bps / 10_000.0)
    for L in range(1, extra_levels + 1):
        p_bid = max(float(MIN_QUOTE_PX), m0 - half * (0.5 + 0.4 * L + rng.random() * 0.1))
        p_ask = max(p_bid + 1e-9, m0 + half * (0.5 + 0.4 * L + rng.random() * 0.1))
        qf = 50.0 + float(rng.random() * 350.0)
        book._rest_bid(p_bid, qf, is_npc=True)
        book._rest_ask(p_ask, 50.0 + float(rng.random() * 350.0), is_npc=True)
    if not book.bids or not book.asks:
        book._rest_bid(max(m0 - half, float(MIN_QUOTE_PX)), 100.0, is_npc=True)
        book._rest_ask(m0 + half, 100.0, is_npc=True)
