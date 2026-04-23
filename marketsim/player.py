from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .amounts import MIN_LOT, QTY_EPS
from .market import Market

ORDER_LOG_MAX = 5000


class Result(str, Enum):
    OK = "ok"
    NO_CASH = "no_cash"
    NO_POSITION = "no_position"
    NOT_FOUND = "not_found"
    BAD_SIZE = "bad_size"
    BAD_PRICE = "bad_price"
    NO_LIQUIDITY = "no_liquidity"


@dataclass
class Player:
    """Fractional *positions* (can be <0 = short with *shorting_enabled*); *locked_cash* for buy limits."""

    cash: float
    positions: dict[str, float] = field(default_factory=dict)
    locked_cash: float = 0.0
    locked_sell: dict[str, float] = field(default_factory=dict)
    fees_paid: float = 0.0
    # Long cost basis (USD) for open lots; average cost = cost_basis/qty
    cost_basis: dict[str, float] = field(default_factory=dict)
    last_trade_tick: dict[str, int] = field(default_factory=dict)
    last_trade_side: dict[str, str] = field(default_factory=dict)
    last_trade_price: dict[str, float] = field(default_factory=dict)
    last_trade_qty: dict[str, float] = field(default_factory=dict)
    # Newest entries at the end; capped for memory (order / audit trail).
    order_log: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=ORDER_LOG_MAX)
    )
    # Copied from GameConfig in *new_session*; governs margin and shorts.
    max_leverage: float = 1.0
    maintenance_margin_rate: float = 0.2
    shorting_enabled: bool = False
    short_borrow_bps_per_sim_day: float = 0.0
    sec_fee_sell_bps: float = 0.0

    def clone(self) -> "Player":
        return Player(
            cash=self.cash,
            positions=dict(self.positions),
            locked_cash=self.locked_cash,
            locked_sell=dict(self.locked_sell),
            fees_paid=self.fees_paid,
            cost_basis=dict(self.cost_basis),
            last_trade_tick=dict(self.last_trade_tick),
            last_trade_side=dict(self.last_trade_side),
            last_trade_price=dict(self.last_trade_price),
            last_trade_qty=dict(self.last_trade_qty),
            order_log=deque(list(self.order_log), maxlen=ORDER_LOG_MAX),
            max_leverage=self.max_leverage,
            maintenance_margin_rate=self.maintenance_margin_rate,
            shorting_enabled=self.shorting_enabled,
            short_borrow_bps_per_sim_day=self.short_borrow_bps_per_sim_day,
            sec_fee_sell_bps=self.sec_fee_sell_bps,
        )

    def append_order_log(self, entry: dict[str, Any]) -> None:
        self.order_log.append(entry)

    def log_fill(
        self,
        *,
        sim_tick: int,
        ticker: str,
        order_type: str,
        side: str,
        price: float,
        qty: float,
    ) -> None:
        self.append_order_log(
            {
                "event": "fill",
                "sim_tick": int(sim_tick),
                "ticker": str(ticker),
                "order_type": str(order_type),
                "side": str(side),
                "price": float(price),
                "qty": float(qty),
                "notional_usd": float(price) * float(qty),
            }
        )

    def log_fee(
        self,
        *,
        sim_tick: int,
        ticker: str,
        order_type: str,
        side: str,
        notional_usd: float,
        fee_usd: float,
        taker_fee_bps: float,
    ) -> None:
        self.append_order_log(
            {
                "event": "fee",
                "sim_tick": int(sim_tick),
                "ticker": str(ticker),
                "order_type": str(order_type),
                "side": str(side),
                "notional_usd": float(notional_usd),
                "fee_usd": float(fee_usd),
                "taker_fee_bps": float(taker_fee_bps),
            }
        )

    def log_reject(
        self,
        *,
        sim_tick: int,
        ticker: str,
        order_type: str,
        side: str,
        result: str,
        message: str = "",
    ) -> None:
        self.append_order_log(
            {
                "event": "reject",
                "sim_tick": int(sim_tick),
                "ticker": str(ticker),
                "order_type": str(order_type),
                "side": str(side),
                "result": str(result),
                "message": str(message) if message else str(result),
            }
        )

    def log_split(self, *, sim_tick: int, ticker: str, ratio: float, source: str) -> None:
        self.append_order_log(
            {
                "event": "split",
                "sim_tick": int(sim_tick),
                "ticker": str(ticker),
                "ratio": float(ratio),
                "source": str(source),
            }
        )

    def log_dividend(
        self,
        *,
        sim_tick: int,
        ticker: str,
        usd_per_share: float,
        cash_received: float,
        source: str,
    ) -> None:
        self.append_order_log(
            {
                "event": "dividend",
                "sim_tick": int(sim_tick),
                "ticker": str(ticker),
                "usd_per_share": float(usd_per_share),
                "cash_received": float(cash_received),
                "source": str(source),
            }
        )

    def log_buyback(
        self,
        *,
        sim_tick: int,
        ticker: str,
        float_fraction: float,
        units_outstanding: float,
        source: str,
    ) -> None:
        self.append_order_log(
            {
                "event": "buyback",
                "sim_tick": int(sim_tick),
                "ticker": str(ticker),
                "float_fraction": float(float_fraction),
                "units_outstanding": float(units_outstanding),
                "source": str(source),
            }
        )

    def apply_cash_dividend_payment(self, ticker: str, usd_per_share: float) -> float:
        """Credit cash dividend for current long; *usd_per_share* = actual ex-div amount."""

        t = str(ticker)
        q = max(0.0, float(self.positions.get(t, 0.0)))
        pay = q * max(0.0, float(usd_per_share))
        if pay > 0.0 and math.isfinite(pay):
            self.cash += pay
        return float(pay)

    def apply_share_buyback_to_holdings(self, ticker: str, mult: float) -> None:
        """Scale down share counts after a float buyback; USD cost basis is unchanged."""

        m = float(mult)
        if not (math.isfinite(m) and 0.0 < m < 1.0 + 1e-9):
            return
        t = str(ticker)
        for d in (self.positions, self.locked_sell):
            if t in d:
                d[t] = float(d[t]) * m
                self._trim(d, t)
        if t in self.last_trade_qty:
            v = float(self.last_trade_qty[t]) * m
            if abs(v) < MIN_LOT:
                self.last_trade_qty.pop(t, None)
            else:
                self.last_trade_qty[t] = v

    def _set_last_trade(
        self,
        ticker: str,
        side: str,
        price: float,
        qty: float,
        sim_tick: int,
    ) -> None:
        self.last_trade_side[ticker] = str(side)
        self.last_trade_price[ticker] = float(price)
        self.last_trade_qty[ticker] = float(qty)
        self.last_trade_tick[ticker] = int(sim_tick)

    def apply_forward_split(self, ticker: str, ratio: float) -> None:
        """*ratio*:1 forward split: share qty × *ratio*; $ cost basis unchanged; last fill px/qty rescaled."""

        r = float(ratio)
        if not (math.isfinite(r) and r > 1.0 + 1e-9):
            return
        t = str(ticker)
        if t in self.positions:
            self.positions[t] = float(self.positions[t]) * r
            self._trim(self.positions, t)
        if t in self.locked_sell:
            self.locked_sell[t] = float(self.locked_sell[t]) * r
            self._trim(self.locked_sell, t)
        if t in self.last_trade_price:
            self.last_trade_price[t] = float(self.last_trade_price[t]) / r
        if t in self.last_trade_qty:
            self.last_trade_qty[t] = float(self.last_trade_qty[t]) * r

    def position_holdings(self, mkt: Market) -> list[dict[str, float | str | int | None]]:
        """Per open long: avg cost, marks, unrealized P/L, last fill metadata (for API / UI)."""

        by = mkt.by_ticker()
        out: list[dict[str, float | str | int | None]] = []
        for t, q in list(self.positions.items()):
            qq = float(q)
            if abs(qq) < MIN_LOT:
                continue
            ins = by.get(t)
            if ins is None:
                continue
            _, mid, _ = mkt.quote(ins.array_index)
            cb = float(self.cost_basis.get(t, 0.0))
            avg = cb / qq if abs(qq) > 1e-15 else 0.0
            mv = qq * float(mid)
            unreal = mv - cb
            unreal_pct = (100.0 * unreal / cb) if cb > 1e-9 else None
            out.append(
                {
                    "ticker": t,
                    "qty": qq,
                    "avg_cost": float(avg),
                    "cost_basis_usd": float(cb),
                    "mark": float(mid),
                    "market_value_usd": float(mv),
                    "unrealized_usd": float(unreal),
                    "unrealized_pct": float(unreal_pct) if unreal_pct is not None else None,
                    "last_side": self.last_trade_side.get(t),
                    "last_price": self.last_trade_price.get(t),
                    "last_qty": self.last_trade_qty.get(t),
                    "last_tick": self.last_trade_tick.get(t),
                }
            )
        out.sort(key=lambda r: str(r["ticker"]))
        return out

    def _trim(self, d: dict[str, float], k: str) -> None:
        if abs(d.get(k, 0.0)) < MIN_LOT:
            d.pop(k, None)

    def free_shares(self, ticker: str) -> float:
        have = float(self.positions.get(ticker, 0.0))
        lock = float(self.locked_sell.get(ticker, 0.0))
        return max(0.0, have) - lock

    def gross_exposure_mtm(self, mkt: Market) -> float:
        """Gross notional: sum |q| * mark (|short| and long)."""

        by = mkt.by_ticker()
        t = 0.0
        for sym, q in self.positions.items():
            ins = by.get(str(sym))
            if ins is None:
                continue
            _, mid, _ = mkt.quote(ins.array_index)
            t += abs(float(q)) * max(0.0, float(mid))
        return float(t)

    def may_afford_buy_gross(self, mkt: Market, need: float) -> bool:
        """With margin (*max_leverage* > 1), allow *need* to push gross toward equity × leverage."""

        if need <= 0.0 or not math.isfinite(need):
            return True
        if self.max_leverage <= 1.0 + 1e-9:
            return self.cash + 1e-6 >= need
        e = self.mark_to_market(mkt)
        g = self.gross_exposure_mtm(mkt)
        return (g + need) <= e * self.max_leverage + 2.0

    def may_afford_sell(
        self, mkt: Market, tkr: str, size: float, dry_sell_notional: float
    ) -> bool:
        """If not shorting, require *free* long shares. Else allow increased gross up to *max_leverage* × equity."""

        t = str(tkr)
        if not self.shorting_enabled:
            return self.free_shares(t) + 1e-6 >= float(size)
        if self.free_shares(t) + 1e-6 >= float(size):
            return True
        ad = max(0.0, float(dry_sell_notional))
        if not math.isfinite(ad):
            return False
        e = self.mark_to_market(mkt)
        g = self.gross_exposure_mtm(mkt)
        lvr = max(1.0, float(self.max_leverage))
        return (g + ad) <= e * lvr + 2.0

    def is_margin_maintained(self, mkt: Market) -> bool:
        if self.max_leverage <= 1.0 + 1e-9 and not self.shorting_enabled:
            return True
        m = max(0.0, float(self.maintenance_margin_rate))
        e = self.mark_to_market(mkt)
        g = self.gross_exposure_mtm(mkt)
        if g < 1.0:
            return True
        return e + 1e-4 >= m * g

    def charge_borrow_on_shorts(
        self, mkt: Market, *, bps_per_sim_day: float, sim_minutes_per_tick: float
    ) -> None:
        """Backward-compatible alias: apply financing with a global short-borrow baseline."""

        self.charge_financing_on_positions(
            mkt,
            base_short_borrow_bps_per_sim_day=bps_per_sim_day,
            sim_minutes_per_tick=sim_minutes_per_tick,
        )

    def charge_financing_on_positions(
        self,
        mkt: Market,
        *,
        base_short_borrow_bps_per_sim_day: float,
        sim_minutes_per_tick: float,
    ) -> None:
        """
        Deduct carrying costs by side and instrument:
        - shorts: base borrow + market dynamic short funding
        - longs: market dynamic long funding (e.g. perp-style funding on crypto)
        """

        day_frac = max(1e-12, float(sim_minutes_per_tick) / 1440.0)
        base_short = max(0.0, float(base_short_borrow_bps_per_sim_day))
        by = mkt.by_ticker()
        for sym, q in list(self.positions.items()):
            qq = float(q)
            if abs(qq) < MIN_LOT:
                continue
            ins = by.get(str(sym))
            if ins is None:
                continue
            _, mid, _ = mkt.quote(ins.array_index)
            notional = abs(qq) * max(0.0, float(mid))
            if notional <= 0.0 or not math.isfinite(notional):
                continue
            long_bps, short_bps = mkt.financing_bps_for_index(ins.array_index)
            bps = float(long_bps) if qq > 0.0 else float(short_bps) + base_short
            if bps <= 0.0:
                continue
            fee = notional * bps / 10_000.0 * day_frac
            if fee > 0.0 and math.isfinite(fee):
                self.cash -= fee
                self.fees_paid += fee

    def charge_sec_fee_sell(self, notional: float) -> float:
        b = max(0.0, float(self.sec_fee_sell_bps)) / 10_000.0
        c = max(0.0, float(notional)) * b
        if c > 0.0:
            self.cash -= c
            self.fees_paid += c
        return float(c)

    def log_liquidation(
        self, *, sim_tick: int, ticker: str, qty: float, price: float, reason: str
    ) -> None:
        self.append_order_log(
            {
                "event": "liquidation",
                "sim_tick": int(sim_tick),
                "ticker": str(ticker),
                "qty": float(qty),
                "price": float(price),
                "reason": str(reason),
            }
        )

    def _flatten_largest_position_mkt(
        self, mkt: Market, *, sim_tick: int, rng: Any
    ) -> bool:
        """Close one position (long or cover short) at mid to restore margin. Returns if did work."""

        by = mkt.by_ticker()
        best_t = None
        best_m = 0.0
        for sym, q in self.positions.items():
            if abs(float(q)) < MIN_LOT:
                continue
            ins = by.get(str(sym))
            if ins is None:
                continue
            _, mid, _ = mkt.quote(ins.array_index)
            n = abs(float(q)) * max(0.0, float(mid))
            if n > best_m + 1e-9:
                best_m = n
                best_t = str(sym)
        if best_t is None:
            return False
        ins0 = by.get(best_t)
        if ins0 is None:
            return False
        _, mid, _ = mkt.quote(ins0.array_index)
        qv = float(self.positions[best_t])
        if qv > MIN_LOT:
            self._apply_sell_to_flat_long(best_t, float(mid), qv, sim_tick)
            self.log_liquidation(
                sim_tick=sim_tick, ticker=best_t, qty=abs(qv), price=float(mid), reason="maintenance"
            )
        elif qv < -MIN_LOT:
            self._apply_buy_leg_cover_short(best_t, float(mid), abs(qv), qv, sim_tick)
            self.log_liquidation(
                sim_tick=sim_tick, ticker=best_t, qty=abs(qv), price=float(mid), reason="maintenance"
            )
        return True

    def _apply_sell_to_flat_long(
        self, t: str, p: float, q: float, sim_tick: int
    ) -> None:
        old_q = float(self.positions.get(t, 0.0))
        old_cb = float(self.cost_basis.get(t, 0.0))
        self.cash += float(p) * float(q)
        v = old_q - float(q)
        if abs(v) < MIN_LOT:
            self.positions.pop(t, None)
            self.cost_basis.pop(t, None)
        else:
            self.positions[t] = v
            if old_q > 1e-12:
                self.cost_basis[t] = old_cb * (v / old_q)
            else:
                self.cost_basis[t] = 0.0
        self._set_last_trade(t, "sell", float(p), float(q), sim_tick)

    def total_cash_balance(self) -> float:
        return float(self.cash + self.locked_cash)

    def charge_taker_fees(self, bps: float, notional: float) -> float:
        """Deduct taker fee on *notional* (sum of |fill px × qty|). Returns fee charged (USD)."""

        c = max(0.0, float(notional)) * max(0.0, float(bps)) / 10_000.0
        if c > 0.0:
            self.cash -= c
            self.fees_paid += c
        return float(c)

    def apply_market_buy_fill(
        self, ticker: str, price: float, qty: float, *, sim_tick: int = -1
    ) -> None:
        t = str(ticker)
        p = float(price)
        q = float(qty)
        oq = float(self.positions.get(t, 0.0))
        if oq < -0.5 * MIN_LOT:
            # Cover short (possibly then open/extend long in same fill)
            cleg = min(q, -oq)
            rem = max(0.0, q - cleg)
            if cleg > QTY_EPS:
                self._apply_buy_leg_cover_short(t, p, cleg, oq, sim_tick)
            if rem > QTY_EPS:
                o2 = float(self.positions.get(t, 0.0))
                self._apply_buy_leg_long(t, p, rem, o2, sim_tick)
            return
        self._apply_buy_leg_long(t, p, q, oq, sim_tick)

    def _apply_buy_leg_long(
        self, t: str, p: float, q: float, oq: float, sim_tick: int
    ) -> None:
        if q <= QTY_EPS:
            return
        cost = p * q
        self.cash -= cost
        old_q = oq
        old_cb = float(self.cost_basis.get(t, 0.0))
        v = old_q + q
        if abs(v) < MIN_LOT:
            self.positions.pop(t, None)
            self.cost_basis.pop(t, None)
        else:
            self.positions[t] = v
            self.cost_basis[t] = old_cb + cost
        self._set_last_trade(t, "buy", p, q, sim_tick)

    def _apply_buy_leg_cover_short(
        self, t: str, p: float, cleg: float, oq: float, sim_tick: int
    ) -> None:
        if cleg <= QTY_EPS:
            return
        cost = p * cleg
        self.cash -= cost
        old_oq = oq
        v = oq + cleg
        old_cb = float(self.cost_basis.get(t, 0.0))
        if abs(v) < MIN_LOT:
            self.positions.pop(t, None)
            self.cost_basis.pop(t, None)
        else:
            self.positions[t] = v
            if v < -1e-12 and old_oq < -1e-12:
                self.cost_basis[t] = old_cb * (v / old_oq)
            else:
                self.cost_basis.pop(t, None)
        self._set_last_trade(t, "buy", p, cleg, sim_tick)

    def apply_market_sell_fill(
        self, ticker: str, price: float, qty: float, *, sim_tick: int = -1
    ) -> None:
        q = float(qty)
        t = str(ticker)
        p = float(price)
        free = self.free_shares(t)
        oq = float(self.positions.get(t, 0.0))
        if not self.shorting_enabled:
            if free < q - MIN_LOT * 0.5:
                raise ValueError("sell over free position (check locked for limits)")
            self._apply_sell_long_leg(t, p, q, oq, sim_tick)
            return
        if free + 1e-9 >= q:
            self._apply_sell_long_leg(t, p, q, oq, sim_tick)
            return
        rem = max(0.0, min(q, free))
        short_add = max(0.0, q - free)
        if rem > QTY_EPS:
            self._apply_sell_long_leg(t, p, rem, oq, sim_tick)
        if short_add > QTY_EPS:
            o2 = float(self.positions.get(t, 0.0))
            self._apply_sell_add_short_leg(t, p, short_add, o2, sim_tick)

    def _apply_sell_long_leg(
        self, t: str, p: float, q: float, old_q: float, sim_tick: int
    ) -> None:
        if q <= QTY_EPS:
            return
        old_cb = float(self.cost_basis.get(t, 0.0))
        self.cash += p * q
        v = old_q - q
        if abs(v) < MIN_LOT:
            self.positions.pop(t, None)
            self.cost_basis.pop(t, None)
        else:
            self.positions[t] = v
            if old_q > 1e-12:
                self.cost_basis[t] = old_cb * (v / old_q)
            else:
                self.cost_basis[t] = 0.0
        self._set_last_trade(t, "sell", p, q, sim_tick)

    def _apply_sell_add_short_leg(
        self, t: str, p: float, q: float, old_q: float, sim_tick: int
    ) -> None:
        if q <= QTY_EPS:
            return
        self.cash += p * q
        v = old_q - q
        old_cb = float(self.cost_basis.get(t, 0.0))
        if abs(v) < MIN_LOT:
            self.positions.pop(t, None)
            self.cost_basis.pop(t, None)
        else:
            self.positions[t] = v
            if v < -1e-12 and old_q < -1e-12:
                self.cost_basis[t] = old_cb * (v / old_q)
            elif v < -1e-12:
                self.cost_basis[t] = -abs(v) * p
        self._set_last_trade(t, "sell", p, q, sim_tick)

    def try_reserve_buy(self, limit_price: float, size: float) -> bool:
        need = float(limit_price) * float(size)
        if self.cash < need - 1e-9:
            return False
        self.cash -= need
        self.locked_cash += need
        return True

    def apply_bought_from_limit(
        self,
        ticker: str,
        exec_price: float,
        fill_qty: float,
        limit_price: float,
        *,
        sim_tick: int = -1,
    ) -> None:
        L = float(limit_price)
        e = float(exec_price)
        q = float(fill_qty)
        self.locked_cash -= L * q
        self.cash += (L - e) * q
        old_q = float(self.positions.get(ticker, 0.0))
        old_cb = float(self.cost_basis.get(ticker, 0.0))
        v = old_q + q
        if abs(v) < MIN_LOT:
            self.positions.pop(ticker, None)
            self.cost_basis.pop(ticker, None)
        else:
            self.positions[ticker] = v
            self.cost_basis[ticker] = old_cb + e * q
        self._set_last_trade(ticker, "buy", e, q, sim_tick)

    def try_lock_shares_for_sell(self, ticker: str, size: float) -> bool:
        if self.free_shares(ticker) < float(size) - MIN_LOT * 0.5:
            return False
        s = float(size)
        self.locked_sell[ticker] = self.locked_sell.get(ticker, 0.0) + s
        return True

    def apply_sold_from_limit(
        self,
        ticker: str,
        exec_price: float,
        fill_qty: float,
        *,
        sim_tick: int = -1,
    ) -> None:
        e = float(exec_price)
        q = float(fill_qty)
        lock = self.locked_sell.get(ticker, 0.0) - q
        if lock < -1e-6:
            raise ValueError("sell lock underflow")
        if abs(lock) < MIN_LOT:
            self.locked_sell.pop(ticker, None)
        else:
            self.locked_sell[ticker] = lock
        old_q = float(self.positions.get(ticker, 0.0))
        old_cb = float(self.cost_basis.get(ticker, 0.0))
        v = old_q - q
        if abs(v) < MIN_LOT:
            self.positions.pop(ticker, None)
            self.cost_basis.pop(ticker, None)
        else:
            self.positions[ticker] = v
            if old_q > 1e-12:
                self.cost_basis[ticker] = old_cb * (v / old_q)
            else:
                self.cost_basis[ticker] = 0.0
        self.cash += e * q
        self._set_last_trade(ticker, "sell", e, q, sim_tick)

    def mark_to_market(self, mkt: Market) -> float:
        total = self.cash + self.locked_cash
        by = mkt.by_ticker()
        for t, q in self.positions.items():
            qq = float(q)
            if abs(qq) < MIN_LOT:
                continue
            ins = by.get(t)
            if ins is None:
                continue
            _, mid, _ = mkt.quote(ins.array_index)
            total += qq * mid
        return float(total)
