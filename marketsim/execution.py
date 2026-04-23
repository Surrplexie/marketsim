from __future__ import annotations

import copy
import math

from .amounts import MAX_NOTIONAL_USD, MIN_LOT, QTY_EPS, is_valid_cash_notional, is_valid_order_size
from .clob import BookFill
from .instrument import Instrument
from .market import Market, Side
from .microstruct import scale_fill_prices, slippage_bps_total, taker_notional_cash
from .player import Player, Result


def _side_s(side: Side) -> str:
    return "buy" if side is Side.BUY else "sell"


def _buying_power_mkt(mkt: Market, pl: Player) -> float:
    """Dollar budget for taker *buys* (margin increases effective buying power)."""

    if pl.max_leverage <= 1.0 + 1e-9:
        return max(0.0, float(pl.cash))
    e = pl.mark_to_market(mkt)
    g = pl.gross_exposure_mtm(mkt)
    h = e * pl.max_leverage - g
    return max(0.0, min(float(MAX_NOTIONAL_USD), h + 1.0e-2))


def _cost_with_synthetic_ask(
    fills: list[BookFill], rem: float, syn_ask: float
) -> float:
    t = 0.0
    for f in fills:
        t += f.price * f.qty
    if rem > QTY_EPS:
        t += syn_ask * rem
    return t


def _dry_market_buy_cost(mkt: Market, i: int, size: float) -> float:
    liq = max(0.0, float(mkt.books[i].total_resting_qty_asks()))
    sz = min(float(size), liq)
    if sz < MIN_LOT - 1e-15:
        return 0.0
    _, _, s_ask = mkt.synthetic_quote(i)
    dry = copy.deepcopy(mkt.books[i])
    f_b, r_b = dry.match_market_buy(sz, taker_is_player=True)
    c = _cost_with_synthetic_ask(f_b, r_b, s_ask)
    mb = _slip_fr_mult_buy(mkt, c)
    return c * mb * (1.0 + float(mkt.taker_fee_bps) / 10_000.0)


def _slip_fr_mult_buy(mkt: Market, pre_slip_notional: float) -> float:
    sl = slippage_bps_total(
        pre_slip_notional,
        mkt.slippage_bps_base,
        mkt.slippage_bps_per_million,
    )
    fr = float(mkt.front_run_bps) if pre_slip_notional >= float(mkt.front_run_notional_usd) else 0.0
    return 1.0 + (sl + fr) / 10_000.0


def _slip_fr_mult_sell(mkt: Market, pre_slip_notional: float) -> float:
    sl = slippage_bps_total(
        pre_slip_notional,
        mkt.slippage_bps_base,
        mkt.slippage_bps_per_million,
    )
    fr = float(mkt.front_run_bps) if pre_slip_notional >= float(mkt.front_run_notional_usd) else 0.0
    return 1.0 - (sl + fr) / 10_000.0  # multiply prices (worse for seller)


def _max_affordable_market_buy_size(
    mkt: Market, i: int, budget: float, *, c_eps: float, max_shares: float | None = None
) -> float | None:
    """
    Max share count with total cost at most *budget*, only against **resting asks**
    (no synthetic infinite ask top-up).
    *c_eps* = cost to buy *MIN_LOT* (caller precomputes; if *budget* below that, return None).
    """

    _, _, s_ask = mkt.synthetic_quote(i)
    ob = mkt.books[i]
    liq = max(0.0, float(ob.total_resting_qty_asks()))
    if liq < MIN_LOT - 1e-12:
        return None
    p0 = float(ob.best_ask()) if ob.best_ask() is not None else float(s_ask)
    if p0 <= 0.0 or not math.isfinite(p0):
        return None
    if c_eps > budget + 1e-9:
        return None

    def cost_sz(sz: float) -> float:
        return _dry_market_buy_cost(mkt, i, sz)

    cap = float(max_shares) if max_shares is not None else liq
    cap = min(cap, liq)
    hi = min(max(budget / p0, MIN_LOT * 2.0), cap)
    guard = 0
    while hi < cap - 1e-15 and cost_sz(hi) <= budget + 1e-9 and guard < 60:
        hi = min(hi * 2.0, cap)
        guard += 1
    hi = min(hi, cap)

    lo = 0.0
    # Tighten upper bound: *cost* must not exceed *budget* (float-safe margin).
    b_limit = max(float(budget) * (1.0 - 1e-10), 0.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if cost_sz(mid) <= b_limit:
            lo = mid
        else:
            hi = mid
    if lo < MIN_LOT - 1e-12:
        return None
    while cost_sz(lo) > b_limit and lo > MIN_LOT * 1.5:
        lo *= 0.9999
    return lo


def execute_market_buy_cash(
    mkt: Market, pl: Player, ins: Instrument, budget: float
) -> Result:
    """Buy as much as **resting asks** and *budget* allow (no synthetic infinite ask)."""

    b = min(float(budget), _buying_power_mkt(mkt, pl), float(MAX_NOTIONAL_USD))
    if b <= 0.0:
        return Result.NO_CASH
    if not is_valid_cash_notional(b):
        return Result.BAD_SIZE
    i = ins.array_index
    liq = max(0.0, float(mkt.books[i].total_resting_qty_asks()))
    if liq < MIN_LOT - 1e-12:
        return Result.NO_LIQUIDITY
    lot = min(MIN_LOT, liq)
    c_lo = _dry_market_buy_cost(mkt, i, lot)
    if c_lo > b + 1e-6:
        return Result.NO_CASH
    size = _max_affordable_market_buy_size(mkt, i, b, c_eps=c_lo, max_shares=liq)
    if size is None:
        return Result.NO_CASH
    b_cap = b
    # Nudge *size* to use almost all of *b_cap* (the order budget, not full wallet when partial).
    for _ in range(36):
        size = min(size, liq)
        c = _dry_market_buy_cost(mkt, i, size)
        if c > b_cap + 1e-3:
            size *= (b_cap * (1.0 - 1e-10)) / max(c, 1e-9)
        elif c < b_cap - 0.25:
            t = 1.0 + min(0.02, 0.5 * (b_cap - c) / max(c, 1.0))
            size2 = min(min(size * t, size * 1.05), liq)
            if _dry_market_buy_cost(mkt, i, size2) <= b_cap * (1.0 - 1e-10):
                size = size2
        else:
            break
    for _ in range(24):
        size = min(size, liq)
        if size < MIN_LOT:
            return Result.NO_CASH
        c2 = _dry_market_buy_cost(mkt, i, size)
        if c2 <= b_cap * (1.0 - 1e-10):
            break
        size *= (b_cap * (1.0 - 1e-10)) / max(c2, 1e-9)
    size = min(size, liq)
    if not is_valid_order_size(size):
        return Result.BAD_SIZE
    return execute_market_order(mkt, pl, ins, Side.BUY, size)


def execute_limit_buy_cash(
    mkt: Market, pl: Player, ins: Instrument, limit_price: float, budget: float
) -> Result:
    """Limit buy: *size* = *budget* / *limit_price* (capped to free cash) so cost ≤ budget."""

    L = float(limit_price)
    if L <= 0 or not math.isfinite(L):
        return Result.BAD_PRICE
    b = min(float(budget), _buying_power_mkt(mkt, pl), float(MAX_NOTIONAL_USD))
    if b <= 0.0:
        return Result.NO_CASH
    if not is_valid_cash_notional(b):
        return Result.BAD_SIZE
    size = b / L
    if size < MIN_LOT - 1e-12:
        return Result.NO_CASH
    if not is_valid_order_size(size):
        return Result.BAD_SIZE
    return execute_limit_buy(mkt, pl, ins, size, L)


def execute_market_order(
    mkt: Market, pl: Player, ins: Instrument, side: Side, size: float
) -> Result:
    if not is_valid_order_size(size):
        return Result.BAD_SIZE
    i = ins.array_index
    st = int(mkt.tick)
    ob = mkt.books[i]
    s_b, _, s_ask = mkt.synthetic_quote(i)
    tkr = ins.ticker
    if side is Side.BUY:
        ask_liq = max(0.0, float(ob.total_resting_qty_asks()))
        size = min(size, ask_liq)
        if size < MIN_LOT - 1e-12:
            return Result.NO_LIQUIDITY
        dry = copy.deepcopy(ob)
        f_b, r_b = dry.match_market_buy(size, taker_is_player=True)
        n0 = taker_notional_cash(f_b, r_b, s_ask)
        mb = _slip_fr_mult_buy(mkt, n0)
        need = n0 * mb * (1.0 + float(mkt.taker_fee_bps) / 10_000.0)
        if not pl.may_afford_buy_gross(mkt, need):
            return Result.NO_CASH
        f, rem = ob.match_market_buy(size, taker_is_player=True)
        rest_ask: list[float] = [float(s_ask)]
        scale_fill_prices(f, rest_ask, mult=mb)
        s_use = rest_ask[0]
        ot = "market"
        s_str = _side_s(side)
        vol_sum = 0.0
        for x in f:
            qx = float(x.qty)
            vol_sum += qx
            pl.apply_market_buy_fill(tkr, x.price, qx, sim_tick=st)
            pl.log_fill(
                sim_tick=st,
                ticker=tkr,
                order_type=ot,
                side=s_str,
                price=float(x.price),
                qty=qx,
            )
        n1 = taker_notional_cash(f, 0.0, s_use)
        fee = pl.charge_taker_fees(float(mkt.taker_fee_bps), n1)
        if fee > 0.0:
            pl.log_fee(
                sim_tick=st,
                ticker=tkr,
                order_type=ot,
                side=s_str,
                notional_usd=n1,
                fee_usd=fee,
                taker_fee_bps=float(mkt.taker_fee_bps),
            )
        mkt.record_tape(i, n1)
        mkt.record_trade_volume(i, vol_sum)
    else:
        bid_liq = max(0.0, float(ob.total_resting_qty_bids()))
        if not pl.shorting_enabled:
            size = min(size, bid_liq)
            if size < MIN_LOT - 1e-12:
                return Result.NO_LIQUIDITY
        f_b, r_b = copy.deepcopy(ob).match_market_sell(size, taker_is_player=True)
        n0s = taker_notional_cash(f_b, r_b, s_b)
        ms = _slip_fr_mult_sell(mkt, n0s)
        apx = n0s * ms
        if not pl.may_afford_sell(mkt, tkr, size, apx):
            return Result.NO_POSITION
        f, rem = ob.match_market_sell(size, taker_is_player=True)
        rest_bid: list[float] = [float(s_b)]
        scale_fill_prices(f, rest_bid, mult=ms)
        s_use = rest_bid[0]
        ot = "market"
        s_str = _side_s(side)
        vol_sum = 0.0
        for x in f:
            qx = float(x.qty)
            vol_sum += qx
            pl.apply_market_sell_fill(tkr, x.price, qx, sim_tick=st)
            pl.log_fill(
                sim_tick=st,
                ticker=tkr,
                order_type=ot,
                side=s_str,
                price=float(x.price),
                qty=qx,
            )
        if rem > QTY_EPS:
            if (not pl.shorting_enabled) and pl.free_shares(tkr) < rem - 0.5 * MIN_LOT:
                return Result.NO_POSITION
            vol_sum += float(rem)
            pl.apply_market_sell_fill(tkr, s_use, rem, sim_tick=st)
            pl.log_fill(
                sim_tick=st,
                ticker=tkr,
                order_type=ot,
                side=s_str,
                price=float(s_use),
                qty=float(rem),
            )
        n1 = taker_notional_cash(f, rem, s_use)
        fee = pl.charge_taker_fees(float(mkt.taker_fee_bps), n1)
        if fee > 0.0:
            pl.log_fee(
                sim_tick=st,
                ticker=tkr,
                order_type=ot,
                side=s_str,
                notional_usd=n1,
                fee_usd=fee,
                taker_fee_bps=float(mkt.taker_fee_bps),
            )
        if pl.sec_fee_sell_bps > 0.0:
            pl.charge_sec_fee_sell(n1)
        mkt.record_tape(i, n1)
        mkt.record_trade_volume(i, vol_sum)
    return Result.OK


def execute_limit_buy(
    mkt: Market, pl: Player, ins: Instrument, size: float, limit_price: float
) -> Result:
    if not is_valid_order_size(size):
        return Result.BAD_SIZE
    L = float(limit_price)
    if L <= 0:
        return Result.BAD_PRICE
    tkr = ins.ticker
    if float(pl.positions.get(tkr, 0.0)) < -0.5 * MIN_LOT:
        return Result.BAD_SIZE
    if not pl.try_reserve_buy(L, size):
        return Result.NO_CASH
    i = ins.array_index
    st = int(mkt.tick)
    ob = mkt.books[i]
    f2, _rem2, _o = ob.add_limit_buy(L, size, taker_is_player=True)
    n_t = 0.0
    ot = "limit"
    s_str = "buy"
    for x in f2:
        n_t += float(x.price) * float(x.qty)
        pl.apply_bought_from_limit(tkr, x.price, x.qty, L, sim_tick=st)
        pl.log_fill(
            sim_tick=st,
            ticker=tkr,
            order_type=ot,
            side=s_str,
            price=float(x.price),
            qty=float(x.qty),
        )
    if n_t > 0.0:
        fee = pl.charge_taker_fees(float(mkt.taker_fee_bps), n_t)
        if fee > 0.0:
            pl.log_fee(
                sim_tick=st,
                ticker=tkr,
                order_type=ot,
                side=s_str,
                notional_usd=n_t,
                fee_usd=fee,
                taker_fee_bps=float(mkt.taker_fee_bps),
            )
        mkt.record_tape(i, n_t)
        mkt.record_trade_volume(i, float(sum(float(x.qty) for x in f2)))
    return Result.OK


def execute_limit_sell(
    mkt: Market, pl: Player, ins: Instrument, size: float, limit_price: float
) -> Result:
    if not is_valid_order_size(size):
        return Result.BAD_SIZE
    L = float(limit_price)
    if L <= 0:
        return Result.BAD_PRICE
    tkr = ins.ticker
    if not pl.try_lock_shares_for_sell(tkr, size):
        return Result.NO_POSITION
    ob = mkt.books[ins.array_index]
    st = int(mkt.tick)
    f2, _rem2, _o = ob.add_limit_sell(L, size, taker_is_player=True)
    n_t = 0.0
    ot = "limit"
    s_str = "sell"
    for x in f2:
        n_t += float(x.price) * float(x.qty)
        pl.apply_sold_from_limit(tkr, x.price, x.qty, sim_tick=st)
        pl.log_fill(
            sim_tick=st,
            ticker=tkr,
            order_type=ot,
            side=s_str,
            price=float(x.price),
            qty=float(x.qty),
        )
    if n_t > 0.0:
        fee = pl.charge_taker_fees(float(mkt.taker_fee_bps), n_t)
        if fee > 0.0:
            pl.log_fee(
                sim_tick=st,
                ticker=tkr,
                order_type=ot,
                side=s_str,
                notional_usd=n_t,
                fee_usd=fee,
                taker_fee_bps=float(mkt.taker_fee_bps),
            )
        mkt.record_tape(ins.array_index, n_t)
        if pl.sec_fee_sell_bps > 0.0 and n_t > 0.0:
            pl.charge_sec_fee_sell(n_t)
        mkt.record_trade_volume(ins.array_index, float(sum(float(x.qty) for x in f2)))
    return Result.OK
