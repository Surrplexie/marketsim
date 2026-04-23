from __future__ import annotations

import math
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from .amounts import MIN_LOT, is_valid_cash_notional, is_valid_order_size
from .engine import Session, new_session
from .instrument import AssetClass
from .market import Side
from .modes import GameModeName, preset
from .player import Result

# One in-memory game per server process; replace with a store if you add auth / multi-tenant
_game: Session | None = None
_step_lock = threading.Lock()

# Cap rows returned in /api/state to keep JSON payloads bounded (full history stays in memory).
ORDER_LOG_STATE_MAX = 400
# New-game starting bankroll (USD); independent of order *MIN_NOTIONAL_USD*.
STARTING_CASH_MIN_USD = 1e-4
STARTING_CASH_MAX_USD = 100_000_000.0


def _lot_step_for_kind(kind: AssetClass) -> float:
    return 1e-6 if kind is AssetClass.CRYPTO else 1e-2


def _tick_size_for_kind(kind: AssetClass, price: float) -> float:
    p = max(0.0, float(price))
    if kind is AssetClass.CRYPTO:
        if p < 1.0:
            return 1e-5
        if p < 100.0:
            return 1e-4
        return 1e-2
    if p < 1.0:
        return 1e-4
    if p < 10.0:
        return 1e-3
    return 1e-2


def get_game() -> Session:
    global _game
    if _game is None:
        _game = new_session(mode=GameModeName.SIMPLE)
    return _game


def set_game(s: Session | None) -> None:
    global _game
    _game = s


def _parse_cash_leg(body: dict) -> str | float | None:
    """
    *None* = order by *size* (shares). *\"all\"* = use all free *cash* (buy). *float* = USD cap.
    """

    if bool(body.get("all_cash")) or bool(body.get("use_all_cash")):
        return "all"
    for k in ("notional", "cash", "cash_usd"):
        if k not in body or body[k] in (None, ""):
            continue
        v = body[k]
        if isinstance(v, str) and v.strip().lower() in ("all", "max", "total"):
            return "all"
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def _result_msg(r: Result) -> str:
    return {
        Result.OK: "ok",
        Result.NO_CASH: "insufficient cash",
        Result.NO_POSITION: "not enough position",
        Result.NOT_FOUND: "unknown ticker",
        Result.BAD_SIZE: "invalid size (fails lot step / min lot) or cash amount (min notional ~1e-6)",
        Result.BAD_PRICE: "bad price (or not aligned to symbol tick size)",
        Result.NO_LIQUIDITY: "not enough size on the book (asks for buys, bids for sells)",
    }.get(r, str(r))


def _state() -> dict[str, Any]:
    s = get_game()
    m = s.market
    p = s.player
    out_ins = []
    gds = m.great_depression_state()
    for ins in m.instruments:
        i = ins.array_index
        b, mid, a = m.quote(i)
        ob = m.books[i]
        ch = m.pct_change_24h_sim(i)
        wick = m.wick_context(i)
        tap = m.micro_tape(i)
        out_ins.append(
            {
                "ticker": ins.ticker,
                "kind": ins.kind.value,
                "sector": ins.sector,
                # Keep watchlist Mcap consistent with chart Mcap mode: listed mid × float.
                "marketcap": float(mid) * float(ins.units_outstanding),
                "units_outstanding": ins.units_outstanding,
                "max_units_outstanding": ins.max_units_outstanding,
                "allow_supply_inflation": bool(ins.allow_supply_inflation),
                "cumulative_volume": m.cumulative_volume(i),
                "ask_liquidity": float(ob.total_resting_qty_asks()),
                "bid_liquidity": float(ob.total_resting_qty_bids()),
                "quote": {"bid": b, "mid": mid, "ask": a},
                "execution_rules": {
                    "lot_step": _lot_step_for_kind(ins.kind),
                    "tick_size": _tick_size_for_kind(ins.kind, mid),
                },
                "financing_bps_per_sim_day": {
                    "long": m.financing_bps_for_index(i)[0],
                    "short": m.financing_bps_for_index(i)[1],
                },
                "book": {"bids": ob.depth_bids(4), "asks": ob.depth_asks(4)},
                "pct_24h": ch,
                "micro": {
                    "tape": tap,
                    "wick": wick,
                },
            }
        )
    mglob = m.micro_global()
    holdings = p.position_holdings(m)
    h_unreal = float(sum(float(h["unrealized_usd"]) for h in holdings))
    olog = list(p.order_log)
    if len(olog) > ORDER_LOG_STATE_MAX:
        olog = olog[-ORDER_LOG_STATE_MAX:]
    return {
        "mode": s.config.label,
        "order_log": olog,
        "great_depression": bool(gds["armed"]),
        "depression": gds,
        "misc": {"chaos": m.chaos_settings()},
        "time": {
            "sim_minutes_per_tick": s.config.sim_minutes_per_tick,
            "stock_fund_annual_return": s.config.stock_fund_annual_return,
            "crypto_top_tier": s.config.crypto_top_tier,
            "ticks_per_sim_24h": m.ticks_per_sim_24h,
            "pct_24h_window_warmed": m.ch24_warmed,
            "sim_minute_of_day": mglob.get("sim_minute_of_day") if isinstance(mglob, dict) else None,
            "taker_fee_bps": s.config.taker_fee_bps,
            "slippage_bps_base": s.config.slippage_bps_base,
            "slippage_bps_per_million": s.config.slippage_bps_per_million,
            "front_run_bps": s.config.front_run_bps,
            "front_run_notional_usd": s.config.front_run_notional_usd,
            "overnight_gap_max_bps": s.config.overnight_gap_max_bps,
        },
        "micro": mglob,
        "tick": m.tick,
        "player": {
            "cash": p.cash,
            "locked_cash": p.locked_cash,
            "total_cash": p.total_cash_balance(),
            "equity": s.equity,
            "positions": dict(p.positions),
            "locked_sell": dict(p.locked_sell),
            "fees_paid": p.fees_paid,
            "holdings": holdings,
            "holdings_unrealized_usd": h_unreal,
            "margin": {
                "max_leverage": p.max_leverage,
                "maintenance_rate": p.maintenance_margin_rate,
                "shorting": p.shorting_enabled,
                "borrow_bps_per_sim_day": p.short_borrow_bps_per_sim_day,
                "sec_fee_sell_bps": p.sec_fee_sell_bps,
                "gross_mtm": p.gross_exposure_mtm(m),
                "maintenance_ok": p.is_margin_maintained(m),
            },
        },
        "trend_override": m.trend_override_state(),
        "volatility_override": m.volatility_override_state(),
        "news": m.news_feed(),
        "instruments": out_ins,
    }


def _read_gui_html() -> str:
    p = Path(__file__).resolve().parent / "static" / "index.html"
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return (
            "<!DOCTYPE html><html><body><h1>marketsim</h1>"
            "<p>GUI file not found. Use an editable install or add "
            "<code>marketsim/static/index.html</code>."
            "</p><a href=\"/api/state\">/api/state</a></body></html>"
        )


app = FastAPI(title="marketsim", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return _read_gui_html()


@app.get("/api/state", response_class=JSONResponse)
def state() -> Any:
    return _state()


@app.post("/api/volatility-override", response_class=JSONResponse)
def volatility_override(body: dict | None = Body(default=None)) -> Any:
    b = body or {}
    s = get_game()
    s.market.set_volatility_override(int(b.get("value", 0) or 0))
    return _state()


@app.post("/api/trend-override", response_class=JSONResponse)
def trend_override(body: dict | None = Body(default=None)) -> Any:
    b = body or {}
    s = get_game()
    v = int(b.get("value", 0) or 0)
    sc = str(b.get("scope", "all") or "all")
    raw = b.get("tickers")
    if isinstance(raw, str) and raw.strip():
        tlist = [x.strip() for x in raw.replace(",", " ").split() if x.strip()]
    elif isinstance(raw, list):
        tlist = [str(x).strip() for x in raw if str(x).strip()]
    else:
        tlist = None
    s.market.set_trend_override(v, sc, tlist)
    return _state()


@app.post("/api/step", response_class=JSONResponse)
def step(body: dict | None = Body(default=None)) -> Any:
    b = body or {}
    s = get_game()
    with _step_lock:
        if b.get("unit") is not None and b.get("n") is not None:
            n = int(b["n"])
            unit = str(b["unit"])
            adv = s.advance_interval(n, unit)
            st = _state()
            st["advance"] = {
                "n": n,
                "unit": unit,
                "steps": adv.ticks,
                "sim_minutes": adv.sim_minutes,
            }
            return st
        n2 = int(b.get("ticks", 1) or 1)
        n2 = max(1, n2)
        # Single request cap only to keep the server responsive; re-post to continue.
        for _ in range(min(n2, 100_000_000)):
            s.step()
        return _state()


@app.post("/api/order", response_class=JSONResponse)
def post_order(body: dict = Body(...)) -> Any:
    s = get_game()
    mkt = s.market
    tck = str(body.get("ticker", "")).upper()
    if not tck:
        raise HTTPException(400, "ticker")
    otype = str(body.get("type", "market")).lower()
    ot = "limit" if otype == "limit" else "market"
    sdir = str(body.get("side", "buy")).lower()
    side = Side.BUY if sdir in ("b", "buy", "bid") else Side.SELL
    sside = "buy" if side is Side.BUY else "sell"
    sim_tick = int(mkt.tick)
    cash_leg = _parse_cash_leg(body)
    sizing = str(body.get("sizing", "")).lower()
    if sizing in ("cash", "notional", "usd", "money") and cash_leg is None:
        raise HTTPException(400, "sizing is cash: pass notional (or all_cash / use_all_cash)")
    if cash_leg is not None and side is not Side.BUY:
        raise HTTPException(400, "notional / cash is only for buy orders")
    if cash_leg is not None:
        budget: float
        if cash_leg == "all":
            budget = float("inf")
        else:
            budget = float(cash_leg)
            if not is_valid_cash_notional(budget):
                raise HTTPException(
                    400,
                    "notional: positive USD amount (min ~1e-6), or use all_cash",
                )
        if otype == "limit":
            pr = float(body.get("price", 0) or 0)
            r = s.order_limit_buy_cash(tck, pr, budget)
        else:
            r = s.order_market_buy_cash(tck, budget)
    else:
        try:
            size = float(body.get("size", 0) or 0)
        except (TypeError, ValueError):
            raise HTTPException(400, "size") from None
        if not is_valid_order_size(size):
            raise HTTPException(
                400, "size: use a positive amount (fractional ok, min ~1e-8), or use notional for cash"
            )
        if otype == "limit":
            pr = float(body.get("price", 0) or 0)
            r = s.order_limit(tck, side, size, pr)
        else:
            r = s.order_market(tck, side, size)
    st = _state()
    st["order_result"] = r.value
    st["ok"] = r is Result.OK
    if r is not Result.OK:
        msg = _result_msg(r)
        st["error"] = msg
        s.player.log_reject(
            sim_tick=sim_tick,
            ticker=tck,
            order_type=ot,
            side=sside,
            result=str(r.value),
            message=msg,
        )
    return st


@app.post("/api/flatten", response_class=JSONResponse)
def post_flatten(body: dict | None = Body(default=None)) -> Any:
    """Flatten one ticker or all open positions via market orders."""

    b = body or {}
    s = get_game()
    mkt = s.market
    t_raw = str(b.get("ticker", "") or "").strip().upper()
    only = [t_raw] if t_raw else [str(t) for t, q in s.player.positions.items() if abs(float(q)) >= MIN_LOT]
    results: list[dict[str, Any]] = []
    for t in only:
        q = float(s.player.positions.get(t, 0.0))
        if abs(q) < MIN_LOT:
            continue
        side = Side.SELL if q > 0.0 else Side.BUY
        r = s.order_market(t, side, abs(q))
        results.append({"ticker": t, "side": "sell" if side is Side.SELL else "buy", "qty": abs(q), "result": r.value})
        if r is not Result.OK:
            s.player.log_reject(
                sim_tick=int(mkt.tick),
                ticker=t,
                order_type="market",
                side="sell" if side is Side.SELL else "buy",
                result=str(r.value),
                message=_result_msg(r),
            )
    st = _state()
    st["flatten"] = {"requested": ("one" if t_raw else "all"), "results": results}
    return st


@app.post("/api/stock_split", response_class=JSONResponse)
def post_stock_split(body: dict = Body(...)) -> Any:
    """Manual forward stock split (N:1) for a chosen ticker; 2 ≤ N ≤ 100. Stocks only."""

    s = get_game()
    m = s.market
    tck = str(body.get("ticker", "")).strip().upper()
    if not tck:
        raise HTTPException(400, "ticker required")
    ins = m.by_ticker().get(tck)
    if ins is None:
        raise HTTPException(404, "unknown ticker")
    if ins.kind is not AssetClass.STOCK:
        raise HTTPException(400, "only common stocks (not funds or crypto)")
    try:
        ratio = float(body.get("ratio", 0))
    except (TypeError, ValueError) as e:
        raise HTTPException(400, "ratio must be a number (2–100 for N:1 forward)") from e
    if not math.isfinite(ratio):
        raise HTTPException(400, "ratio must be finite")
    n = int(round(ratio))
    if n < 2 or n > 100:
        raise HTTPException(400, "ratio must be between 2 and 100 (N:1 forward split)")
    j = ins.array_index
    if m.apply_forward_split_for_index(j, float(n)):
        s.player.apply_forward_split(tck, float(n))
        s.player.log_split(
            sim_tick=int(m.tick), ticker=tck, ratio=float(n), source="manual"
        )
        st = _state()
        st["split_applied"] = {"ticker": tck, "ratio": n}
        return st
    raise HTTPException(500, "split could not be applied")


@app.post("/api/stock_dividend", response_class=JSONResponse)
def post_stock_dividend(body: dict = Body(...)) -> Any:
    """Manual cash dividend (USD per share) for a *stock*; ex-div price drop. Not for funds or crypto."""

    s = get_game()
    m = s.market
    tck = str(body.get("ticker", "")).strip().upper()
    if not tck:
        raise HTTPException(400, "ticker required")
    ins = m.by_ticker().get(tck)
    if ins is None:
        raise HTTPException(404, "unknown ticker")
    if ins.kind is not AssetClass.STOCK:
        raise HTTPException(400, "only common stocks (not funds or crypto)")
    try:
        dps = float(body.get("usd_per_share", 0) or 0)
    except (TypeError, ValueError) as e:
        raise HTTPException(400, "usd_per_share must be a number") from e
    if not math.isfinite(dps):
        raise HTTPException(400, "usd_per_share must be finite")
    if dps < 1e-9 or dps > 1_000_000.0:
        raise HTTPException(400, "usd_per_share out of range (roughly 1e-9 … 1e6)")
    j = ins.array_index
    ok, d = m.apply_cash_dividend_for_index(j, dps)
    if not ok or d <= 0.0:
        raise HTTPException(
            400,
            "dividend could not be applied (exceeds price floor or not a valid amount)",
        )
    cash = s.player.apply_cash_dividend_payment(tck, d)
    s.player.log_dividend(
        sim_tick=int(m.tick),
        ticker=tck,
        usd_per_share=d,
        cash_received=cash,
        source="manual",
    )
    st = _state()
    st["dividend_applied"] = {
        "ticker": tck,
        "usd_per_share": d,
        "cash_received": cash,
    }
    return st


@app.post("/api/stock_buyback", response_class=JSONResponse)
def post_stock_buyback(body: dict = Body(...)) -> Any:
    """Retire a fraction of the common-stock float; scales player shares and the CLOB. Stocks only."""

    s = get_game()
    m = s.market
    tck = str(body.get("ticker", "")).strip().upper()
    if not tck:
        raise HTTPException(400, "ticker required")
    ins = m.by_ticker().get(tck)
    if ins is None:
        raise HTTPException(404, "unknown ticker")
    if ins.kind is not AssetClass.STOCK:
        raise HTTPException(400, "only common stocks (not funds or crypto)")
    try:
        frac = float(
            body.get("float_fraction", body.get("fraction", body.get("frac", 0)))
            or 0
        )
    except (TypeError, ValueError) as e:
        raise HTTPException(400, "float_fraction must be a number") from e
    if not math.isfinite(frac):
        raise HTTPException(400, "float_fraction must be finite")
    if frac < 0.0005 or frac > 0.2:
        raise HTTPException(400, "float_fraction must be between 0.0005 and 0.2 (0.05%–20%)")
    j = ins.array_index
    if m.apply_share_buyback_for_index(j, frac):
        s.player.apply_share_buyback_to_holdings(tck, 1.0 - frac)
        s.player.log_buyback(
            sim_tick=int(m.tick),
            ticker=tck,
            float_fraction=frac,
            units_outstanding=float(ins.units_outstanding),
            source="manual",
        )
        st = _state()
        st["buyback_applied"] = {
            "ticker": tck,
            "float_fraction": frac,
            "units_outstanding": float(ins.units_outstanding),
        }
        return st
    raise HTTPException(500, "buyback could not be applied")


def _parse_mode(m: str) -> GameModeName:
    m = m.lower()
    for e in GameModeName:
        if e.value == m:
            return e
    return GameModeName.SIMPLE


@app.post("/api/reset", response_class=JSONResponse)
def reset(body: dict | None = Body(default=None)) -> Any:
    body = body or {}
    prev_cfg = get_game().config
    mode = str(body.get("mode", "simple") or "simple")
    c = preset(_parse_mode(mode))
    # Preserve launch-time universe sizes (e.g. CLI overrides) unless caller explicitly overrides.
    c = replace(
        c,
        n_stocks=int(prev_cfg.n_stocks),
        n_funds=int(prev_cfg.n_funds),
        n_crypto=int(prev_cfg.n_crypto),
    )
    raw_n_stocks = body.get("n_stocks", body.get("nStocks"))
    if raw_n_stocks is not None and str(raw_n_stocks).strip() != "":
        try:
            n_stocks = int(raw_n_stocks)
        except (TypeError, ValueError) as e:
            raise HTTPException(400, "n_stocks must be an integer") from e
        c = replace(c, n_stocks=max(0, n_stocks))
    raw_n_funds = body.get("n_funds", body.get("nFunds"))
    if raw_n_funds is not None and str(raw_n_funds).strip() != "":
        try:
            n_funds = int(raw_n_funds)
        except (TypeError, ValueError) as e:
            raise HTTPException(400, "n_funds must be an integer") from e
        c = replace(c, n_funds=max(0, n_funds))
    raw_n_crypto = body.get("n_crypto", body.get("nCrypto"))
    if raw_n_crypto is not None and str(raw_n_crypto).strip() != "":
        try:
            n_crypto = int(raw_n_crypto)
        except (TypeError, ValueError) as e:
            raise HTTPException(400, "n_crypto must be an integer") from e
        c = replace(c, n_crypto=max(0, n_crypto))
    if bool(body.get("great_depression")):
        c = replace(c, great_depression=True)
    chaos_map = {
        "chaos_flash_crash": "chaos_flash_crash",
        "chaos_meme_squeeze": "chaos_meme_squeeze",
        "chaos_fat_finger": "chaos_fat_finger",
        "chaos_exchange_halt": "chaos_exchange_halt",
        "chaos_rumor_mill": "chaos_rumor_mill",
        "chaos_sector_rotation": "chaos_sector_rotation",
        "chaos_funding_panic": "chaos_funding_panic",
        "chaos_liquidity_drought": "chaos_liquidity_drought",
        "chaos_whale_rebalance": "chaos_whale_rebalance",
        "chaos_crypto_weekend_mania": "chaos_crypto_weekend_mania",
    }
    for req_key, cfg_key in chaos_map.items():
        if req_key in body:
            c = replace(c, **{cfg_key: bool(body.get(req_key))})
    raw_sc = body.get("starting_cash", body.get("startingCash"))
    if raw_sc is not None and str(raw_sc).strip() != "":
        try:
            sc = float(raw_sc)
        except (TypeError, ValueError) as e:
            raise HTTPException(400, "starting_cash must be a number") from e
        if not math.isfinite(sc):
            raise HTTPException(400, "starting_cash must be finite")
        if sc < STARTING_CASH_MIN_USD or sc > STARTING_CASH_MAX_USD:
            raise HTTPException(
                400,
                f"starting_cash must be between {STARTING_CASH_MIN_USD} and {STARTING_CASH_MAX_USD} USD",
            )
        c = replace(c, starting_cash=float(sc))
    raw_shorting = body.get("shorting", body.get("shorting_enabled"))
    if raw_shorting is not None:
        c = replace(c, shorting_enabled=bool(raw_shorting))
    raw_lev = body.get("max_leverage", body.get("maxLeverage"))
    if raw_lev is not None and str(raw_lev).strip() != "":
        try:
            lev = float(raw_lev)
        except (TypeError, ValueError) as e:
            raise HTTPException(400, "max_leverage must be a number") from e
        if not math.isfinite(lev):
            raise HTTPException(400, "max_leverage must be finite")
        if lev < 1.0 or lev > 10.0:
            raise HTTPException(400, "max_leverage must be between 1.0 and 10.0")
        c = replace(c, max_leverage=lev)
    raw_maint = body.get("maintenance_rate", body.get("maintenance"))
    if raw_maint is not None and str(raw_maint).strip() != "":
        try:
            maint = float(raw_maint)
        except (TypeError, ValueError) as e:
            raise HTTPException(400, "maintenance_rate must be a number") from e
        if not math.isfinite(maint):
            raise HTTPException(400, "maintenance_rate must be finite")
        if maint < 0.0 or maint > 0.95:
            raise HTTPException(400, "maintenance_rate must be between 0.0 and 0.95")
        c = replace(c, maintenance_margin_rate=maint)
    raw_borrow = body.get("short_borrow_bps_per_sim_day", body.get("short_borrow_bps"))
    if raw_borrow is not None and str(raw_borrow).strip() != "":
        try:
            bbps = float(raw_borrow)
        except (TypeError, ValueError) as e:
            raise HTTPException(400, "short_borrow_bps_per_sim_day must be a number") from e
        if not math.isfinite(bbps):
            raise HTTPException(400, "short_borrow_bps_per_sim_day must be finite")
        if bbps < 0.0 or bbps > 5000.0:
            raise HTTPException(400, "short_borrow_bps_per_sim_day must be between 0 and 5000")
        c = replace(c, short_borrow_bps_per_sim_day=bbps)
    set_game(new_session(custom=c))
    return _state()


@app.post("/api/starting-cash", response_class=JSONResponse)
def post_starting_cash(body: dict = Body(...)) -> Any:
    """
    Set bankroll while **sim tick is still 0** and the book is flat (no positions, no locks).
    Does not rebuild the universe (use *POST /api/reset* for a full new game).
    """

    s = get_game()
    if int(s.market.tick) != 0:
        raise HTTPException(400, "starting cash can only be changed at sim tick 0 (before any time step)")
    p = s.player
    for q in p.positions.values():
        if abs(float(q)) > float(MIN_LOT) * 100:
            raise HTTPException(
                400,
                "starting cash cannot be changed after opening positions (still tick 0)",
            )
    if float(p.locked_cash) > 1e-6:
        raise HTTPException(400, "cancel or fill open limit buys (locked cash) before changing starting cash")
    if any(abs(float(v)) > float(MIN_LOT) * 100 for v in p.locked_sell.values()):
        raise HTTPException(400, "clear locked sells before changing starting cash")
    raw = body.get("cash", body.get("starting_cash", body.get("startingCash")))
    try:
        v = float(raw)
    except (TypeError, ValueError) as e:
        raise HTTPException(400, "cash must be a number") from e
    if not math.isfinite(v):
        raise HTTPException(400, "cash must be finite")
    if v < STARTING_CASH_MIN_USD or v > STARTING_CASH_MAX_USD:
        raise HTTPException(
            400,
            f"cash must be between {STARTING_CASH_MIN_USD} and {STARTING_CASH_MAX_USD} USD",
        )
    p.cash = float(v)
    s.config = replace(s.config, starting_cash=float(v))
    return _state()


@app.get("/api/chart/{ticker}", response_class=JSONResponse)
def chart_series(
    ticker: str,
    bucket: int = 4,
    max_bars: int = 0,
) -> Any:
    """OHLC candles from stored quote mids; *bucket* = sim samples per candle (non-overlapping).
    *max_bars* 0 = return full history for this session; *max_bars* > 0 caps the latest N bars.
    """

    s = get_game()
    m = s.market
    tck = ticker.strip().upper()
    ins = m.by_ticker().get(tck)
    if ins is None:
        raise HTTPException(404, "unknown ticker")
    b = max(1, min(int(bucket), 35_040))
    mb = int(max_bars)
    if mb < 0:
        mb = 0
    elif mb > 0:
        mb = min(mb, 100_000_000)
    candles = m.chart_ohlc(ins.array_index, b, max_bars=mb)
    u = max(0.0, float(ins.units_outstanding))
    if u > 0.0 and candles:
        candles_mcap = [
            {
                "time": int(c["time"]),
                "open": float(c["open"]) * u,
                "high": float(c["high"]) * u,
                "low": float(c["low"]) * u,
                "close": float(c["close"]) * u,
                "volume": float(c.get("volume", 0.0)),
            }
            for c in candles
        ]
    else:
        candles_mcap = []
    return {
        "ticker": tck,
        "bucket": b,
        "sim_tick": m.tick,
        "candles": candles,
        "candles_mcap": candles_mcap,
        "units_outstanding": float(ins.units_outstanding),
    }


@app.get("/api/health", response_class=PlainTextResponse)
def health() -> str:
    return "ok"
