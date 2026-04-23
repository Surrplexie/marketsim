from __future__ import annotations

from dataclasses import dataclass, field

from .execution import (
    execute_limit_buy,
    execute_limit_sell,
    execute_market_buy_cash,
    execute_limit_buy_cash,
    execute_market_order,
)
from .instrument import AssetClass, make_universe
from .market import Market, Side
from .modes import GameConfig, GameModeName, preset
from .player import Player, Result
from .sim_time import TimeAdvance, interval_to_ticks
from .sim_tr import new_rng

__all__ = ["Session", "new_session"]


@dataclass
class Session:
    """Holds *market*, *player*, and *config*; advances time; routes orders to the CLOB."""

    config: GameConfig
    market: Market
    player: Player
    _rng: object
    _sim_day_ordinal: int = field(default=0, init=False, repr=False)

    def step(self) -> None:
        mpt = self.config.sim_minutes_per_tick
        t0 = int(self.market.tick)
        d_cur = int(t0 * mpt // 1440) if t0 > 0 else 0
        if (
            d_cur > self._sim_day_ordinal
            and self._sim_day_ordinal >= 0
            and self.config.overnight_gap_max_bps > 0.0
        ):
            self.market.apply_overnight_gaps_bps(
                self._rng, self.config.overnight_gap_max_bps
            )
        self._sim_day_ordinal = max(0, d_cur)
        self.market.step()
        self._maybe_liquidation()
        if self.player.short_borrow_bps_per_sim_day > 0.0:
            self.player.charge_borrow_on_shorts(
                self.market,
                bps_per_sim_day=self.player.short_borrow_bps_per_sim_day,
                sim_minutes_per_tick=self.config.sim_minutes_per_tick,
            )
        self._maybe_random_stock_splits()
        self._maybe_random_dividend_buyback()

    def _maybe_liquidation(self) -> None:
        m, p = self.market, self.player
        st = int(m.tick)
        if p.is_margin_maintained(m):
            return
        for _ in range(24):
            if p.is_margin_maintained(m):
                break
            if not p._flatten_largest_position_mkt(m, sim_tick=st, rng=self._rng):
                break

    def _maybe_random_dividend_buyback(self) -> None:
        """Random cash dividends and share buybacks on *stock* names only; funds & crypto skip."""

        m = self.market
        p = self.player
        p_div = 6.0e-5
        p_bb = 3.0e-5
        for j, ins in enumerate(m.instruments):
            if ins.kind is not AssetClass.STOCK:
                continue
            mid = float(m._mids[j])
            tkr = ins.ticker
            st = int(m.tick)
            if self._rng.random() < p_div:
                dps = mid * float(self._rng.uniform(0.0004, 0.0025))
                ok, d = m.apply_cash_dividend_for_index(j, dps)
                if ok and d > 0.0:
                    cash = p.apply_cash_dividend_payment(tkr, d)
                    p.log_dividend(
                        sim_tick=st,
                        ticker=tkr,
                        usd_per_share=d,
                        cash_received=cash,
                        source="auto",
                    )
            if self._rng.random() < p_bb:
                f = float(self._rng.uniform(0.0008, 0.012))
                if m.apply_share_buyback_for_index(j, f):
                    mlt = 1.0 - f
                    p.apply_share_buyback_to_holdings(tkr, mlt)
                    p.log_buyback(
                        sim_tick=st,
                        ticker=tkr,
                        float_fraction=f,
                        units_outstanding=float(ins.units_outstanding),
                        source="auto",
                    )

    def _maybe_random_stock_splits(self) -> None:
        """Random forward splits for *stock* tickers with mid in [1200, 2000] USD; funds & crypto skip."""

        m = self.market
        lo, hi = 1200.0, 2000.0
        p_split = 5.0e-5
        for j, ins in enumerate(m.instruments):
            if ins.kind is not AssetClass.STOCK:
                continue
            mid = float(m._mids[j])
            if not (lo <= mid <= hi):
                continue
            if self._rng.random() >= p_split:
                continue
            ratio = float(self._rng.choice([2, 2, 2, 2, 2, 3, 3, 4, 5]))
            if m.apply_forward_split_for_index(j, ratio):
                self.player.apply_forward_split(ins.ticker, ratio)
                self.player.log_split(
                    sim_tick=int(m.tick),
                    ticker=ins.ticker,
                    ratio=ratio,
                    source="auto",
                )

    def advance_interval(self, count: int, unit: str) -> TimeAdvance:
        """Move sim clock forward by *count* *unit* (see *parse_run_line* for spellings)."""

        ta = interval_to_ticks(
            count, unit, sim_minutes_per_tick=self.config.sim_minutes_per_tick
        )
        for _ in range(ta.ticks):
            self.step()
        return ta

    def _ins(self, ticker: str):
        return self.market.by_ticker().get(ticker)

    def order_market(self, ticker: str, side: Side, size: float) -> Result:
        ins = self._ins(ticker)
        if ins is None:
            return Result.NOT_FOUND
        return execute_market_order(self.market, self.player, ins, side, float(size))

    def order_limit(self, ticker: str, side: Side, size: float, price: float) -> Result:
        ins = self._ins(ticker)
        if ins is None:
            return Result.NOT_FOUND
        if side is Side.BUY:
            return execute_limit_buy(
                self.market, self.player, ins, float(size), float(price)
            )
        return execute_limit_sell(
            self.market, self.player, ins, float(size), float(price)
        )

    def order_market_buy_cash(self, ticker: str, budget: float) -> Result:
        ins = self._ins(ticker)
        if ins is None:
            return Result.NOT_FOUND
        return execute_market_buy_cash(self.market, self.player, ins, float(budget))

    def order_limit_buy_cash(self, ticker: str, limit_price: float, budget: float) -> Result:
        ins = self._ins(ticker)
        if ins is None:
            return Result.NOT_FOUND
        return execute_limit_buy_cash(
            self.market, self.player, ins, float(limit_price), float(budget)
        )

    def order(self, ticker: str, side: Side, size: float) -> Result:
        """Shorthand: market order."""

        return self.order_market(ticker, side, size)

    @property
    def equity(self) -> float:
        return self.player.mark_to_market(self.market)


def new_session(
    mode: GameModeName | str | None = None,
    *,
    custom: GameConfig | None = None,
) -> Session:
    """Build universe + market + player from a preset *mode* or *custom* config."""

    if custom is not None:
        c = custom
    elif mode is not None:
        name = mode if isinstance(mode, GameModeName) else GameModeName(str(mode).lower())
        c = preset(name)
    else:
        c = preset(GameModeName.SIMPLE)

    sim_rng = new_rng(c.seed)
    instruments = make_universe(c.n_stocks, c.n_funds, c.n_crypto, sim_rng)
    mkt = Market(
        instruments=instruments,
        spread_bps=c.spread_bps,
        vol_multiplier=c.vol_multiplier,
        drift_per_tick=c.drift_bias * max(c.sim_time_scale, 0.01),
        dt=0.25 * max(c.sim_time_scale, 0.01),
        seed=None if c.seed is None else c.seed + 1337,
        sim_minutes_per_tick=c.sim_minutes_per_tick,
        stock_fund_annual_return=c.stock_fund_annual_return,
        crypto_top_tier=c.crypto_top_tier,
        crypto_tier_vol_mult=c.crypto_tier_vol_mult,
        crypto_mcap_vol_power=c.crypto_mcap_vol_power,
        crypto_mcap_ref_usd=c.crypto_mcap_ref_usd,
        crypto_vol_max_mult=c.crypto_vol_max_mult,
        great_depression=c.great_depression,
        taker_fee_bps=c.taker_fee_bps,
        slippage_bps_base=c.slippage_bps_base,
        slippage_bps_per_million=c.slippage_bps_per_million,
        front_run_bps=c.front_run_bps,
        front_run_notional_usd=c.front_run_notional_usd,
        round_magnet_bps=c.round_magnet_bps,
        tod_signature_bps=c.tod_signature_bps,
        micro_tape_ema=c.micro_tape_ema,
        micro_wick_lookback=c.micro_wick_lookback,
    )
    p = Player(cash=c.starting_cash)
    p.max_leverage = c.max_leverage
    p.maintenance_margin_rate = c.maintenance_margin_rate
    p.shorting_enabled = c.shorting_enabled
    p.short_borrow_bps_per_sim_day = c.short_borrow_bps_per_sim_day
    p.sec_fee_sell_bps = c.sec_fee_sell_bps
    return Session(config=c, market=mkt, player=p, _rng=sim_rng)
