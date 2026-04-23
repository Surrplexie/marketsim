from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .amounts import MIN_QUOTE_PX
from .clob import OrderBook, seed_liquidity
from .enums import Side  # re-export
from .instrument import (
    AssetClass,
    EW_C1F_EXP,
    EW_S10_PREFIX,
    EW_S2F_EXP,
    Instrument,
    MEGA_FUND_BASKET_N1,
    MEGA_FUND_BASKET_N2,
    MEGA_SECTOR_ETF_BASKET_N,
    STOCK_SECTOR_KEYS,
)
from .microstruct import (
    round_magnet_delta,
    sim_minute_of_day,
    tod_drift_shape,
    wick_ohlc,
)
from .sim_tr import Array, new_rng, batch_gbm

# Same floor as *MIN_QUOTE_PX* in *amounts* (CLOB + GBM; keeps listed bid/ask positive).
MIN_MID = float(MIN_QUOTE_PX)

__all__ = ("Market", "Side", "MIN_MID")

@dataclass
class Market:
    """
    Vector mid prices, bid/ask from CLOB, GBM per tick.
    Stocks & funds: long-run exponential bias toward *stock_fund_annual_return* per *sim* year
    (*drift_per_tick* / mode *drift_bias* applies only to crypto, so equities keep long-run growth).
    Crypto: *sigma* rescaled from live mcap (top tier calmer, smaller cap → more vol).
    Volatility override slider scales equity *μ* and **σ** for listed stocks, funds, and crypto.
    """

    instruments: list[Instrument]
    spread_bps: float
    vol_multiplier: float
    drift_per_tick: float
    dt: float
    seed: int | None
    sim_minutes_per_tick: float
    stock_fund_annual_return: float
    crypto_top_tier: int
    crypto_tier_vol_mult: float
    crypto_mcap_vol_power: float
    crypto_mcap_ref_usd: float
    crypto_vol_max_mult: float
    great_depression: bool = False
    taker_fee_bps: float = 2.0
    slippage_bps_base: float = 1.2
    slippage_bps_per_million: float = 2.0
    front_run_bps: float = 0.4
    front_run_notional_usd: float = 20_000.0
    round_magnet_bps: float = 0.25
    tod_signature_bps: float = 1.0
    micro_tape_ema: float = 0.2
    micro_wick_lookback: int = 8
    # Opening "liquidity calendar": temporarily wider NPC half-spread (fades over sim minutes).
    open_liquidity_extra_bps_peak: float = 14.0
    open_liquidity_fade_sim_minutes: float = 105.0
    # Mean sim ticks between random headline shocks (geometric-ish draw each tick).
    headline_mean_ticks: float = 2400.0
    chaos_flash_crash: bool = False
    chaos_meme_squeeze: bool = False
    chaos_fat_finger: bool = False
    chaos_exchange_halt: bool = False
    chaos_rumor_mill: bool = False
    chaos_sector_rotation: bool = False
    chaos_funding_panic: bool = False
    chaos_liquidity_drought: bool = False
    chaos_whale_rebalance: bool = False
    chaos_crypto_weekend_mania: bool = False
    _rng: np.random.Generator = field(init=False, repr=False)
    _mids: Array = field(init=False, repr=False)
    _mu: Array = field(init=False, repr=False)
    _sigma: Array = field(init=False, repr=False)
    _sigma_baseline: Array = field(init=False, repr=False)
    _crypto_index: list[int] = field(init=False, repr=False)
    _mu_equity_addon: float = field(init=False, repr=False)
    _tick: int = field(default=0, init=False)
    books: list[OrderBook] = field(default_factory=list, init=False, repr=False)
    # Rolling quote mids for % change vs last *sim* 24h (see *_ch24_window_ticks*).
    _ch24_window_ticks: int = field(default=0, init=False, repr=False)
    _mid_hist: list[deque[float]] = field(default_factory=list, init=False, repr=False)
    # For candlestick / chart: *(sim_tick, mid)* (listed quote mid) after each *step()*; unbounded
    # for the session (long runs can use a lot of memory).
    _chart_hist: list[deque[tuple[int, float]]] = field(default_factory=list, init=False, repr=False)
    _vol_hist: list[deque[tuple[int, float]]] = field(default_factory=list, init=False, repr=False)
    _vol_snapshot: np.ndarray = field(init=False, repr=False)
    # Great depression: one random crash in [500,1000], then *recovery* ticks pull prices toward 99% of pre
    # for ~99% of names; the rest (≈1%) stay impaired.
    _gd_armed: bool = field(default=False, init=False, repr=False)
    _gd_trigger: int | None = field(default=None, init=False, repr=False)
    _gd_fired: bool = field(default=False, init=False, repr=False)
    _gd_pre: np.ndarray | None = field(default=None, init=False, repr=False)
    _gd_impaired: set[int] = field(default_factory=set, init=False, repr=False)
    _gd_recovery_left: int = field(default=0, init=False, repr=False)
    _gd_lerp: float = field(default=0.05, init=False, repr=False)
    # Microstructure: tape, peer, TOD, magnet
    _tape_ema: np.ndarray = field(init=False, repr=False)
    _flow_bump: np.ndarray = field(init=False, repr=False)
    _tape_a: float = field(default=0.2, init=False, repr=False)
    _peer_a: int = field(default=0, init=False, repr=False)
    _peer_b: int = field(default=0, init=False, repr=False)
    _mqa: deque = field(init=False, repr=False)
    _mqb: deque = field(init=False, repr=False)
    _reta: deque = field(init=False, repr=False)
    _retb: deque = field(init=False, repr=False)
    _peer_div_bps: float = field(default=0.0, init=False, repr=False)
    _peer_corr: float = field(default=0.0, init=False, repr=False)
    _cal_bid_sh: np.ndarray = field(init=False, repr=False)
    _cal_ask_sh: np.ndarray = field(init=False, repr=False)
    _headline_vol_addon: np.ndarray = field(init=False, repr=False)
    _news: deque = field(init=False, repr=False)
    _financing_long_bps: np.ndarray = field(init=False, repr=False)
    _financing_short_bps: np.ndarray = field(init=False, repr=False)
    # Optional UI/API drift bias: -10 = strong downward + noisy, +10 = strong upward, 0 = off.
    _trend_value: int = field(default=0, init=False, repr=False)
    _trend_scope: str = field(default="all", init=False, repr=False)
    _trend_custom: set[str] = field(default_factory=set, init=False, repr=False)
    # -10..+10, 0 = off: scales listed equity *μ* (long-run growth); **σ** for crypto *and* stocks/funds (≈10× at +10).
    _volatility_value: int = field(default=0, init=False, repr=False)
    _chaos_once_flash_done: bool = field(default=False, init=False, repr=False)
    _chaos_halt_until_tick: np.ndarray = field(init=False, repr=False)
    _chaos_price_shock_bps: np.ndarray = field(init=False, repr=False)
    _chaos_sigma_addon: np.ndarray = field(init=False, repr=False)
    _chaos_funding_panic_until: int = field(default=0, init=False, repr=False)
    _chaos_liquidity_drought_until: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = new_rng(self.seed)
        if self.great_depression:
            self._gd_armed = True
            self._gd_trigger = int(self._rng.integers(500, 1001))
        else:
            self._gd_armed = False
            self._gd_trigger = None
        n = len(self.instruments)
        self._mids = np.array([x.last for x in self.instruments], dtype=np.float64)
        self._mids = np.maximum(self._mids, float(MIN_MID))
        self._mu = np.zeros(n, dtype=np.float64)
        self._sigma = np.zeros(n, dtype=np.float64)
        self._sigma_baseline = np.zeros(n, dtype=np.float64)
        mpt = max(float(self.sim_minutes_per_tick), 0.1)
        steps_per_sim_year = (365.0 * 24.0 * 60.0) / mpt
        # *steps_per_sim_year * dt* = SDE time advanced in one *sim* year* of *step()* calls
        t_y = steps_per_sim_year * self.dt
        r = max(float(self.stock_fund_annual_return), 1e-6)
        self._mu_equity_addon = float(np.log(1.0 + r) / max(t_y, 1e-9))

        self._crypto_index = [i for i, x in enumerate(self.instruments) if x.kind is AssetClass.CRYPTO]
        self._stock_indices = [i for i, x in enumerate(self.instruments) if x.kind is AssetClass.STOCK]
        self._fund_indices = [i for i, x in enumerate(self.instruments) if x.kind is AssetClass.FUND]

        for i, ins in enumerate(self.instruments):
            if ins.is_listed_equity:
                # Mode drift_bias must not erode the equity long-run target (e.g. complex had negative bias).
                self._mu[i] = self._mu_equity_addon + float(self._rng.standard_normal() * 0.00002)
            else:
                self._mu[i] = self.drift_per_tick + float(
                    self._rng.standard_normal() * 0.00002
                )
            ann = ins.base_annual_vol * self.vol_multiplier
            per_step = ann * 0.02 * np.sqrt(max(self.dt, 1e-9))
            self._sigma_baseline[i] = max(per_step, 1e-8)

        self._apply_crypto_sigma_mcaps(self._mids)
        for i in range(n):
            if i not in self._crypto_index:
                self._sigma[i] = self._sigma_baseline[i]

        self.books = [OrderBook() for _ in self.instruments]
        for k, _ins in enumerate(self.instruments):
            seed_liquidity(self.books[k], float(self._mids[k]), self.spread_bps, self._rng)
        mpt_24h = max(float(self.sim_minutes_per_tick), 1e-9)
        self._ch24_window_ticks = max(1, int(math.ceil(24.0 * 60.0 / mpt_24h)))
        w = self._ch24_window_ticks
        self._mid_hist = [deque(maxlen=w + 1) for _ in range(n)]
        self._append_ch24_history()
        self._chart_hist = [deque() for _ in range(n)]
        self._vol_hist = [deque() for _ in range(n)]
        self._seed_chart_history()
        self._tape_ema = np.zeros(n, dtype=np.float64)
        self._flow_bump = np.zeros(n, dtype=np.float64)
        self._tape_a = min(0.5, 0.05 + 0.3 * max(0.0, float(self.micro_tape_ema)))
        idxm = sorted(
            range(n),
            key=lambda j: -float(self._mids[j] * self.instruments[j].units_outstanding),
        )
        self._peer_a = int(idxm[0])
        self._peer_b = int(idxm[1] if n > 1 else idxm[0])
        self._mqa = deque(maxlen=21)
        self._mqb = deque(maxlen=21)
        self._reta = deque(maxlen=36)
        self._retb = deque(maxlen=36)
        self._peer_div_bps = 0.0
        self._peer_corr = 0.0
        self._cal_bid_sh = np.zeros(n, dtype=np.float64)
        self._cal_ask_sh = np.zeros(n, dtype=np.float64)
        self._headline_vol_addon = np.zeros(n, dtype=np.float64)
        self._news = deque(maxlen=40)
        self._cumulative_volume = np.zeros(n, dtype=np.float64)
        self._vol_snapshot = self._cumulative_volume.copy()
        self._financing_long_bps = np.zeros(n, dtype=np.float64)
        self._financing_short_bps = np.zeros(n, dtype=np.float64)
        self._chaos_halt_until_tick = np.zeros(n, dtype=np.int64)
        self._chaos_price_shock_bps = np.zeros(n, dtype=np.float64)
        self._chaos_sigma_addon = np.zeros(n, dtype=np.float64)

    def record_trade_volume(self, j: int, qty: float) -> None:
        """Add *qty* from CLOB fills to session volume (also grows from natural turnover each *step*)."""

        if 0 <= int(j) < int(self._cumulative_volume.size):
            self._cumulative_volume[int(j)] += max(0.0, float(qty))

    def cumulative_volume(self, j: int) -> float:
        return float(self._cumulative_volume[j]) if 0 <= int(j) < int(self._cumulative_volume.size) else 0.0

    def _listed_equity_and_fund_supply_flow(self) -> None:
        """
        Small stochastic change in shares outstanding for stocks and funds, with **price scaled**
        so *price × units* (≈ mcap) is unchanged before other dynamics—mirrors quiet issuance / buybacks.
        """

        specs: list[tuple[int, float, float, float]] = []
        for j in self._stock_indices:
            specs.append((j, 4.2e-6, 0.992, 1.008))
        for j in self._fund_indices:
            specs.append((j, 2.0e-6, 0.996, 1.004))
        for j, sigz, lo_b, hi_b in specs:
            ins = self.instruments[j]
            cap = ins.max_units_outstanding
            u0 = float(ins.units_outstanding)
            if u0 <= 0.0 or not math.isfinite(u0):
                continue
            z = float(self._rng.normal(0.0, sigz))
            f = math.exp(z)
            u1 = u0 * f
            lo = max(u0 * lo_b, 1e4)
            hi = u0 * hi_b
            if cap is not None and math.isfinite(float(cap)):
                hi = min(hi, float(cap))
            u1 = max(lo, min(u1, hi))
            if abs(u1 - u0) < u0 * 1e-10:
                continue
            self._mids[j] = max(float(MIN_MID), float(self._mids[j]) * (u0 / u1))
            ins.units_outstanding = u1
            ins.last = float(self._mids[j])
            self._cumulative_volume[j] += abs(u1 - u0) * float(self._rng.uniform(0.25, 1.05))

    def _apply_natural_volume_tick(self, t0: np.ndarray) -> None:
        """Add turnover not tied to the player: scales with σ, realized move, tape, and float."""

        n = int(self._mids.size)
        tpd = max(1.0, float(self.ticks_per_sim_24h))
        for j in range(n):
            ins = self.instruments[j]
            u = float(ins.units_outstanding)
            if u <= 0.0 or not math.isfinite(u):
                continue
            sig = float(self._sigma[j])
            m0 = max(float(MIN_MID), float(t0[j]))
            m1 = max(float(MIN_MID), float(self._mids[j]))
            ar = abs(m1 - m0) / max(m0, 1e-12)
            tape_n = float(self._tape_ema[j]) / max(m0, 1.0)
            h = float(self._headline_vol_addon[j]) if j < int(self._headline_vol_addon.size) else 0.0
            intraday = 0.0055 + min(
                0.038,
                600.0 * sig + 15.0 * ar + min(0.014, tape_n * 1.1e-6) + 0.0022 * min(h, 2.0),
            )
            sh = u * intraday / tpd * float(self._rng.uniform(0.38, 1.72))
            self._cumulative_volume[j] += max(0.0, sh)

    def _mint_inflation_crypto(self) -> None:
        """
        Inflate circulating units on mintable cryptos; **price scales down** by old/new float
        (dilution, constant-ish market cap before other dynamics).
        """

        for j in self._crypto_index:
            ins = self.instruments[j]
            if not ins.allow_supply_inflation:
                continue
            u0 = float(ins.units_outstanding)
            cap = ins.max_units_outstanding
            if cap is not None and math.isfinite(float(cap)) and u0 >= float(cap) - 1e-6:
                continue
            bump = 1.0 + 0.32 * float(self._headline_vol_addon[j]) + 22.0 * float(self._sigma[j])
            bump = max(0.55, min(3.2, bump))
            du = u0 * float(self._rng.uniform(4e-8, 2.5e-6)) * bump
            u1 = u0 + du
            if cap is not None and math.isfinite(float(cap)):
                u1 = min(u1, float(cap))
            du_act = u1 - u0
            if du_act <= 1e-18:
                continue
            self._mids[j] = max(float(MIN_MID), float(self._mids[j]) * (u0 / u1))
            ins.units_outstanding = u1
            ins.last = float(self._mids[j])
            self._cumulative_volume[j] += du_act * float(self._rng.uniform(0.75, 1.2))

    @property
    def tick(self) -> int:
        return self._tick

    def set_trend_override(
        self, value: int, scope: str, tickers: list[str] | None = None
    ) -> None:
        """-10..+10 bias on GBM *mu* (per *step*), scoped to asset groups or a custom multi-ticker set."""

        v = int(max(-10, min(10, int(value))))
        s = (scope or "all").lower().strip()
        allowed = ("all", "stocks", "funds", "crypto", "equity", "custom")
        if s not in allowed:
            s = "all"
        self._trend_value = v
        self._trend_scope = s
        self._trend_custom = set()
        if s == "custom" and tickers:
            valid = {x.ticker.upper() for x in self.instruments}
            for t in tickers:
                u = str(t).strip().upper()
                if u in valid:
                    self._trend_custom.add(u)

    def _trend_affects_index(self, j: int) -> bool:
        s = str(self._trend_scope)
        ins = self.instruments[j]
        if s == "all":
            return True
        if s == "stocks" and ins.kind is AssetClass.STOCK:
            return True
        if s == "funds" and ins.kind is AssetClass.FUND:
            return True
        if s == "crypto" and ins.kind is AssetClass.CRYPTO:
            return True
        if s == "equity" and ins.is_listed_equity:
            return True
        if s == "custom" and ins.ticker in self._trend_custom:
            return True
        return False

    def trend_override_state(self) -> dict[str, int | str | list[str]]:
        return {
            "value": int(self._trend_value),
            "scope": str(self._trend_scope),
            "custom_tickers": sorted(self._trend_custom),
        }

    def set_volatility_override(self, value: int) -> None:
        """-10..+10; 0 = off. Scales equity *μ* (long-run growth) and **σ** for crypto, stocks, and funds."""

        self._volatility_value = int(max(-10, min(10, int(value))))

    def volatility_override_state(self) -> dict[str, int | float | str]:
        v = int(self._volatility_value)
        if v == 0:
            return {
                "value": 0,
                "active": "off",
                "equity_mu_mult": 1.0,
                "crypto_sigma_mult": 1.0,
                "listed_equity_sigma_mult": 1.0,
            }
        if v < 0:
            em = max(0.05, 1.0 + 0.095 * float(v))
            cm = max(0.04, 1.0 + 0.096 * float(v))
        else:
            em = 1.0 + 0.20 * float(v)
            cm = 1.0 + 0.9 * float(v)
        sm = float(cm)
        return {
            "value": v,
            "active": "on",
            "equity_mu_mult": float(em),
            "crypto_sigma_mult": sm,
            "listed_equity_sigma_mult": sm,
        }

    def _volatility_equity_mu_mult(self) -> float:
        v = int(self._volatility_value)
        if v == 0:
            return 1.0
        if v < 0:
            return max(0.05, 1.0 + 0.095 * float(v))
        return 1.0 + 0.20 * float(v)

    def _volatility_slider_sigma_mult(self) -> float:
        """σ multiplier from the volatility slider; applied to crypto and listed stocks/funds."""

        v = int(self._volatility_value)
        if v == 0:
            return 1.0
        if v < 0:
            return max(0.04, 1.0 + 0.096 * float(v))
        return 1.0 + 0.9 * float(v)

    def _volatility_crypto_sigma_mult(self) -> float:
        return self._volatility_slider_sigma_mult()

    def mid(self, i: int) -> float:
        return float(self._mids[i])

    def spread_ratio(self) -> float:
        return self.spread_bps / 10_000.0

    def synthetic_quote(self, i: int) -> tuple[float, float, float]:
        m = self.mid(i)
        half = m * 0.5 * self.spread_ratio()
        return m - half, m, m + half

    def quote(self, i: int) -> tuple[float, float, float]:
        s_b, s_m, s_a = self.synthetic_quote(i)
        ob = self.books[i]
        bo = ob.best_bid()
        ao = ob.best_ask()
        b = max(float(MIN_MID), float(bo) if bo is not None else s_b)
        a = max(float(MIN_MID), float(ao) if ao is not None else s_a)
        if a <= b + 1e-15:
            return s_b, s_m, s_a
        m = 0.5 * (b + a)
        m = max(float(MIN_MID), m)
        return b, m, a

    def mcap(self, i: int) -> float:
        return float(self._mids[i] * self.instruments[i].units_outstanding)

    def apply_forward_split_for_index(self, j: int, ratio: float) -> bool:
        """
        Apply a *ratio*:1 forward split (e.g. 2.0 → 2:1) to instrument *j* only if it is
        a common stock. Adjusts mid, float, order book, and rolling **price** history so
        % change and chart stay consistent. Returns *True* if applied.
        """
        n = int(self._mids.size)
        if j < 0 or j >= n:
            return False
        ins = self.instruments[j]
        if ins.kind is not AssetClass.STOCK:
            return False
        r = float(ratio)
        if not math.isfinite(r) or r <= 1.0 + 1e-9:
            return False
        cap = ins.max_units_outstanding
        new_u = float(ins.units_outstanding) * r
        if cap is not None and math.isfinite(float(cap)) and new_u > float(cap) + 1e-2:
            return False
        self._mids[j] = max(float(MIN_MID), float(self._mids[j]) / r)
        ins.units_outstanding = new_u
        ins.last = float(self._mids[j])
        self.books[j].apply_forward_split(r)
        ch = self._chart_hist[j]
        self._chart_hist[j] = deque(
            ((int(t), float(m) / r) for (t, m) in ch),
            maxlen=ch.maxlen,
        )
        mh = self._mid_hist[j]
        self._mid_hist[j] = deque((float(x) / r for x in mh), maxlen=mh.maxlen)
        if self._gd_pre is not None and 0 <= j < int(self._gd_pre.size):
            self._gd_pre[j] = float(self._gd_pre[j]) / r
        if j == int(self._peer_a):
            self._mqa = deque(
                (float(x) / r for x in self._mqa), maxlen=self._mqa.maxlen
            )
        if j == int(self._peer_b):
            self._mqb = deque(
                (float(x) / r for x in self._mqb), maxlen=self._mqb.maxlen
            )
        return True

    def _bump_mids_in_histories(self, j: int, delta: float) -> None:
        """Add *delta* to stored price series (chart, 24h, GD, peer deques) for *j*."""

        dd = float(delta)
        if not math.isfinite(dd) or abs(dd) < 1e-15:
            return
        ch = self._chart_hist[j]
        self._chart_hist[j] = deque(
            ((int(t), max(float(MIN_MID), float(m) + dd)) for (t, m) in ch),
            maxlen=ch.maxlen,
        )
        mh = self._mid_hist[j]
        self._mid_hist[j] = deque(
            (max(float(MIN_MID), float(x) + dd) for x in mh), maxlen=mh.maxlen
        )
        if self._gd_pre is not None and 0 <= j < int(self._gd_pre.size):
            self._gd_pre[j] = max(float(MIN_MID), float(self._gd_pre[j]) + dd)
        if j == int(self._peer_a):
            self._mqa = deque(
                (max(float(MIN_MID), float(x) + dd) for x in self._mqa),
                maxlen=self._mqa.maxlen,
            )
        if j == int(self._peer_b):
            self._mqb = deque(
                (max(float(MIN_MID), float(x) + dd) for x in self._mqb),
                maxlen=self._mqb.maxlen,
            )

    def apply_overnight_gaps_bps(
        self, rng: object, max_abs_bps: float, *, listed_only: bool = True
    ) -> None:
        """At sim-day roll: random *mid* gap (±*max_abs_bps*), usually listed assets only."""

        gmax = max(0.0, float(max_abs_bps))
        if gmax <= 0.0:
            return
        n = int(self._mids.size)
        for j in range(n):
            if listed_only and not self.instruments[j].is_listed_equity:
                continue
            g = float(rng.uniform(-gmax, gmax))  # type: ignore[attr-defined]
            if not math.isfinite(g) or abs(g) < 1e-8:
                continue
            m0 = float(self._mids[j])
            d = m0 * g / 10_000.0
            if not math.isfinite(d) or abs(d) < 1e-12:
                continue
            self.books[j].shift_price_levels(d)
            self._mids[j] = max(float(MIN_MID), m0 + d)
            self.instruments[j].last = float(self._mids[j])
            self._bump_mids_in_histories(j, d)

    def _ex_div_adjust_price_histories(self, j: int, d: float) -> None:
        """Shift stored mids down by *d* (ex-div) for chart / 24h / GD / peer deques."""

        dd = float(d)
        if dd <= 0.0 or not math.isfinite(dd):
            return
        ch = self._chart_hist[j]
        self._chart_hist[j] = deque(
            ((int(t), max(float(MIN_MID), float(m) - dd)) for (t, m) in ch),
            maxlen=ch.maxlen,
        )
        mh = self._mid_hist[j]
        self._mid_hist[j] = deque(
            (max(float(MIN_MID), float(x) - dd) for x in mh), maxlen=mh.maxlen
        )
        if self._gd_pre is not None and 0 <= j < int(self._gd_pre.size):
            self._gd_pre[j] = max(float(MIN_MID), float(self._gd_pre[j]) - dd)
        if j == int(self._peer_a):
            self._mqa = deque(
                (max(float(MIN_MID), float(x) - dd) for x in self._mqa),
                maxlen=self._mqa.maxlen,
            )
        if j == int(self._peer_b):
            self._mqb = deque(
                (max(float(MIN_MID), float(x) - dd) for x in self._mqb),
                maxlen=self._mqb.maxlen,
            )

    def apply_cash_dividend_for_index(
        self, j: int, usd_per_share: float
    ) -> tuple[bool, float]:
        """
        Common-stock cash dividend: *usd_per_share* (capped so mid stays above *MIN_MID*).
        Ex-div: listed mid and all CLOB price levels fall by the paid amount. Returns
        (*applied*, *actual_usd_per_share* for player payment).
        """

        n = int(self._mids.size)
        if j < 0 or j >= n:
            return (False, 0.0)
        ins = self.instruments[j]
        if ins.kind is not AssetClass.STOCK:
            return (False, 0.0)
        dps = float(usd_per_share)
        if not math.isfinite(dps) or dps <= 0.0:
            return (False, 0.0)
        mid0 = float(self._mids[j])
        d = min(dps, max(0.0, mid0 - float(MIN_MID) * 1.000_001 - 1e-9))
        if d <= 0.0:
            return (False, 0.0)
        self.books[j].shift_price_levels(-d)
        self._mids[j] = max(float(MIN_MID), mid0 - d)
        ins.last = float(self._mids[j])
        self._ex_div_adjust_price_histories(j, d)
        return (True, d)

    def apply_share_buyback_for_index(self, j: int, float_fraction: float) -> bool:
        """
        Retire a fraction of the common-stock float: *units_outstanding* and all resting
        share sizes in the CLOB scale by (1 − *float_fraction*). Mid unchanged. *float_fraction* in (0, 0.5).
        Player positions must be scaled by the same factor separately.
        """

        n = int(self._mids.size)
        if j < 0 or j >= n:
            return False
        ins = self.instruments[j]
        if ins.kind is not AssetClass.STOCK:
            return False
        f = float(float_fraction)
        if not math.isfinite(f) or f <= 0.0 or f >= 0.5:
            return False
        u0 = float(ins.units_outstanding)
        if u0 <= 0.0:
            return False
        m = 1.0 - f
        ins.units_outstanding = u0 * m
        self.books[j].scale_resting_shares(m)
        return True

    def _append_ch24_history(self) -> None:
        """After books/mids are consistent, store mid used for 24h % (listed quote)."""

        for k in range(len(self.instruments)):
            _, mid, _ = self.quote(k)
            self._mid_hist[k].append(float(mid))

    @property
    def ticks_per_sim_24h(self) -> int:
        """*step()* count per *sim* 24h, derived from *sim_minutes_per_tick* (1440 min = 1 sim day)."""

        return self._ch24_window_ticks

    @property
    def ch24_warmed(self) -> bool:
        """True once at least *sim* 24h of ticks have passed (full rolling window)."""

        return self._tick >= self._ch24_window_ticks

    def pct_change_24h_sim(self, i: int) -> float | None:
        """
        Percent change in **listed** mid vs oldest mid kept in the rolling *sim* 24h window.
        If fewer than two samples exist, returns *None*; until the window is full, the baseline
        is still the left end of the deque (session start for early game).
        """

        d = self._mid_hist[i]
        if len(d) < 2:
            return None
        _, now, _ = self.quote(i)
        past = d[0]
        if past <= 1e-12:
            return None
        return 100.0 * (now - past) / past

    def _seed_chart_history(self) -> None:
        t = int(self._tick)
        for k in range(len(self.instruments)):
            _, mid, _ = self.quote(k)
            self._chart_hist[k].append((t, float(mid)))
            self._vol_hist[k].append((t, 0.0))

    def _append_chart_row(self) -> None:
        t = int(self._tick)
        for k in range(len(self.instruments)):
            _, mid, _ = self.quote(k)
            self._chart_hist[k].append((t, float(mid)))
            dv = max(0.0, float(self._cumulative_volume[k] - self._vol_snapshot[k]))
            self._vol_hist[k].append((t, dv))
            self._vol_snapshot[k] = float(self._cumulative_volume[k])

    def chart_ohlc(
        self,
        i: int,
        bucket: int,
        *,
        max_bars: int = 0,
    ) -> list[dict[str, int | float]]:
        """
        Aggregate stored *(tick, mid)* into OHLC candles, *bucket* points per bar (non-overlapping).
        *time* in each bar is 1_000_000_000 + end tick (for the chart UI as UTCTimestamp seconds).
        *max_bars* ≤ 0 returns the full series; *max_bars* > 0 returns at most that many **latest** bars.
        """

        dq = self._chart_hist[i]
        if not dq:
            return []
        pts = list(dq)
        vols = list(self._vol_hist[i]) if 0 <= i < len(self._vol_hist) else []
        b = max(1, int(bucket))
        out: list[dict[str, int | float]] = []
        j = 0
        while j < len(pts):
            seg = pts[j : j + b]
            if not seg:
                break
            j += b
            mids = [p[1] for p in seg]
            t_end = int(seg[-1][0])
            o, h, lo, c = float(mids[0]), float(max(mids)), float(min(mids)), float(mids[-1])
            vseg = vols[j : j + b] if vols else []
            v = float(sum(float(x[1]) for x in vseg)) if vseg else 0.0
            out.append(
                {
                    "time": 1_000_000_000 + t_end,
                    "open": o,
                    "high": h,
                    "low": lo,
                    "close": c,
                    "volume": v,
                }
            )
        if not out:
            return []
        lim = int(max_bars)
        if lim <= 0:
            return out
        mbar = min(lim, len(out))
        return out[-mbar:]

    def great_depression_state(self) -> dict[str, int | bool | None]:
        """Exposed in API: armed = crash not yet hit; *ticks_to_event* when still armed."""

        t = int(self._tick)
        if self._gd_armed and self._gd_trigger is not None and not self._gd_fired:
            return {
                "armed": True,
                "fired": False,
                "in_recovery": False,
                "ticks_to_event": max(0, int(self._gd_trigger) - t),
            }
        if self._gd_fired and self._gd_recovery_left > 0:
            return {
                "armed": False,
                "fired": True,
                "in_recovery": True,
                "ticks_to_event": None,
            }
        if self._gd_fired:
            return {
                "armed": False,
                "fired": True,
                "in_recovery": False,
                "ticks_to_event": None,
            }
        return {
            "armed": False,
            "fired": False,
            "in_recovery": False,
            "ticks_to_event": None,
        }

    def chaos_settings(self) -> dict[str, bool]:
        return {
            "flash_crash": bool(self.chaos_flash_crash),
            "meme_squeeze": bool(self.chaos_meme_squeeze),
            "fat_finger": bool(self.chaos_fat_finger),
            "exchange_halt": bool(self.chaos_exchange_halt),
            "rumor_mill": bool(self.chaos_rumor_mill),
            "sector_rotation": bool(self.chaos_sector_rotation),
            "funding_panic": bool(self.chaos_funding_panic),
            "liquidity_drought": bool(self.chaos_liquidity_drought),
            "whale_rebalance": bool(self.chaos_whale_rebalance),
            "crypto_weekend_mania": bool(self.chaos_crypto_weekend_mania),
        }

    def _chaos_emit_news(self, line: str, tag: str) -> None:
        self._news.append({"tick": int(self._tick), "line": str(line), "tag": str(tag)})

    def _chaos_apply_price_shock_bps(self, j: int, bps: float) -> None:
        if not (0 <= int(j) < int(self._mids.size)):
            return
        b = float(bps)
        if not math.isfinite(b) or abs(b) < 1e-9:
            return
        m0 = float(self._mids[j])
        d = m0 * b / 10_000.0
        if not math.isfinite(d) or abs(d) < 1e-12:
            return
        self.books[j].shift_price_levels(d)
        self._mids[j] = max(float(MIN_MID), m0 + d)
        self.instruments[j].last = float(self._mids[j])
        self._bump_mids_in_histories(j, d)

    def _apply_chaos_events_pre_gbm(self) -> None:
        n = int(self._mids.size)
        if n <= 0:
            return
        stocks = list(self._stock_indices)
        cryptos = list(self._crypto_index)
        # 1) Flash crash roulette (one-shot each session).
        if self.chaos_flash_crash and (not self._chaos_once_flash_done) and self._tick > 150:
            if float(self._rng.random()) < 1.2e-4:
                j = int(self._rng.choice(np.array(stocks if stocks else list(range(n)), dtype=np.int64)))
                drop = float(self._rng.uniform(2000.0, 6000.0))
                self._chaos_apply_price_shock_bps(j, -drop)
                self._chaos_once_flash_done = True
                self._chaos_emit_news(f"{self.instruments[j].ticker}: flash-crash roulette hit ({drop/100:.1f}%).", "chaos:flash_crash")
        # 2) Meme squeeze.
        if self.chaos_meme_squeeze and stocks and float(self._rng.random()) < 4.0e-4:
            j = int(self._rng.choice(np.array(stocks, dtype=np.int64)))
            self._chaos_price_shock_bps[j] += float(self._rng.uniform(65.0, 230.0))
            self._chaos_sigma_addon[j] = min(2.5, float(self._chaos_sigma_addon[j]) + float(self._rng.uniform(0.22, 0.65)))
            self._chaos_emit_news(f"{self.instruments[j].ticker}: meme squeeze impulse.", "chaos:meme_squeeze")
        # 3) Fat finger event.
        if self.chaos_fat_finger and float(self._rng.random()) < 4.0e-4:
            pool = stocks + cryptos
            if not pool:
                pool = list(range(n))
            j = int(self._rng.choice(np.array(pool, dtype=np.int64)))
            bps = float(self._rng.uniform(-180.0, 180.0))
            self._chaos_apply_price_shock_bps(j, bps)
            self._chaos_emit_news(f"{self.instruments[j].ticker}: fat-finger sweep ({bps:+.0f} bps).", "chaos:fat_finger")
        # 4) Exchange halt.
        if self.chaos_exchange_halt and stocks and float(self._rng.random()) < 2.0e-4:
            j = int(self._rng.choice(np.array(stocks, dtype=np.int64)))
            dur = int(self._rng.integers(10, 41))
            self._chaos_halt_until_tick[j] = max(int(self._chaos_halt_until_tick[j]), int(self._tick + dur))
            self._chaos_emit_news(f"{self.instruments[j].ticker}: exchange halt ({dur} ticks).", "chaos:halt")
        # 5) Rumor mill.
        if self.chaos_rumor_mill and stocks and float(self._rng.random()) < 6.0e-4:
            j = int(self._rng.choice(np.array(stocks, dtype=np.int64)))
            sign = -1.0 if float(self._rng.random()) < 0.5 else 1.0
            self._chaos_price_shock_bps[j] += sign * float(self._rng.uniform(40.0, 170.0))
            self._chaos_sigma_addon[j] = min(2.5, float(self._chaos_sigma_addon[j]) + float(self._rng.uniform(0.15, 0.45)))
            self._chaos_emit_news(f"{self.instruments[j].ticker}: rumor-mill shock ({'up' if sign > 0 else 'down'}).", "chaos:rumor")
        # 6) Sector rotation storm.
        if self.chaos_sector_rotation and stocks and float(self._rng.random()) < 3.5e-4:
            keys = sorted({str(self.instruments[j].sector) for j in stocks if str(self.instruments[j].sector) in STOCK_SECTOR_KEYS})
            if len(keys) >= 2:
                out_k = str(self._rng.choice(np.array(keys, dtype=object)))
                in_k = str(self._rng.choice(np.array([k for k in keys if k != out_k], dtype=object)))
                for j in stocks:
                    sk = str(self.instruments[j].sector)
                    if sk == in_k:
                        self._chaos_price_shock_bps[j] += float(self._rng.uniform(18.0, 55.0))
                    elif sk == out_k:
                        self._chaos_price_shock_bps[j] -= float(self._rng.uniform(18.0, 55.0))
                self._chaos_emit_news(f"Sector rotation storm: into {in_k}, out of {out_k}.", "chaos:rotation")
        # 7) Funding panic window.
        if self.chaos_funding_panic and float(self._rng.random()) < 2.5e-4:
            dur = int(self._rng.integers(25, 95))
            self._chaos_funding_panic_until = max(int(self._chaos_funding_panic_until), int(self._tick + dur))
            self._chaos_emit_news(f"Funding panic: borrow/funding spike for {dur} ticks.", "chaos:funding")
        # 8) Liquidity drought window.
        if self.chaos_liquidity_drought and float(self._rng.random()) < 2.5e-4:
            dur = int(self._rng.integers(20, 85))
            self._chaos_liquidity_drought_until = max(int(self._chaos_liquidity_drought_until), int(self._tick + dur))
            self._chaos_emit_news(f"Liquidity drought: widened spreads for {dur} ticks.", "chaos:drought")
        # 9) Whale rebalance.
        if self.chaos_whale_rebalance and stocks and float(self._rng.random()) < 2.8e-4:
            ranked = sorted(stocks, key=lambda j0: -self._mcap(j0))
            k = max(2, min(len(ranked), max(2, len(ranked) // 8)))
            for idx, j in enumerate(ranked[:k]):
                sgn = 1.0 if idx < (k // 2) else -1.0
                self._chaos_price_shock_bps[j] += sgn * float(self._rng.uniform(16.0, 52.0))
            self._chaos_emit_news("Whale rebalance: heavyweight basket rotation.", "chaos:whale")
        # 10) Crypto weekend mania.
        if self.chaos_crypto_weekend_mania and cryptos:
            day_idx = int((int(self._tick) * max(float(self.sim_minutes_per_tick), 1e-9)) // 1440) % 7
            is_weekend = day_idx in (5, 6)
            if is_weekend:
                for j in cryptos:
                    self._chaos_sigma_addon[j] = min(2.5, float(self._chaos_sigma_addon[j]) + 0.08)
                if float(self._rng.random()) < 8.0e-4:
                    j = int(self._rng.choice(np.array(cryptos, dtype=np.int64)))
                    self._chaos_apply_price_shock_bps(j, float(self._rng.uniform(-220.0, 220.0)))
                    self._chaos_emit_news(f"{self.instruments[j].ticker}: weekend mania wick.", "chaos:weekend")

    def _apply_crypto_sigma_mcaps(self, mids: Array) -> None:
        """Set *_sigma* for each crypto from current mcap (vs. peers) and *baseline* vol."""

        top = max(int(self.crypto_top_tier), 0)
        ref = max(float(self.crypto_mcap_ref_usd), 1.0)
        power = max(float(self.crypto_mcap_vol_power), 0.0)
        mx = max(float(self.crypto_vol_max_mult), 1.0)
        tier = float(self.crypto_tier_vol_mult)
        C = list(self._crypto_index)
        if not C:
            return
        mc: list[tuple[int, float]] = []
        for i in C:
            m = float(mids[i]) * float(self.instruments[i].units_outstanding)
            mc.append((i, m))
        mc.sort(key=lambda t: t[1], reverse=True)
        for rank, (i, mcap) in enumerate(mc):
            b = self._sigma_baseline[i]
            if rank < top and top > 0:
                self._sigma[i] = b * tier
            else:
                m0 = max(mcap, 1.0)
                mscale = (ref / m0) ** power
                mscale = min(max(mscale, 1.0), mx)
                self._sigma[i] = b * mscale

    def _mcap(self, j: int) -> float:
        return float(self._mids[j]) * float(self.instruments[j].units_outstanding)

    def _sort_indices_by_performance_or_mcap(self, indices: list[int]) -> list[int]:
        """When *ch24* is warm, rank by 24h % (desc), then mcap. Else by mcap only."""

        if not indices:
            return []
        if self.ch24_warmed:

            def key(i: int) -> tuple[float, float]:
                p = self.pct_change_24h_sim(i)
                p24 = float(p) if p is not None and math.isfinite(float(p)) else -1.0e30
                return (p24, self._mcap(i))

            return sorted(indices, key=key, reverse=True)
        return sorted(indices, key=lambda i: self._mcap(i), reverse=True)

    def _ew_basket_top_indices(self, n: int, pool: str) -> list[int]:
        """*pool* = 'stock' or 'crypto'; top *n* by performance (when warm) or mcap."""

        if pool == "stock":
            raw = list(self._stock_indices)
        elif pool == "crypto":
            raw = list(self._crypto_index)
        else:
            return []
        ranked = self._sort_indices_by_performance_or_mcap(raw)
        if not ranked:
            return []
        k = min(int(n), len(ranked))
        return ranked[:k]

    def _apply_opening_calendar_spreads(self) -> None:
        """Session-aware NPC spread shape by asset class (listed session + crypto weekend profile)."""

        n = int(self._mids.size)
        if n <= 0:
            return
        minute = float(sim_minute_of_day(int(self._tick), self.sim_minutes_per_tick))
        fade_m = max(1.0, float(self.open_liquidity_fade_sim_minutes))
        fade = max(0.0, 1.0 - minute / fade_m)
        extra_bps = max(0.0, float(self.open_liquidity_extra_bps_peak)) * fade
        if self.chaos_liquidity_drought and int(self._tick) < int(self._chaos_liquidity_drought_until):
            extra_bps += 16.0
        day_idx = int((int(self._tick) * max(float(self.sim_minutes_per_tick), 1e-9)) // 1440) % 7
        is_weekend = day_idx in (5, 6)
        listed_open = 570.0 <= minute <= 960.0  # ~09:30-16:00
        for k in range(n):
            ob = self.books[k]
            lb = float(self._cal_bid_sh[k])
            la = float(self._cal_ask_sh[k])
            if lb != 0.0 or la != 0.0:
                ob.reprice_npc_sides(-lb, -la)
            mid0 = max(float(MIN_MID), float(self._mids[k]))
            ins = self.instruments[k]
            if ins.is_listed_equity:
                # Open auction wider, then regular session tighter, after-hours wider again.
                if listed_open:
                    eb = extra_bps
                else:
                    eb = max(extra_bps, 7.5)
            elif ins.kind is AssetClass.CRYPTO:
                # 24/7 crypto: mild weekend liquidity deterioration.
                eb = 1.6 if is_weekend else 0.6
            else:
                eb = extra_bps
            w = mid0 * (max(0.0, eb) / 10_000.0) * 0.5
            self._cal_bid_sh[k] = -w
            self._cal_ask_sh[k] = w
            if w > 1e-14:
                ob.reprice_npc_sides(-w, w)

    def _decay_headline_vol_addon(self) -> None:
        self._headline_vol_addon *= 0.9888

    def _maybe_roll_headline(self) -> None:
        n = int(self._mids.size)
        stocks = list(self._stock_indices)
        if n < 2 or not stocks:
            return
        mean_ticks = max(200.0, float(self.headline_mean_ticks))
        if float(self._rng.random()) >= (1.0 / mean_ticks):
            return
        keys_in_play = [
            k
            for k in STOCK_SECTOR_KEYS
            if any(str(self.instruments[j].sector) == k for j in stocks)
        ]
        if not keys_in_play:
            keys_in_play = list(STOCK_SECTOR_KEYS)
        mode = int(self._rng.integers(0, 3))
        bump = float(self._rng.uniform(0.14, 0.38))
        line = ""
        tag = ""
        if mode == 0:
            line = "Tape: macro headline — vol lifts across listed names."
            tag = "macro"
            m = bump * 0.52
            for j in range(n):
                if self.instruments[j].is_listed_equity:
                    self._headline_vol_addon[j] = min(
                        2.2, float(self._headline_vol_addon[j]) + m
                    )
        elif mode == 1:
            sk = str(self._rng.choice(keys_in_play))
            line = f"Sector wire: {sk} group in focus after headline flow."
            tag = f"sector:{sk}"
            for j in stocks:
                if str(self.instruments[j].sector) == sk:
                    self._headline_vol_addon[j] = min(
                        2.2, float(self._headline_vol_addon[j]) + bump
                    )
        else:
            j = int(self._rng.choice(np.array(stocks, dtype=np.int64)))
            tk = self.instruments[j].ticker
            line = f"{tk}: idiosyncratic headline — two-way interest."
            tag = f"sym:{tk}"
            self._headline_vol_addon[j] = min(
                2.2, float(self._headline_vol_addon[j]) + bump * 1.08
            )
        self._news.append({"tick": int(self._tick), "line": line, "tag": tag})

    def news_feed(self) -> list[dict[str, int | str]]:
        return [dict(x) for x in self._news]

    def _sync_ew_basket_fund_mids(self) -> None:
        """
        Mega T16 / T25 / C3 / S10 *fund* share prices: each tick, mid = equal-weight
        mean of the **current** top-N constituents (N=16, 25, 3, 10 sector) ranked by
        24h % return once the window is full; until then, by mcap. Independent
        GBM on those fund rows is overwritten so NAV tracks the actual basket.
        """

        for j, ins in enumerate(self.instruments):
            if ins.kind is not AssetClass.FUND:
                continue
            sec = str(ins.sector)
            members: list[int] = []
            if sec == "ew_top16_stocks":
                members = self._ew_basket_top_indices(int(MEGA_FUND_BASKET_N1), "stock")
            elif sec == "ew_top25_stocks":
                members = self._ew_basket_top_indices(int(MEGA_FUND_BASKET_N2), "stock")
            elif sec == "ew_top3_crypto":
                members = self._ew_basket_top_indices(3, "crypto")
            elif sec.startswith(EW_S10_PREFIX):
                key = sec[len(EW_S10_PREFIX) :]
                raw = [
                    i
                    for i in self._stock_indices
                    if str(self.instruments[i].sector) == key
                ]
                members = self._sort_indices_by_performance_or_mcap(raw)[
                    : int(MEGA_SECTOR_ETF_BASKET_N)
                ]
            elif sec == EW_S2F_EXP:
                raw = list(self._stock_indices)
                ranked = self._sort_indices_by_performance_or_mcap(raw)
                take = max(1, int(math.ceil((2.0 * len(ranked)) / 5.0))) if ranked else 0
                members = ranked[:take]
            elif sec == EW_C1F_EXP:
                raw = list(self._crypto_index)
                ranked = self._sort_indices_by_performance_or_mcap(raw)
                take = max(1, int(math.ceil((1.0 * len(ranked)) / 5.0))) if ranked else 0
                members = ranked[:take]
            else:
                continue
            if not members:
                continue
            if sec in (EW_S2F_EXP, EW_C1F_EXP):
                # Exponential-ish mcap weighting: larger caps get disproportionate weight.
                vals: list[tuple[float, float]] = []
                for i in members:
                    cap = max(1.0, float(self._mids[i]) * float(self.instruments[i].units_outstanding))
                    w = cap ** 1.35
                    vals.append((w, float(self._mids[i])))
                tw = float(sum(w for w, _ in vals))
                nav = (
                    float(sum(w * px for w, px in vals) / tw)
                    if tw > 1e-12
                    else float(sum(float(self._mids[i]) for i in members) / float(len(members)))
                )
            else:
                acc = 0.0
                for i in members:
                    acc += float(self._mids[i])
                nav = acc / float(len(members))
            if math.isfinite(nav) and nav > 0.0:
                # Funds should track basket NAV, but avoid violent one-tick jumps.
                prev = max(float(MIN_MID), float(self._mids[j]))
                tgt = float(nav)
                max_step = 0.06  # 6% max one-tick reprice for fund rows.
                lo = prev * (1.0 - max_step)
                hi = prev * (1.0 + max_step)
                clipped = min(max(tgt, lo), hi)
                self._mids[j] = max(float(MIN_MID), clipped)

    def _update_financing_regime(self) -> None:
        """
        Per-instrument financing bps/day:
        - crypto: perp-like dynamic funding (can charge longs or shorts)
        - listed: short-carry premium that rises with vol/off-hours stress
        """

        n = int(self._mids.size)
        minute = float(sim_minute_of_day(int(self._tick), self.sim_minutes_per_tick))
        listed_open = 570.0 <= minute <= 960.0  # ~09:30-16:00
        for j in range(n):
            ins = self.instruments[j]
            mid = max(float(MIN_MID), float(self._mids[j]))
            sig = float(self._sigma[j])
            tape_norm = min(2.0, float(self._tape_ema[j]) / max(mid * 2_000_000.0, 1.0))
            self._financing_long_bps[j] = 0.0
            self._financing_short_bps[j] = 0.0
            if ins.kind is AssetClass.CRYPTO:
                p24 = self.pct_change_24h_sim(j)
                p24r = float(p24) / 100.0 if p24 is not None and math.isfinite(float(p24)) else 0.0
                # Positive flow => longs pay shorts; negative flow => shorts pay longs.
                signed = math.tanh(4.0 * p24r + 0.8 * tape_norm) * 8.0
                if signed >= 0.0:
                    self._financing_long_bps[j] = signed
                else:
                    self._financing_short_bps[j] = -signed
                # Structural borrow premium for hard-to-borrow crypto shorts.
                self._financing_short_bps[j] += min(8.0, 0.4 + 180.0 * sig)
            else:
                off_hr = 0.35 if not listed_open else 0.0
                self._financing_short_bps[j] = min(5.0, 0.1 + 80.0 * sig + off_hr)
            if self.chaos_funding_panic and int(self._tick) < int(self._chaos_funding_panic_until):
                self._financing_short_bps[j] *= 2.6
                self._financing_long_bps[j] += 0.5

    def financing_bps_for_index(self, j: int) -> tuple[float, float]:
        if 0 <= int(j) < int(self._mids.size):
            return (float(self._financing_long_bps[int(j)]), float(self._financing_short_bps[int(j)]))
        return (0.0, 0.0)

    def financing_snapshot(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for ins in self.instruments:
            lb, sb = self.financing_bps_for_index(ins.array_index)
            out[str(ins.ticker)] = {"long_bps_per_sim_day": lb, "short_bps_per_sim_day": sb}
        return out

    def step(self) -> None:
        self._apply_opening_calendar_spreads()
        self._apply_chaos_events_pre_gbm()
        self._decay_headline_vol_addon()
        self._maybe_roll_headline()
        t0 = self._mids.copy()
        n = int(self._mids.size)
        if self._gd_armed and self._gd_trigger is not None and int(self._tick) == int(self._gd_trigger):
            pre = self._mids.copy()
            u_eq = float(self._rng.uniform(0.38, 0.52))
            u_cr = float(self._rng.uniform(0.20, 0.36))
            for j in range(n):
                if self.instruments[j].is_listed_equity:
                    self._mids[j] *= 1.0 - u_eq
                else:
                    self._mids[j] *= 1.0 - u_cr
            self._mids = np.maximum(self._mids, float(MIN_MID))
            self._gd_pre = pre
            k_imp = min(n, (n + 99) // 100)  # ≈1% of names (ceiling), never 0 when n > 0
            self._gd_impaired = (
                set() if k_imp < 1 else set(self._rng.permutation(n)[:k_imp].astype(int).tolist())
            )
            self._gd_recovery_left = int(self._rng.integers(180, 400))
            self._gd_lerp = float(self._rng.uniform(0.045, 0.07))
            self._gd_armed = False
            self._gd_fired = True

        self._apply_crypto_sigma_mcaps(self._mids)
        for j in range(n):
            if j not in self._crypto_index:
                self._sigma[j] = self._sigma_baseline[j]
        v_sig = self._volatility_slider_sigma_mult()
        if v_sig != 1.0:
            for j in range(n):
                ins = self.instruments[j]
                if j in self._crypto_index or ins.is_listed_equity:
                    self._sigma[j] = float(max(self._sigma[j] * v_sig, 1e-12))
        ha = np.clip(self._headline_vol_addon, 0.0, 2.5)
        self._sigma = self._sigma * (1.0 + ha)
        if int(self._chaos_sigma_addon.size) == int(self._sigma.size):
            self._sigma = self._sigma * (1.0 + np.clip(self._chaos_sigma_addon, 0.0, 2.5))
        _min_d = sim_minute_of_day(int(self._tick), self.sim_minutes_per_tick)
        _tshape = tod_drift_shape(_min_d)
        _todw = (float(self.tod_signature_bps) / 10_000.0) * _tshape
        v_mu = self._volatility_equity_mu_mult()
        for j in range(n):
            if self.instruments[j].is_listed_equity:
                self._mu[j] = self._mu_equity_addon * v_mu + float(
                    self._rng.standard_normal() * 1e-6
                )
            else:
                self._mu[j] = self.drift_per_tick + float(self._rng.standard_normal() * 1e-6)
            self._mu[j] += 2.2e-5 * _todw

        if self._stock_indices:
            uniq = {str(self.instruments[j].sector) for j in self._stock_indices}
            sector_z = {
                s: float(self._rng.standard_normal() * 5e-5)
                for s in uniq
                if s in STOCK_SECTOR_KEYS
            }
            for j in self._stock_indices:
                ss = str(self.instruments[j].sector)
                if ss in sector_z:
                    self._mu[j] += 0.48 * sector_z[ss]

        tr = int(self._trend_value)
        if tr != 0:
            kf = abs(tr) / 10.0
            mag = 1.4e-4 * kf
            for j in range(n):
                if not self._trend_affects_index(j):
                    continue
                if tr > 0:
                    w = float(self._rng.standard_normal() * 1.2e-5 * kf)
                    self._mu[j] += mag + w
                else:
                    w = float(self._rng.standard_normal() * 4.0e-5 * kf)
                    self._mu[j] -= mag * 0.9 + w

        before = self._mids.copy()
        self._mids = batch_gbm(self._mids, self._mu, self._sigma, dt=self.dt, rng=self._rng)
        self._mids = np.maximum(self._mids, float(MIN_MID))
        if int(self._chaos_price_shock_bps.size) == int(self._mids.size):
            shock_ratio = np.clip(self._chaos_price_shock_bps, -8000.0, 8000.0) / 10_000.0
            self._mids = np.maximum(float(MIN_MID), self._mids * (1.0 + shock_ratio))
            self._chaos_price_shock_bps *= 0.60
        if int(self._chaos_sigma_addon.size) == int(self._mids.size):
            self._chaos_sigma_addon *= 0.88
        if int(self._chaos_halt_until_tick.size) == int(self._mids.size):
            for j in range(n):
                if int(self._chaos_halt_until_tick[j]) > int(self._tick):
                    self._mids[j] = float(t0[j])
        for j in range(n):
            dmag = round_magnet_delta(float(self._mids[j]), float(self.round_magnet_bps))
            self._mids[j] = max(float(MIN_MID), float(self._mids[j] + dmag))
        if self._gd_recovery_left > 0 and self._gd_pre is not None:
            pre = self._gd_pre
            lam = self._gd_lerp
            for j in range(n):
                if j in self._gd_impaired:
                    target = 0.40 * pre[j]
                else:
                    target = 0.99 * pre[j]
                self._mids[j] = (1.0 - lam) * self._mids[j] + lam * target
            self._mids = np.maximum(self._mids, float(MIN_MID))
            self._gd_recovery_left -= 1
        self._listed_equity_and_fund_supply_flow()
        self._mint_inflation_crypto()
        self._sync_ew_basket_fund_mids()
        self._mids = np.maximum(self._mids, float(MIN_MID))
        for k, ins in enumerate(self.instruments):
            d = float(self._mids[k] - t0[k])
            self.books[k].reprice_npcs(d)
            ins.last = float(self._mids[k])
        self._step_micro_tape_and_peer(t0)
        self._update_financing_regime()
        self._apply_natural_volume_tick(t0)
        self._append_ch24_history()
        self._tick += 1
        self._append_chart_row()

    def record_tape(self, i: int, notional: float) -> None:
        """Add player-originated *notional* to order-flow (testing: tape + front-run context)."""

        if 0 <= int(i) < int(self._mids.size):
            self._flow_bump[int(i)] = float(self._flow_bump[int(i)] + max(0.0, float(notional)) * 0.45)

    def wick_context(self, j: int) -> dict[str, float]:
        w = max(2, int(self.micro_wick_lookback))
        dq = self._chart_hist[j]
        if len(dq) < 2:
            return {"upper": 0.0, "lower": 0.0, "body": 0.0}
        take = list(dq)[-w:]
        mids = [p[1] for p in take]
        o, h, lo, c = float(mids[0]), float(max(mids)), float(min(mids)), float(mids[-1])
        ohc = wick_ohlc(o, h, lo, c)
        return ohc

    def _step_micro_tape_and_peer(self, t0: np.ndarray) -> None:
        n = int(self._mids.size)
        a, b = self._peer_a, self._peer_b
        for j in range(n):
            inst = float(abs(self._mids[j] - t0[j]) * max(1.0, float(t0[j])))
            ta = self._tape_a
            self._tape_ema[j] = ta * self._tape_ema[j] + (1.0 - ta) * inst
            self._tape_ema[j] += 0.2 * (float(self._flow_bump[j]) / 1.0) * max(1.0, float(t0[j])) / 1_000_000.0
        self._flow_bump *= 0.88
        self._mqa.append(float(self._mids[a]))
        self._mqb.append(float(self._mids[b]))
        if len(self._mqa) >= 21 and len(self._mqb) >= 21:
            ma0, ma1 = float(self._mqa[0]), float(self._mids[a])
            mb0, mb1 = float(self._mqb[0]), float(self._mids[b])
            if ma0 > 1e-9 and mb0 > 1e-9:
                sa = math.log(max(ma1, 1e-9) / ma0)
                sb = math.log(max(mb1, 1e-9) / mb0)
                self._peer_div_bps = 10000.0 * (sa - sb)
        if len(self._mqa) >= 2 and len(self._mqb) >= 2:
            if self._mqa[-2] > 1e-9 and self._mqb[-2] > 1e-9:
                ra1 = math.log(self._mqa[-1] / self._mqa[-2])
                rb1 = math.log(self._mqb[-1] / self._mqb[-2])
                self._reta.append(ra1)
                self._retb.append(rb1)
        if len(self._reta) >= 8 and len(self._reta) == len(self._retb):
            xs = list(self._reta)[-28:]
            ys = list(self._retb)[-28:]
            if len(xs) == len(ys) and len(xs) >= 5:
                c = np.corrcoef(
                    np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
                )
                if c.shape == (2, 2):
                    v = float(c[0, 1])
                    if math.isfinite(v):
                        self._peer_corr = v

    def micro_tape(self, j: int) -> float:
        return float(self._tape_ema[j]) if 0 <= j < int(self._tape_ema.size) else 0.0

    def micro_global(self) -> dict[str, float | int | str | dict[str, str | float] | None]:
        m = sim_minute_of_day(int(self._tick), self.sim_minutes_per_tick)
        a = int(self._peer_a)
        b = int(self._peer_b)
        fade_m = max(1.0, float(self.open_liquidity_fade_sim_minutes))
        open_cal = max(0.0, 1.0 - float(m) / fade_m) * max(
            0.0, float(self.open_liquidity_extra_bps_peak)
        )
        last_news = self._news[-1] if self._news else None
        return {
            "sim_minute_of_day": float(m),
            "tod_shape": float(tod_drift_shape(m)),
            "open_calendar_extra_bps": float(open_cal),
            "news_last": dict(last_news) if isinstance(last_news, dict) else None,
            "lagging_peer": {
                "leader": str(self.instruments[a].ticker),
                "lagger": str(self.instruments[b].ticker),
                "divergence_20t_bps": float(self._peer_div_bps),
                "return_corr_ew": float(self._peer_corr),
            },
        }

    def by_ticker(self) -> dict[str, Instrument]:
        return {i.ticker: i for i in self.instruments}
