from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from statistics import mean

# Max float vs initial *listed* shares / fund units (splits cannot exceed this).
LISTED_MAX_SUPPLY_MULT_STOCK: float = 50.0
LISTED_MAX_SUPPLY_MULT_FUND: float = 50.0
# Bitcoin-style hard cap for the single capped crypto in a universe.
CRYPTO_HARD_CAP_UNITS: float = 21_000_000.0

class AssetClass(str, Enum):
    STOCK = "stock"
    FUND = "fund"  # ETF / mutual fund: same long-run growth model as listed stocks
    CRYPTO = "crypto"


# Mega universe: 32+4+8 (random US megas, four index-style funds, tiered cryptos)
MEGA_N_STOCKS = 32
MEGA_N_FUNDS = 4
MEGA_N_CRYPTO = 8
# Equal-weight index funds: top-N by 24h % (when warm) else mcap (replaces old top-10 / top-20).
MEGA_FUND_BASKET_N1 = 16
MEGA_FUND_BASKET_N2 = 25
# Sector ETF: equal-weight top-N **stocks** in one GICS-style bucket (see *STOCK_SECTOR_KEYS*).
MEGA_SECTOR_ETF_BASKET_N = 10
# Labels assigned to mega / classic stocks for correlated drift and the S10 sector fund.
STOCK_SECTOR_KEYS = ("TECH", "HLTH", "FIN", "ENRG", "CNSM", "INDU", "MATR", "UTIL")
EW_S10_PREFIX = "ew_s10_"
EW_ALL_MINI_PREFIX = "ew_allmini_"
EW_SPX_PREFIX = "ew_spx_"
EW_S2F_EXP = "ew_s2f_exp"
EW_C1F_EXP = "ew_c1f_exp"


# Generic pools for name generation; extend anytime
STOCK_PREFIXES = ("ARC", "NOR", "SUM", "WEST", "BLUE", "IRON", "NOVA", "CROWN")
STOCK_SUFFIXES = ("LABS", "CAP", "GROUP", "IND", "SYS", "ENERGY", "BIO", "MEDIA")
FUND_PREFIXES = ("IDX", "BETA", "CORE", "US", "GLOB", "RET", "BAL", "GROW")
FUND_SUFFIXES = ("ETF", "TR", "FUND", "MKT", "500", "CAP", "IDX", "BAL")
CRYPTO_PREFIXES = ("moon", "zero", "meta", "flux", "dark", "neo", "hyper", "quant")
CRYPTO_SUFFIXES = ("coin", "cash", "bit", "token", "link", "pad", "swap", "fi")


def _build_unique_tickers(
    n: int,
    rng,
    prefixes: tuple[str, ...],
    suffixes: tuple[str, ...],
    *,
    combine,
    fallback_prefix: str,
    start: int = 1,
) -> list[str]:
    """
    Build up to *n* unique tickers from prefix/suffix pools, then fall back to
    numbered symbols if caller asks for more than the cartesian product can provide.
    """

    if n <= 0:
        return []
    combos = [combine(p, s) for p in prefixes for s in suffixes]
    # Preserve seed-determinism by using RNG for the base cartesian shuffle.
    rng.shuffle(combos)
    chosen = combos[:n]
    if len(chosen) < n:
        used = set(chosen)
        k = start
        while len(chosen) < n:
            t = f"{fallback_prefix}{k:03d}"
            if t not in used:
                chosen.append(t)
                used.add(t)
            k += 1
    return sorted(chosen)


def _ticker_stocks(n: int, rng) -> list[str]:
    return _build_unique_tickers(
        n,
        rng,
        STOCK_PREFIXES,
        STOCK_SUFFIXES,
        combine=lambda a, b: f"{a}{b[:3]}",
        fallback_prefix="STK",
    )


def _ticker_funds(n: int, rng) -> list[str]:
    return _build_unique_tickers(
        n,
        rng,
        FUND_PREFIXES,
        FUND_SUFFIXES,
        combine=lambda a, b: f"{a}{b}",
        fallback_prefix="FND",
    )


def _ticker_crypto(n: int, rng) -> list[str]:
    return _build_unique_tickers(
        n,
        rng,
        CRYPTO_PREFIXES,
        CRYPTO_SUFFIXES,
        combine=lambda a, b: f"{a[:4]}{b}".upper(),
        fallback_prefix="CRY",
    )


@dataclass
class Instrument:
    """Name with a mid, supply for *marketcap = price * units_shares*."""

    id: int
    ticker: str
    kind: AssetClass
    # Annual rough scale; per-tick *sigma* is built in *Market* (crypto updated by mcap)
    base_annual_vol: float
    # Price
    last: float
    # Outstanding shares (stocks, funds) or circulating / indexed units (crypto)
    units_outstanding: float
    # Optional tag
    sector: str = "general"
    array_index: int = 0
    # None = no hard ceiling (mintable cryptos). Otherwise *units_outstanding* must stay ≤ this.
    max_units_outstanding: float | None = None
    # If True, *Market* may inflate *units_outstanding* with a dilution-style price scale each tick.
    allow_supply_inflation: bool = False

    @property
    def market_cap(self) -> float:
        return max(0.0, float(self.last)) * max(0.0, float(self.units_outstanding))

    @property
    def is_listed_equity(self) -> bool:
        return self.kind in (AssetClass.STOCK, AssetClass.FUND)


def _exp_weighted_nav(items: list[Instrument], power: float = 1.35) -> float:
    """
    Exponential-ish mcap weighting: weight ~ mcap**power (power>1 overweights larger caps).
    Returns simple mean fallback if totals are degenerate.
    """

    if not items:
        return 0.0
    p = max(1.0, float(power))
    ws: list[float] = []
    ps: list[float] = []
    for x in items:
        m = max(1.0, float(x.market_cap))
        w = m ** p
        ws.append(w)
        ps.append(float(x.last))
    tw = float(sum(ws))
    if tw <= 1e-12 or not (tw > 0.0):
        return float(mean(ps))
    return float(sum(w * px for w, px in zip(ws, ps)) / tw)


def _auto_all_stock_topn_ladder(n_stocks: int) -> list[int]:
    """
    SALL tracks only the top third of all stocks (dynamic basket).
    Keep list-return shape for existing call sites.
    """

    n = max(0, int(n_stocks))
    if n < 10:
        return []
    return [max(1, int(round(float(n) / 3.0)))]


def _make_mega_cap_universe(rng) -> list[Instrument]:
    """
    32 US mega-cap style stocks (800B–1.5T mcap), 8 cryptos (3 large 500B–750B, 5 mid/small 10B–100B),
    4 index-style funds: T16 / T25 / C3 / S10SEC; *Market* sets each fund's NAV to the equal-weight mean of
    the *current* top-16 / top-25 / top-3 (stocks or crypto) and top-10 stocks in the fund's sector,
    ranked by 24h % when warm, else by mcap.
    """

    out: list[Instrument] = []
    n1, n2 = MEGA_FUND_BASKET_N1, MEGA_FUND_BASKET_N2
    stocksyms = _ticker_stocks(MEGA_N_STOCKS, rng)
    cryposyms = _ticker_crypto(MEGA_N_CRYPTO, rng)
    # --- Stocks: 800B - 1.5T, then order by mcap ---
    sector_ring = list(STOCK_SECTOR_KEYS)
    rng.shuffle(sector_ring)
    stocks: list[Instrument] = []
    for si, t in enumerate(stocksyms):
        mcap = float(rng.uniform(800_000_000_000.0, 1_500_000_000_000.0))
        price = float(rng.uniform(20.0, 480.0))
        units = mcap / max(price, 1e-3)
        sec = sector_ring[si % len(sector_ring)]
        stocks.append(
            Instrument(
                id=0,
                ticker=t,
                kind=AssetClass.STOCK,
                base_annual_vol=float(rng.uniform(0.12, 0.38)),
                last=price,
                units_outstanding=units,
                sector=sec,
                array_index=0,
            )
        )
    stocks.sort(key=lambda s: s.market_cap, reverse=True)
    for i, s in enumerate(stocks):
        s.id = i
        s.array_index = i
        s.max_units_outstanding = float(s.units_outstanding) * float(LISTED_MAX_SUPPLY_MULT_STOCK)
        s.allow_supply_inflation = False
        out.append(s)
    nav_t16 = float(mean([s.last for s in stocks[:n1]]))
    nav_t25 = float(mean([s.last for s in stocks[:n2]]))
    counts: dict[str, int] = {}
    for s in stocks:
        k = str(s.sector)
        if k in STOCK_SECTOR_KEYS:
            counts[k] = counts.get(k, 0) + 1
    eligible = [k for k, v in counts.items() if v >= 4]
    etf_key = str(rng.choice(eligible if eligible else list(STOCK_SECTOR_KEYS)))
    in_sector = [s for s in stocks if str(s.sector) == etf_key]
    in_sector.sort(key=lambda s: s.market_cap, reverse=True)
    basket = in_sector[: int(MEGA_SECTOR_ETF_BASKET_N)]
    nav_s10 = float(mean([s.last for s in basket])) if basket else nav_t16
    # --- Crypto: 3 large (500B–750B) + 5 small (10B–100B) ---
    top3: list[Instrument] = []
    for _ in range(3):
        mcap = float(rng.uniform(500_000_000_000.0, 750_000_000_000.0))
        price = float(rng.uniform(0.2, 250.0))
        supply = mcap / max(price, 1e-6)
        top3.append(
            Instrument(
                id=0,
                ticker="",
                kind=AssetClass.CRYPTO,
                base_annual_vol=float(rng.uniform(0.28, 0.8)),
                last=price,
                units_outstanding=supply,
                sector="crypto",
                array_index=0,
            )
        )
    small: list[Instrument] = []
    for _ in range(5):
        mcap = float(rng.uniform(10_000_000_000.0, 100_000_000_000.0))
        price = float(rng.uniform(0.1, 80.0))
        supply = mcap / max(price, 1e-6)
        small.append(
            Instrument(
                id=0,
                ticker="",
                kind=AssetClass.CRYPTO,
                base_annual_vol=float(rng.uniform(0.5, 1.15)),
                last=price,
                units_outstanding=supply,
                sector="crypto",
                array_index=0,
            )
        )
    for i, t in enumerate(cryposyms[:3]):
        top3[i].ticker = t
    for i, t in enumerate(cryposyms[3:8]):
        small[i].ticker = t
    all_crypto = top3 + small
    all_crypto.sort(key=lambda c: c.market_cap, reverse=True)
    for ci, c in enumerate(all_crypto):
        if ci == 0:
            cap = float(CRYPTO_HARD_CAP_UNITS)
            u = float(c.units_outstanding)
            if u > cap:
                c.last = float(c.last) * (u / cap)
                c.units_outstanding = cap
            c.max_units_outstanding = cap
            c.allow_supply_inflation = False
        else:
            c.max_units_outstanding = None
            c.allow_supply_inflation = True
    nav_top3c = float(mean([c.last for c in all_crypto[:3]]))
    cbase = MEGA_N_STOCKS
    for i, c in enumerate(all_crypto):
        c.id = cbase + i
        c.array_index = cbase + i
        out.append(c)
    # --- 3 index funds: NAV = basket average; choose mcap 20B–80B for float ---
    fbase = cbase + MEGA_N_CRYPTO
    fund_specs: list[tuple[str, float, str]] = [
        ("T16AVG", nav_t16, "ew_top16_stocks"),
        ("T25AVG", nav_t25, "ew_top25_stocks"),
        ("C3MKT", nav_top3c, "ew_top3_crypto"),
        ("S10SEC", nav_s10, f"{EW_S10_PREFIX}{etf_key}"),
    ]
    # Auto basket funds:
    # - 5th: top 2/5 of stocks (exp mcap weighted)
    # - 6th: top 1/5 of cryptos (exp mcap weighted)
    top_s = max(1, int(math.ceil((2.0 * len(stocks)) / 5.0)))
    top_c = max(1, int(math.ceil((1.0 * len(all_crypto)) / 5.0)))
    nav_s2f = _exp_weighted_nav(stocks[:top_s], power=1.35)
    nav_c1f = _exp_weighted_nav(all_crypto[:top_c], power=1.35)
    fund_specs.extend(
        [
            ("S2FEXP", nav_s2f, EW_S2F_EXP),
            ("C1FEXP", nav_c1f, EW_C1F_EXP),
        ]
    )
    mini_topn = _auto_all_stock_topn_ladder(len(stocks))
    for topn in mini_topn:
        nav_mini = float(mean([s.last for s in stocks[: int(topn)]])) if stocks else nav_t16
        fund_specs.append((f"SALL{int(topn):03d}", nav_mini, f"{EW_ALL_MINI_PREFIX}{int(topn)}"))
    # Broad stock index tiers (S&P-style mini indexes): top 10/20/30 by rank.
    for spn in (10, 20, 30):
        if len(stocks) >= spn:
            nav_sp = float(mean([s.last for s in stocks[:spn]]))
            fund_specs.append((f"SPX{spn:02d}", nav_sp, f"{EW_SPX_PREFIX}{spn}"))
    for j, (ftick, last0, ssec) in enumerate(fund_specs):
        f_mcap = float(rng.uniform(20_000_000_000.0, 80_000_000_000.0))
        units = f_mcap / max(last0, 1e-6)
        out.append(
            Instrument(
                id=fbase + j,
                ticker=ftick,
                kind=AssetClass.FUND,
                base_annual_vol=float(rng.uniform(0.08, 0.22)),
                last=last0,
                units_outstanding=units,
                sector=ssec,
                array_index=fbase + j,
            )
        )
        fu = out[-1]
        fu.max_units_outstanding = float(fu.units_outstanding) * float(LISTED_MAX_SUPPLY_MULT_FUND)
        fu.allow_supply_inflation = False
    return out


def _make_classic_universe(
    n_stocks: int, n_funds: int, n_crypto: int, rng
) -> list[Instrument]:
    """Original procedural universe (broad mcap and price ranges)."""

    stocksyms = _ticker_stocks(n_stocks, rng) if n_stocks else []
    fundsyms = _ticker_funds(n_funds, rng) if n_funds else []
    cryposyms = _ticker_crypto(n_crypto, rng) if n_crypto else []
    out: list[Instrument] = []
    stocks_created: list[Instrument] = []
    cryptos_created: list[Instrument] = []
    idx = 0
    for t in stocksyms:
        start = float(rng.uniform(20.0, 240.0))
        vol = float(rng.uniform(0.15, 0.45))
        shares = float(rng.integers(8_000_000, 600_000_000))
        out.append(
            Instrument(
                id=idx,
                ticker=t,
                kind=AssetClass.STOCK,
                base_annual_vol=vol,
                last=start,
                units_outstanding=shares,
                sector=str(rng.choice(STOCK_SECTOR_KEYS)),
                array_index=idx,
            )
        )
        stocks_created.append(out[-1])
        idx += 1
    for t in fundsyms:
        start = float(rng.uniform(12.0, 180.0))
        vol = float(rng.uniform(0.10, 0.32))
        shares = float(rng.integers(40_000_000, 2_200_000_000))
        out.append(
            Instrument(
                id=idx,
                ticker=t,
                kind=AssetClass.FUND,
                base_annual_vol=vol,
                last=start,
                units_outstanding=shares,
                sector="fund",
                array_index=idx,
            )
        )
        idx += 1
    for t in cryposyms:
        start = float(rng.uniform(0.3, 120.0))
        vol = float(rng.uniform(0.4, 1.2))
        supply = float(10 ** rng.uniform(4.0, 10.0))
        out.append(
            Instrument(
                id=idx,
                ticker=t,
                kind=AssetClass.CRYPTO,
                base_annual_vol=vol,
                last=start,
                units_outstanding=supply,
                sector="crypto",
                array_index=idx,
            )
        )
        cryptos_created.append(out[-1])
        idx += 1
    # Auto basket funds:
    # - 5th style fund: top 2/5 stocks (exp mcap weighted)
    # - 6th style fund: top 1/5 cryptos (exp mcap weighted)
    if stocks_created:
        ss = sorted(stocks_created, key=lambda x: x.market_cap, reverse=True)
        k = max(1, int(math.ceil((2.0 * len(ss)) / 5.0)))
        nav = _exp_weighted_nav(ss[:k], power=1.35)
        f_mcap = float(rng.uniform(20_000_000_000.0, 80_000_000_000.0))
        units = f_mcap / max(nav, 1e-6)
        out.append(
            Instrument(
                id=idx,
                ticker="S2FEXP",
                kind=AssetClass.FUND,
                base_annual_vol=float(rng.uniform(0.08, 0.22)),
                last=nav,
                units_outstanding=units,
                sector=EW_S2F_EXP,
                array_index=idx,
            )
        )
        idx += 1
    mini_topn = _auto_all_stock_topn_ladder(len(stocks_created))
    for topn in mini_topn:
        ranked = sorted(stocks_created, key=lambda x: x.market_cap, reverse=True)
        use = ranked[: int(topn)]
        if not use:
            continue
        nav = float(mean([x.last for x in use]))
        f_mcap = float(rng.uniform(10_000_000_000.0, 50_000_000_000.0))
        units = f_mcap / max(nav, 1e-6)
        out.append(
            Instrument(
                id=idx,
                ticker=f"SALL{int(topn):03d}",
                kind=AssetClass.FUND,
                base_annual_vol=float(rng.uniform(0.08, 0.22)),
                last=nav,
                units_outstanding=units,
                sector=f"{EW_ALL_MINI_PREFIX}{int(topn)}",
                array_index=idx,
            )
        )
        idx += 1
    ranked_all = sorted(stocks_created, key=lambda x: x.market_cap, reverse=True)
    for spn in (10, 20, 30):
        use = ranked_all[:spn]
        if len(use) < spn:
            continue
        nav = float(mean([x.last for x in use]))
        f_mcap = float(rng.uniform(12_000_000_000.0, 55_000_000_000.0))
        units = f_mcap / max(nav, 1e-6)
        out.append(
            Instrument(
                id=idx,
                ticker=f"SPX{spn:02d}",
                kind=AssetClass.FUND,
                base_annual_vol=float(rng.uniform(0.08, 0.20)),
                last=nav,
                units_outstanding=units,
                sector=f"{EW_SPX_PREFIX}{spn}",
                array_index=idx,
            )
        )
        idx += 1
    if cryptos_created:
        cc = sorted(cryptos_created, key=lambda x: x.market_cap, reverse=True)
        k = max(1, int(math.ceil((1.0 * len(cc)) / 5.0)))
        nav = _exp_weighted_nav(cc[:k], power=1.35)
        f_mcap = float(rng.uniform(8_000_000_000.0, 45_000_000_000.0))
        units = f_mcap / max(nav, 1e-6)
        out.append(
            Instrument(
                id=idx,
                ticker="C1FEXP",
                kind=AssetClass.FUND,
                base_annual_vol=float(rng.uniform(0.10, 0.30)),
                last=nav,
                units_outstanding=units,
                sector=EW_C1F_EXP,
                array_index=idx,
            )
        )
        idx += 1
    cryptos = [x for x in out if x.kind is AssetClass.CRYPTO]
    cryptos.sort(key=lambda c: c.market_cap, reverse=True)
    cap_rank = {id(c): i for i, c in enumerate(cryptos)}
    for ins in out:
        if ins.kind is AssetClass.STOCK:
            ins.max_units_outstanding = (
                float(ins.units_outstanding) * float(LISTED_MAX_SUPPLY_MULT_STOCK)
            )
            ins.allow_supply_inflation = False
        elif ins.kind is AssetClass.FUND:
            ins.max_units_outstanding = (
                float(ins.units_outstanding) * float(LISTED_MAX_SUPPLY_MULT_FUND)
            )
            ins.allow_supply_inflation = False
        elif ins.kind is AssetClass.CRYPTO and cryptos:
            if cap_rank.get(id(ins), 999) == 0:
                cap = float(CRYPTO_HARD_CAP_UNITS)
                u = float(ins.units_outstanding)
                if u > cap:
                    ins.last = float(ins.last) * (u / cap)
                    ins.units_outstanding = cap
                ins.max_units_outstanding = cap
                ins.allow_supply_inflation = False
            else:
                ins.max_units_outstanding = None
                ins.allow_supply_inflation = True
    return out


def make_universe(
    n_stocks: int,
    n_funds: int,
    n_crypto: int,
    rng,
) -> list[Instrument]:
    """Procedural tickers, float, supply; *rng*-deterministic. Special mega-cap for 32+4+8."""

    if (
        n_stocks == MEGA_N_STOCKS
        and n_funds == MEGA_N_FUNDS
        and n_crypto == MEGA_N_CRYPTO
    ):
        return _make_mega_cap_universe(rng)
    return _make_classic_universe(n_stocks, n_funds, n_crypto, rng)
