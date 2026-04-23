from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .instrument import MEGA_N_CRYPTO, MEGA_N_FUNDS, MEGA_N_STOCKS

class GameModeName(str, Enum):
    """Preset difficulty and pacing."""

    FREE = "free"  # user-tuned; built from custom params
    CUSTOM = "custom"
    EASY = "easy"
    HARD = "hard"
    COMPLEX = "complex"
    SIMPLE = "simple"


@dataclass(frozen=True, slots=True)
class GameConfig:
    """Tunable simulation: speeds, vol, count of names, and starting capital."""

    label: str
    # Wall-clock seconds per simulation tick (ultra-slow to ultra-fast)
    wall_seconds_per_tick: float
    # How much simulated "wall time" one tick represents (1 = abstract unit; use for long vs short)
    sim_time_scale: float
    # Price dynamics multiplier on base daily vol
    vol_multiplier: float
    # Drift bias per tick (rough trend); can be +/-
    drift_bias: float
    # Starting cash
    starting_cash: float
    # Number of tradeable names (stocks + crypto)
    n_stocks: int
    n_crypto: int
    # Wider spread in harder modes (slippage feel)
    spread_bps: float
    # Seed; None = OS entropy
    seed: int | None = None
    # For *run* / calendar commands: how many *sim* minutes one *step()* is worth (UI scale only;
    # does not have to match GBM *dt*). E.g. 15 → 96 steps ≈ 1 sim day.
    sim_minutes_per_tick: float = 15.0
    n_funds: int = 0
    # Expected long-run appreciation for stocks & fund shares per *sim* year (exponential, GBM drift).
    stock_fund_annual_return: float = 0.08
    # If True for a new session: a crash fires once at a random tick in [500,1000], then reverts; option is one-shot.
    great_depression: bool = False
    # Optional chaos toggles (misc panel).
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
    # Crypto: top *N* by mcap are calmer; below that, *higher* vol as mcap is smaller
    crypto_top_tier: int = 3
    crypto_tier_vol_mult: float = 0.72
    crypto_mcap_vol_power: float = 0.22
    crypto_mcap_ref_usd: float = 1.0e8
    crypto_vol_max_mult: float = 3.5
    # Microstructure: fees, slippage, tape, wick, TOD, magnet, front-run (taker; see *Market*).
    taker_fee_bps: float = 2.0
    slippage_bps_base: float = 1.2
    slippage_bps_per_million: float = 2.0
    front_run_bps: float = 0.4
    front_run_notional_usd: float = 20_000.0
    round_magnet_bps: float = 0.25
    tod_signature_bps: float = 1.0
    micro_tape_ema: float = 0.2
    micro_wick_lookback: int = 8
    # Realism (optional): *max_leverage* 1.0 = cash only; 2.0 = gross exposure ≤ 2× equity. Shorts
    # increase gross. *maintenance_margin_rate*: if equity < rate × gross, forced liquidation. Borrow
    # cost is charged per sim day on |short| notional. *overnight_gap_max_bps* = random |gap| on each sim day.
    max_leverage: float = 1.0
    maintenance_margin_rate: float = 0.2
    shorting_enabled: bool = False
    short_borrow_bps_per_sim_day: float = 0.0
    sec_fee_sell_bps: float = 0.0
    overnight_gap_max_bps: float = 0.0


def preset(mode: GameModeName) -> GameConfig:
    """Return a baseline config for a named mode."""

    u = (MEGA_N_STOCKS, MEGA_N_FUNDS, MEGA_N_CRYPTO)  # 32+4+8: same for every mode
    if mode is GameModeName.SIMPLE:
        return GameConfig(
            label="simple",
            wall_seconds_per_tick=0.35,
            sim_time_scale=1.0,
            vol_multiplier=0.5,
            drift_bias=0.0,
            starting_cash=25_000_000.0,
            n_stocks=u[0],
            n_crypto=u[2],
            spread_bps=8.0,
            seed=None,
            sim_minutes_per_tick=15.0,
            n_funds=u[1],
            crypto_top_tier=3,
        )
    if mode is GameModeName.EASY:
        return GameConfig(
            label="easy",
            wall_seconds_per_tick=0.45,
            sim_time_scale=0.5,
            vol_multiplier=0.6,
            drift_bias=0.00005,
            starting_cash=40_000_000.0,
            n_stocks=u[0],
            n_crypto=u[2],
            spread_bps=6.0,
            n_funds=u[1],
            crypto_top_tier=3,
        )
    if mode is GameModeName.HARD:
        return GameConfig(
            label="hard",
            wall_seconds_per_tick=0.2,
            sim_time_scale=2.0,
            vol_multiplier=1.4,
            drift_bias=0.0,
            starting_cash=8_000_000.0,
            n_stocks=u[0],
            n_crypto=u[2],
            spread_bps=18.0,
            n_funds=u[1],
            crypto_top_tier=3,
            taker_fee_bps=2.2,
            slippage_bps_base=1.6,
            front_run_bps=0.6,
            max_leverage=1.5,
            maintenance_margin_rate=0.22,
            shorting_enabled=True,
            short_borrow_bps_per_sim_day=1.2,
            sec_fee_sell_bps=0.05,
            overnight_gap_max_bps=0.8,
        )
    if mode is GameModeName.COMPLEX:
        return GameConfig(
            label="complex",
            wall_seconds_per_tick=0.15,
            sim_time_scale=1.5,
            vol_multiplier=1.1,
            drift_bias=-0.00002,
            starting_cash=18_000_000.0,
            n_stocks=u[0],
            n_crypto=u[2],
            spread_bps=12.0,
            n_funds=u[1],
            crypto_top_tier=3,
            max_leverage=1.25,
            maintenance_margin_rate=0.2,
            shorting_enabled=True,
            short_borrow_bps_per_sim_day=0.6,
            sec_fee_sell_bps=0.02,
            overnight_gap_max_bps=0.4,
        )
    if mode in (GameModeName.FREE, GameModeName.CUSTOM):
        return GameConfig(
            label="free",
            wall_seconds_per_tick=0.3,
            sim_time_scale=1.0,
            vol_multiplier=0.9,
            drift_bias=0.0,
            starting_cash=25_000_000.0,
            n_stocks=u[0],
            n_crypto=u[2],
            spread_bps=10.0,
            n_funds=u[1],
            crypto_top_tier=3,
        )
    raise ValueError(f"Unknown mode: {mode}")


def build_custom(
    *,
    wall_seconds_per_tick: float = 0.3,
    sim_time_scale: float = 1.0,
    vol_multiplier: float = 1.0,
    drift_bias: float = 0.0,
    starting_cash: float = 25_000_000.0,
    n_stocks: int = MEGA_N_STOCKS,
    n_funds: int = MEGA_N_FUNDS,
    n_crypto: int = MEGA_N_CRYPTO,
    spread_bps: float = 10.0,
    seed: int | None = None,
    sim_minutes_per_tick: float = 15.0,
    stock_fund_annual_return: float = 0.08,
    great_depression: bool = False,
    chaos_flash_crash: bool = False,
    chaos_meme_squeeze: bool = False,
    chaos_fat_finger: bool = False,
    chaos_exchange_halt: bool = False,
    chaos_rumor_mill: bool = False,
    chaos_sector_rotation: bool = False,
    chaos_funding_panic: bool = False,
    chaos_liquidity_drought: bool = False,
    chaos_whale_rebalance: bool = False,
    chaos_crypto_weekend_mania: bool = False,
    crypto_top_tier: int = 3,
    crypto_tier_vol_mult: float = 0.72,
    crypto_mcap_vol_power: float = 0.22,
    crypto_mcap_ref_usd: float = 1.0e8,
    crypto_vol_max_mult: float = 3.5,
    taker_fee_bps: float = 2.0,
    slippage_bps_base: float = 1.2,
    slippage_bps_per_million: float = 2.0,
    front_run_bps: float = 0.4,
    front_run_notional_usd: float = 20_000.0,
    round_magnet_bps: float = 0.25,
    tod_signature_bps: float = 1.0,
    micro_tape_ema: float = 0.2,
    micro_wick_lookback: int = 8,
    max_leverage: float = 1.0,
    maintenance_margin_rate: float = 0.2,
    shorting_enabled: bool = False,
    short_borrow_bps_per_sim_day: float = 0.0,
    sec_fee_sell_bps: float = 0.0,
    overnight_gap_max_bps: float = 0.0,
) -> GameConfig:
    """Arbitrary 'free' or 'custom' run."""

    return GameConfig(
        label="custom",
        wall_seconds_per_tick=wall_seconds_per_tick,
        sim_time_scale=sim_time_scale,
        vol_multiplier=vol_multiplier,
        drift_bias=drift_bias,
        starting_cash=starting_cash,
        n_stocks=n_stocks,
        n_crypto=n_crypto,
        spread_bps=spread_bps,
        seed=seed,
        sim_minutes_per_tick=sim_minutes_per_tick,
        n_funds=n_funds,
        stock_fund_annual_return=stock_fund_annual_return,
        crypto_top_tier=crypto_top_tier,
        crypto_tier_vol_mult=crypto_tier_vol_mult,
        crypto_mcap_vol_power=crypto_mcap_vol_power,
        crypto_mcap_ref_usd=crypto_mcap_ref_usd,
        crypto_vol_max_mult=crypto_vol_max_mult,
        great_depression=great_depression,
        chaos_flash_crash=chaos_flash_crash,
        chaos_meme_squeeze=chaos_meme_squeeze,
        chaos_fat_finger=chaos_fat_finger,
        chaos_exchange_halt=chaos_exchange_halt,
        chaos_rumor_mill=chaos_rumor_mill,
        chaos_sector_rotation=chaos_sector_rotation,
        chaos_funding_panic=chaos_funding_panic,
        chaos_liquidity_drought=chaos_liquidity_drought,
        chaos_whale_rebalance=chaos_whale_rebalance,
        chaos_crypto_weekend_mania=chaos_crypto_weekend_mania,
        taker_fee_bps=taker_fee_bps,
        slippage_bps_base=slippage_bps_base,
        slippage_bps_per_million=slippage_bps_per_million,
        front_run_bps=front_run_bps,
        front_run_notional_usd=front_run_notional_usd,
        round_magnet_bps=round_magnet_bps,
        tod_signature_bps=tod_signature_bps,
        micro_tape_ema=micro_tape_ema,
        micro_wick_lookback=micro_wick_lookback,
        max_leverage=max_leverage,
        maintenance_margin_rate=maintenance_margin_rate,
        shorting_enabled=shorting_enabled,
        short_borrow_bps_per_sim_day=short_borrow_bps_per_sim_day,
        sec_fee_sell_bps=sec_fee_sell_bps,
        overnight_gap_max_bps=overnight_gap_max_bps,
    )
