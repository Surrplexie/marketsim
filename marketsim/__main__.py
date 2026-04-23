from __future__ import annotations

import argparse
import time
from dataclasses import replace

from rich.console import Console

from .engine import new_session, Session
from .modes import GameConfig, GameModeName, build_custom, preset
from .market import Side
from .player import Result
from .sim_time import parse_run_line
from .tui import render_session, HELP

MODES = [m.value for m in GameModeName]


def _result_msg(r: Result) -> str:
    return {
        Result.OK: "ok",
        Result.NO_CASH: "insufficient cash",
        Result.NO_POSITION: "not enough position",
        Result.NOT_FOUND: "unknown ticker",
        Result.BAD_SIZE: "size / notional invalid (min lot ~1e-8, min notional ~1e-6)",
        Result.BAD_PRICE: "bad price",
        Result.NO_LIQUIDITY: "not enough resting liquidity on the book",
    }.get(r, str(r))


def _make_config(ns: argparse.Namespace) -> GameConfig:
    if ns.custom:
        return build_custom(
            wall_seconds_per_tick=ns.wall if ns.wall is not None else 0.3,
            sim_time_scale=ns.time_scale if ns.time_scale is not None else 1.0,
            vol_multiplier=ns.vol,
            drift_bias=ns.drift,
            starting_cash=ns.cash if ns.cash is not None else 25_000_000.0,
            n_stocks=ns.n_stocks,
            n_funds=ns.n_funds,
            n_crypto=ns.n_crypto,
            spread_bps=ns.spread,
            seed=ns.seed,
            sim_minutes_per_tick=ns.mpt
            if ns.mpt is not None
            else 15.0,
            great_depression=bool(getattr(ns, "great_depression", False)),
        )
    c = preset(GameModeName(ns.mode))
    if ns.seed is not None:
        c = replace(c, seed=ns.seed)
    if ns.wall is not None:
        c = replace(c, wall_seconds_per_tick=ns.wall)
    if ns.time_scale is not None:
        c = replace(c, sim_time_scale=ns.time_scale)
    if ns.cash is not None and ns.custom is False:
        c = replace(c, starting_cash=ns.cash)
    if ns.mpt is not None and ns.custom is False:
        c = replace(c, sim_minutes_per_tick=ns.mpt)
    if bool(getattr(ns, "great_depression", False)):
        c = replace(c, great_depression=True)
    return c


def _run_tui(s: Session) -> None:
    c = Console()
    while True:
        c.clear()
        render_session(c, s)
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            time.sleep(0.05)
            continue
        low = line.lower()
        if low in ("q", "quit", "exit"):
            return
        if low == "h":
            c.print(HELP)
            input("[dim]Press Enter...[/dim]")
            continue
        if low in ("n", "next", ""):
            s.step()
            time.sleep(max(s.config.wall_seconds_per_tick, 0) * 0.25)
            continue
        if low in ("a", "auto5"):
            for _ in range(5):
                s.step()
                time.sleep(s.config.wall_seconds_per_tick)
            continue
        parts = line.split()
        if len(parts) >= 3 and parts[0].lower() == "run":
            try:
                n, u = parse_run_line([parts[1], parts[2]])
            except ValueError as e:
                c.print(f"[red]{e}[/]")
                time.sleep(0.2)
                continue
            if n <= 0:
                c.print("[dim]nothing to do (n<=0)[/]" if n < 0 else "[dim]nothing to do (n=0)[/]")
                continue
            adv = s.advance_interval(n, u)
            c.print(
                f"[green]ok  +{adv.ticks} step(s)  (~{adv.sim_minutes:.1f} sim min  "
                f"· {s.config.sim_minutes_per_tick:g} min/step)[/]"
            )
            time.sleep(0.08)
            continue
        if len(parts) == 3 and parts[0].lower() == "bc":
            tck = parts[1].upper()
            if parts[2].lower() == "all":
                r = s.order_market_buy_cash(tck, float("inf"))
            else:
                try:
                    cash = float(parts[2])
                except ValueError:
                    c.print(f"[red]bad cash: {parts[2]!r}[/]")
                    input("[dim]Enter...[/dim]")
                    continue
                r = s.order_market_buy_cash(tck, cash)
            if r is not Result.OK:
                c.print(f"[red]order: {_result_msg(r)}[/]")
            else:
                c.print("[green]filled[/]")
            time.sleep(0.12)
            continue
        if len(parts) == 3 and parts[0].lower() in ("b", "buy", "s", "sell"):
            side = Side.BUY if parts[0].lower() in ("b", "buy") else Side.SELL
            tck = parts[1].upper()
            qraw = parts[2]
            if side is Side.BUY and qraw.startswith("$"):
                try:
                    cash = float(qraw[1:].replace(",", ""))
                except ValueError:
                    c.print(f"[red]bad cash: {qraw!r}[/]")
                    input("[dim]Enter...[/dim]")
                    continue
                r = s.order_market_buy_cash(tck, cash)
            else:
                try:
                    n = float(qraw)
                except ValueError:
                    c.print(f"[red]bad quantity: {parts[2]}[/]")
                    input("[dim]Enter...[/dim]")
                    continue
                r = s.order(tck, side, n)
            if r is not Result.OK:
                c.print(f"[red]order: {_result_msg(r)}[/]")
            else:
                c.print("[green]filled[/]")
            time.sleep(0.12)
            continue
        if len(parts) == 4 and parts[0].lower() == "lbc":
            tck = parts[1].upper()
            try:
                if parts[2].startswith("$"):
                    budget = float(parts[2][1:].replace(",", ""))
                else:
                    budget = float(parts[2])
                px = float(parts[3])
            except ValueError:
                c.print(f"[red]bad args: {parts!r}[/]")
                time.sleep(0.4)
                continue
            r = s.order_limit_buy_cash(tck, px, budget)
            if r is not Result.OK:
                c.print(f"[red]order: {_result_msg(r)}[/]")
            else:
                c.print("[green]accepted[/]")
            time.sleep(0.12)
            continue
        if len(parts) == 4 and parts[0].lower() in ("lb", "ls"):
            tck = parts[1].upper()
            try:
                n = float(parts[2])
                px = float(parts[3])
            except ValueError:
                c.print(f"[red]bad args: {parts!r}[/]")
                time.sleep(0.4)
                continue
            if parts[0].lower() == "lb":
                r = s.order_limit(tck, Side.BUY, n, px)
            else:
                r = s.order_limit(tck, Side.SELL, n, px)
            if r is not Result.OK:
                c.print(f"[red]order: {_result_msg(r)}[/]")
            else:
                c.print("[green]accepted[/]")
            time.sleep(0.12)
            continue
        c.print(f"[red]Unknown: {line!r}. Try: {HELP}[/]")
        time.sleep(0.4)


def _run_headless(s: Session, ticks: int) -> None:
    for _ in range(ticks):
        s.step()
    print(f"tick={s.market.tick} equity={s.equity:.2f} cash={s.player.cash:.2f}")


def _run_headless_advance(s: Session, unit: str, n: int) -> None:
    adv = s.advance_interval(n, unit)
    print(
        f"tick={s.market.tick} +{adv.ticks} step(s)  "
        f"({n} {unit!r}  ~{adv.sim_minutes:.0f} sim min  min/step={s.config.sim_minutes_per_tick:g})  "
        f"equity={s.equity:.2f} cash={s.player.cash:.2f}"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Terminal market sim — stocks + crypto-style dynamics (GBM, seeded).",
    )
    p.add_argument(
        "--mode",
        default="simple",
        choices=MODES,
        help="Preset difficulty/pace",
    )
    p.add_argument("--custom", action="store_true", help="Use --vol, --n-stocks, etc. as full custom config")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--wall", type=float, default=None, help="Override wall seconds per tick")
    p.add_argument("--time-scale", type=float, default=None, dest="time_scale")
    p.add_argument("--vol", type=float, default=1.0, help="[custom] vol multiplier")
    p.add_argument("--drift", type=float, default=0.0, help="[custom] drift bias")
    p.add_argument("--cash", type=float, default=None)
    p.add_argument(
        "--n-stocks",
        type=int,
        default=20,
        dest="n_stocks",
        help="[--custom] count (32+4+8 = mega-cap universe; other mixes use classic generator)",
    )
    p.add_argument(
        "--n-funds",
        type=int,
        default=3,
        dest="n_funds",
        help="[--custom] fund count (use 4 with 32 stocks + 8 crypto for mega T16/T25/C3/S10 index set)",
    )
    p.add_argument(
        "--n-crypto",
        type=int,
        default=5,
        dest="n_crypto",
        help="[--custom] cryptos (8 with 32+4 = mega tiered crypto + four index funds)",
    )
    p.add_argument("--spread", type=float, default=10.0, help="[custom] bid/ask spread in bps")
    p.add_argument(
        "--ticks",
        type=int,
        default=0,
        help="If >0, run that many sim ticks and print stats (headless, no TUI).",
    )
    p.add_argument(
        "--advance",
        nargs=2,
        default=None,
        metavar=("INTERVAL", "N"),
        help="Headless: advance by sim time, e.g.  --advance day 2  or  --advance 2 day",
    )
    p.add_argument(
        "--mpt",
        type=float,
        default=None,
        dest="mpt",
        help="Sim minutes per engine step (for run/--advance; default 15 → ~96 step/day).",
    )
    p.add_argument(
        "--great-depression",
        action="store_true",
        dest="great_depression",
        help="Schedule a one-time crash in 500–1000 ticks, then 99% recovery for most names.",
    )
    p.add_argument("--web", action="store_true", help="Run FastAPI + browser UI (uvicorn)")
    p.add_argument("--host", type=str, default="127.0.0.1", help="[--web] bind address")
    p.add_argument("--port", type=int, default=8000, help="[--web] port")
    args = p.parse_args()
    if args.web:
        import uvicorn
        from .api import app
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        return
    if args.ticks and args.ticks > 0 and args.advance is not None:
        raise SystemExit("Use only one of --ticks and --advance")
    cfg = _make_config(args)
    s = new_session(custom=cfg)
    if args.advance is not None:
        try:
            n, u = parse_run_line([args.advance[0], args.advance[1]])
        except ValueError as e:
            raise SystemExit(str(e)) from e
        _run_headless_advance(s, u, n)
        return
    if args.ticks and args.ticks > 0:
        _run_headless(s, args.ticks)
        return
    _run_tui(s)


if __name__ == "__main__":
    main()
