from __future__ import annotations

from datetime import datetime, timezone

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .engine import Session
from .instrument import AssetClass

HELP = (
    "[bold]Commands[/]: [cyan]n[/] tick  |  [cyan]a[/] x5 ticks  |  "
    "[cyan]run UNIT N[/] or [cyan]run N UNIT[/] sim time (e.g. [cyan]run day 2[/], [cyan]run 3 hour[/])  |  "
    "[cyan]b TICK N[/] mkt buy (sh)  |  [cyan]b TICK $X[/] or [cyan]bc TICK X[/] mkt buy by $  |  [cyan]bc TICK all[/]  |  "
    "[cyan]s TICK N[/] mkt sell  |  "
    "[cyan]lb TICK N P[/] limit buy  |  [cyan]lbc TICK $ P[/] limit buy (cash @ limit)  |  [cyan]ls TICK N P[/] limit sell  |  "
    "[cyan]h[/] help  |  [cyan]q[/] quit"
)


def _fmt_money(x: float) -> str:
    return f"{x:,.2f}"


def _fmt_qty(x: float) -> str:
    s = f"{x:.8f}".rstrip("0").rstrip(".")
    return s or "0"


def _fmt_pct_24h(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:+.2f}%"


def _fmt_mcap(x: float) -> str:
    ax = abs(x)
    if ax >= 1e12:
        return f"{x/1e12:.2f}T"
    if ax >= 1e9:
        return f"{x/1e9:.2f}B"
    if ax >= 1e6:
        return f"{x/1e6:.2f}M"
    if ax >= 1e3:
        return f"{x/1e3:.1f}k"
    return f"{x:,.0f}"


def render_session(console: Console, s: Session) -> None:
    """Draw quotes, portfolio, and optional book depth (first row per name)."""

    m = s.market
    t = Table(title="Market (book + synthetic if empty)", box=box.ROUNDED, show_lines=False)
    t.add_column("K", style="dim", width=3)
    t.add_column("Ticker", style="bold")
    t.add_column("Mcap", justify="right", style="dim", width=8)
    t.add_column("Bid", justify="right")
    t.add_column("Mid", justify="right", style="white")
    t.add_column("Ask", justify="right")
    t.add_column("24h", justify="right", style="dim", width=7)
    t.add_column("Depth b/a", style="dim", overflow="fold")
    t.add_column("t", style="dim", width=4)
    for ins in m.instruments:
        if ins.kind is AssetClass.STOCK:
            tag = "S"
        elif ins.kind is AssetClass.FUND:
            tag = "F"
        else:
            tag = "C"
        b, mid, a = m.quote(ins.array_index)
        ob = m.books[ins.array_index]
        db = ob.depth_bids(2)
        da = ob.depth_asks(2)
        dtxt = f"{str(db)[:20]} {str(da)[:20]}"
        p24 = m.pct_change_24h_sim(ins.array_index)
        style_24 = "green" if (p24 is not None and p24 > 0) else ("red" if (p24 is not None and p24 < 0) else "dim")
        t.add_row(
            tag,
            ins.ticker,
            _fmt_mcap(ins.market_cap),
            _fmt_money(b),
            _fmt_money(mid),
            _fmt_money(a),
            Text(_fmt_pct_24h(p24), style=style_24),
            dtxt,
            f"{m.tick}",
        )
    ptab = Table(title="Portfolio", box=box.ROUNDED)
    ptab.add_column("Ticker")
    ptab.add_column("Qty", justify="right")
    ptab.add_column("Mkt", justify="right")
    if not s.player.positions and not s.player.locked_sell:
        ptab.add_row("—", "0", "—")
    for tk, q in sorted(s.player.positions.items()):
        ins = m.by_ticker().get(tk)
        if ins is None:
            continue
        _, mid, _ = m.quote(ins.array_index)
        ptab.add_row(tk, _fmt_qty(q), _fmt_money(q * mid))

    hdr = Text()
    hdr.append("marketsim", style="bold green")
    hdr.append("  │  mode ")
    hdr.append(s.config.label, style="magenta")
    hdr.append("  │  t=")
    hdr.append(str(m.tick), style="yellow")
    hdr.append("  │  wall ")
    hdr.append(f"{s.config.wall_seconds_per_tick:.2f}s", style="dim")
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    tot = s.player.cash + s.player.locked_cash
    mpt = s.config.sim_minutes_per_tick
    sub = (
        f"[dim]{now}[/]  ·  sim [dim]{mpt:g} min/step[/]  "
        f"·  free cash [green]{_fmt_money(s.player.cash)}[/]  "
        f"·  locked (buy) [yellow]{_fmt_money(s.player.locked_cash)}[/]  "
        f"·  cash total [white]{_fmt_money(tot)}[/]  "
        f"·  equity [bold cyan]{_fmt_money(s.equity)}[/]\n"
        + HELP
    )
    g = Group(Panel.fit(hdr), t, ptab, Panel(sub, title="Status", style="dim"))
    console.print(g)
