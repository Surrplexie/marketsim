# marketsim

`marketsim` is a single-player, in-memory stock/fund/crypto market sandbox.

It is designed to be:
- easy to start for first-time users,
- fast to iterate for tinkerers,
- transparent enough to learn from (API JSON, deterministic seeds, readable code).

You can run it three ways:
- **GUI (browser, recommended)**,
- **TUI (terminal interface)**,
- **headless CLI**.

---

## 1) What You Need

- **Python 3.10+**
- `pip`
- dependencies listed in `pyproject.toml` / `requirements.txt` (NumPy, Rich, FastAPI, Uvicorn)

---

## 2) Install (Start Here)

From the repository root:

```bash
pip install -e .
```

Alternative:

```bash
pip install -r requirements.txt
```

Check installation:

```bash
python -m marketsim --help
```

If editable install succeeded, this may also work:

```bash
marketsim --help
```

---

## 3) Quick Start (Fastest Path)

Run the web GUI:

```bash
python -m marketsim --web
```

Open:

- [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

Health check:

- [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

Stop server:

- `Ctrl+C` in terminal

Custom bind:

```bash
python -m marketsim --web --host 127.0.0.1 --port 8000
```

---

## 4) Run Modes (GUI / TUI / Headless)

### A) GUI (Browser)

```bash
python -m marketsim --web
```

Best for:
- charting,
- order ticket workflow,
- overlays/panels,
- debugging with live API JSON.

### B) TUI (Terminal App)

```bash
python -m marketsim
python -m marketsim --mode hard --seed 42
```

Best for:
- keyboard-only flow,
- fast stepping,
- low-overhead remote sessions.

### C) Headless CLI

```bash
python -m marketsim --mode simple --ticks 500
python -m marketsim --advance day 2
```

Best for:
- quick experiments,
- scripted runs,
- reproducible snapshots.

---

## 5) First Session Walkthrough (GUI)

When you open the GUI:

1. **Set Starting Cash (USD)** (if desired).
2. Open **Preset · New game** (bottom-left panel), choose mode.
3. Open **> miscellaneous** panel, toggle stress events if wanted.
4. Click **New game**.
5. Use `+1`, `+5`, `+50`, `Run`, or **Auto** to move time.
6. Trade from the order ticket.
7. Double-click symbols/positions to open chart.

### Important startup rules

- `Apply cash` only works at **tick 0** and when account is flat/unlocked.
- `New game` applies:
  - selected `mode`,
  - starting cash,
  - margin/short settings,
  - miscellaneous toggles (Great Depression + chaos options).

---

## 6) GUI Layout Guide

### Left Column (session controls)

Panels are overlay-style (summary row in rail, full form in modal):

- `Volatility regime`
- `Trend override`
- `Starting cash (USD)`
- `Margin & shorts`
- `> miscellaneous`
- `Preset · New game` (bottom-left)

### Center Column

- Watchlist table (sortable/filterable, density, kind filters)
- Double-click row to open chart

### Right Column

- Order ticket
- Order history & audit
- Manual stock split
- Dividend & buyback (stocks)
- News / micro / API JSON / positions

Most dense sections can be opened in larger overlays.

---

## 7) Trading Basics

- **Market buy/sell**
- **Limit buy/sell**
- Buy sizing modes:
  - shares,
  - cash notional,
  - all cash
- Fast presets (`+1`, `+10`, `+100`, `25% cash`, `50% cash`, `all cash`)
- `Flatten all` / per-position flatten

Orders are checked against:
- symbol lot size,
- symbol tick size,
- cash / position / margin constraints,
- available book liquidity.

---

## 8) GUI Hotkeys (Website)

Hotkeys are global while on the webpage, with guardrails:
- they are ignored while typing in `input` / `select` / `textarea`,
- `Esc` closes topmost panel/chart,
- `Tab` is intentionally disabled in the page.

Bindings:

- `Enter` — run interval (`Interval × N`)
- `/` — focus symbol selector
- `B` — market buy current symbol
- `S` — market sell current symbol
- `L` — limit buy current symbol
- `K` — limit sell current symbol
- `N` or `1` — `+1` tick
- `5` — `+5` ticks
- `0` — `+50` ticks
- `9` or `D` — `+96` ticks (about 1 sim day at 15 min/tick)
- `F` — flatten all positions
- `R` — toggle auto-run
- `[` / `]` — auto-run rate down/up
- `C` — open chart for selected ticker

---

## 9) TUI Command Cheat Sheet

At the `>` prompt:

- `n` (or Enter): one tick
- `a`: five ticks
- `run day 2` or `run 2 day`: advance by simulated calendar time
- `b TICK N`: market buy shares
- `b TICK $5000` or `bc TICK 5000`: market buy by USD
- `bc TICK all`: buy with all free cash
- `s TICK N`: market sell shares
- `lb TICK N P`: limit buy
- `ls TICK N P`: limit sell
- `lbc TICK 5000 100`: limit buy with cash budget at limit
- `h`: help
- `q`: quit

---

## 10) CLI Flags You’ll Use Most

General:
- `--mode simple|easy|hard|complex|free|custom`
- `--seed 42`
- `--web`
- `--host`, `--port`

Headless:
- `--ticks N`
- `--advance day 2` (or similar unit/n pair)

Custom config:
- `--custom`
- `--n-stocks`, `--n-funds`, `--n-crypto`
- `--vol`, `--drift`, `--spread`, `--cash`
- `--mpt` (sim minutes per tick)

Realism:
- `--shorting`
- `--max-leverage`
- `--maintenance`
- `--short-borrow-bps`
- `--great-depression`

---

## 11) Modes and Universe

Preset mode names:
- `simple`, `easy`, `hard`, `complex`, `free`, `custom`

Default mega universe (when using 32/4/8 layout):
- **32** stocks
- **4** funds (basket/index-style)
- **8** crypto assets

Custom universe example:

```bash
python -m marketsim --web --mode complex --n-stocks 100 --n-funds 8 --n-crypto 20
```

---

## 12) Miscellaneous Panel (Great Depression + Chaos)

`> miscellaneous` includes:

- Great Depression
- Flash crash roulette
- Meme squeeze
- Fat finger event
- Exchange halt
- Rumor mill
- Sector rotation storm
- Funding panic
- Liquidity drought
- Whale rebalance
- Crypto weekend mania

These are **session config toggles** and apply on **New game**.

---

## 13) API Quick Reference

Base URL: `http://127.0.0.1:8000` (default)

- `GET /` — GUI page
- `GET /api/health` — liveness
- `GET /api/state` — full game state JSON
- `POST /api/step` — advance ticks or interval
- `POST /api/order` — submit order
- `POST /api/reset` — new game with settings
- `POST /api/flatten` — flatten one/all positions
- `POST /api/starting-cash` — change starting cash (tick 0 only)
- `POST /api/trend-override`
- `POST /api/volatility-override`
- `POST /api/stock_split`
- `POST /api/stock_dividend`
- `POST /api/stock_buyback`
- `GET /api/chart/{ticker}` — OHLC (+ volume, + mcap candles)

`/api/state` includes:
- mode/tick/time,
- instruments + quote/book/micro fields,
- player state + holdings,
- order log,
- news feed,
- depression status,
- miscellaneous chaos settings.

---

## 14) Troubleshooting

### Port already in use

Use another port:

```bash
python -m marketsim --web --port 8001
```

### GUI loads but nothing works

- open `/api/health`
- open `/api/state`
- ensure dependencies are installed and server terminal has no traceback

### Orders keep getting rejected

Check:
- lot step / tick size,
- available cash/position,
- liquidity on book,
- margin settings.

Use `order_log` and API JSON to inspect exact reject reason.

### Want deterministic runs

Use:

```bash
python -m marketsim --seed 42
```

---

## 15) Project Map (For Curious Newbies)

- `marketsim/__main__.py` — CLI entry (web/tui/headless)
- `marketsim/api.py` — FastAPI routes
- `marketsim/static/index.html` — GUI
- `marketsim/engine.py` — session lifecycle and stepping
- `marketsim/market.py` — core simulation dynamics/events
- `marketsim/player.py` — account/positions/margin/order log
- `marketsim/execution.py` — order execution logic
- `marketsim/instrument.py` — universe generation
- `marketsim/modes.py` — presets/config
- `marketsim/clob.py` — order book model

---

## 16) Disclaimer

`marketsim` is a learning sandbox and simulation toy, not brokerage software or investment advice.
