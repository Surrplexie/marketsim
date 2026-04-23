# marketsim

**marketsim** is a single-player, in-memory **stock + ETF + crypto** market toy: vectorized GBM-style prices, a simple CLOB (limit book + NPC liquidity), optional margin/shorting, and presets from **easy** through **hard**. Use the **browser UI**, a **terminal UI**, or run **headless** batches.

---

## Requirements

- **Python 3.10+**
- Dependencies: **NumPy**, **Rich** (TUI), **FastAPI** + **Uvicorn** (web). Declared in `pyproject.toml` / `requirements.txt`.

---

## Installation

From the repository root:

```bash
pip install -e .
```

(or `pip install -r requirements.txt` if you prefer not to install the package metadata).

Verify:

```bash
python -m marketsim --help
```

Optional CLI entry point (after editable install):

```bash
marketsim --help
```

---

## Three ways to run

### 1) Browser UI (recommended for new users)

```bash
python -m marketsim --web
```

Then open **http://127.0.0.1:8000/** (or your `--host` / `--port`). Static assets are bundled from `marketsim/static/index.html`.

- **Health check:** `GET http://127.0.0.1:8000/api/health`
- Stop the server with **Ctrl+C** in the terminal.

```bash
python -m marketsim --web --host 127.0.0.1 --port 8000
```

### 2) Terminal UI (TUI)

```bash
python -m marketsim
# or with a preset and seed:
python -m marketsim --mode hard --seed 42
```

At the `>` prompt:

| Input | Action |
|--------|--------|
| `n` or Enter | One simulation tick |
| `a` | Five ticks in a row |
| `run day 2` or `run 2 day` | Advance by **simulated** calendar time (see `--mpt`) |
| `b TICK N` | Market buy *N* shares |
| `bc TICK 5000` / `b TICK $5000` | Market buy with **USD** cap |
| `bc TICK all` | Market buy using all free cash |
| `s TICK N` | Market sell *N* shares |
| `lb TICK N P` / `ls TICK N P` | Limit buy / sell |
| `h` | Help |
| `q` | Quit |

### 3) Headless (no UI)

```bash
# Run N engine steps and print a one-line summary
python -m marketsim --mode simple --ticks 500

# Advance by sim time (same grammar as TUI `run`)
python -m marketsim --advance day 2
```

---

## Game modes and universe

Presets live in **`marketsim/modes.py`**. Names: **`simple`**, **`easy`**, **`hard`**, **`complex`**, **`free`**, **`custom`**. In the API, **`free`** and **`custom`** both resolve to the same baseline “free-style” config (tune further via CLI `--custom` or by editing code).

**Default “mega” universe** (used by presets when stock/fund/crypto counts match **32 + 4 + 8**):

- **32** large-cap-style **stocks** (sectors, correlated drift)
- **4** equal-weight **index-style funds** (e.g. top-16 / top-25 stocks, top-3 crypto, sector basket)
- **8** **cryptos** (tiered volatility by mcap; one capped at **21M** units, others can inflate slowly with dilution-style pricing)

Other universe sizes use the **classic** procedural generator (`make_universe` in `marketsim/instrument.py`).

**CLI custom universe** (only with `--custom`):

```bash
python -m marketsim --custom --n-stocks 10 --n-funds 2 --n-crypto 3 --vol 1.1 --spread 12
```

For mega-cap, use **32 / 4 / 8** together so the special generator activates.

---

## Web UI: first-time walkthrough

### Before the first time step

1. **Starting cash (USD)** — Enter any amount from **$0.0001** to **$100,000,000**.
   - **Apply cash** — Updates bankroll while **sim tick = 0** and you have **no positions** and **no locked cash/sells** (same bounds).
   - **New game** — Rebuilds the session with the selected **preset** and optional **Great Depression** flag, and applies **starting_cash** from the field.

2. **Preset** — `simple` / `easy` / `hard` / `complex` / `free` / `custom`; applied on **New game** (`POST /api/reset` with `mode` and optional `starting_cash` or `startingCash`).

3. **Margin & shorts** (optional) — In the Session panel, you can enable shorting (including crypto), set max leverage, maintenance rate, and short borrow bps/day; applied on **New game**.

4. **Great Depression** (checkbox) — One scripted crash window (then recovery behavior); read the in-UI hint.

### Advancing time

- **+1 tick / +5 / +50** — Single-process `POST /api/step` with `{"ticks": N}`.
- **Run** — `POST /api/step` with `{"unit":"day","n":1}` style body (simulated calendar).
- **Auto** — Checkbox + rate slider to step repeatedly (client-side timer).

### Trading

- Pick a **symbol**, **size** or **cash** for buys, optional **limit price**.
- **Market / limit** buy and sell buttons call `POST /api/order` (see API below).
- Order ticket supports quick presets (`+1`, `+10`, `+100`, `25% cash`, `50% cash`, `all cash`), flatten actions, optional large-order confirm, and keyboard shortcuts.
- Orders enforce symbol execution rules (lot step and tick size); invalid increments are rejected.
- Non-abbreviated numeric displays use thousands separators in the GUI (e.g., `43,000.00`), while abbreviated views remain abbreviated (`43K`, `4T`, etc.).
- If the book has no size on the side you need, you may get **insufficient liquidity** (market orders are capped by **resting** bid/ask size unless shorts allow synthetic remainder on sells).

### Watchlist and chart

- Rows are sortable/filterable; favorites can be pinned first, and single-click selects the ticker in the order ticket.
- Double-click a watchlist row (or click a position row) to open the chart.
- **Vol** blends **printed** volume with a **model turnover** component tied to volatility, moves, tape, and supply flow.
- **Ask sz** — Total resting ask size (rough cap for a market **buy** without new limits).
- Chart supports **Price**, **Mcap**, and **Stats** views, plus bottom volume bars in candle view. You can set a default open mode.
- **API JSON** at the bottom mirrors `GET /api/state` for debugging.

### Session controls (left column)

- **Trend override** — Biases drift for scoped names (`POST /api/trend-override`).
- **Volatility regime** — Adjusts long-run equity **μ** and **σ** for **stocks, funds, and crypto** (`POST /api/volatility-override`).
- **News** line — Random headline shocks when they fire (also in JSON `news`).

### Layout tip

On a wide screen the **header** and **side panels** stay fixed while the **watchlist** scrolls; each sidebar scrolls on its own.

---

## HTTP API (overview)

Base URL: same host/port as the web server (default `http://127.0.0.1:8000`).

| Method | Path | Purpose |
|--------|------|--------|
| `GET` | `/` | Browser UI (HTML) |
| `GET` | `/api/state` | Full JSON snapshot (mode, tick, instruments, player, orders, overrides, `news`, …) |
| `GET` | `/api/health` | Plain-text liveness |
| `POST` | `/api/step` | Advance sim (`ticks` and/or `unit`+`n`) |
| `POST` | `/api/order` | Place market/limit orders (JSON body) |
| `POST` | `/api/reset` | New game: `mode`, optional `great_depression`, optional `starting_cash`, optional margin/short settings |
| `POST` | `/api/flatten` | Flatten one ticker or all open positions |
| `POST` | `/api/starting-cash` | Set cash at **tick 0** only (flat book); `{"cash": 12345}` |
| `POST` | `/api/trend-override` | Trend slider + scope |
| `POST` | `/api/volatility-override` | Volatility slider |
| `POST` | `/api/stock_split` | Manual forward split (stocks) |
| `POST` | `/api/stock_dividend` | Cash dividend (stocks) |
| `POST` | `/api/stock_buyback` | Buyback fraction (stocks) |
| `GET` | `/api/chart/{ticker}` | OHLC + per-candle volume + optional mcap view |

There is **one in-memory game per server process**; restarting Uvicorn clears it.

---

## Simulation behavior (short)

- **Prices:** Listed **stocks** and **funds** follow GBM-style dynamics with a long-run drift tied to `stock_fund_annual_return` (see `GameConfig`). **Crypto** σ is rescaled by **mcap tiers** (see `crypto_*` fields in config).
- **Funds:** Many fund mids are **overwritten each tick** to track equal-weight basket NAVs (see `Market._sync_ew_basket_fund_mids`).
- **Supply:** Instruments can carry **`max_units_outstanding`**; crypto may **mint** with dilution; stocks/funds get a tiny **float flow**; caps affect splits and issuance-style logic.
- **Volume:** Session **volume** grows from **fills** plus **synthetic turnover** and float-related flow so the tape is not only player-driven.
- **Execution realism:** Orders follow symbol-specific **lot step** and **tick size** constraints.
- **Financing realism:** Carry is charged per position each tick (dynamic instrument funding + configured short borrow baseline).
- **Calendar / liquidity:** Session-aware spread behavior (listed open/after-hours profile, 24/7 crypto with weekend liquidity effects) and optional listed-only overnight gaps.
- **Optional realism:** `GameConfig` / session can include **leverage**, **maintenance margin**, **shorting**, **borrow cost**, **SEC-style fee on non-crypto sells**, **overnight gaps** — exposed in `/api/state` when enabled.

For exact numbers and formulas, read **`marketsim/market.py`**, **`marketsim/execution.py`**, and **`marketsim/modes.py`**.

---

## Project layout (orientation)

| Path | Role |
|------|------|
| `marketsim/__main__.py` | CLI: TUI / web / headless |
| `marketsim/api.py` | FastAPI app and routes |
| `marketsim/static/index.html` | Browser UI |
| `marketsim/engine.py` | `Session`, `new_session`, day roll, optional corp actions |
| `marketsim/market.py` | GBM step, books, microstructure hooks, volume/supply helpers |
| `marketsim/player.py` | Cash, positions, margin, order log |
| `marketsim/execution.py` | Order routing and fills |
| `marketsim/instrument.py` | Universe builders, `Instrument` |
| `marketsim/modes.py` | `GameConfig`, presets, `build_custom` |
| `marketsim/clob.py` | Order book |

---

## Troubleshooting

| Issue | What to try |
|-------|----------------|
| **Address already in use** (`--web`) | `--port 8001` or stop the other process |
| **Blank page** | Ensure editable install so `marketsim/static` is on the path; open `/api/state` to see JSON |
| **Orders rejected** | Check `cash`, `NO_LIQUIDITY`, margin messages in UI or `order_log` in state |
| **Want reproducible runs** | `--seed 42` (CLI) or pass seed through custom config if you extend the API |

---

## Original goal (design intent)

Aim: a **rich, stochastic** toy market with **multiple difficulty presets**, **fast or slow** stepping, **short and long** simulated horizons, and enough hooks (overrides, corp actions, optional stress events) to experiment — **not** a substitute for real market data or brokerage software.
