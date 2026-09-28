#!/usr/bin/env python3
import csv, gzip, io, json, math, os, sys, urllib.request
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta

SAMPLE_URL = os.environ.get(
    "KOL_SAMPLE_URL",
    "https://supabase.madeonsol.com/storage/v1/object/public/dataset-samples/sol-kol-trades/sol-kol-trades_sample.csv.gz",
)

STARTING_CASH = 1000.0
POSITION_USD = 20.0
MAX_POSITIONS = 5
COST_RATE_EACH_SIDE = 0.02
STOP_MULT = 0.60
MAX_HOLD = timedelta(hours=24)
TRAIL_START = 2.0
TRAIL_DRAWDOWN = 0.35
TPS = [(2.0, 0.20), (4.0, 0.25), (10.0, 0.25)]

def parse_ts(s):
    if not s:
        return None
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        d = datetime.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.astimezone(timezone.utc)
    except Exception:
        return None

def fnum(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None

def download_rows():
    req = urllib.request.Request(SAMPLE_URL, headers={"User-Agent":"Mozilla/5.0 historical-backtest"})
    with urllib.request.urlopen(req, timeout=60) as r:
        blob = r.read()
    try:
        raw = gzip.decompress(blob).decode("utf-8-sig")
    except Exception:
        raw = blob.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(raw))
    rows = []
    for i, row in enumerate(reader):
        ts = parse_ts(row.get("traded_at") or row.get("block_time") or row.get("timestamp"))
        price = fnum(row.get("price_usd_at_trade") or row.get("price_usd"))
        mint = (row.get("token_mint") or row.get("mint") or "").strip()
        wallet = (row.get("wallet_address") or row.get("wallet") or "").strip()
        side = (row.get("action") or row.get("side") or "").strip().lower()
        if side in ("b", "buy", "bought"): side = "buy"
        if side in ("s", "sell", "sold"): side = "sell"
        if ts and price and price > 0 and mint and wallet and side in ("buy","sell"):
            rows.append({
                "ts": ts,
                "price": price,
                "mint": mint,
                "wallet": wallet,
                "side": side,
                "symbol": (row.get("token_symbol") or row.get("symbol") or mint[:6]).strip(),
                "sol_amount": fnum(row.get("sol_amount")) or 0.0,
                "raw": row,
            })
    rows.sort(key=lambda x: x["ts"])
    return rows, list(reader.fieldnames or [])

class Position:
    def __init__(self, mint, symbol, ts, raw_price, spend, signal_k, wallets):
        self.mint = mint
        self.symbol = symbol
        self.entry_ts = ts
        self.entry_raw = raw_price
        self.entry_fill = raw_price * (1 + COST_RATE_EACH_SIDE)
        self.spend = spend
        self.initial_qty = spend / self.entry_fill
        self.qty = self.initial_qty
        self.realized = 0.0
        self.signal_k = signal_k
        self.wallets = list(wallets)
        self.peak_mult = (raw_price * (1-COST_RATE_EACH_SIDE)) / self.entry_fill
        self.tp_done = [False]*len(TPS)
        self.last_raw = raw_price
        self.last_ts = ts
        self.exit_reason = None
        self.closed_ts = None
    def mult(self, raw):
        return (raw * (1-COST_RATE_EACH_SIDE)) / self.entry_fill
    def sell_fraction(self, raw, frac):
        q = self.qty * frac
        proceeds = q * raw * (1-COST_RATE_EACH_SIDE)
        self.qty -= q
        self.realized += proceeds
        return proceeds
    def sell_all(self, raw):
        q = self.qty
        proceeds = q * raw * (1-COST_RATE_EACH_SIDE)
        self.qty = 0.0
        self.realized += proceeds
        return proceeds
    @property
    def pnl(self):
        return self.realized - self.spend

def run_backtest(rows, k, window_minutes, allow_repeat_after_hours=24):
    cash = STARTING_CASH
    equity_peak = STARTING_CASH
    max_dd = 0.0
    positions = {}
    closed = []
    buy_windows = defaultdict(deque)
    last_signal_at = {}
    last_price = {}
    exposure = 0.0

    def equity(mark_prices):
        e = cash
        for p in positions.values():
            raw = mark_prices.get(p.mint, p.last_raw)
            e += p.qty * raw * (1-COST_RATE_EACH_SIDE) + p.realized
        return e

    def close_position(mint, ts, raw, reason):
        nonlocal cash, exposure
        p = positions[mint]
        p.last_raw, p.last_ts = raw, ts
        proceeds = p.sell_all(raw)
        cash += proceeds + p.realized - proceeds  # add all realized proceeds accumulated across partial exits
        # p.realized contains all exits; but partial TP proceeds were already added to cash at execution.
        # The line above only adds final proceeds; historical partial proceeds are handled at TP time.
        # Undo the over-add of previous partial realized:
        previous_partial = p.realized - proceeds
        cash -= previous_partial
        p.exit_reason = reason
        p.closed_ts = ts
        closed.append(p)
        exposure -= p.spend
        del positions[mint]

    # Instead of close_position's bookkeeping complexity, use helper below with cash updates at each sale.
    closed.clear()
    positions.clear()
    cash = STARTING_CASH
    exposure = 0.0
    equity_peak = STARTING_CASH
    max_dd = 0.0
    buy_windows.clear()
    last_signal_at.clear()
    last_price.clear()

    for row in rows:
        ts, mint, wallet, side, raw = row["ts"], row["mint"], row["wallet"], row["side"], row["price"]
        last_price[mint] = raw

        # Time exits for all overdue positions, using each token's last observed price.
        overdue = []
        for m,p in list(positions.items()):
            if ts - p.entry_ts >= MAX_HOLD:
                overdue.append(m)
        for m in overdue:
            p = positions[m]
            px = last_price.get(m, p.last_raw)
            proceeds = p.sell_all(px)
            cash += proceeds
            p.exit_reason = "max_hold_24h"
            p.closed_ts = ts
            closed.append(p)
            exposure -= p.spend
            del positions[m]

        # Mark/manage an open position whenever this token gets a priced KOL trade tick.
        if mint in positions:
            p = positions[mint]
            p.last_raw, p.last_ts = raw, ts
            m = p.mult(raw)
            p.peak_mult = max(p.peak_mult, m)

            # Stop first.
            if m <= STOP_MULT:
                proceeds = p.sell_all(raw); cash += proceeds
                p.exit_reason = "stop_-40%"
                p.closed_ts = ts
                closed.append(p); exposure -= p.spend; del positions[mint]
            else:
                # Execute each newly crossed TP in order, fractions of remaining quantity.
                for j,(target, frac) in enumerate(TPS):
                    if mint not in positions: break
                    if (not p.tp_done[j]) and m >= target:
                        proceeds = p.sell_fraction(raw, frac); cash += proceeds
                        p.tp_done[j] = True
                if mint in positions and p.peak_mult >= TRAIL_START and m <= p.peak_mult*(1-TRAIL_DRAWDOWN):
                    proceeds = p.sell_all(raw); cash += proceeds
                    p.exit_reason = "trailing_35%"
                    p.closed_ts = ts
                    closed.append(p); exposure -= p.spend; del positions[mint]

        # Signal detection uses buys only; distinct wallets inside rolling window.
        if side == "buy":
            dq = buy_windows[mint]
            cutoff = ts - timedelta(minutes=window_minutes)
            while dq and dq[0][0] < cutoff:
                dq.popleft()
            dq.append((ts, wallet))
            # keep only latest event per wallet for distinct counting
            latest = {}
            for t,w in dq:
                latest[w] = t
            distinct = sorted(latest.items(), key=lambda kv: kv[1])

            if len(distinct) >= k and mint not in positions and cash >= POSITION_USD and len(positions) < MAX_POSITIONS and exposure + POSITION_USD <= 100.0001:
                prev = last_signal_at.get(mint)
                if prev is None or ts - prev >= timedelta(hours=allow_repeat_after_hours):
                    signal_wallets = [w for w,_ in distinct]
                    p = Position(mint, row["symbol"], ts, raw, POSITION_USD, len(distinct), signal_wallets)
                    positions[mint] = p
                    cash -= POSITION_USD
                    exposure += POSITION_USD
                    last_signal_at[mint] = ts

        e = equity(last_price)
        equity_peak = max(equity_peak, e)
        if equity_peak > 0:
            max_dd = max(max_dd, (equity_peak - e)/equity_peak)

    # Final close at last observed KOL-trade price in sample.
    end_ts = rows[-1]["ts"] if rows else datetime.now(timezone.utc)
    for mint,p in list(positions.items()):
        raw = last_price.get(mint, p.last_raw)
        proceeds = p.sell_all(raw); cash += proceeds
        p.exit_reason = "sample_end"
        p.closed_ts = end_ts
        closed.append(p); exposure -= p.spend; del positions[mint]

    pnls = [p.pnl for p in closed]
    wins = sum(1 for x in pnls if x > 0)
    losses = sum(1 for x in pnls if x < 0)
    return {
        "strategy": f"k>={k}_within_{window_minutes}m",
        "starting_cash": STARTING_CASH,
        "ending_cash": round(cash, 4),
        "pnl_usd": round(cash-STARTING_CASH, 4),
        "return_pct": round((cash/STARTING_CASH-1)*100, 4),
        "closed_positions": len(closed),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round((wins/len(closed)*100) if closed else 0, 2),
        "max_drawdown_pct": round(max_dd*100, 2),
        "best_trade_usd": round(max(pnls),4) if pnls else None,
        "worst_trade_usd": round(min(pnls),4) if pnls else None,
        "exit_reasons": {r:sum(1 for p in closed if p.exit_reason==r) for r in sorted(set(p.exit_reason for p in closed))},
        "notes": "Event-driven on priced KOL trade ticks; not full DEX candle path. 2% execution drag each side. No Jev API/deployer/dump-cluster features in the public sample.",
    }

def main():
    rows, fields = download_rows()
    if not rows:
        raise SystemExit("No usable rows found in public sample")
    times = [r["ts"] for r in rows]
    summary = {
        "sample": {
            "source": SAMPLE_URL,
            "usable_rows": len(rows),
            "fields": fields,
            "from": min(times).isoformat(),
            "to": max(times).isoformat(),
            "span_hours": round((max(times)-min(times)).total_seconds()/3600,2),
            "unique_wallets": len({r["wallet"] for r in rows}),
            "unique_tokens": len({r["mint"] for r in rows}),
            "buys": sum(r["side"]=="buy" for r in rows),
            "sells": sum(r["side"]=="sell" for r in rows),
        },
        "assumptions": {
            "starting_cash_usd": STARTING_CASH,
            "position_usd": POSITION_USD,
            "max_positions": MAX_POSITIONS,
            "max_exposure_usd": 100,
            "execution_drag_each_side_pct": COST_RATE_EACH_SIDE*100,
            "stop_loss_pct": -40,
            "max_hold_hours": 24,
            "tp": [{"multiple":m,"fraction_remaining":f} for m,f in TPS],
            "trailing_after_multiple": TRAIL_START,
            "trailing_drawdown_pct": TRAIL_DRAWDOWN*100,
        },
        "results": [
            run_backtest(rows, 2, 5),
            run_backtest(rows, 2, 15),
            run_backtest(rows, 3, 15),
            run_backtest(rows, 5, 15),
        ],
    }
    print("BACKTEST_RESULT_JSON_START")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("BACKTEST_RESULT_JSON_END")
    with open("backtest_result.json","w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

if __name__ == "__main__":
    main()
