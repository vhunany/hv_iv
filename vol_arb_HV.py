"""
HV vs IV Options Straddle Strategy
===================================
Strategy Logic:
1. At the start of each trading month, calculate HV - IV for all options
   with monthly expirations.
2. Rank the differences into deciles.
3. Go LONG straddles on top decile (HV >> IV → options are "cheap").
4. Go SHORT straddles on bottom decile (IV >> HV → options are "expensive").

Data Source: yfinance (free) for price history + implied vol from option chains.
             Replace with your broker/data provider API for live trading.

Dependencies:
    pip install yfinance pandas numpy scipy matplotlib seaborn
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta
import calendar
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Optional

import requests

no_dividend_tickers = pd.read_csv("sp500_no_dividends.csv", header=None)[0].tolist()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# CONFIG = {
#     "tickers": [                        # Universe of underlyings to scan
#         "SPY", "QQQ", "IWM", "GLD", "TLT",
#         "AAPL", "MSFT", "AMZN", "GOOGL", "META",
#         "NVDA", "TSLA", "JPM", "BAC", "XOM",
#     ],
#     "hv_window": 21,                    # Trading days for Historical Volatility (1-month)
#     "hv_annualize": 252,                # Trading days in a year
#     "iv_moneyness_tol": 0.03,           # ATM tolerance: |K/S - 1| <= this value
#     "top_decile_n": 1,                  # Number of top/bottom deciles to trade
#     "notional_per_straddle": 10_000,    # $ notional per straddle leg
#     "risk_free_rate": 0.05,             # Used for BSM delta (not HV/IV calc)
#     "results_dir": ".",
# }

CONFIG = {
    "tickers": no_dividend_tickers,
    "hv_window": 21,                    # Trading days for Historical Volatility (1-month)
    "hv_annualize": 252,                # Trading days in a year
    "iv_moneyness_tol": 0.03,           # ATM tolerance: |K/S - 1| <= this value
    "top_decile_n": 1,                  # Number of top/bottom deciles to trade
    "notional_per_straddle": 10_000,    # $ notional per straddle leg
    "risk_free_rate": 0.05,             # Used for BSM delta (not HV/IV calc)
    "results_dir": ".",
}


# ─────────────────────────────────────────────
# HELPERS: DATE UTILITIES
# ─────────────────────────────────────────────

def get_monthly_expiration(year: int, month: int) -> datetime:
    """Return 3rd Friday of the given month (standard monthly options expiry)."""
    c = calendar.monthcalendar(year, month)
    fridays = [week[4] for week in c if week[4] != 0]
    third_friday = fridays[2]
    return datetime(year, month, third_friday)


def get_start_of_trading_month(year: int, month: int) -> datetime:
    """Return the first trading day (Mon-Fri) of the month."""
    d = datetime(year, month, 1)
    while d.weekday() >= 5:           # skip Sat/Sun
        d += timedelta(days=1)
    return d


def get_next_monthly_expiry(reference_date: datetime) -> datetime:
    """Return the nearest upcoming monthly expiry after reference_date."""
    y, m = reference_date.year, reference_date.month
    exp = get_monthly_expiration(y, m)
    if exp <= reference_date:
        m += 1
        if m > 12:
            m, y = 1, y + 1
        exp = get_monthly_expiration(y, m)
    return exp


# ─────────────────────────────────────────────
# HISTORICAL VOLATILITY
# ─────────────────────────────────────────────

def compute_hv(ticker: str, as_of_date: datetime, window: int = 21,
               annualize: int = 252) -> Optional[float]:
    """
    Annualized Historical Volatility = std of log-returns × sqrt(annualize).
    Uses `window` trading days ending on as_of_date.
    """
    start = as_of_date - timedelta(days=window * 2 + 10)   # buffer for weekends
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                     end=(as_of_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                     auto_adjust=True, progress=False)
    if df.empty or len(df) < window + 1:
        return None

    closes = df["Close"].dropna().iloc[-(window + 1):]
    log_rets = np.log(closes / closes.shift(1)).dropna()
    hv = log_rets.std() * np.sqrt(annualize)
    return float(hv)


# ─────────────────────────────────────────────
# IMPLIED VOLATILITY  (Black-Scholes inversion)
# ─────────────────────────────────────────────

def bsm_price(S, K, T, r, sigma, option_type="call") -> float:
    """Black-Scholes-Merton option price."""
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if option_type == "call" else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(market_price: float, S: float, K: float, T: float,
                r: float, option_type: str = "call",
                tol: float = 1e-6, max_iter: int = 200) -> Optional[float]:
    """Newton-Raphson IV solver."""
    if T <= 0 or market_price <= 0:
        return None
    intrinsic = max(0.0, (S - K) if option_type == "call" else (K - S))
    if market_price < intrinsic:
        return None

    sigma = 0.25                        # initial guess
    for _ in range(max_iter):
        price = bsm_price(S, K, T, r, sigma, option_type)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        if vega < 1e-10:
            return None
        diff = price - market_price
        sigma -= diff / vega
        sigma = max(0.001, min(sigma, 20.0))
        if abs(diff) < tol:
            return sigma
    return sigma                        # best estimate even if not converged


def get_atm_iv(ticker: str, as_of_date: datetime,
               expiry: datetime, r: float = 0.05,
               moneyness_tol: float = 0.03) -> Optional[float]:
    """
    Fetch the option chain for `ticker`, select near-ATM options expiring
    on or near `expiry`, and return the average IV of the ATM call + put.
    Uses yfinance; replace with live data feed for production.
    """
    tk = yf.Ticker(ticker)

    # Find the closest available expiry string
    try:
        available = tk.options                      # list of "YYYY-MM-DD" strings
    except Exception:
        return None
    if not available:
        return None

    expiry_str = expiry.strftime("%Y-%m-%d")
    # Pick the nearest available expiry within ±10 calendar days
    best = None
    best_delta = 9999
    for exp_s in available:
        delta = abs((datetime.strptime(exp_s, "%Y-%m-%d") - expiry).days)
        if delta < best_delta:
            best_delta, best = delta, exp_s
    if best is None or best_delta > 10:
        return None

    try:
        chain = tk.option_chain(best)
    except Exception:
        return None

    # Current stock price
    hist = tk.history(period="1d")
    if hist.empty:
        return None
    S = float(hist["Close"].iloc[-1])

    ivs = []
    T = max((datetime.strptime(best, "%Y-%m-%d") - as_of_date).days / 365.0, 1 / 365)

    for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
        if df is None or df.empty:
            continue
        # Filter near-ATM strikes
        df = df.copy()
        df["moneyness"] = (df["strike"] - S).abs() / S
        atm = df[df["moneyness"] <= moneyness_tol].copy()
        if atm.empty:
            atm = df.nsmallest(1, "moneyness")

        for _, row in atm.iterrows():
            mid = (row.get("bid", 0) + row.get("ask", 0)) / 2
            if mid <= 0:
                mid = row.get("lastPrice", 0)
            if mid <= 0:
                # fall back to implied vol column if available
                iv_col = row.get("impliedVolatility", None)
                if iv_col and iv_col > 0:
                    ivs.append(float(iv_col))
                continue
            iv = implied_vol(mid, S, float(row["strike"]), T, r, opt_type)
            if iv and 0.01 < iv < 20:
                ivs.append(iv)

    return float(np.mean(ivs)) if ivs else None


# ─────────────────────────────────────────────
# CORE: BUILD HV-IV SNAPSHOT
# ─────────────────────────────────────────────

def build_hv_iv_snapshot(tickers: list[str],
                          as_of_date: datetime,
                          cfg: dict) -> pd.DataFrame:
    """
    For each ticker, compute HV and ATM IV; return a DataFrame with
    HV, IV, Spread (HV-IV) and decile rank.
    """
    expiry = get_next_monthly_expiry(as_of_date)
    records = []

    print(f"\n{'='*60}")
    print(f"Snapshot Date : {as_of_date.strftime('%Y-%m-%d')}")
    print(f"Target Expiry : {expiry.strftime('%Y-%m-%d')}")
    print(f"{'='*60}")

    for ticker in tickers:
        print(f"  Processing {ticker:<8}", end=" ", flush=True)
        hv = compute_hv(ticker, as_of_date,
                        window=cfg["hv_window"],
                        annualize=cfg["hv_annualize"])
        iv = get_atm_iv(ticker, as_of_date, expiry,
                        r=cfg["risk_free_rate"],
                        moneyness_tol=cfg["iv_moneyness_tol"])

        if hv is None or iv is None:
            print(f"→ SKIP (HV={hv}, IV={iv})")
            continue

        spread = hv - iv
        print(f"→ HV={hv:.1%}  IV={iv:.1%}  Spread={spread:+.1%}")
        records.append({
            "ticker":      ticker,
            "expiry":      expiry.strftime("%Y-%m-%d"),
            "snapshot_dt": as_of_date.strftime("%Y-%m-%d"),
            "HV":          round(hv, 4),
            "IV":          round(iv, 4),
            "spread":      round(spread, 4),
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("spread", ascending=False).reset_index(drop=True)

    # Decile rank: 10 = largest positive spread (cheapest vol), 1 = largest negative
    df["decile"] = pd.qcut(df["spread"], q=10, labels=False, duplicates="drop") + 1
    return df


# ─────────────────────────────────────────────
# PORTFOLIO CONSTRUCTION
# ─────────────────────────────────────────────

def build_portfolio(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Assign trade direction based on decile:
      - Top decile (10): LONG straddle  (HV >> IV → buy cheap vol)
      - Bottom decile (1): SHORT straddle (IV >> HV → sell expensive vol)
    """
    if df.empty:
        return pd.DataFrame()

    max_decile = df["decile"].max()
    min_decile = df["decile"].min()

    conditions = [
        df["decile"] >= max_decile - cfg["top_decile_n"] + 1,
        df["decile"] <= min_decile + cfg["top_decile_n"] - 1,
    ]
    choices = ["LONG STRADDLE", "SHORT STRADDLE"]
    df["position"] = np.select(conditions, choices, default="FLAT")

    portfolio = df[df["position"] != "FLAT"].copy()
    portfolio["notional"] = cfg["notional_per_straddle"]

    # Approximate straddle cost: ~0.8 * IV * sqrt(T/252) * S
    # Here we express it relative to notional for sizing reference
    portfolio["approx_straddle_pct"] = (
        0.8 * portfolio["IV"] * np.sqrt(21 / 252)
    ).round(4)

    return portfolio[["ticker", "snapshot_dt", "expiry",
                       "HV", "IV", "spread", "decile",
                       "position", "notional", "approx_straddle_pct"]]


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_snapshot(df: pd.DataFrame,
                  portfolio: pd.DataFrame,
                  as_of_date: datetime):
    """Bar chart of HV-IV spreads colored by position."""

    if df.empty:
        return

    # ── limit for readability ───────────────────────────────
    df = df.head(40)

    # Precompute sets (faster than repeated filtering)
    long_set = set(portfolio[portfolio["position"] == "LONG STRADDLE"]["ticker"])
    short_set = set(portfolio[portfolio["position"] == "SHORT STRADDLE"]["ticker"])

    # ── figure layout ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 14))
    fig.suptitle(
        f"HV vs IV Analysis  |  {as_of_date.strftime('%B %Y')}",
        fontsize=14,
        fontweight="bold"
    )

    # ── Left: spread bar chart ──────────────────────────────
    ax = axes[0]

    colors = []
    for ticker in df["ticker"]:
        if ticker in long_set:
            colors.append("#2ecc71")   # green = buy vol
        elif ticker in short_set:
            colors.append("#e74c3c")   # red = sell vol
        else:
            colors.append("#95a5a6")   # grey = flat

    ax.barh(
        df["ticker"],
        df["spread"] * 100,
        color=colors,
        edgecolor="white"
    )

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("HV − IV  (%)")
    ax.set_title("Spread by Ticker  (green=Long, red=Short)")
    ax.invert_yaxis()

    ax.tick_params(axis="y", labelsize=6)
    ax.tick_params(axis="x", labelsize=8)

    # ── Right: HV vs IV heatmap ─────────────────────────────
    ax2 = axes[1]

    pivot = df[["ticker", "HV", "IV"]].set_index("ticker")

    import seaborn as sns
    sns.heatmap(
        pivot.multiply(100).round(1),
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        linewidths=0.5,
        ax=ax2,
        cbar_kws={"label": "Volatility (%)"}
    )

    ax2.set_title("HV vs IV Heatmap")
    ax2.tick_params(axis="y", labelsize=6)
    ax2.tick_params(axis="x", labelsize=8)

    plt.tight_layout()

    # ── save inside project folder ──────────────────────────
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "outputs")
    os.makedirs(save_dir, exist_ok=True)

    fname = os.path.join(
        save_dir,
        f"hv_iv_snapshot_{as_of_date.strftime('%Y%m')}.png"
    )

    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved → {fname}")
    plt.close()

# ─────────────────────────────────────────────
# BACKTESTING LOOP  (multi-month)
# ─────────────────────────────────────────────

class StraddleBacktester:
    """
    Simplified P&L backtester.
    P&L for a straddle approximated as:
        Long  straddle: realized_move - premium_paid
        Short straddle: premium_received - realized_move
    where realized_move = |S_expiry / S_entry - 1|
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.results: list[dict] = []

    def run(self, months: list[tuple[int, int]]) -> pd.DataFrame:
        all_snapshots = []
        all_portfolios = []

        for year, month in months:
            entry_date = get_start_of_trading_month(year, month)
            expiry_dt  = get_next_monthly_expiry(entry_date)

            snapshot  = build_hv_iv_snapshot(self.cfg["tickers"], entry_date, self.cfg)
            if snapshot.empty:
                continue
            portfolio = build_portfolio(snapshot, self.cfg)

            all_snapshots.append(snapshot.assign(month=f"{year}-{month:02d}"))

            # Compute realized P&L at expiry
            pnl_records = []
            for _, row in portfolio.iterrows():
                pnl = self._compute_pnl(row, entry_date, expiry_dt)
                pnl_records.append({**row.to_dict(), "pnl": pnl})

            if pnl_records:
                month_df = pd.DataFrame(pnl_records)
                all_portfolios.append(month_df)

        if not all_portfolios:
            return pd.DataFrame()

        full = pd.concat(all_portfolios, ignore_index=True)
        self._print_summary(full)
        return full

    def _compute_pnl(self, row: pd.Series,
                     entry: datetime, expiry: datetime) -> float:
        """Simplified straddle P&L based on realized vs implied move."""
        ticker = row["ticker"]
        notional = row["notional"]
        iv = row["IV"]
        position = row["position"]

        # Download price at entry and expiry
        try:
            hist = yf.download(
                ticker,
                start=(entry - timedelta(days=3)).strftime("%Y-%m-%d"),
                end=(expiry + timedelta(days=3)).strftime("%Y-%m-%d"),
                auto_adjust=True, progress=False
            )["Close"]
            if hist.empty or len(hist) < 2:
                return 0.0

            S_entry  = float(hist.asof(entry))
            S_expiry = float(hist.asof(expiry))
        except Exception:
            return 0.0

        T = max((expiry - entry).days / 365.0, 1 / 365)
        realized_move = abs(S_expiry / S_entry - 1)
        implied_move  = 0.8 * iv * np.sqrt(T)      # expected straddle break-even

        # P&L as % of notional
        pnl_pct = (realized_move - implied_move) if position == "LONG STRADDLE" \
             else (implied_move - realized_move)

        return round(pnl_pct * notional, 2)

    @staticmethod
    def _print_summary(df: pd.DataFrame):
        print("\n" + "=" * 60)
        print("BACKTEST SUMMARY")
        print("=" * 60)
        for pos in ["LONG STRADDLE", "SHORT STRADDLE"]:
            sub = df[df["position"] == pos]
            if sub.empty:
                continue
            total_pnl   = sub["pnl"].sum()
            avg_pnl     = sub["pnl"].mean()
            win_rate    = (sub["pnl"] > 0).mean()
            print(f"\n{pos}")
            print(f"  Trades    : {len(sub)}")
            print(f"  Total P&L : ${total_pnl:,.0f}")
            print(f"  Avg P&L   : ${avg_pnl:,.0f}")
            print(f"  Win Rate  : {win_rate:.1%}")

        print(f"\nCOMBINED TOTAL P&L : ${df['pnl'].sum():,.0f}")
        print("=" * 60)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    cfg = CONFIG

    print("\n" + "╔" + "═"*58 + "╗")
    print("║   HV vs IV Straddle Portfolio Strategy" + " "*19 + "║")
    print("╚" + "═"*58 + "╝")

    # ── Single-month snapshot (current month) ───────────────
    now = datetime.today()
    snapshot = build_hv_iv_snapshot(cfg["tickers"], now, cfg)

    if snapshot.empty:
        print("\nNo data retrieved. Check internet connection / tickers.")
        return

    portfolio = build_portfolio(snapshot, cfg)

    print("\n" + "─"*60)
    print("FULL SNAPSHOT (sorted by spread):")
    print("─"*60)
    print(snapshot.to_string(index=False))

    print("\n" + "─"*60)
    print("PORTFOLIO TRADES:")
    print("─"*60)
    print(portfolio.to_string(index=False) if not portfolio.empty
          else "  No trades (insufficient tickers for decile split).")

    
    save_dir = os.path.expanduser("~/hv_iv_strategy")   # creates ~/hv_iv_strategy/
    os.makedirs(save_dir, exist_ok=True)
    snapshot.to_csv(os.path.join(save_dir, "hv_iv_snapshot.csv"), index=False)
    portfolio.to_csv(os.path.join(save_dir, "hv_iv_portfolio.csv"), index=False)
    print("\n  CSVs saved → hv_iv_snapshot.csv, hv_iv_portfolio.csv")

    # Plot
    plot_snapshot(snapshot, portfolio, now)

    # ── Optional: 3-month backtest ───────────────────────────
    run_backtest = True          # Set True to run historical backtest
    if run_backtest:
        backtester = StraddleBacktester(cfg)
        backtest_months = [
            (2024, 1), (2024, 2), (2024, 3),
            (2024, 4), (2024, 5), (2024, 6),
        ]
        results = backtester.run(backtest_months)
        if not results.empty:
            results.to_csv("backtest_results.csv", index=False)
            print("\nBacktest results → backtest_results.csv")




if __name__ == "__main__":
    main()