# backtest_oos.py
# Corrected out-of-sample backtest of the four-pair strategy.
# It reuses compute_paper_pair from daily_update.py, so the backtest and the live paper
# portfolio run exactly the same trading logic.
#
# Fixes compared with the first notebook version:
#   1. Kalman noise level is set from pre-2024 data only (no peeking at the test period).
#   2. Position sizing uses the hedge ratio known at the previous close (no same-day peeking).
#   3. The test starts flat on the first test day, like the paper portfolio does.
#
# Run from the project folder:  python backtest_oos.py
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # draw to a file, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sqlalchemy import text

import daily_update as du

DATA_DIR = Path(__file__).resolve().parent / "data"
OOS_START = pd.Timestamp("2024-01-02")     # first out-of-sample day (pairs were chosen on data up to 2023)
N_PAIRS_TESTED = 61                        # every pair tested for cointegration in the selection step
HEADLINE_COST_BPS = 10
COST_STRESS_BPS = (5, 10, 20)
EULER_GAMMA = 0.5772156649


def load_prices():
    tickers = sorted({t for pair in du.SELECTED_PAIRS for t in pair})
    df = pd.read_sql(text("SELECT symbol, date, adj_close FROM prices ORDER BY date"), du.engine)
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="symbol", values="adj_close")[tickers].dropna(how="any")


def run_backtest(prices, cost_bps):
    du.GO_LIVE = OOS_START                 # same switch the paper portfolio uses
    du.COST_BPS = cost_bps
    calib = prices.loc[prices.index <= du.IN_SAMPLE_END]
    rets = {}
    for t1, t2 in du.SELECTED_PAIRS:
        obs_cov = sm.OLS(calib[t1], sm.add_constant(calib[t2])).fit().resid.var()
        res = du.compute_paper_pair(prices[t1], prices[t2], obs_cov)
        rets[f"{t1}/{t2}"] = res.loc[res.index >= OOS_START, "net_return"]
    r = pd.DataFrame(rets)
    r["PORTFOLIO"] = r.mean(axis=1)        # equal weight across the four pairs
    return r


def metrics(r):
    eq = (1 + r).cumprod()
    n = len(r)
    total = eq.iloc[-1] - 1
    annual = eq.iloc[-1] ** (252 / n) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * np.sqrt(252)
    sortino = r.mean() * 252 / downside if downside > 0 else np.nan
    max_dd = (eq / eq.cummax() - 1).min()
    var95 = np.percentile(r, 5)
    cvar95 = r[r <= var95].mean()
    return {
        "total_return": total, "annual_return": annual, "annual_vol": vol,
        "sharpe": sharpe, "sortino": sortino, "max_drawdown": max_dd,
        "skew": r.skew(), "excess_kurtosis": r.kurtosis(),
        "var95_daily": var95, "cvar95_daily": cvar95,
    }


def psr(r, sr_benchmark=0.0):
    """Probabilistic Sharpe ratio: chance the true Sharpe beats sr_benchmark (daily units),
    allowing for the sample length and the skew and fat tails of the returns."""
    t = len(r)
    sr = r.mean() / r.std()
    skew = r.skew()
    kurt = r.kurtosis() + 3                # raw kurtosis
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(t - 1) / denom))


def expected_max_sharpe(n_trials, var_sr):
    """Best Sharpe you would expect from n_trials strategies with no real skill."""
    return np.sqrt(var_sr) * (
        (1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials)
        + EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * np.e))
    )


def pct(x):
    return f"{x * 100:6.2f}%"


def main():
    prices = load_prices()
    print(f"Prices loaded: {prices.index.min().date()} to {prices.index.max().date()}")

    r = run_backtest(prices, HEADLINE_COST_BPS)
    print(f"Out-of-sample window: {r.index.min().date()} to {r.index.max().date()} ({len(r)} trading days)\n")

    table = pd.DataFrame({col: metrics(r[col]) for col in r.columns}).T
    show = table.copy()
    for c in ["total_return", "annual_return", "annual_vol", "max_drawdown", "var95_daily", "cvar95_daily"]:
        show[c] = show[c].map(pct)
    for c in ["sharpe", "sortino", "skew", "excess_kurtosis"]:
        show[c] = show[c].map(lambda v: f"{v:6.2f}")
    print(f"=== Out-of-sample results, {HEADLINE_COST_BPS}bps per leg ===")
    print(show.to_string())

    stats = {"window_start": str(r.index.min().date()), "window_end": str(r.index.max().date()),
             "trading_days": int(len(r)), "cost_sensitivity": [], "deflated": []}
    print("\n=== Cost sensitivity (portfolio) ===")
    for bps in COST_STRESS_BPS:
        m = metrics(run_backtest(prices, bps)["PORTFOLIO"])
        stats["cost_sensitivity"].append({"bps": bps, "total_return": m["total_return"],
                                          "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"]})
        print(f"{bps:>3}bps per leg: total return {pct(m['total_return'])}, Sharpe {m['sharpe']:5.2f}, max drawdown {pct(m['max_drawdown'])}")
    run_backtest(prices, HEADLINE_COST_BPS)   # leave the module at the headline setting

    # Statistical confidence in the Sharpe ratio
    port = r["PORTFOLIO"]
    pair_sr_daily = [r[c].mean() / r[c].std() for c in r.columns if c != "PORTFOLIO" and r[c].std() > 0]
    var_sr = float(np.var(pair_sr_daily, ddof=1))
    print("\n=== How much should you trust the Sharpe ratio? ===")
    print(f"Probabilistic Sharpe (chance the true Sharpe is above zero): {psr(port):.1%}")
    stats["psr"] = psr(port)
    stats["daily_sharpe"] = float(port.mean() / port.std())
    for n in (len(pair_sr_daily), N_PAIRS_TESTED):
        sr0 = expected_max_sharpe(n, var_sr)
        stats["deflated"].append({"n_trials": int(n), "dsr": psr(port, sr0), "luck_daily_sharpe": float(sr0)})
        print(f"Deflated Sharpe, allowing for {n:>2} strategies tried: {psr(port, sr0):.1%}  "
              f"(luck alone would give a daily Sharpe of about {sr0:.3f}; yours is {port.mean() / port.std():.3f})")
    print("Read it like a probability. Near 50% or below means the result is not distinguishable from luck.")

    # Save outputs for the dashboard and the write-up
    DATA_DIR.mkdir(exist_ok=True)
    r.to_csv(DATA_DIR / "backtest_oos_returns.csv", index_label="date")
    table.to_csv(DATA_DIR / "backtest_oos_summary.csv", index_label="series")
    (DATA_DIR / "backtest_oos_stats.json").write_text(json.dumps(stats, indent=2))
    eq = (1 + r).cumprod()
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in eq.columns:
        if col == "PORTFOLIO":
            ax.plot(eq.index, eq[col], color="black", linewidth=2.2, label="Equal-weight portfolio")
        else:
            ax.plot(eq.index, eq[col], alpha=0.35, label=col)
    ax.axhline(1.0, color="gray", linestyle=":")
    ax.set_title(f"Out-of-sample equity curves ({HEADLINE_COST_BPS}bps per leg, corrected method)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DATA_DIR / "backtest_oos_equity.png", dpi=150)
    print("\nSaved to the data folder: backtest_oos_returns.csv, backtest_oos_summary.csv, backtest_oos_stats.json, backtest_oos_equity.png")


if __name__ == "__main__":
    main()
