# daily_update.py
# Daily job: refresh prices, recompute latest signals, and extend the paper portfolio log.
import pandas as pd
import numpy as np
import yfinance as yf
from sqlalchemy import create_engine, text
import statsmodels.api as sm
from pykalman import KalmanFilter
import json
import logging
import os
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent     # the project folder, so cron and manual runs behave the same
DATA_DIR = HERE / "data"

logging.basicConfig(
    filename=str(HERE / "daily_update.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def get_db_url():
    """Database address comes from the DB_URL setting, or from a local file called .db_url
    (kept out of GitHub). No password is stored in this code."""
    url = os.environ.get("DB_URL")
    if url:
        return url
    url_file = HERE / ".db_url"
    if url_file.exists():
        return url_file.read_text().strip()
    raise RuntimeError("No database address found. Set DB_URL or create a .db_url file in the project folder.")


engine = create_engine(get_db_url())

ALL_TICKERS = [
    "HSBA.L","BARC.L","LLOY.L","NWG.L","STAN.L","RIO.L","BHP.L","AAL.L","GLEN.L","ANTO.L",
    "AV.L","LGEN.L","PRU.L","ADM.L","ABDN.L","TSCO.L","SBRY.L","MKS.L","NXT.L","JD.L",
    "NG.L","SSE.L","SVT.L","UU.L","CNA.L","BP.L","SHEL.L","AZN.L","GSK.L","RR.L","BA.L",
    "SMIN.L","VOD.L","BT-A.L","ULVR.L","DGE.L","RKT.L","LAND.L","BLND.L","IAG.L","EZJ.L"
]

SELECTED_PAIRS = [("RIO.L","ANTO.L"), ("PRU.L","ABDN.L"), ("NG.L","SVT.L"), ("SVT.L","UU.L")]
IN_SAMPLE_END = "2023-12-31"

# Paper portfolio settings
GO_LIVE = pd.Timestamp("2026-10-12")   # first day of the forward paper record; portfolio starts flat
COST_BPS = 10                          # per leg, charged on every position change
ENTRY_Z, EXIT_Z, Z_WINDOW = 2.0, 0.5, 60


def update_prices():
    logging.info("Starting price update")
    for symbol in ALL_TICKERS:
        try:
            df = yf.download(symbol, period="5d", progress=False, auto_adjust=False)
            if df.empty:
                logging.warning(f"No data for {symbol}")
                continue
            df = df.reset_index()
            df.columns = [c[0].lower().replace(" ", "_") if isinstance(c, tuple) else c.lower().replace(" ", "_") for c in df.columns]
            df["symbol"] = symbol
            records = df[["symbol","date","open","high","low","close","adj_close","volume"]].to_dict("records")
            with engine.begin() as conn:
                for r in records:
                    conn.execute(text("""
                        INSERT INTO prices (symbol, date, open, high, low, close, adj_close, volume)
                        VALUES (:symbol, :date, :open, :high, :low, :close, :adj_close, :volume)
                        ON CONFLICT (symbol, date) DO UPDATE SET
                            open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
                            close=EXCLUDED.close, adj_close=EXCLUDED.adj_close, volume=EXCLUDED.volume;
                    """), r)
        except Exception as e:
            logging.error(f"Failed to update {symbol}: {e}")
    logging.info("Price update complete")


def kalman_hedge_ratio(y, x, obs_cov):
    delta = 1e-5
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.vstack([x, np.ones(len(x))]).T[:, np.newaxis]
    kf = KalmanFilter(
        n_dim_obs=1, n_dim_state=2, initial_state_mean=[0, 0],
        initial_state_covariance=np.ones((2, 2)) * 10, transition_matrices=np.eye(2),
        observation_matrices=obs_mat, observation_covariance=obs_cov, transition_covariance=trans_cov
    )
    state_means, _ = kf.filter(y.values)
    return state_means[:, 0]


def recompute_signals():
    logging.info("Recomputing signals")
    df = pd.read_sql(text("SELECT symbol, date, adj_close FROM prices ORDER BY date"), engine)
    df["date"] = pd.to_datetime(df["date"])
    prices = df.pivot(index="date", columns="symbol", values="adj_close").dropna(how="any")

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS daily_signals (
                pair TEXT, date DATE, spread DOUBLE PRECISION,
                zscore DOUBLE PRECISION, signal INTEGER,
                PRIMARY KEY (pair, date)
            );
        """))

    for t1, t2 in SELECTED_PAIRS:
        x_reg = sm.add_constant(prices[t2])
        residual_var = sm.OLS(prices[t1], x_reg).fit().resid.var()
        hedge_ratios = kalman_hedge_ratio(prices[t1], prices[t2], residual_var)
        spread = prices[t1].values - hedge_ratios * prices[t2].values
        spread_s = pd.Series(spread, index=prices.index)

        rolling_mean = spread_s.rolling(60).mean()
        rolling_std = spread_s.rolling(60).std()
        zscore = (spread_s - rolling_mean) / rolling_std

        latest_date = prices.index[-1]
        latest_z = zscore.iloc[-1]
        signal = 1 if latest_z < -2.0 else (-1 if latest_z > 2.0 else 0)

        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO daily_signals (pair, date, spread, zscore, signal)
                VALUES (:pair, :date, :spread, :zscore, :signal)
                ON CONFLICT (pair, date) DO UPDATE SET
                    spread=EXCLUDED.spread, zscore=EXCLUDED.zscore, signal=EXCLUDED.signal;
            """), {"pair": f"{t1}/{t2}", "date": latest_date, "spread": float(spread_s.iloc[-1]),
                   "zscore": float(latest_z), "signal": signal})

    logging.info("Signal recompute complete")


def compute_paper_pair(p1, p2, obs_cov):
    """Simulated trading of one pair. Starts flat on GO_LIVE.
    A signal is read at the close of day D; that position earns returns from D+1.
    The hedge ratio used for sizing is the one known at the previous close (no look-ahead).
    Returns a DataFrame indexed by date: position (held at that day's close), hedge_ratio, net_return."""
    hedge = pd.Series(kalman_hedge_ratio(p1, p2, obs_cov), index=p1.index)
    spread = p1 - hedge * p2
    z = (spread - spread.rolling(Z_WINDOW).mean()) / spread.rolling(Z_WINDOW).std()

    pos = pd.Series(0, index=p1.index, dtype=int)
    position = 0
    for i, d in enumerate(p1.index):
        zi = z.iloc[i]
        if d < GO_LIVE or np.isnan(zi):
            continue
        if position == 0:
            if zi > ENTRY_Z:
                position = -1      # spread too high: short it
            elif zi < -ENTRY_Z:
                position = 1       # spread too low: long it
        elif abs(zi) < EXIT_Z:
            position = 0           # back near the mean: close
        pos.iloc[i] = position

    r1 = p1.pct_change()
    r2 = p2.pct_change()
    h_prev = hedge.shift(1)
    notional = 1 + h_prev.abs()
    gross = pos.shift(1).fillna(0) * (r1 / notional - (h_prev / notional) * r2)
    cost = pos.diff().abs().fillna(0) * (COST_BPS / 10000) * 2
    net = gross.fillna(0) - cost
    return pd.DataFrame({"position": pos, "hedge_ratio": hedge, "net_return": net})


def update_paper_portfolio():
    """Adds any new days to the paper_portfolio table. Existing rows are never changed,
    so the table is an append-only forward record."""
    logging.info("Updating paper portfolio")
    tickers = sorted({t for pair in SELECTED_PAIRS for t in pair})

    df = pd.read_sql(text("SELECT symbol, date, adj_close FROM prices ORDER BY date"), engine)
    df["date"] = pd.to_datetime(df["date"])
    prices = df.pivot(index="date", columns="symbol", values="adj_close")[tickers].dropna(how="any")

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS paper_portfolio (
                pair TEXT, date DATE, position INTEGER,
                hedge_ratio DOUBLE PRECISION, net_return DOUBLE PRECISION,
                PRIMARY KEY (pair, date)
            );
        """))

    calib = prices.loc[prices.index <= IN_SAMPLE_END]   # noise level set from pre-2024 data only
    added = 0
    for t1, t2 in SELECTED_PAIRS:
        obs_cov = sm.OLS(calib[t1], sm.add_constant(calib[t2])).fit().resid.var()
        res = compute_paper_pair(prices[t1], prices[t2], obs_cov)
        res = res[res.index >= GO_LIVE]
        if res.empty:
            continue
        with engine.begin() as conn:
            for d, r in res.iterrows():
                result = conn.execute(text("""
                    INSERT INTO paper_portfolio (pair, date, position, hedge_ratio, net_return)
                    VALUES (:pair, :date, :position, :hedge_ratio, :net_return)
                    ON CONFLICT (pair, date) DO NOTHING;
                """), {"pair": f"{t1}/{t2}", "date": d.date(), "position": int(r["position"]),
                       "hedge_ratio": float(r["hedge_ratio"]), "net_return": float(r["net_return"])})
                added += result.rowcount

    if added == 0 and prices.index[-1] < GO_LIVE:
        logging.info(f"Paper portfolio not live yet (starts {GO_LIVE.date()})")
    else:
        logging.info(f"Paper portfolio updated, {added} new rows")


def export_dashboard_data():
    """Writes small CSV files into ./data so the public dashboard never needs the database.
    The dashboard (app.py) reads only these files."""
    logging.info("Exporting dashboard data")
    DATA_DIR.mkdir(exist_ok=True)
    tickers = sorted({t for pair in SELECTED_PAIRS for t in pair})

    df = pd.read_sql(text("SELECT symbol, date, adj_close FROM prices ORDER BY date"), engine)
    df["date"] = pd.to_datetime(df["date"])
    prices = df.pivot(index="date", columns="symbol", values="adj_close")[tickers].dropna(how="any")
    calib = prices.loc[prices.index <= IN_SAMPLE_END]

    # Full history of hedge ratio, spread and z-score for each pair (for the charts)
    frames = []
    for t1, t2 in SELECTED_PAIRS:
        obs_cov = sm.OLS(calib[t1], sm.add_constant(calib[t2])).fit().resid.var()
        hedge = pd.Series(kalman_hedge_ratio(prices[t1], prices[t2], obs_cov), index=prices.index)
        spread = prices[t1] - hedge * prices[t2]
        z = (spread - spread.rolling(Z_WINDOW).mean()) / spread.rolling(Z_WINDOW).std()
        frames.append(pd.DataFrame({"pair": f"{t1}/{t2}", "date": prices.index,
                                    "hedge_ratio": hedge.values, "spread": spread.values, "zscore": z.values}))
    pd.concat(frames).to_csv(DATA_DIR / "pair_history.csv", index=False)

    for table in ("paper_portfolio", "daily_signals"):
        try:
            pd.read_sql(text(f"SELECT * FROM {table} ORDER BY pair, date"), engine).to_csv(DATA_DIR / f"{table}.csv", index=False)
        except Exception:
            logging.exception(f"Could not export {table}")

    meta = {"last_price_date": str(prices.index[-1].date()), "go_live": str(GO_LIVE.date()),
            "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "cost_bps": COST_BPS,
            "entry_z": ENTRY_Z, "exit_z": EXIT_Z, "z_window": Z_WINDOW, "pairs": [f"{a}/{b}" for a, b in SELECTED_PAIRS]}
    (DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    logging.info("Dashboard data exported")


if __name__ == "__main__":
    logging.info(f"=== Daily update run: {datetime.now()} ===")
    for step in (update_prices, recompute_signals, update_paper_portfolio, export_dashboard_data):
        try:
            step()
        except Exception:
            logging.exception(f"Step failed: {step.__name__}")
    logging.info("=== Run finished ===")
