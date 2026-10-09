# app.py
# Cointegrated: UK equity pairs-trading dashboard.
# Reads only the CSV/JSON files in ./data (written by daily_update.py and backtest_oos.py),
# so it runs anywhere, including Streamlit Community Cloud, with no database.
#
# Run locally:  streamlit run app.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).resolve().parent / "data"

st.set_page_config(page_title="Cointegrated - UK pairs trading", page_icon="📈", layout="wide")


# ----------------------------------------------------------------- loading
@st.cache_data(ttl=600)
def load_csv(name, parse_dates=("date",)):
    path = DATA / name
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=list(parse_dates) if parse_dates else None)


@st.cache_data(ttl=600)
def load_json(name):
    path = DATA / name
    return json.loads(path.read_text()) if path.exists() else None


meta = load_json("meta.json")
bt_stats = load_json("backtest_oos_stats.json")
history = load_csv("pair_history.csv")
paper = load_csv("paper_portfolio.csv")
bt_returns = load_csv("backtest_oos_returns.csv")
bt_summary = load_csv("backtest_oos_summary.csv", parse_dates=None)

if meta is None or history is None or bt_returns is None or bt_stats is None:
    st.error("Data files are missing from the data folder. Run `python daily_update.py` and `python backtest_oos.py` first.")
    st.stop()

bt_returns = bt_returns.set_index("date")
PAIRS = meta["pairs"]


# ----------------------------------------------------------------- helpers
def pct(x, digits=2):
    return f"{x * 100:.{digits}f}%"


def drawdown(r):
    eq = (1 + r).cumprod()
    return eq / eq.cummax() - 1


def rolling_sharpe(r, window=126):
    return r.rolling(window).mean() / r.rolling(window).std() * np.sqrt(252)


def line_fig(series_dict, title, yaxis=None, hlines=(), height=380):
    fig = go.Figure()
    for name, s in series_dict.items():
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=name, line=dict(width=2)))
    for y, dash in hlines:
        fig.add_hline(y=y, line_dash=dash, line_color="gray", opacity=0.6)
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=50, b=10),
                      yaxis_title=yaxis, legend=dict(orientation="h", y=-0.15))
    return fig


# ----------------------------------------------------------------- header
st.title("Cointegrated")
st.caption("UK equity pairs-trading research project. Cointegration pair selection, Kalman-filter hedge ratio, "
           "walk-forward out-of-sample backtest and a forward paper-trading record. Simulation only, no real money.")
st.caption(f"Prices up to {meta['last_price_date']}  |  dashboard data refreshed {meta['exported_at']}")

tab_over, tab_live, tab_sig, tab_bt, tab_lim = st.tabs(
    ["Overview", "Live paper record", "Signals and hedge ratio", "Backtest", "Method and limitations"])

port = bt_returns["PORTFOLIO"]
port_row = bt_summary.set_index("series").loc["PORTFOLIO"] if bt_summary is not None else None
dsr4 = next((d for d in bt_stats["deflated"] if d["n_trials"] == len(PAIRS)), bt_stats["deflated"][0])
dsr_many = bt_stats["deflated"][-1]

# ----------------------------------------------------------------- overview
with tab_over:
    st.subheader("The honest headline")
    st.write(
        "The strategy trades four pairs of UK shares whose prices have moved together historically. "
        "When the gap between a pair widens abnormally it bets the gap will close. "
        "The pairs were chosen using data up to 31 Dec 2023 only. Everything after that date is out-of-sample."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Out-of-sample return", pct(port_row["total_return"]), help=f"{bt_stats['window_start']} to {bt_stats['window_end']}, {meta['cost_bps']}bps per leg")
    c2.metric("Sharpe ratio", f"{port_row['sharpe']:.2f}")
    c3.metric("Max drawdown", pct(port_row["max_drawdown"]))
    c4.metric("Deflated Sharpe", f"{dsr4['dsr']:.0%}",
              help="Probability the result beats what luck alone would produce, allowing for several strategies tried. Near or below 50% means not distinguishable from luck.")
    st.info(
        f"**Reading this fairly:** {'a modest positive return' if port_row['total_return'] > 0 else 'a negative return'}, and the deflated Sharpe is {dsr4['dsr']:.0%} "
        f"(only {dsr_many['dsr']:.0%} if you count all {dsr_many['n_trials']} pairs originally tested), so it cannot be told apart from luck. "
        "The edge also disappears when trading costs double. This is a research and engineering project, not a proven money-maker."
    )
    eq = (1 + bt_returns).cumprod()
    st.plotly_chart(line_fig({"Equal-weight portfolio": eq["PORTFOLIO"]}, "Out-of-sample growth of 1 (portfolio)", "Growth of 1", hlines=[(1.0, "dot")]))
    n_live = 0 if paper is None or paper.empty else paper["date"].nunique()
    if n_live == 0:
        st.write(f"**Forward paper record:** starts {meta['go_live']}. It is an append-only log written by a scheduled job, so it cannot be edited after the fact.")
    else:
        st.write(f"**Forward paper record:** {n_live} trading days logged so far. See the Live paper record tab.")

# ----------------------------------------------------------------- live record
with tab_live:
    st.subheader("Forward paper-trading record")
    st.write("A scheduled job logs the strategy's simulated positions and returns every weekday after the market closes. "
             "Rows are never changed once written. This is the real test of the strategy, because it cannot be fitted to the past.")
    if paper is None or paper.empty:
        st.info(f"No live rows yet. The record starts on {meta['go_live']} and the portfolio begins flat. "
                "The first rows appear after the first scheduled run on or after that date.")
    else:
        daily = paper.groupby("date")["net_return"].mean().sort_index()
        eq_live = (1 + daily).cumprod()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trading days logged", f"{len(daily)}")
        m2.metric("Total return", pct(eq_live.iloc[-1] - 1))
        m3.metric("Max drawdown", pct(drawdown(daily).min()))
        if len(daily) >= 60 and daily.std() > 0:
            m4.metric("Sharpe ratio", f"{daily.mean() / daily.std() * np.sqrt(252):.2f}")
        else:
            m4.metric("Sharpe ratio", "n/a", help="Needs at least 60 trading days to mean anything.")
        st.plotly_chart(line_fig({"Paper portfolio": eq_live}, "Paper portfolio growth of 1", "Growth of 1", hlines=[(1.0, "dot")]))
        st.plotly_chart(line_fig({"Drawdown": drawdown(daily)}, "Drawdown", "Drawdown"))

        st.markdown("**Current positions**")
        last = paper.sort_values("date").groupby("pair").tail(1).set_index("pair")
        label = {1: "Long the spread", -1: "Short the spread", 0: "Flat"}
        show = pd.DataFrame({"As of": last["date"].dt.date, "Position": last["position"].map(label),
                             "Hedge ratio": last["hedge_ratio"].round(3)})
        st.dataframe(show)
        if len(daily) < 60:
            st.caption("Fewer than 60 trading days so far: treat these numbers as an early log, not evidence.")

# ----------------------------------------------------------------- signals
with tab_sig:
    st.subheader("Where each pair stands today")
    latest = history.sort_values("date").groupby("pair").tail(1).set_index("pair")

    def read_signal(z):
        if pd.isna(z):
            return "n/a"
        if z > meta["entry_z"]:
            return "Entry: short the spread"
        if z < -meta["entry_z"]:
            return "Entry: long the spread"
        return "No new entry"

    table = pd.DataFrame({"Date": latest["date"].dt.date, "Z-score": latest["zscore"].round(2),
                          "Hedge ratio": latest["hedge_ratio"].round(3),
                          "Signal at today's z-score": latest["zscore"].map(read_signal)})
    st.dataframe(table)
    st.caption(f"Enter when |z| > {meta['entry_z']}, exit when |z| < {meta['exit_z']}, z measured over a {meta['z_window']}-day window. "
               "The table shows the entry rule only. A pair already in a trade stays in it until it exits, so see the paper record for actual positions.")

    pair = st.selectbox("Pair", PAIRS)
    h = history[history["pair"] == pair].set_index("date")
    years = st.slider("Years of history to show", 1, 8, 2)
    h = h[h.index >= h.index.max() - pd.DateOffset(years=years)]
    st.plotly_chart(line_fig({"Z-score": h["zscore"]}, f"{pair}: spread z-score", "Z-score",
                             hlines=[(meta["entry_z"], "dash"), (-meta["entry_z"], "dash"), (0, "dot")]))
    st.plotly_chart(line_fig({"Kalman hedge ratio": h["hedge_ratio"]}, f"{pair}: adaptive hedge ratio (Kalman filter)", "Shares of 2nd stock per share of 1st"))
    st.caption("A fixed hedge ratio would be a flat line. The Kalman filter lets it drift as the relationship between the two companies changes.")

# ----------------------------------------------------------------- backtest
with tab_bt:
    st.subheader("Out-of-sample backtest")
    st.write(f"Window: {bt_stats['window_start']} to {bt_stats['window_end']} ({bt_stats['trading_days']} trading days). "
             f"Pairs chosen on earlier data only. Costs: {meta['cost_bps']}bps per leg on every position change. "
             "Signals use the close of day D and earn returns from day D+1.")
    eq = (1 + bt_returns).cumprod()
    fig = go.Figure()
    for col in eq.columns:
        if col == "PORTFOLIO":
            fig.add_trace(go.Scatter(x=eq.index, y=eq[col], name="Equal-weight portfolio", line=dict(width=3, color="black")))
        else:
            fig.add_trace(go.Scatter(x=eq.index, y=eq[col], name=col, opacity=0.5))
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray")
    fig.update_layout(title="Growth of 1 by pair and combined", height=400, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig)

    st.markdown("**Results table**")
    s = bt_summary.set_index("series")
    nice = pd.DataFrame({
        "Total return": s["total_return"].map(pct), "Annual return": s["annual_return"].map(pct),
        "Volatility": s["annual_vol"].map(pct), "Sharpe": s["sharpe"].map(lambda v: f"{v:.2f}"),
        "Sortino": s["sortino"].map(lambda v: f"{v:.2f}"), "Max drawdown": s["max_drawdown"].map(pct),
        "Skew": s["skew"].map(lambda v: f"{v:.2f}"), "Excess kurtosis": s["excess_kurtosis"].map(lambda v: f"{v:.1f}"),
        "VaR 95% (1 day)": s["var95_daily"].map(pct), "CVaR 95% (1 day)": s["cvar95_daily"].map(pct)})
    st.dataframe(nice)
    st.caption("Diversification matters: single pairs had drawdowns near -25%, the combined portfolio about -10%. "
               "NG/SVT's extreme kurtosis comes from one event, the National Grid share-price drop in May 2024 (a corporate-event shock), which caused almost all of that pair's loss.")

    c1, c2 = st.columns(2)
    c1.plotly_chart(line_fig({"Rolling 6-month Sharpe": rolling_sharpe(port)}, "Portfolio rolling Sharpe (126 days)", "Sharpe", hlines=[(0, "dot")], height=320))
    c2.plotly_chart(line_fig({"Drawdown": drawdown(port)}, "Portfolio drawdown", "Drawdown", height=320))

    st.markdown("**How sensitive is it to trading costs?**")
    cs = pd.DataFrame(bt_stats["cost_sensitivity"])
    cs_show = pd.DataFrame({"Cost per leg": cs["bps"].map(lambda b: f"{b} bps"), "Total return": cs["total_return"].map(pct),
                            "Sharpe": cs["sharpe"].map(lambda v: f"{v:.2f}"), "Max drawdown": cs["max_drawdown"].map(pct)}).set_index("Cost per leg")
    st.dataframe(cs_show)

    st.markdown("**How much should the Sharpe ratio be trusted?**")
    d1, d2, d3 = st.columns(3)
    d1.metric("Probabilistic Sharpe", f"{bt_stats['psr']:.0%}", help="Chance the true Sharpe is above zero, given sample length and fat tails.")
    d2.metric(f"Deflated Sharpe ({dsr4['n_trials']} strategies)", f"{dsr4['dsr']:.0%}")
    d3.metric(f"Deflated Sharpe ({dsr_many['n_trials']} strategies)", f"{dsr_many['dsr']:.0%}")
    st.caption("Deflated Sharpe (Bailey and Lopez de Prado) asks whether the result beats what the best of several tries would show by luck alone. "
               "Near or below 50% means it is not distinguishable from luck.")

# ----------------------------------------------------------------- limitations
with tab_lim:
    st.subheader("Method")
    st.markdown(
        "1. **Universe:** 41 large UK shares (FTSE index members).\n"
        "2. **Pair selection:** Engle-Granger cointegration test on data up to 31 Dec 2023, 61 pairs tested, 4 kept.\n"
        "3. **Hedge ratio:** Kalman filter, so it adapts over time. Noise level set from pre-2024 data only.\n"
        "4. **Signals:** z-score of the spread over 60 days. Enter beyond 2, exit inside 0.5.\n"
        "5. **Sizing:** dollar-neutral, each pair gets equal capital, the four pairs are equal-weighted.\n"
        "6. **Timing:** signal at the close of day D, returns earned from D+1, hedge ratio from the previous close.\n"
        "7. **Costs:** 10bps per leg on every position change.\n"
        "8. **Automation:** a scheduled job refreshes prices, signals and the paper log every weekday evening."
    )
    st.subheader("Limitations")
    st.warning(
        "- **Not distinguishable from luck.** The deflated Sharpe is below 50%.\n"
        "- **Costs decide it.** The edge disappears at 20bps per leg.\n"
        "- **Stamp duty ignored.** UK stamp duty (0.5% on buying UK shares, as far as I know) is not in the cost model, so real trading would look worse. Shorting also needs borrow and financing costs.\n"
        "- **Survivorship bias.** Only current index members are used, so companies that dropped out are missing.\n"
        "- **Optimistic fills.** Trades are assumed at the closing price on the signal day.\n"
        "- **No stop-loss.** One corporate-event shock (National Grid, May 2024) caused essentially the whole NG/SVT loss.\n"
        "- **Small sample.** Four pairs and under three years out-of-sample.\n"
        "- **Reversal in selection.** The pairs that looked best in-sample did not rank the same out-of-sample.\n"
        "- **Simulation only.** No real orders are placed and no real money is involved."
    )
    st.caption("This page is a research project and not investment advice.")
