# Cointegrated: UK equity pairs trading

**Live dashboard:** https://cointegrated.streamlit.app

A research project that tests a statistical-arbitrage strategy on UK shares, end to end: data pipeline, pair selection, adaptive hedge ratio, out-of-sample backtest with costs, risk analytics, and a forward paper-trading record that updates every weekday. Simulation only. No real money is involved and nothing here is investment advice.

## The honest result

Out-of-sample window 2 Jan 2024 to 9 Oct 2026 (694 trading days), equal-weight portfolio of four pairs, 10bps cost per leg:

| Metric | Value |
|---|---|
| Total return | 5.48% |
| Sharpe ratio | 0.36 |
| Max drawdown | -9.57% |
| Probabilistic Sharpe | 72% |
| Deflated Sharpe (4 strategies tried) | 41% |
| Deflated Sharpe (all 61 pairs tested) | 11% |

What this does and does not show:

- The return is modest and positive, but the deflated Sharpe is below 50%, so it **cannot be told apart from luck**.
- The edge **disappears when costs double**: at 20bps per leg the portfolio return is -0.66%.
- Combining the four pairs cut the maximum drawdown from roughly -25% per pair to -9.6%.
- The pairs that looked best in-sample did not rank the same way out-of-sample.
- NG/SVT lost 11% out-of-sample. Almost all of it came from one event: National Grid fell 10.9% on 23 May 2024 (rights-issue news, as far as I know) and the spread kept widening. The strategy has no stop-loss.

I treat this as a research and engineering project, not a proven strategy. The forward paper record, which starts on 12 Oct 2026, is the real test because it cannot be fitted to the past.

## Method

1. **Universe:** 41 large UK shares (FTSE index members), daily prices from Yahoo Finance via `yfinance`.
2. **Pair selection:** Engle-Granger cointegration test on data up to 31 Dec 2023 only. 61 pairs tested, 4 kept: RIO/ANTO, PRU/ABDN, NG/SVT, SVT/UU.
3. **Hedge ratio:** Kalman filter (`pykalman`) so the ratio adapts as the relationship drifts. The noise level is set from pre-2024 data only.
4. **Signal:** z-score of the spread over a 60-day window. Enter when |z| > 2.0, exit when |z| < 0.5.
5. **Sizing:** dollar-neutral. Each pair is weighted 1/(1+|h|) and h/(1+|h|), and the four pairs are equal-weighted.
6. **Timing:** a signal read at the close of day D earns returns from D+1. The hedge ratio used for sizing is the one known at the previous close. A test that cuts off future prices confirms earlier results do not change.
7. **Costs:** 10bps per leg on every position change.
8. **Risk analytics:** Sharpe, Sortino, drawdown, skew, kurtosis, VaR and CVaR at 95%, probabilistic Sharpe and deflated Sharpe (Bailey and Lopez de Prado).
9. **Automation:** a scheduled job (`cron`) refreshes prices, recomputes signals, appends to the paper-trading log (existing rows are never changed) and exports CSV files for the dashboard.

The backtest and the paper portfolio use the same function (`compute_paper_pair`), so they trade identically.
