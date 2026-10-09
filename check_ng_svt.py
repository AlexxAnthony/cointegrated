# check_ng_svt.py
# Looks at the worst days of the NG/SVT pair to see if the extreme tail is a real move or a data glitch.
# Run from the project folder:  python check_ng_svt.py
import pandas as pd
from sqlalchemy import text

import daily_update as du

PAIR = "NG.L/SVT.L"

r = pd.read_csv("backtest_oos_returns.csv", index_col="date", parse_dates=True)[PAIR]
worst = r.nsmallest(5)
print("=== 5 worst days for", PAIR, "===")
print((worst * 100).round(2).astype(str) + "%")

df = pd.read_sql(text("SELECT symbol, date, adj_close, close FROM prices WHERE symbol IN ('NG.L','SVT.L') ORDER BY date"), du.engine)
df["date"] = pd.to_datetime(df["date"])
adj = df.pivot(index="date", columns="symbol", values="adj_close")
raw = df.pivot(index="date", columns="symbol", values="close")

print("\n=== Daily price moves on those days (adjusted close) ===")
moves = adj.pct_change()
for d in worst.index:
    print(d.date(), "| NG move:", f"{moves.loc[d, 'NG.L'] * 100:6.2f}%", "| SVT move:", f"{moves.loc[d, 'SVT.L'] * 100:6.2f}%")

print("\n=== Raw close vs adjusted close around the worst day ===")
d0 = worst.index[0]
window = pd.concat([raw.add_suffix(" close"), adj.add_suffix(" adj")], axis=1)
i = window.index.get_loc(d0)
print(window.iloc[max(i - 3, 0): i + 4].round(2).to_string())

print("\n=== Largest absolute daily moves in the whole history ===")
big = moves.abs().stack().nlargest(6)
print((big * 100).round(2).astype(str) + "%")
