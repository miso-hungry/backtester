import datetime
from datetime import datetime as datetime2
import glob
import json
import math
import pytz
import requests

import streamlit as st
st.set_page_config(layout="wide")
import numpy as np
import pandas as pd
import plotly.express as pltx
import plotly.graph_objects as go
from plotly.subplots import make_subplots


time_zone = pytz.timezone('UTC')

avail_symbols = glob.glob('demo_data/*')
avail_symbols = sorted([x.split('_')[3] for x in avail_symbols])
# avail_symbols

symbols = st.multiselect('Symbol', options=avail_symbols, default=['SOLUSDT'])
cols = st.columns(2)
avail_datasets = ["2021-01-01_2024-08-01"]
dataset = cols[0].selectbox('Dataset', options=avail_datasets)

cols = st.columns(2)
now_time = datetime2.now(tz=time_zone)
start_time = now_time - datetime.timedelta(days=30 * 3)

start_date = cols[0].date_input(
    "Start Date", value=start_time.date()
)
end_date = cols[1].date_input(
    "End Date", value=now_time.date()
)

start_time = time_zone.localize(datetime2.combine(start_date, datetime.time()))
end_time = time_zone.localize(datetime2.combine(end_date, datetime.time()))

cols = st.columns(6)
capital = cols[0].number_input("Capital", min_value=0, step=50_000, value=100_000)
vol_target = cols[1].number_input("Vol Target", min_value=0.0, step=0.1, value=0.45)
simulations = (1, 2, 4, 8, 16, 32, 64)
display_forecast = cols[2].selectbox("Display Forecast", options=simulations, index=simulations.index(4))
forecast_cap = cols[3].number_input("Forecast Cap", min_value=0, value=2)
fee_bps = cols[4].number_input("Fee (bps)", min_value=0.0, step=0.5, value=3.0)
candle_len = cols[5].selectbox("Candle Length", ["1d", "1h"])

buf = []
# avail_symbols = ['SOLUSDT', 'BTCUSDT', 'ETHUSDT', 'DOGEUSDT', 'XRPUSDT']
symbols = symbols or avail_symbols
for symbol in symbols:
    fpath = f'demo_data/{dataset}_{symbol}_{candle_len}.csv.gz'
    df = pd.read_csv(fpath)
    df['symbol'] = symbol
    buf.append(df)

df = pd.concat(buf, ignore_index=True)
df['time'] = pd.to_datetime(df['time'])
df["log_price"] = np.log(df["close_price"])
df["log_return"] = df.groupby('symbol')['log_price'].diff()
df["fast_vol"] = df.groupby('symbol')["log_return"].transform(lambda grp: grp.ewm(span=32).std() * 16)
df["slow_vol"] = df.groupby('symbol')['log_return'].transform(lambda grp: grp.ewm(span=365).std() * 16)
df["vol"] = 0.7 * df["fast_vol"] + 0.3 * df["slow_vol"]
# df

reference = df
buf = []
rollups = []
return_corrs = []

for ma in simulations:
    df = reference.copy()
    df["slow_ema"] = df.groupby('symbol')["close_price"].transform(lambda grp: grp.ewm(span=ma * 4).mean())
    df["fast_ema"] = df.groupby('symbol')["close_price"].transform(lambda grp: grp.ewm(span=ma).mean())
    df["forecast"] = (df["fast_ema"] - df["slow_ema"]) / (df["vol"] / 16 * df["close_price"])
    mean_score = df["forecast"].abs().mean()
    df["forecast_norm"] = np.clip(df["forecast"] / mean_score, -forecast_cap, forecast_cap) / forecast_cap
    df["returns"] = df.groupby('symbol')["forecast_norm"].shift(1) * df["log_return"]
    df['fees'] = df.groupby('symbol')["forecast_norm"].diff().abs() * fee_bps * 1e-4
    df['returns'] = df['returns'] - df['fees']
    df["cum_returns"] = df.groupby('symbol')["returns"].cumsum()
    df['simulation'] = ma
    buf.append(df)

master = pd.concat(buf, ignore_index=True)

if len(symbols) > 1:
    fig = pltx.line(master, x="time", y="cum_returns", color="symbol", facet_col="simulation", facet_col_wrap=4)
else:
    fig = pltx.line(master, x="time", y="cum_returns", color="simulation")
st.plotly_chart(fig, use_container_width=True)

aggr = master.groupby(["simulation", "time"]).agg(dict(returns='mean', forecast_norm='mean')).reset_index()
trading_days = (aggr['time'].max() - aggr['time'].min()).days

for ma in simulations:
    df = aggr[aggr['simulation'] == ma]
    rollup = dict(
        simulation=ma,
        tot_return=df['returns'].sum(),
        annualized_return=df['returns'].sum() / trading_days * 365,
        sharpe=df['returns'].mean() / df['returns'].std() * 16,
        turnover=df['returns'].diff().abs().sum() / trading_days
    )
    rollups.append(rollup)
    return_corrs.append(df['returns'].reset_index(drop=True))

rollups = pd.DataFrame(rollups)
corrs = pd.concat(return_corrs, axis=1, keys=simulations)
corrs = corrs.corr()
cols = st.columns(2)
cols[0].subheader('Summary')
cols[0].dataframe(rollups)
cols[1].subheader('Return Correlations')
cols[1].dataframe(corrs)

if len(symbols) > 1:
    cross = master[master["simulation"] == display_forecast]
    cross = cross.groupby("symbol").last().sort_values("cum_returns").reset_index()
    st.plotly_chart(pltx.bar(cross, x="symbol", y="cum_returns"), use_container_width=True)

    aggr['cum_returns'] = aggr.groupby('simulation')['returns'].cumsum()
    st.plotly_chart(
        pltx.line(aggr, x="time", y="cum_returns", facet_col="simulation", facet_col_wrap=4),
        use_container_width=True,
    )

fig = make_subplots(rows=4, cols=1, specs=[[{}], [{}], [{"secondary_y": True}], [{}]])

cross_symbol = st.selectbox('Cross Section', options=symbols)
cross = master[(master['symbol'] == cross_symbol)]
reference = cross[cross['simulation'] == 1]
fig.add_trace(
    go.Candlestick(
        x=reference["time"],
        open=reference["open_price"],
        high=reference["high_price"],
        low=reference["low_price"],
        close=reference["close_price"],
        name="price",
    )
)

for ma in simulations:
    df = cross[cross['simulation'] == ma]
    fig.add_trace(go.Scatter(x=df["time"], y=df["fast_ema"], name=f"ema{ma}"))

fig.add_trace(
    go.Scatter(x=reference["time"], y=reference["vol"], name="annualized vol"), row=2, col=1,
)

df = cross[cross['simulation'] == display_forecast]
fig.add_trace(
    go.Scatter(x=df["time"], y=df["forecast"], name=f"forecast{display_forecast}"), row=3, col=1,
)
fig.add_trace(
    go.Scatter(x=df["time"], y=df["forecast_norm"], name=f"forecast_norm{display_forecast}"), row=3, col=1,
)

for ma in simulations:
    df = cross[cross['simulation'] == ma]
    fig.add_trace(go.Scatter(x=df["time"], y=df["cum_returns"], name=f"cum_returns{ma}"), row=4, col=1)

# for ma in simulations:
#     df = cache[ma]
#     fig.add_trace(go.Scatter(x=df["time"], y=df["cum_fees"], name=f"fees{ma}"), row=5, col=1)

fig.update_layout(xaxis_rangeslider_visible=False, height=4 * 400)
st.plotly_chart(fig, use_container_width=True)


