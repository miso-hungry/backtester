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


@st.cache_data(ttl=3600)
def get_market(base, start_time: datetime2, end_time: datetime2, api_len: str, api_key: str):
    ohlcv_url = "https://api.coinalyze.net/v1/ohlcv-history"
    if isinstance(base, str):
        base = [base]

    params = dict(
        api_key=api_key,
        symbols=",".join([f"{b}USDT_PERP.A" for b in base]),
        interval=api_len,
        to=str(int(end_time.timestamp())),
    )
    params["from"] = str(int(start_time.timestamp()))
    print(params)

    rsp = requests.get(ohlcv_url, params=params)

    rsp_json = rsp.json()
    df_data = []

    if not rsp_json or "message" in rsp_json:
        print(params)
        print(rsp)
        st.error(rsp_json)
        st.stop()

    for symbol_data in rsp_json:
        sym = symbol_data["symbol"]
        candles = symbol_data["history"]
        df = pd.DataFrame(candles)
        df["sym"] = sym
        df_data.append(df)

    df = pd.concat(df_data, ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True, unit="s")
    return df


all_bases = sorted(
    ['LOOM', 'DODO', 'ORBS', 'GALA', 'AGIX', 'ZRX', 'FIL', 'APT', 'APE', 'ARK', 'STORJ', 'TIA', 'FTM', 'STPT', 'SUSHI', 'WAXP', 'DOGE', 'XRP', 'NEO', 'KLAY', 'FRONT', 'IMX', 'LRC', 'SUI', 'C98', 'LINA', 'DOT', 'FLOKI', 'STMX', 'MINA', 'MOVR', 'LTC', 'UMA', 'MASK', 'STX', 'FTT', 'SSV', 'SAND', 'MATIC', 'ADA', 'UNI', 'GMX', 'OP', 'ETC', 'SATS', 'BTC', 'HIFI', 'BAND', 'AMB', 'AAVE', 'TRU', 'SOL', 'ORDI', 'BNB', 'BONK', 'WLD', 'RATS', 'IOST', 'LQTY', 'QTUM', 'ACE', 'GRT', 'COCOS', 'FLM', 'POWR', 'PERP', 'CFX', 'PEOPLE', 'UNFI', 'NMR', 'TRB', 'API3', 'BLZ', 'BNT', 'ENS', 'LINK', 'MANTA', 'LDO', 'ATOM', 'AR', 'VET', 'MANA', 'BCH', 'BAKE', 'WAVES', 'OCEAN', 'DUSK', 'ETH', 'CYBER', 'SXP', 'RUNE', 'CELR', 'BLUR', 'YGG', 'MAGIC', 'ICX', 'GMT', 'ARB', 'ID', 'INJ', 'AVAX', 'AXS', 'LUNA', 'NEAR', 'CHZ', 'RVN', 'ICP', 'SHIB', 'FET', 'SNX', 'BOND', 'CTSI', 'FLOW', 'ONT', 'LUNC', 'SEI', 'SKL', 'DYX', 'XLM', 'RNDR', 'STG', 'LPT', 'OGN', 'USTC', 'EOS', 'LEVER', 'MEME', 'YFI', 'CRV', 'COMP', 'GAS', 'PEPE', 'BIGT', 'XAI', 'BSV', 'JTO']
)
base = st.selectbox('Base Currency', options=all_bases, index=all_bases.index('SOL'))
params = dict(base=base)
precomputed = st.checkbox("Demo Dataset", value=True)
cols = st.columns(2)
dataset = cols[0].selectbox('Dataset', options=["2023-01-01_2023-12-31"], disabled=not precomputed)
api_key = cols[1].text_input('API Key', disabled=precomputed)

cols = st.columns(2)
now_time = datetime2.now(tz=time_zone)
start_time = now_time - datetime.timedelta(days=30 * 3)

start_date = cols[0].date_input(
    "Start Date", value=start_time.date(), disabled=precomputed
)
end_date = cols[1].date_input(
    "End Date", value=now_time.date(), disabled=precomputed
)

start_time = time_zone.localize(datetime2.combine(start_date, datetime.time()))
end_time = time_zone.localize(datetime2.combine(end_date, datetime.time()))

cols = st.columns(6)
capital = cols[0].number_input("Capital", min_value=0, step=50_000, value=100_000, disabled=precomputed)
vol_target = cols[1].number_input("Vol Target", min_value=0.0, step=0.1, value=0.45, disabled=precomputed)
simulations = (1, 2, 4, 8, 16, 32, 64)
display_forecast = cols[2].selectbox("Display Forecast", options=simulations, index=simulations.index(4))
forecast_cap = cols[3].number_input("Forecast Cap", min_value=0, value=2, disabled=precomputed)
fee_bps = cols[4].number_input("Fee (bps)", min_value=0.0, step=0.5, value=2.0, disabled=precomputed)
candle_len = cols[5].selectbox("Candle Length", ["1d", "1h"])

if candle_len == "1d":
    api_len = "daily"
    annualize_factor = math.sqrt(365)
else:
    api_len = "1hour"
    annualize_factor = math.sqrt(24 * 365)

with st.expander('Advanced'):
    debug = st.checkbox("Debug")
    if debug:
        precomputed_data_dir = f"/Users/xiaoyidou/backtester/demo_data"
    else:
        precomputed_data_dir = f"demo_data"

def calc_vol(df: pd.DataFrame):
    df["fast_vol"] = df["log_diffs"].ewm(span=32).std() * 16
    if df["t"].max() - df["t"].min() >= datetime.timedelta(days=32 * 3 * 2):
        df["slow_vol"] = df.rolling(32 * 3)["fast_vol"].mean()
    else:
        df["slow_vol"] = df["fast_vol"].mean()
    df["vol"] = 0.7 * df["fast_vol"] + 0.3 * df["slow_vol"]

def read_json(fpath: str):
    with open(fpath) as f:
        json_data = json.load(f)

    rollups = pd.DataFrame(json_data['rollups'])
    corrs = pd.DataFrame(json_data['corrs'])

    return rollups, corrs


if not precomputed:
    if not api_key:
        st.error(f'api key required')
        st.stop()

    df = get_market(base, start_time, end_time, api_len, api_key)
    df["log_prices"] = np.log(df["c"])
    df["log_diffs"] = df["log_prices"] - df["log_prices"].shift(1)

    if candle_len == "1h":
        vol_df = get_market(base, start_time, end_time, "daily")
        vol_df["log_prices"] = np.log(vol_df["c"])
        vol_df["log_diffs"] = vol_df["log_prices"] - vol_df["log_prices"].shift(1)
    else:
        vol_df = df

    calc_vol(vol_df)

    if candle_len == "1h":
        df = df.merge(vol_df[["sym", "t", "vol"]], how="left")
        df["vol"] = df["vol"].ffill()

    cache = {}
    reference = df.dropna()
    return_corrs = []
    rollups = []

    for ma in simulations:
        df = reference.copy()
        df["slow_ema"] = df["c"].ewm(span=ma * 4).mean()
        df["fast_ema"] = df["c"].ewm(span=ma).mean()

        # want avg absolute value of forecast to correspond with vol target
        df["forecast"] = (df["fast_ema"] - df["slow_ema"]) / (df["vol"] / 16 * df["c"])
        df["unsigned_forecast"] = df["forecast"].abs()
        df["forecast"] = df["forecast"] / df["unsigned_forecast"].mean()
        df["forecast"] = np.clip(df["forecast"], -forecast_cap, forecast_cap)
        df["target"] = df["forecast"] * capital * vol_target / (df["vol"])
        df["pnl"] = df["target"].shift(1) * df["log_diffs"]
        df["cum_pnl"] = df["pnl"].cumsum()
        df["cum_volume"] = (df["target"] - df["target"].shift(1)).abs().cumsum()
        df["cum_fees"] = df["cum_volume"] * fee_bps * 1e-4
        cache[ma] = df

        rollup = dict(
            simulation=ma,
            tot_return=df['return'].sum(),
            sharpe=df['return'].mean() / df['return'].std() * annualize_factor,
        )
        rollups.append(rollup)
        return_corrs.append(df['return'].reset_index(drop=True))

    rollups = pd.DataFrame(rollups)
    corrs = pd.concat(return_corrs, axis=1, keys=simulations)
    corrs = corrs.corr()
else:
    results_csv = f"{precomputed_data_dir}/{dataset}_{base}_{candle_len}_results.csv.gz"
    df = pd.read_csv(results_csv)

    cache = {}
    reference = df[df["simulation"] == display_forecast]
    for ma in simulations:
        cache[ma] = df[df["simulation"] == ma]

    rollups_json = f'{precomputed_data_dir}/{dataset}_{base}_{candle_len}_rollups.json'
    rollups, corrs = read_json(rollups_json)

fig = make_subplots(rows=5, cols=1, specs=[[{}], [{}], [{"secondary_y": True}], [{}], [{}]])

fig.add_trace(
    go.Candlestick(
        x=reference["t"],
        open=reference["o"],
        high=reference["h"],
        low=reference["l"],
        close=reference["c"],
        name="price",
    )
)

for ma in simulations:
    df = cache[ma]
    fig.add_trace(go.Scatter(x=df["t"], y=df["fast_ema"], name=f"ema{ma}"))

fig.add_trace(
    go.Scatter(x=reference["t"], y=reference["vol"], name="annualized vol"), row=2, col=1,
)

df = cache[display_forecast]
fig.add_trace(
    go.Scatter(x=df["t"], y=df["forecast"], name=f"forecast{display_forecast}"), row=3, col=1,
)
fig.add_trace(
    go.Scatter(x=df["t"], y=df["target"], name=f"target{display_forecast}"), row=3, col=1, secondary_y=True,
)

for ma in simulations:
    df = cache[ma]
    fig.add_trace(go.Scatter(x=df["t"], y=df["cum_pnl"], name=f"pnl{ma}"), row=4, col=1)

for ma in simulations:
    df = cache[ma]
    fig.add_trace(go.Scatter(x=df["t"], y=df["cum_fees"], name=f"fees{ma}"), row=5, col=1)

fig.update_layout(xaxis_rangeslider_visible=False, height=5 * 400)
st.plotly_chart(fig, use_container_width=True)


def plot_rollups(rollups: pd.DataFrame, corrs: pd.DataFrame):
    cols = st.columns(2)
    cols[0].subheader('Summary')
    cols[0].dataframe(rollups)
    cols[1].subheader('Return Correlations')
    cols[1].dataframe(corrs)

plot_rollups(rollups, corrs)


def read_precomputed(data_dir: str, dataset: str, candle_len: str):
    df_data = []
    for fpath in glob.glob(f"{data_dir}/{dataset}_*_{candle_len}_results.csv.gz"):
        df = pd.read_csv(fpath)
        df_data.append(df)

    df = pd.concat(df_data, ignore_index=True)
    df = df.sort_values(["simulation", "sym", "t"])
    return df


aggregate = st.checkbox("Aggregate", disabled=not precomputed)
if aggregate:
    st.divider()
    df = read_precomputed(precomputed_data_dir, dataset, candle_len)
    symbols = sorted(df["sym"].unique())
    default_exclude = [x for x in ["COCOSUSDT_PERP.A", "FTTUSDT_PERP.A"] if x in symbols]
    exclude = st.multiselect("Exclude Visual", options=symbols, default=default_exclude)
    if exclude:
        df = df[~df["sym"].isin(exclude)]
    ds = df
    st.plotly_chart(
        pltx.line(ds, x="t", y="cum_pnl", color="sym", facet_col="simulation", facet_col_wrap=4,),
        use_container_width=True,
    )

    aggr = df.groupby(["simulation", "t"]).agg(dict(cum_pnl="sum", cum_fees="sum")).reset_index()
    aggr["cum_pnl_net"] = aggr["cum_pnl"] - aggr["cum_fees"]
    st.plotly_chart(
        pltx.line(aggr, x="t", y=["cum_pnl", "cum_fees", "cum_pnl_net"], facet_col="simulation", facet_col_wrap=4),
        use_container_width=True,
    )

    rollups_json = f'{precomputed_data_dir}/{dataset}_{candle_len}_rollups.json'
    rollups, corrs = read_json(rollups_json)
    plot_rollups(rollups, corrs)

    df = df[df["simulation"] == display_forecast]
    df = df.groupby("sym").last().sort_values("cum_pnl").reset_index()
    st.plotly_chart(pltx.bar(df, x="sym", y="cum_pnl"), use_container_width=True)
