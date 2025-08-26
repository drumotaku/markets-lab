import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

# 画面設定
st.set_page_config(page_title="マーケッツラボ（無料版）", layout="wide")
st.title("📈 マーケッツラボ — 株式・暗号資産（完全無料）")
st.caption("Streamlit + yfinance + pandas + plotly（データ: Yahoo Finance）")

# --------- データ取得 ----------
@st.cache_data(ttl=600)
def load_yf(tickers: list[str], period: str = "1y", interval: str = "1d") -> dict[str, pd.DataFrame]:
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, period=period, interval=interval, auto_adjust=True, progress=False)
            if not df.empty:
                data[t] = df.dropna()
        except Exception:
            pass
    return data

def add_indicators(df: pd.DataFrame, sma_short: int = 50, sma_long: int = 200) -> pd.DataFrame:
    d = df.copy()
    price_col = "Close" if "Close" in d.columns else ("Adj Close" if "Adj Close" in d.columns else "close")
    d["短期SMA"] = d[price_col].rolling(sma_short).mean()
    d["長期SMA"] = d[price_col].rolling(sma_long).mean()
    ma = d[price_col].rolling(20).mean()
    std = d[price_col].rolling(20).std()
    d["BB上限"], d["BB中央値"], d["BB下限"] = ma + 2*std, ma, ma - 2*std
    diff = d[price_col].diff()
    up = diff.clip(lower=0).rolling(14).mean()
    dn = (-diff.clip(upper=0)).rolling(14).mean()
    rs = up / dn
    d["RSI"] = 100 - (100/(1+rs))
    d["price_col"] = price_col
    return d

def plot_price(df: pd.DataFrame, title: str):
    price_col = df.get("price_col", "Close")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[price_col], name="価格"))
    for col, label, dash in [("短期SMA","短期SMA",None), ("長期SMA","長期SMA",None),
                             ("BB上限","BB上限","dot"), ("BB中央値","BB中央値","dot"), ("BB下限","BB下限","dot")]:
        if col in df:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=label,
                                     line=dict(dash=dash) if dash else None))
    fig.update_layout(title=title, legend=dict(orientation="h"), margin=dict(l=0,r=0,t=40,b=0))
    return fig

def backtest_sma(df: pd.DataFrame, s: int = 50, l: int = 200, fee: float = 0.001):
    d = add_indicators(df, s, l).dropna().copy()
    price_col = d.get("price_col","Close")
    d["シグナル"] = (d["短期SMA"] > d["長期SMA"]).astype(int)
    d["ポジ変"] = d["シグナル"].diff().fillna(0)
    ret = d[price_col].pct_change().fillna(0)
    cost = abs(d["ポジ変"]) * fee
    strat = d["シグナル"]*ret - cost
    d["資産曲線"] = (1+strat).cumprod()
    final = d["資産曲線"].iloc[-1]
    dd = d["資産曲線"]/d["資産曲線"].cummax() - 1
    stats = {
        "最終損益(%)": round((final-1)*100, 2),
        "最大ドローダウン(%)": round(-dd.min()*100, 2),
        "取引回数": int(abs(d["ポジ変"]).sum())
    }
    return d, stats

def corr_matrix(price_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cols = []
    for t, df in price_dict.items():
        price_col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else "close")
        cols.append(df[price_col].rename(t))
    if not cols:
        return pd.DataFrame()
    prices = pd.concat(cols, axis=1).dropna()
    rets = prices.pct_change().dropna()
    return rets.corr()

# --------- サイドバー ----------
st.sidebar.header("設定")
mode = st.sidebar.radio("データ種別", ["株式", "暗号資産"])
period = st.sidebar.selectbox("期間", ["1mo","3mo","6mo","1y","2y","5y","ytd","max"], index=3)
interval = st.sidebar.selectbox("足", ["1d","1h","30m","15m","5m"], index=0)

default_stock = "AAPL"
default_crypto = "BTC-USD"
ticker = st.sidebar.text_input("ティッカー（例: AAPL / BTC-USD）",
                               value=(default_crypto if mode=="暗号資産" else default_stock)).strip()

sma_s = st.sidebar.number_input("短期SMA日数", min_value=2, max_value=200, value=50, step=1)
sma_l = st.sidebar.number_input("長期SMA日数", min_value=5, max_value=400, value=200, step=5)
fee = st.sidebar.number_input("片道手数料（比率）", min_value=0.0, max_value=0.01, value=0.001, step=0.0005, format="%0.4f")

st.sidebar.markdown("---")
multi_input = st.sidebar.text_area("複数ティッカー（相関用）", value="AAPL,MSFT,GOOGL")

# --------- タブ ----------
tab1, tab2, tab3 = st.tabs(["📊 チャート", "🧪 バックテスト", "🔗 相関分析"])

with tab1:
    st.subheader("価格チャート + 指標")
    if ticker:
        dct = load_yf([ticker], period=period, interval=interval)
        if dct.get(ticker) is not None:
            df = add_indicators(dct[ticker], sma_s, sma_l)
            st.plotly_chart(plot_price(df, f"{ticker} — {period}/{interval}"), use_container_width=True)
            rsi_fig = go.Figure(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
            rsi_fig.add_hline(y=70, line_dash="dot"); rsi_fig.add_hline(y=30, line_dash="dot")
            rsi_fig.update_layout(margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(rsi_fig, use_container_width=True)
        else:
            st.warning("データ取得に失敗しました。ティッカーを確認してください。")

with tab2:
    st.subheader("SMAクロス簡易バックテスト")
    if ticker:
        dct = load_yf([ticker], period=period, interval=interval)
        if dct.get(ticker) is not None and not dct[ticker].empty:
            hist, stats = backtest_sma(dct[ticker], s=int(sma_s), l=int(sma_l), fee=float(fee))
            colA, colB = st.columns([2,1])
            with colA:
                eq_fig = go.Figure(go.Scatter(x=hist.index, y=hist["資産曲線"], name="資産曲線"))
                eq_fig.update_layout(margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(eq_fig, use_container_width=True)
            with colB:
                st.write("成績")
                st.table(pd.DataFrame(stats, index=[0]).T.rename(columns={0:"値"}))
            st.caption("注: 税金・スリッページ等は未考慮。教育・研究目的の簡易検証です。")
        else:
            st.warning("データ取得に失敗しました。")

with tab3:
    st.subheader("相関分析（複数資産の日次リターン）")
    tickers = [t.strip() for t in multi_input.split(",") if t.strip()]
    if tickers:
        dct = load_yf(tickers, period=period, interval=interval)
        if dct:
            corr = corr_matrix(dct)
            if not corr.empty:
                st.dataframe(corr.style.format("{:.2f}"))
                heat = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorbar=dict(title="相関")))
                heat.update_layout(margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(heat, use_container_width=True)
            else:
                st.warning("相関を計算できませんでした。")
        else:
            st.warning("いずれのティッカーも取得できませんでした。")

st.markdown("---")
st.markdown("完全無料: Streamlit / yfinance / pandas / numpy / plotly（非商用・学術目的推奨）")
