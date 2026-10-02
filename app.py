import io
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Nifty500 Short Scanner", layout="wide")
URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
FALLBACK = ("RELIANCE,TCS,HDFCBANK,INFY,ICICIBANK,SBIN,ITC,LT,AXISBANK,KOTAKBANK,TATAMOTORS,"
            "TATASTEEL,ADANIENT,BAJFINANCE,MARUTI,SUNPHARMA,WIPRO,HCLTECH,ONGC,NTPC,"
            "COALINDIA,JSWSTEEL,HINDALCO,VEDL,ZOMATO,DLF,IDEA,YESBANK,PNB,BANKBARODA").split(",")


@st.cache_data(ttl=86400)
def universe():
    try:
        r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text))[["Symbol", "Industry"]].dropna(subset=["Symbol"])
    except Exception:
        return pd.DataFrame({"Symbol": FALLBACK, "Industry": "Unknown"})


@st.cache_data(ttl=86400)
def fo_universe():
    try:
        r = requests.get("https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip() for c in df.columns]
        return set(df["SYMBOL"].astype(str).str.strip())
    except Exception:
        return set()


@st.cache_data(ttl=300, show_spinner=False)
def nifty(period, interval):
    try:
        return yf.download("^NSEI", period=period, interval=interval, progress=False,
                           auto_adjust=False)["Close"].squeeze().dropna()
    except Exception:
        return None


def secret(k):
    try:
        return st.secrets[k]
    except Exception:
        return ""


def tg_send(token, chat, text):
    try:
        return requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                             json={"chat_id": chat, "text": text}, timeout=10).ok
    except Exception:
        return False


@st.cache_data(ttl=300, show_spinner=False)
def fetch(syms, period, interval):
    out = {}
    for i in range(0, len(syms), 100):
        chunk = [s + ".NS" for s in syms[i:i + 100]]
        try:
            raw = yf.download(chunk, period=period, interval=interval, group_by="ticker",
                              progress=False, threads=True, auto_adjust=False)
        except Exception:
            continue
        for t in chunk:
            try:
                x = raw[t].dropna()
                if len(x) > 60:
                    out[t[:-3]] = x
            except Exception:
                pass
    return out


def day_chg(d):
    """Aaj ka % change vs pichhle session ka close."""
    dates = pd.Series(d.index.date, index=d.index)
    prev = d.Close[dates < dates.iloc[-1]]
    return d.Close.iloc[-1] / prev.iloc[-1] - 1 if len(prev) else np.nan


def ind(d):
    c, h, l, v = d.Close, d.High, d.Low, d.Volume
    d["e20"] = c.ewm(span=20, adjust=False).mean()
    d["e50"] = c.ewm(span=50, adjust=False).mean()
    dl = c.diff()
    up = dl.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-dl.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    m = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["mh"] = m - m.ewm(span=9, adjust=False).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    u, w = h.diff(), -l.diff()
    pdm = pd.Series(np.where((u > w) & (u > 0), u, 0.0), index=d.index)
    mdm = pd.Series(np.where((w > u) & (w > 0), w, 0.0), index=d.index)
    d["pdi"] = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean() / d.atr
    d["mdi"] = 100 * mdm.ewm(alpha=1 / 14, adjust=False).mean() / d.atr
    dx = 100 * (d.pdi - d.mdi).abs() / (d.pdi + d.mdi)
    d["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    d["vr"] = v / v.rolling(20).mean()
    d["low20"] = l.rolling(20).min().shift(1)
    d["sc"] = ((c < d.e20) * 15 + (d.e20 < d.e50) * 15 + ((d.rsi < 50) & (d.rsi > 30)) * 15
               + (d.mh < 0) * 10 + (d.mh < d.mh.shift()) * 5
               + ((d.mdi > d.pdi) & (d.adx > 20)) * 15 + (c < d.low20) * 15 + (d.vr > 1.5) * 10)
    return d


def backtest(d, thr, k, rr, hold, cost):
    d = ind(d.copy())
    sig = ((d.sc >= thr) & (d.sc.shift() < thr)).values
    o, h, l, c, a = (d[x].values for x in ["Open", "High", "Low", "Close", "atr"])
    n, i, rows = len(d), 60, []
    while i < n - 1:
        if sig[i] and a[i] > 0:
            e, r = o[i + 1], k * a[i]
            sl, tp, end = e + r, e - rr * r, min(i + 1 + hold, n - 1)
            for j in range(i + 1, end + 1):
                if h[j] >= sl:
                    ex, x = sl, j
                    break
                if l[j] <= tp:
                    ex, x = tp, j
                    break
            else:
                ex, x = c[end], end
            rows.append((d.index[i + 1], e, ex, (e - ex) / e - 2 * cost / 100))
            i = x
        i += 1
    return pd.DataFrame(rows, columns=["Date", "Entry", "Exit", "Ret"])


st.title("📉 Nifty 500 Short Scanner")
tab1, tab2 = st.tabs(["🔍 Live Scanner", "🧪 Backtest"])
udf = universe()
syms_all = udf["Symbol"].tolist()
sec_map = dict(zip(udf["Symbol"], udf["Industry"]))

with tab1:
    c1, c2 = st.columns(2)
    tf = c1.selectbox("Timeframe", ["Daily", "1 Hour", "15 Min"])
    thr = c2.slider("Min short score", 40, 100, 70, 5)
    c3, c4 = st.columns(2)
    cap = c3.number_input("Capital ₹", 10000, 10000000, 200000, 10000)
    rk = c4.number_input("Risk per trade %", 0.1, 5.0, 1.0, 0.1)
    c5, c6 = st.columns(2)
    minp = c5.number_input("Min price ₹", 0, 5000, 50)
    atrk = c6.number_input("Stop = ATR ×", 0.5, 4.0, 1.5, 0.1)
    n = st.slider("Kitne stocks scan karne hain", 20, len(syms_all), min(len(syms_all), 500), 10)
    p, iv = {"Daily": ("1y", "1d"), "1 Hour": ("3mo", "1h"), "15 Min": ("1mo", "15m")}[tf]
    fo_only = st.checkbox("Sirf F&O stocks (overnight short possible)", value=True)
    rs_only = st.checkbox("Sirf Nifty se weak stocks (RS < 0, last 20 candles)", value=True)
    weak_n = st.slider("Kitne weakest sectors highlight karne hain", 1, 10, 4)
    sec_only = st.checkbox("Sirf weak sectors ke stocks dikhao", value=False)
    tg = st.checkbox("Telegram alert bhejo")
    tok = chat = ""
    if tg:
        tok = st.text_input("Bot token", secret("TG_TOKEN"), type="password")
        chat = st.text_input("Chat ID", secret("TG_CHAT"))
    if st.button("🚀 Scan / Refresh", type="primary", use_container_width=True):
        fetch.clear()
        fo = fo_universe()
        if fo_only and not fo:
            st.warning("F&O list load nahi hui, filter skip kiya.")
        pool = [s for s in syms_all if s in fo] if (fo_only and fo) else syms_all
        with st.spinner("Live data aa raha hai..."):
            data = fetch(tuple(pool[:n]), p, iv)
            nf = nifty(p, iv)
        try:
            nr = nf.iloc[-1] / nf.iloc[-21] - 1
        except Exception:
            nr = None
        rows, sec = [], []
        for s, d in data.items():
            d = ind(d.copy())
            x = d.iloc[-1]
            sector = sec_map.get(s, "Unknown")
            sec.append((sector, day_chg(d)))
            rs = (x.Close / d.Close.iloc[-21] - 1 - nr) * 100 if nr is not None else np.nan
            if x.sc >= thr and x.Close >= minp and not (rs_only and rs >= 0):
                risk = atrk * x.atr
                qty = int((cap * rk / 100) / risk) if risk > 0 else 0
                rows.append({"Stock": s, "Sector": sector, "LTP": round(x.Close, 2), "Score": int(x.sc),
                             "RS%": round(rs, 2), "RSI": round(x.rsi, 1), "ADX": round(x.adx, 1),
                             "VolX": round(x.vr, 2), "SL": round(x.Close + risk, 2),
                             "T1 (1:2)": round(x.Close - 2 * risk, 2),
                             "T2 (1:3)": round(x.Close - 3 * risk, 2), "Qty": qty})
        sd = pd.DataFrame(sec, columns=["Sector", "chg"]).dropna()
        sg = sd.groupby("Sector").agg(Stocks=("chg", "size"), AvgChg=("chg", lambda v: v.mean() * 100),
                                      Red=("chg", lambda v: (v < 0).mean() * 100))
        sg = sg[sg.Stocks >= 3]
        sg["r"] = sg.AvgChg.rank() + sg.Red.rank(ascending=False)
        sg = sg.sort_values("r")
        weak = [i for i in sg.index[:weak_n] if sg.AvgChg[i] < 0]
        st.subheader("🏭 Sector Weakness (aaj)")
        st.dataframe(sg.drop(columns="r").round(2).rename(columns={"AvgChg": "Avg chg %", "Red": "Red stocks %"}),
                     use_container_width=True)
        if weak:
            st.info("🔻 Selling sabse zyada: " + ", ".join(
                f"**{w}** ({sg.AvgChg[w]:.2f}%, {sg.Red[w]:.0f}% stocks red)" for w in weak)
                + "\n\nIn sectors ke weak stocks me short setups ko priority do.")
        else:
            st.info("Koi sector clearly weak nahi hai abhi, short selection me extra savdhani rakho.")
        st.success(f"{len(data)} stocks scan hue, {len(rows)} short setups mile")
        if rows:
            res = pd.DataFrame(rows)
            res["Sec%"] = res.Sector.map(sg.AvgChg).round(2)
            res["Weak Sec"] = np.where(res.Sector.isin(weak), "🔻", "")
            res["Final"] = res.Score + 10 * res.Sector.isin(weak)
            if sec_only:
                res = res[res.Sector.isin(weak)]
            res = res.sort_values(["Final", "Score"], ascending=False)
            st.dataframe(res, hide_index=True, use_container_width=True)
            if tg and len(res):
                if tok and chat:
                    lines = [f"{r['Stock']} ({r['Sector']}) | LTP {r['LTP']} | Final {r['Final']} | SL {r['SL']} | T1 {r['T1 (1:2)']} | Qty {r['Qty']}"
                             for _, r in res.head(10).iterrows()]
                    head = f"📉 Short setups ({tf})\nWeak sectors: {', '.join(weak) or 'None'}\n"
                    ok = tg_send(tok, chat, head + "\n".join(lines))
                    st.success("Telegram alert bhej diya ✅") if ok else st.error("Telegram alert fail hua, token/chat ID check karo")
                else:
                    st.warning("Telegram token aur chat ID daalo.")
    st.caption("Last candle live hoti hai (yfinance ~15 min delayed ho sakta hai). "
               "Overnight short sirf F&O stocks me; cash me sirf intraday (MIS). Educational use only.")

with tab2:
    c1, c2 = st.columns(2)
    bn = c1.slider("Stocks", 10, len(syms_all), 50, 10)
    yrs = c2.selectbox("History", ["2y", "5y", "10y"], index=1)
    c3, c4 = st.columns(2)
    bthr = c3.slider("Entry score >=", 40, 100, 70, 5, key="b1")
    bk = c4.number_input("SL ATR ×", 0.5, 4.0, 1.5, 0.1, key="b2")
    c5, c6 = st.columns(2)
    brr = c5.number_input("Reward:Risk", 1.0, 5.0, 2.0, 0.5)
    bh = c6.number_input("Max hold (candles)", 1, 30, 7)
    c7, c8 = st.columns(2)
    cost = c7.number_input("Cost per side %", 0.0, 1.0, 0.05, 0.01)
    alloc = c8.number_input("Capital per trade %", 1, 100, 10)
    if st.button("▶️ Run Backtest", type="primary", use_container_width=True):
        with st.spinner("Backtest chal raha hai..."):
            data = fetch(tuple(syms_all[:bn]), yrs, "1d")
            frames = []
            for s, d in data.items():
                t = backtest(d, bthr, bk, brr, bh, cost)
                if len(t):
                    t.insert(0, "Stock", s)
                    frames.append(t)
        if not frames:
            st.warning("Koi trade nahi bana. Score kam karke try karo.")
        else:
            T = pd.concat(frames).sort_values("Date").reset_index(drop=True)
            eq = (1 + T.Ret * alloc / 100).cumprod()
            dd = (eq / eq.cummax() - 1).min()
            gp, gl = T.Ret[T.Ret > 0].sum(), -T.Ret[T.Ret < 0].sum()
            m = st.columns(2)
            m[0].metric("Trades", len(T))
            m[1].metric("Win rate", f"{(T.Ret > 0).mean() * 100:.1f}%")
            m = st.columns(2)
            m[0].metric("Profit factor", f"{gp / gl:.2f}" if gl else "∞")
            m[1].metric("Avg trade", f"{T.Ret.mean() * 100:.2f}%")
            m = st.columns(2)
            m[0].metric("Total return", f"{(eq.iloc[-1] - 1) * 100:.1f}%")
            m[1].metric("Max drawdown", f"{dd * 100:.1f}%")
            st.line_chart(pd.Series(eq.values, index=T.Date, name="Equity"))
            st.dataframe(T, hide_index=True, use_container_width=True)
            st.download_button("⬇️ Trades CSV", T.to_csv(index=False), "trades.csv")
    st.caption("Entry: signal candle ke next open par short. Stop/target ATR based, stop pehle check hota hai (conservative).")
            
