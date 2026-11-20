from datetime import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="股票基本面、布林通道與當日波動分析",
    layout="wide",
)
st.title("📈 股票基本面 × 布林通道預估 × 當日波動圖")

# --- 側邊欄參數設定 ---
with st.sidebar:
  st.header("⚙️ 參數設定")
  tickers_input = st.text_input(
      "股票代號（以逗號分隔）",
      value="2330.TW, SPY, NVDA",
  )
  start_date = st.date_input("歷史區間開始日期", value=datetime(2024, 1, 1))
  use_today = st.checkbox("歷史區間結束日期使用今天", value=True)
  custom_end_date = st.date_input(
      "自訂歷史區間結束日期",
      value=datetime.today(),
      disabled=use_today,
  )
  window = st.number_input(
      "移動平均天數 (Window)",
      min_value=5,
      max_value=500,
      value=252,
  )
  intraday_interval = st.selectbox(
      "當日波動圖時間顆粒",
      options=["5m", "1m", "15m", "30m"],
      index=0,
      help="5m 適合觀察全日波動節奏；1m 細節最細但資料點較多。",
  )
  amplitude_threshold = st.number_input(
      "當日振幅提醒門檻 (%)",
      min_value=0.5,
      max_value=30.0,
      value=3.0,
      step=0.5,
      help="當（當日最高價 - 當日最低價）/ 當日最低價 達此百分比時，顯示盤中波動提醒。",
  )
  run_btn = st.button(
      "🚀 開始分析",
      type="primary",
      use_container_width=True,
  )

  st.divider()
  st.header("⏱️ 盤中自動刷新")
  auto_refresh = st.checkbox(
      "啟用自動刷新（定期更新當日波動圖）",
      value=False,
      help="開啟後會依設定間隔自動重新抓取各檔股票當日分時資料並更新畫面。",
  )
  refresh_interval = st.number_input(
      "刷新間隔（秒）",
      min_value=15,
      max_value=600,
      value=60,
      step=15,
      disabled=not auto_refresh,
  )


@st.cache_data(ttl=3600)
def download_history_with_retry(
    symbol: str, start: str, end: str, retries: int = 3
) -> pd.DataFrame:
  """安全下載歷史股價，含重試與 MultiIndex 欄位處理"""
  for attempt in range(1, retries + 1):
    try:
      data = yf.download(
          symbol,
          start=start,
          end=end,
          progress=False,
          auto_adjust=False,
          threads=False,
      )
      if data is not None and not data.empty:
        if isinstance(data.columns, pd.MultiIndex):
          data.columns = data.columns.get_level_values(0)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
          if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.dropna(subset=["High", "Low", "Close"])
        if not data.empty:
          return data
    except Exception:
      pass
    if attempt < retries:
      time.sleep(2)
  return pd.DataFrame()


@st.cache_data(ttl=60)
def download_intraday_data(
    symbol: str, interval: str = "5m"
) -> tuple[pd.DataFrame, str]:
  """下載當日分時波動資料；若無當日資料則自動改抓近 5 日分時作備援"""
  attempts = [
      ("1d", interval, "當日盤中/最近交易日分時"),
      ("5d", "15m" if interval in ("1m", "5m") else interval, "近 5 日分時（備援）"),
  ]
  for period, iv, label in attempts:
    try:
      df = yf.download(
          symbol,
          period=period,
          interval=iv,
          progress=False,
          auto_adjust=False,
          threads=False,
      )
      if df is not None and not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
          df.columns = df.columns.get_level_values(0)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
          if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["High", "Low", "Close"])
        if not df.empty:
          if period == "1d":
            last_day = df.index[-1].date()
            day_df = df[df.index.date == last_day]
            if not day_df.empty:
              return day_df, f"{label}（{last_day}，間隔 {iv}）"
          return df, f"{label}（間隔 {iv}）"
    except Exception:
      continue
  return pd.DataFrame(), "無分時資料"


@st.cache_data(ttl=3600)
def get_fundamental_info(ticker_symbol: str) -> dict | None:
  """抓取單一股票基本面指標"""
  try:
    stock = yf.Ticker(ticker_symbol)
    info = stock.info or {}
    bs = stock.balance_sheet

    if not info and (bs is None or bs.empty):
      return None

    currency = info.get("currency", "")
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")
    company_nature = (
        f"{sector} - {industry}"
        if (sector and sector != "N/A") or (industry and industry != "N/A")
        else "無資料"
    )

    eps = info.get("trailingEps", None)
    eps_str = f"{eps:.2f} ({currency})" if isinstance(eps, (int, float)) else "N/A"

    pe_ratio = info.get("trailingPE", None)
    pe_str = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "N/A"

    gross_margin = info.get("grossMargins", None)
    gm_str = f"{gross_margin * 100:.2f}%" if isinstance(gross_margin, (int, float)) else "N/A"

    profit_margin = info.get("profitMargins", None)
    pm_str = f"{profit_margin * 100:.2f}%" if isinstance(profit_margin, (int, float)) else "N/A"

    ca_str = "N/A"
    debt_ratio_str = "N/A"
    if bs is not None and not bs.empty:
      latest_bs = bs.iloc[:, 0]
      if "Current Assets" in latest_bs and pd.notna(latest_bs["Current Assets"]):
        ca_val = float(latest_bs["Current Assets"])
        ca_str = f"{ca_val / 1e8:,.2f} 億 ({currency})"
      if (
          "Total Liabilities Net Minority Interest" in latest_bs
          and "Total Assets" in latest_bs
      ):
        total_liab = latest_bs["Total Liabilities Net Minority Interest"]
        total_assets = latest_bs["Total Assets"]
        if (
            pd.notna(total_liab)
            and pd.notna(total_assets)
            and float(total_assets) != 0
        ):
          debt_ratio_str = f"{(float(total_liab) / float(total_assets)) * 100:.2f}%"

    short_name = info.get("shortName", "")
    col_title = f"{ticker_symbol} ({short_name})" if short_name else ticker_symbol

    return {
        "col_title": col_title,
        "metrics": [
            company_nature,
            eps_str,
            pe_str,
            gm_str,
            pm_str,
            ca_str,
            debt_ratio_str,
        ],
    }
  except Exception:
    return None


def compute_analysis(df: pd.DataFrame, window: int) -> tuple[dict, pd.DataFrame]:
  """計算歷史極值、布林通道指標與單檔分析摘要"""
  work_df = df.copy()
  absolute_high = float(work_df["High"].max())
  absolute_low = float(work_df["Low"].min())
  high_date = work_df["High"].idxmax().strftime("%Y-%m-%d")
  low_date = work_df["Low"].idxmin().strftime("%Y-%m-%d")
  current_price = float(work_df["Close"].iloc[-1])

  has_enough_window = len(work_df) >= window
  if has_enough_window:
    work_df["SMA"] = work_df["Close"].rolling(window=window).mean()
    work_df["STD"] = work_df["Close"].rolling(window=window).std()
    work_df["Estimated_High"] = work_df["SMA"] + (2 * work_df["STD"])
    work_df["Estimated_Low"] = work_df["SMA"] - (2 * work_df["STD"])

    latest_sma_val = float(work_df["SMA"].iloc[-1])
    latest_est_high_val = float(work_df["Estimated_High"].iloc[-1])
    latest_est_low_val = float(work_df["Estimated_Low"].iloc[-1])

    latest_sma = f"{latest_sma_val:.2f}"
    latest_est_high = f"{latest_est_high_val:.2f}"
    latest_est_low = f"{latest_est_low_val:.2f}"
  else:
    latest_sma_val = None
    latest_est_high_val = None
    latest_est_low_val = None
    latest_sma = "資料不足"
    latest_est_high = "資料不足"
    latest_est_low = "資料不足"

  summary = {
      "最新收盤價": f"{current_price:.2f}",
      "區間最高價": f"{absolute_high:.2f}",
      "區間最高價日期": high_date,
      "區間最低價": f"{absolute_low:.2f}",
      "區間最低價日期": low_date,
      f"基準均價 (SMA {window}日)": latest_sma,
      "預估壓力高點 (+2SD)": latest_est_high,
      "預估支撐低點 (-2SD)": latest_est_low,
      "has_enough_window": has_enough_window,
      "latest_est_high_val": latest_est_high_val,
      "latest_est_low_val": latest_est_low_val,
  }
  return summary, work_df


def evaluate_intraday_alert(
    ticker_symbol: str,
    intra_df: pd.DataFrame,
    threshold_pct: float,
    est_high_2sd: float | None,
    est_low_2sd: float | None,
    has_bollinger: bool,
) -> dict:
  """彙整當日振幅與最新分時價對歷史布林通道（+2SD / -2SD）的突破/跌破狀態"""
  if intra_df is None or intra_df.empty:
    return {
        "has_data": False,
        "has_alert": False,
        "amp_exceeded": False,
        "broke_upper": False,
        "broke_lower": False,
        "amplitude_pct": 0.0,
        "headline": f"目前無法取得 {ticker_symbol} 分時資料。",
        "details": [],
    }

  day_high = float(intra_df["High"].max())
  day_low = float(intra_df["Low"].min())
  high_time = intra_df["High"].idxmax().strftime("%H:%M")
  low_time = intra_df["Low"].idxmin().strftime("%H:%M")
  latest_price = float(intra_df["Close"].iloc[-1])

  amplitude_pts = day_high - day_low
  amplitude_pct = (amplitude_pts / day_low * 100) if day_low != 0 else 0.0
  amp_exceeded = amplitude_pct >= threshold_pct

  dist_to_high_pct = ((day_high - latest_price) / day_high * 100) if day_high != 0 else 0.0
  dist_from_low_pct = ((latest_price - day_low) / day_low * 100) if day_low != 0 else 0.0
  range_pos_pct = (
      ((latest_price - day_low) / amplitude_pts * 100)
      if amplitude_pts > 0
      else 50.0
  )

  pos_notes = []
  if dist_to_high_pct <= 0.25:
    pos_notes.append(f"接近／觸及當日高點 {day_high:.2f}（{high_time}）")
  if dist_from_low_pct <= 0.25:
    pos_notes.append(f"接近／觸及當日低點 {day_low:.2f}（{low_time}）")
  pos_desc = "、".join(pos_notes) if pos_notes else f"位於當日高低區間約 {range_pos_pct:.0f}% 位置"

  broke_upper = False
  broke_lower = False
  details = []

  if amp_exceeded:
    details.append(
        f"當日振幅 **{amplitude_pct:.2f}%**（高低點 **{day_low:.2f} ~ {day_high:.2f}**，價差 **{amplitude_pts:.2f}**）已達／超過設定門檻 **{threshold_pct:.2f}%**。"
    )
  else:
    details.append(
        f"當日振幅 **{amplitude_pct:.2f}%**（高低點 **{day_low:.2f} ~ {day_high:.2f}**，價差 **{amplitude_pts:.2f}**），低於門檻 **{threshold_pct:.2f}%**（尚差 {threshold_pct - amplitude_pct:.2f}%）。"
    )

  if has_bollinger and est_high_2sd is not None and est_low_2sd is not None:
    if latest_price > est_high_2sd:
      broke_upper = True
      diff_pts = latest_price - est_high_2sd
      diff_pct = (diff_pts / est_high_2sd * 100) if est_high_2sd != 0 else 0.0
      details.append(
          f"📈 **突破歷史布林通道預估高點 (+2SD)**：最新分時價 **{latest_price:.2f}** 高於預估壓力高點 **{est_high_2sd:.2f}**（超出 **+{diff_pts:.2f}** / **+{diff_pct:.2f}%**）。"
      )
    elif latest_price < est_low_2sd:
      broke_lower = True
      diff_pts = est_low_2sd - latest_price
      diff_pct = (diff_pts / est_low_2sd * 100) if est_low_2sd != 0 else 0.0
      details.append(
          f"📉 **跌破歷史布林通道預估低點 (-2SD)**：最新分時價 **{latest_price:.2f}** 低於預估支撐低點 **{est_low_2sd:.2f}**（跌破 **-{diff_pts:.2f}** / **-{diff_pct:.2f}%**）。"
      )
    else:
      up_gap_pct = ((est_high_2sd - latest_price) / latest_price * 100) if latest_price != 0 else 0.0
      dn_gap_pct = ((latest_price - est_low_2sd) / latest_price * 100) if latest_price != 0 else 0.0
      details.append(
          f"在歷史布林通道區間內（支撐 **-2SD {est_low_2sd:.2f}** ~ 壓力 **+2SD {est_high_2sd:.2f}**）：距上界尚有 **{up_gap_pct:.2f}%**，距下界尚有 **{dn_gap_pct:.2f}%**。"
      )
  else:
    details.append("歷史交易日數不足設定的 Window，暫無完整布林通道（+2SD / -2SD）可供比對。")

  details.append(f"最新分時價 **{latest_price:.2f}**（{pos_desc}）。")

  has_alert = amp_exceeded or broke_upper or broke_lower
  alert_tags = []
  if amp_exceeded:
    alert_tags.append("振幅超標")
  if broke_upper:
    alert_tags.append("突破 +2SD")
  if broke_lower:
    alert_tags.append("跌破 -2SD")

  headline = (
      f"🚨 **{ticker_symbol} 盤中提醒（{'／'.join(alert_tags)}）**"
      if has_alert
      else f"🔔 **{ticker_symbol} 盤中波動狀態正常**"
  )

  return {
      "has_data": True,
      "has_alert": has_alert,
      "amp_exceeded": amp_exceeded,
      "broke_upper": broke_upper,
      "broke_lower": broke_lower,
      "amplitude_pct": amplitude_pct,
      "headline": headline,
      "details": details,
  }


def render_bollinger_chart(
    ticker_symbol: str, calc_df: pd.DataFrame, window: int, end_label: str
):
  """繪製歷史股價與布林通道高低點預估圖"""
  fig, ax = plt.subplots(figsize=(14, 4.8))
  ax.plot(calc_df.index, calc_df["Close"], label="Close Price", color="black", alpha=0.75)

  if {"SMA", "Estimated_High", "Estimated_Low"}.issubset(calc_df.columns):
    ax.plot(calc_df.index, calc_df["SMA"], label=f"{window}-Day SMA", color="blue", linestyle="--")
    ax.plot(calc_df.index, calc_df["Estimated_High"], label="Estimated High (+2SD)", color="red", alpha=0.65)
    ax.plot(calc_df.index, calc_df["Estimated_Low"], label="Estimated Low (-2SD)", color="green", alpha=0.65)
    ax.fill_between(
        calc_df.index,
        calc_df["Estimated_High"],
        calc_df["Estimated_Low"],
        color="gray",
        alpha=0.15,
        label="95% Volatility Range",
    )

  ax.set_title(f"{ticker_symbol} High & Low Estimation Model ({end_label})", fontsize=13)
  ax.set_xlabel("Date")
  ax.set_ylabel("Price")
  ax.legend(loc="upper left")
  ax.grid(True, alpha=0.3)
  plt.tight_layout()
  st.pyplot(fig)
  plt.close(fig)


def render_intraday_volatility(
    ticker_symbol: str,
    intra_df: pd.DataFrame,
    source_desc: str,
    threshold_pct: float,
    est_high_2sd: float | None,
    est_low_2sd: float | None,
    has_bollinger: bool,
):
  """繪製當日波動圖並顯示盤中波動提醒（含 +2SD / -2SD 突破與跌破提示）"""
  st.markdown(f"#### ⚡ {ticker_symbol} 當日波動圖與盤中提醒（{source_desc}）")

  if intra_df is None or intra_df.empty:
    st.info("目前無法取得該股票的分時波動資料（可能處於休市或資料源暫時無回應）。")
    return

  alert_info = evaluate_intraday_alert(
      ticker_symbol=ticker_symbol,
      intra_df=intra_df,
      threshold_pct=threshold_pct,
      est_high_2sd=est_high_2sd,
      est_low_2sd=est_low_2sd,
      has_bollinger=has_bollinger,
  )

  banner_body = f"{alert_info['headline']}\n\n" + "\n".join(
      f"- {line}" for line in alert_info["details"]
  )
  if alert_info["has_alert"]:
    st.warning(banner_body)
  else:
    st.info(banner_body)

  work = intra_df.copy()
  day_high = float(work["High"].max())
  day_low = float(work["Low"].min())
  high_time = work["High"].idxmax().strftime("%H:%M")
  low_time = work["Low"].idxmin().strftime("%H:%M")
  latest_price = float(work["Close"].iloc[-1])
  open_price = (
      float(work["Open"].iloc[0])
      if "Open" in work.columns and pd.notna(work["Open"].iloc[0])
      else float(work["Close"].iloc[0])
  )

  amplitude_pts = day_high - day_low
  amplitude_pct = alert_info["amplitude_pct"]
  change_pct = ((latest_price - open_price) / open_price * 100) if open_price != 0 else 0.0

  if has_bollinger and est_high_2sd is not None and est_low_2sd is not None:
    if alert_info["broke_upper"]:
      bb_status_label = "突破 +2SD"
      bb_delta = f"最新 {latest_price:.2f} > 高界 {est_high_2sd:.2f}"
    elif alert_info["broke_lower"]:
      bb_status_label = "跌破 -2SD"
      bb_delta = f"最新 {latest_price:.2f} < 低界 {est_low_2sd:.2f}"
    else:
      bb_status_label = "通道區間內"
      bb_delta = f"-2SD {est_low_2sd:.2f} ~ +2SD {est_high_2sd:.2f}"
  else:
    bb_status_label = "資料不足"
    bb_delta = "需達 Window 天數"

  m1, m2, m3, m4, m5 = st.columns(5)
  m1.metric("最新分時價", f"{latest_price:.2f}", f"相對首筆 {change_pct:+.2f}%")
  m2.metric("當日最高價", f"{day_high:.2f}", f"出現於 {high_time}")
  m3.metric("當日最低價", f"{day_low:.2f}", f"出現於 {low_time}")
  m4.metric(
      "當日高低振幅",
      f"{amplitude_pct:.2f}%",
      f"價差 {amplitude_pts:.2f}（門檻 {threshold_pct:.1f}%）",
  )
  m5.metric("布林通道狀態", bb_status_label, bb_delta)

  typical_price = (work["High"] + work["Low"] + work["Close"]) / 3.0
  if "Volume" in work.columns and work["Volume"].fillna(0).sum() > 0:
    vol = work["Volume"].fillna(0)
    cum_vol = vol.cumsum().replace(0, np.nan)
    vwap = (typical_price * vol).cumsum() / cum_vol
    vwap_label = "Intraday VWAP"
  else:
    vwap = typical_price.expanding().mean()
    vwap_label = "Avg Typical Price"

  x_labels = work.index.strftime("%m-%d %H:%M")
  x_pos = np.arange(len(work))

  has_volume = "Volume" in work.columns and work["Volume"].fillna(0).sum() > 0
  if has_volume:
    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1, figsize=(14, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
  else:
    fig, ax_price = plt.subplots(figsize=(14, 4.5))
    ax_vol = None

  ax_price.fill_between(
      x_pos,
      work["Low"],
      work["High"],
      color="steelblue",
      alpha=0.18,
      label="Bar High-Low Range",
  )
  ax_price.plot(x_pos, work["Close"], color="#111827", linewidth=1.8, label="Intraday Close")
  ax_price.plot(x_pos, vwap, color="#d97706", linestyle="--", linewidth=1.4, label=vwap_label)
  ax_price.axhline(day_high, color="#dc2626", linestyle=":", linewidth=1.2, label=f"Day High ({day_high:.2f})")
  ax_price.axhline(day_low, color="#16a34a", linestyle=":", linewidth=1.2, label=f"Day Low ({day_low:.2f})")

  if has_bollinger and est_high_2sd is not None and est_low_2sd is not None:
    ax_price.axhline(
        est_high_2sd,
        color="#991b1b",
        linestyle="-.",
        linewidth=1.3,
        label=f"Hist +2SD ({est_high_2sd:.2f})",
    )
    ax_price.axhline(
        est_low_2sd,
        color="#166534",
        linestyle="-.",
        linewidth=1.3,
        label=f"Hist -2SD ({est_low_2sd:.2f})",
    )

  title_tags = []
  if alert_info["amp_exceeded"]:
    title_tags.append(f"AMP>={threshold_pct:.1f}%")
  if alert_info["broke_upper"]:
    title_tags.append("BREAK +2SD")
  if alert_info["broke_lower"]:
    title_tags.append("BREAK -2SD")
  title_suffix = f" [{' | '.join(title_tags)}]" if title_tags else ""

  ax_price.set_title(
      f"{ticker_symbol} Intraday Volatility ({source_desc}){title_suffix}", fontsize=13
  )
  ax_price.set_ylabel("Price")
  ax_price.legend(loc="upper left", ncol=2)
  ax_price.grid(True, alpha=0.3)

  if ax_vol is not None:
    colors = np.where(
        work["Close"] >= work["Open"]
        if "Open" in work.columns
        else work["Close"].diff().fillna(0) >= 0,
        "#ef4444",
        "#22c55e",
    )
    ax_vol.bar(x_pos, work["Volume"].fillna(0), color=colors, alpha=0.65, width=0.65)
    ax_vol.set_ylabel("Volume")
    ax_vol.grid(True, alpha=0.3)

  step = max(1, len(x_pos) // 10)
  tick_idx = list(range(0, len(x_pos), step))
  if (len(x_pos) - 1) not in tick_idx:
    tick_idx.append(len(x_pos) - 1)
  (ax_vol if ax_vol is not None else ax_price).set_xticks(tick_idx)
  (ax_vol if ax_vol is not None else ax_price).set_xticklabels(
      [x_labels[i] for i in tick_idx], rotation=25, ha="right"
  )

  plt.tight_layout()
  st.pyplot(fig)
  plt.close(fig)


def run_full_analysis():
  """執行完整分析並存入 session_state"""
  end_date_str = (
      datetime.today().strftime("%Y-%m-%d")
      if use_today
      else custom_end_date.strftime("%Y-%m-%d")
  )
  start_date_str = start_date.strftime("%Y-%m-%d")
  tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

  if not tickers:
    st.session_state["analysis_bundle"] = None
    st.error("❌ 請至少輸入一個股票代號（例如 2330.TW 或 SPY）。")
    return

  index_labels = [
      "公司性質",
      "每股盈餘 (EPS)",
      "本益比 (P/E)",
      "毛利率",
      "淨利率",
      "流動資產 (最新期)",
      "負債率 (總負債/總資產)",
  ]
  combined_fundamentals = {}
  per_ticker_results = {}

  progress = st.progress(0, text="準備抓取資料...")
  total_steps = len(tickers)

  for idx, t in enumerate(tickers):
    progress.progress(
        idx / total_steps,
        text=f"⏳ 正在分析 {t} 基本面、歷史區間與當日波動（{idx + 1}/{total_steps}）...",
    )

    fund_info = get_fundamental_info(t)
    if fund_info:
      combined_fundamentals[fund_info["col_title"]] = fund_info["metrics"]

    hist_df = download_history_with_retry(t, start_date_str, end_date_str)
    intra_df, intra_desc = download_intraday_data(t, interval=intraday_interval)

    if hist_df.empty:
      per_ticker_results[t] = {
          "status": "empty",
          "message": f"❌ {t} 在 {start_date_str} ~ {end_date_str} 期間無法取得歷史股價資料。",
          "intra_df": intra_df,
          "intra_desc": intra_desc,
      }
    else:
      summary, calc_df = compute_analysis(hist_df, window=int(window))
      per_ticker_results[t] = {
          "status": "ok",
          "summary": summary,
          "calc_df": calc_df,
          "intra_df": intra_df,
          "intra_desc": intra_desc,
      }

    if idx < len(tickers) - 1:
      time.sleep(0.5)

  progress.progress(1.0, text="✅ 分析完成")
  time.sleep(0.2)
  progress.empty()

  fund_df = (
      pd.DataFrame(combined_fundamentals, index=index_labels)
      if combined_fundamentals
      else pd.DataFrame()
  )
  if not fund_df.empty:
    fund_df.index.name = "財務指標"

  st.session_state["analysis_bundle"] = {
      "tickers": tickers,
      "start_date_str": start_date_str,
      "end_date_str": end_date_str,
      "window": int(window),
      "intraday_interval": intraday_interval,
      "fund_df": fund_df,
      "per_ticker_results": per_ticker_results,
      "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
  }


def refresh_intraday_only():
  """僅重新抓取各檔股票的當日波動資料，保留既有基本面與歷史分析結果"""
  bundle = st.session_state.get("analysis_bundle")
  if not bundle:
    return

  download_intraday_data.clear()
  tickers = bundle["tickers"]
  iv = bundle.get("intraday_interval", intraday_interval)
  per_ticker_results = bundle["per_ticker_results"]

  for t in tickers:
    intra_df, intra_desc = download_intraday_data(t, interval=iv)
    if t in per_ticker_results:
      per_ticker_results[t]["intra_df"] = intra_df
      per_ticker_results[t]["intra_desc"] = intra_desc

  bundle["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  st.session_state["analysis_bundle"] = bundle


# --- 觸發完整分析或自動刷新 ---
if run_btn:
  st.session_state["last_refresh_ts"] = time.time()
  run_full_analysis()
elif auto_refresh and st.session_state.get("analysis_bundle") is None:
  st.session_state["last_refresh_ts"] = time.time()
  run_full_analysis()


bundle = st.session_state.get("analysis_bundle")

if bundle:
  tickers = bundle["tickers"]
  start_date_str = bundle["start_date_str"]
  end_date_str = bundle["end_date_str"]
  win = bundle["window"]
  fund_df = bundle["fund_df"]
  per_ticker_results = bundle["per_ticker_results"]
  last_updated = bundle.get("last_updated", "")

  status_text = (
      f"🕒 最後更新時間：**{last_updated}**"
      f" ｜ 🎯 振幅提醒門檻：**{float(amplitude_threshold):.1f}%**"
      " ｜ 📐 比對基準：**布林通道 ±2SD**"
      + (
          f" ｜ 🟢 自動刷新已啟用（每 {int(refresh_interval)} 秒更新當日波動圖）"
          if auto_refresh
          else " ｜ ⚪ 自動刷新已關閉"
      )
  )
  st.caption(status_text)

  # 1. 全檔基本面比較總表
  if not fund_df.empty:
    st.subheader("📋 股票基本面比較總表")
    st.dataframe(fund_df, use_container_width=True)
    st.download_button(
        "📥 下載基本面比較總表 (CSV)",
        data=fund_df.to_csv().encode("utf-8-sig"),
        file_name=f"股票基本面比較表_{end_date_str}.csv",
        mime="text/csv",
        key="download_all_fundamentals",
    )

  st.divider()
  st.subheader("🗂️ 多檔股票分頁詳情、當日波動圖與獨立匯出")

  # 2. 多檔股票 Tab 分頁切換（有振幅超標或突破/跌破 ±2SD 則加上 🚨 標記）
  tab_titles = []
  for t in tickers:
    res = per_ticker_results.get(t, {})
    summary = res.get("summary", {})
    alert_eval = evaluate_intraday_alert(
        ticker_symbol=t,
        intra_df=res.get("intra_df", pd.DataFrame()),
        threshold_pct=float(amplitude_threshold),
        est_high_2sd=summary.get("latest_est_high_val"),
        est_low_2sd=summary.get("latest_est_low_val"),
        has_bollinger=bool(summary.get("has_enough_window", False)),
    )
    alert_prefix = "🚨 " if alert_eval["has_alert"] else ""
    tab_titles.append(f"{alert_prefix}📌 {t}")

  tabs = st.tabs(tab_titles)

  for tab, t in zip(tabs, tickers):
    with tab:
      res = per_ticker_results.get(t)
      if not res:
        st.warning(f"❌ 找不到 {t} 的分析資料。")
        continue

      summary = res.get("summary", {})

      # 區塊 A：當日波動圖與盤中波動提醒（含布林通道 +2SD / -2SD 突破與跌破比對）
      render_intraday_volatility(
          ticker_symbol=t,
          intra_df=res.get("intra_df", pd.DataFrame()),
          source_desc=res.get("intra_desc", "無分時資料"),
          threshold_pct=float(amplitude_threshold),
          est_high_2sd=summary.get("latest_est_high_val"),
          est_low_2sd=summary.get("latest_est_low_val"),
          has_bollinger=bool(summary.get("has_enough_window", False)),
      )

      st.divider()

      # 區塊 B：歷史極值與布林通道高低點預估
      if res["status"] != "ok":
        st.warning(res["message"])
        continue

      calc_df = res["calc_df"]

      st.markdown(f"#### 📉 {t} 歷史極值與布林通道（{start_date_str} ~ {end_date_str}）")

      r1_c1, r1_c2, r1_c3 = st.columns(3)
      r1_c1.metric("最新收盤價", summary["最新收盤價"])
      r1_c2.metric("區間最高價", summary["區間最高價"], f"發生於 {summary['區間最高價日期']}")
      r1_c3.metric("區間最低價", summary["區間最低價"], f"發生於 {summary['區間最低價日期']}")

      if summary["has_enough_window"]:
        r2_c1, r2_c2, r2_c3 = st.columns(3)
        r2_c1.metric(f"基準均價 (SMA {win}日)", summary[f"基準均價 (SMA {win}日)"])
        r2_c2.metric("預估壓力高點 (+2SD)", summary["預估壓力高點 (+2SD)"])
        r2_c3.metric("預估支撐低點 (-2SD)", summary["預估支撐低點 (-2SD)"])
      else:
        st.info(
            f"⚠️ 有效交易日（{len(calc_df)} 天）少於設定的 window={win} 天，"
            "目前僅顯示歷史極值，尚未產生完整布林通道上下界。"
        )

      render_bollinger_chart(t, calc_df, window=win, end_label=end_date_str)

      # 區塊 C：單檔獨立匯出（排除內部數值/旗標欄位）
      internal_keys = {
          "has_enough_window",
          "latest_est_high_val",
          "latest_est_low_val",
      }
      summary_rows = [
          {"項目": k, "數值": v}
          for k, v in summary.items()
          if k not in internal_keys
      ]
      summary_df = pd.DataFrame(summary_rows)

      history_export_df = calc_df.copy()
      history_export_df.index = history_export_df.index.strftime("%Y-%m-%d")
      history_export_df.index.name = "Date"

      dl_col1, dl_col2 = st.columns(2)
      with dl_col1:
        st.download_button(
            f"📄 匯出 {t} 分析摘要 (CSV)",
            data=summary_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{t}_分析摘要_{start_date_str}_{end_date_str}.csv",
            mime="text/csv",
            key=f"dl_summary_{t}",
            use_container_width=True,
        )
      with dl_col2:
        st.download_button(
            f"📊 匯出 {t} 歷史股價與指標 (CSV)",
            data=history_export_df.to_csv().encode("utf-8-sig"),
            file_name=f"{t}_歷史股價與布林通道_{start_date_str}_{end_date_str}.csv",
            mime="text/csv",
            key=f"dl_hist_{t}",
            use_container_width=True,
        )

      with st.expander(f"檢視 {t} 歷史股價與指標明細"):
        st.dataframe(history_export_df.tail(30), use_container_width=True)

  # --- 自動刷新計時器 ---
  if auto_refresh:
    last_ts = st.session_state.get("last_refresh_ts", time.time())
    elapsed = time.time() - last_ts
    remaining = max(0, int(refresh_interval - elapsed))
    countdown_slot = st.sidebar.empty()
    countdown_slot.caption(f"⏳ 下次更新當日波動圖約剩 {remaining} 秒")

    time.sleep(1)
    if time.time() - last_ts >= int(refresh_interval):
      st.session_state["last_refresh_ts"] = time.time()
      refresh_intraday_only()
    st.rerun()
else:
  st.info("👈 請在左側設定股票代號與參數後，點擊「開始分析」。")
