"""
dashboard/app.py
----------------
Streamlit dashboard for the Stock Price Forecasting project.

Run with:
    streamlit run dashboard/app.py

Features
--------
- Upload your own CSV or use demo data
- Historical price & volume chart
- Technical indicators panel (SMA, EMA, RSI, MACD, Bollinger Bands)
- Train models and view predictions
- Model comparison table
- Buy / Sell / Hold signal panel
"""

# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import sys, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(BASE_DIR, 'src')

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data_preprocessing import preprocess, dataset_summary
from feature_engineering import build_features, get_feature_columns
from model_training import (
    time_based_split, scale_features,
    train_linear_regression, train_random_forest,
    train_xgboost, train_arima, predict_arima_steps,
)
from evaluation import (
    compute_metrics, build_comparison_table,
    generate_signals,
)
from utils import generate_sample_stock_data

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="📈 Stock Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f0f4f8;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border-left: 4px solid #1565C0;
    }
    .signal-buy  { color: #2e7d32; font-weight: bold; font-size: 1.4em; }
    .signal-sell { color: #c62828; font-weight: bold; font-size: 1.4em; }
    .signal-hold { color: #f57f17; font-weight: bold; font-size: 1.4em; }
    .section-header { font-size: 1.1em; font-weight: bold; color: #1565C0;
                      border-bottom: 2px solid #1565C0; padding-bottom: 4px; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/stock-market.png", width=80)
    st.title("Stock Forecasting")
    st.markdown("---")

    st.subheader("📂 Data Source")
    data_source = st.radio("Choose data source:", ["Demo Data", "Upload CSV"])

    raw_df = None

    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload stock CSV", type=["csv"])
        if uploaded:
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                raw_df = preprocess(tmp_path)
                st.success(f"✅ Loaded {len(raw_df)} rows")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        ticker = st.selectbox("Select ticker:", ["AAPL", "TSLA", "MSFT", "AMZN", "Custom"])
        seed_map = {"AAPL": 42, "TSLA": 7, "MSFT": 13, "AMZN": 99, "Custom": 0}
        raw_df = generate_sample_stock_data(
            n_days=1500,
            ticker=ticker if ticker != "Custom" else "DEMO",
            seed=seed_map[ticker],
        )
        # Apply preprocessing
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            raw_df.to_csv(tmp.name, index=False)
            raw_df = preprocess(tmp.name)

    st.markdown("---")
    st.subheader("⚙️ Settings")
    train_ratio    = st.slider("Train / Test Split", 0.60, 0.90, 0.80, 0.05)
    indicator_days = st.slider("Chart lookback (days)", 90, 730, 252, 30)

    st.markdown("---")
    run_models = st.button("🚀 Train & Predict", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("⚠️ For educational use only. Not financial advice.")


# ─────────────────────────────────────────────
# Main content – only if data loaded
# ─────────────────────────────────────────────
if raw_df is None:
    st.info("👈 Upload a CSV or select demo data from the sidebar to get started.")
    st.stop()

df_clean = raw_df.copy()

# ── Header ──
ticker_label = df_clean["Ticker"].iloc[0] if "Ticker" in df_clean.columns else "Stock"
st.title(f"📈 {ticker_label} – Stock Price Forecasting Dashboard")

summary = dataset_summary(df_clean)
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Trading Days",  summary["trading_days"])
col2.metric("Price Range",   f"${summary['close_min']:.2f} – ${summary['close_max']:.2f}")
col3.metric("Mean Close",    f"${summary['close_mean']:.2f}")
col4.metric("Avg Volume",    f"{summary['avg_daily_volume']:,.0f}")
total_ret = (df_clean["Close"].iloc[-1] / df_clean["Close"].iloc[0] - 1) * 100
col5.metric("Total Return",  f"{total_ret:+.1f}%",
            delta_color="normal" if total_ret >= 0 else "inverse")

st.markdown("---")

# ─────────────────────────────────────────────
# Tab layout
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Historical Data",
    "📉 Technical Indicators",
    "🤖 Predictions",
    "🚦 Signals",
])


# ──────────────────────────────────────────────
# TAB 1: Historical Data
# ──────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-header">Price & Volume History</p>', unsafe_allow_html=True)

    tail = df_clean.tail(indicator_days)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28],
        subplot_titles=["Candlestick Chart", "Volume"],
        vertical_spacing=0.05,
    )
    fig.add_trace(go.Candlestick(
        x=tail["Date"], open=tail["Open"],
        high=tail["High"], low=tail["Low"], close=tail["Close"],
        name="OHLC",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    volume_colors = ["#26a69a" if c >= o else "#ef5350"
                     for c, o in zip(tail["Close"], tail["Open"])]
    fig.add_trace(go.Bar(
        x=tail["Date"], y=tail["Volume"],
        name="Volume", marker_color=volume_colors, opacity=0.8,
    ), row=2, col=1)

    fig.update_layout(
        height=550,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Raw Data Table"):
        st.dataframe(
            df_clean[["Date","Open","High","Low","Close","Volume","Daily_Return"]]
            .tail(indicator_days)
            .set_index("Date")
            .style.format({
                "Open": "{:.2f}", "High": "{:.2f}", "Low": "{:.2f}",
                "Close": "{:.2f}", "Volume": "{:,.0f}", "Daily_Return": "{:.4f}",
            }),
            height=300,
        )


# ──────────────────────────────────────────────
# TAB 2: Technical Indicators
# ──────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-header">Technical Analysis</p>', unsafe_allow_html=True)

    # Build features (needed for indicators)
    @st.cache_data(show_spinner="Building features…")
    def get_features(data_hash):
        return build_features(df_clean)

    df_feat = get_features(hash(str(df_clean.shape) + str(df_clean["Close"].sum())))
    tail_f  = df_feat.tail(indicator_days)

    indicator_choice = st.selectbox(
        "Select indicator panel:",
        ["Moving Averages", "RSI", "MACD", "Bollinger Bands"],
    )

    if indicator_choice == "Moving Averages":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tail_f["Date"], y=tail_f["Close"],
                                 name="Close", line=dict(color="#37474F", width=1.5)))
        for col, color in [("SMA_7","#EF5350"),("SMA_14","#42A5F5"),("SMA_30","#66BB6A")]:
            if col in tail_f.columns:
                fig.add_trace(go.Scatter(x=tail_f["Date"], y=tail_f[col],
                                         name=col, line=dict(color=color, width=1.3)))
        fig.update_layout(title="Simple Moving Averages", template="plotly_white",
                          height=450, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    elif indicator_choice == "RSI":
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4])
        fig.add_trace(go.Scatter(x=tail_f["Date"], y=tail_f["Close"],
                                 name="Close", line=dict(color="#1565C0")), row=1, col=1)
        fig.add_trace(go.Scatter(x=tail_f["Date"], y=tail_f["RSI"],
                                 name="RSI", line=dict(color="#9C27B0")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red",   row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=2, col=1)
        fig.update_layout(title="RSI (14)", template="plotly_white",
                          height=500, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    elif indicator_choice == "MACD":
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4])
        fig.add_trace(go.Scatter(x=tail_f["Date"], y=tail_f["Close"],
                                 name="Close", line=dict(color="#1565C0")), row=1, col=1)
        fig.add_trace(go.Scatter(x=tail_f["Date"], y=tail_f["MACD"],
                                 name="MACD", line=dict(color="#F57C00")), row=2, col=1)
        fig.add_trace(go.Scatter(x=tail_f["Date"], y=tail_f["MACD_Signal"],
                                 name="Signal", line=dict(color="#EF5350")), row=2, col=1)
        colors = ["#4CAF50" if v >= 0 else "#F44336" for v in tail_f["MACD_Hist"]]
        fig.add_trace(go.Bar(x=tail_f["Date"], y=tail_f["MACD_Hist"],
                             name="Histogram", marker_color=colors), row=2, col=1)
        fig.update_layout(title="MACD (12,26,9)", template="plotly_white",
                          height=500, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    elif indicator_choice == "Bollinger Bands":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tail_f["Date"], y=tail_f["Close"],
                                 name="Close", line=dict(color="#1565C0", width=1.5)))
        fig.add_trace(go.Scatter(x=tail_f["Date"], y=tail_f["BB_Upper"],
                                 name="Upper", line=dict(color="#EF5350", dash="dot")))
        fig.add_trace(go.Scatter(x=tail_f["Date"], y=tail_f["BB_Lower"],
                                 name="Lower", line=dict(color="#66BB6A", dash="dot"),
                                 fill="tonexty", fillcolor="rgba(128,128,128,0.08)"))
        fig.add_trace(go.Scatter(x=tail_f["Date"], y=tail_f["BB_Mid"],
                                 name="Mid SMA", line=dict(color="#FFA726", dash="dash")))
        fig.update_layout(title="Bollinger Bands (20-day, 2σ)", template="plotly_white",
                          height=450, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# TAB 3: Predictions
# ──────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-header">Model Training & Predictions</p>',
                unsafe_allow_html=True)

    if not run_models:
        st.info("👈 Click **Train & Predict** in the sidebar to train models.")
    else:
        # Ensure features available
        if "df_feat" not in dir():
            df_feat = build_features(df_clean)

        feature_cols = get_feature_columns(df_feat)

        with st.spinner("Training models… this may take a minute."):
            X_train, X_test, y_train, y_test, train_df, test_df = time_based_split(
                df_feat, feature_cols, train_ratio=train_ratio
            )
            X_tr_sc, X_te_sc, scaler = scale_features(X_train, X_test)

            # Train
            lr    = train_linear_regression(X_tr_sc, y_train)
            rf    = train_random_forest(X_tr_sc, y_train, n_estimators=100)
            xgb_m = train_xgboost(X_tr_sc, y_train, n_estimators=200)
            arima = train_arima(train_df["Close"])

            preds = {
                "LinearRegression": lr.predict(X_te_sc),
                "RandomForest":     rf.predict(X_te_sc),
                "XGBoost":         xgb_m.predict(X_te_sc),
                "ARIMA":           predict_arima_steps(arima, len(test_df)),
            }

        st.success("✅ Models trained!")

        # Metrics table
        rows = []
        for name, pred in preds.items():
            m = compute_metrics(y_test, np.asarray(pred), name)
            rows.append(m)
        comp_df = pd.DataFrame(rows).sort_values("RMSE").set_index("Model")

        st.subheader("📊 Model Comparison")
        st.dataframe(
            comp_df.style.background_gradient(subset=["RMSE","MAE","MAPE (%)"],
                                               cmap="RdYlGn_r")
                         .format("{:.4f}"),
            use_container_width=True,
        )

        # Plotly interactive chart
        st.subheader("📈 Actual vs Predicted")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=test_df["Date"], y=y_test,
            name="Actual", line=dict(color="#1565C0", width=2),
        ))
        colors = ["#FF5722","#4CAF50","#9C27B0","#FF9800"]
        for (name, pred), color in zip(preds.items(), colors):
            p = np.asarray(pred)
            fig.add_trace(go.Scatter(
                x=test_df["Date"][:len(p)], y=p,
                name=name, line=dict(color=color, dash="dash", width=1.4),
                opacity=0.85,
            ))
        fig.update_layout(
            height=480,
            template="plotly_white",
            hovermode="x unified",
            title="Actual vs Predicted – All Models (Test Set)",
            legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Store for signals tab
        st.session_state["preds"]    = preds
        st.session_state["y_test"]   = y_test
        st.session_state["test_df"]  = test_df
        st.session_state["comp_df"]  = comp_df
        st.session_state["xgb_pred"] = preds.get("XGBoost")


# ──────────────────────────────────────────────
# TAB 4: Signals
# ──────────────────────────────────────────────
with tab4:
    st.markdown('<p class="section-header">Buy / Sell / Hold Signals</p>',
                unsafe_allow_html=True)

    if "xgb_pred" not in st.session_state:
        st.info("👈 Train models first (Predictions tab).")
    else:
        best_pred  = np.asarray(st.session_state["xgb_pred"])
        test_df_s  = st.session_state["test_df"]
        threshold  = st.slider("Signal threshold (%)", 0.1, 2.0, 0.5, 0.1) / 100

        sig_df = generate_signals(test_df_s, best_pred, threshold_pct=threshold)

        # Latest signal banner
        latest      = sig_df.iloc[-1]
        latest_sig  = latest["Signal"]
        sig_color   = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}
        sig_emoji   = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}

        col_l, col_m, col_r = st.columns([1, 2, 1])
        with col_m:
            st.markdown(f"""
            <div style='background:#f5f5f5; border-radius:12px; padding:24px; text-align:center;'>
                <div style='font-size:2em'>{sig_emoji[latest_sig]}</div>
                <div class='{sig_color[latest_sig]}'>{latest_sig}</div>
                <div style='color:#555; margin-top:8px;'>
                    Current: <b>${latest['Close']:.2f}</b> →
                    Predicted: <b>${latest['Predicted_Close']:.2f}</b><br>
                    Expected return: <b>{latest['Predicted_Return']*100:+.2f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # Signal overlay chart
        buy_mask  = sig_df["Signal"] == "BUY"
        sell_mask = sig_df["Signal"] == "SELL"
        hold_mask = sig_df["Signal"] == "HOLD"

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sig_df["Date"], y=sig_df["Close"],
                                 name="Close", line=dict(color="#37474F", width=1.5)))
        fig.add_trace(go.Scatter(x=sig_df["Date"], y=sig_df["Predicted_Close"],
                                 name="Predicted", line=dict(color="#0288D1", dash="dash", width=1.2)))
        fig.add_trace(go.Scatter(
            x=sig_df["Date"][buy_mask], y=sig_df["Close"][buy_mask],
            mode="markers", name="BUY",
            marker=dict(symbol="triangle-up", color="#4CAF50", size=10),
        ))
        fig.add_trace(go.Scatter(
            x=sig_df["Date"][sell_mask], y=sig_df["Close"][sell_mask],
            mode="markers", name="SELL",
            marker=dict(symbol="triangle-down", color="#F44336", size=10),
        ))
        fig.add_trace(go.Scatter(
            x=sig_df["Date"][hold_mask], y=sig_df["Close"][hold_mask],
            mode="markers", name="HOLD",
            marker=dict(symbol="circle", color="#FFC107", size=5, opacity=0.4),
        ))
        fig.update_layout(
            title="Buy / Sell / Hold Signals on Price Chart",
            template="plotly_white", height=470,
            hovermode="x unified",
            legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        counts = sig_df["Signal"].value_counts()
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("🟢 BUY signals",  counts.get("BUY", 0))
        sc2.metric("🟡 HOLD signals", counts.get("HOLD", 0))
        sc3.metric("🔴 SELL signals", counts.get("SELL", 0))

        

        with st.expander("📋 Recent Signals Table"):
            st.dataframe(
                sig_df[["Date","Close","Predicted_Close","Predicted_Return","Signal"]]
                .tail(30)
                .set_index("Date")
                .style.applymap(
                    lambda v: "color: green; font-weight: bold" if v == "BUY"
                    else ("color: red; font-weight: bold" if v == "SELL" else ""),
                    subset=["Signal"]
                )
                .format({"Close": "{:.2f}", "Predicted_Close": "{:.2f}",
                         "Predicted_Return": "{:.4f}"}),
                use_container_width=True,
            )
