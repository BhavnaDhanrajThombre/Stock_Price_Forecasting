"""
utils.py
--------
Shared utilities: EDA visualisations, logging setup, colour palettes,
sample data generator, and plotting helpers used across notebooks and scripts.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# Global style
# ─────────────────────────────────────────────

def set_plot_style():
    """Apply a clean, professional matplotlib style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 120,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

set_plot_style()

PLOT_DIR = "outputs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def get_logger(name: str = "stock_forecasting") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


logger = get_logger()


# ─────────────────────────────────────────────
# Sample Data Generator (for demo / testing)
# ─────────────────────────────────────────────

def generate_sample_stock_data(n_days: int = 1000,
                               ticker: str = "DEMO",
                               start_price: float = 100.0,
                               seed: int = 42) -> pd.DataFrame:
    """
    Simulate realistic OHLCV data using Geometric Brownian Motion.

    Parameters
    ----------
    n_days      : number of trading days
    ticker      : stock ticker label
    start_price : initial close price
    seed        : random seed for reproducibility
    """
    np.random.seed(seed)
    mu    = 0.0003     # daily drift (≈7.5% annually)
    sigma = 0.015      # daily volatility

    returns = np.random.normal(mu, sigma, n_days)
    close   = start_price * np.cumprod(1 + returns)

    # Simulate OHLV from close
    high   = close * (1 + np.abs(np.random.normal(0, 0.005, n_days)))
    low    = close * (1 - np.abs(np.random.normal(0, 0.005, n_days)))
    open_  = np.roll(close, 1)
    open_[0] = start_price
    volume = np.random.randint(1_000_000, 10_000_000, n_days)

    dates = pd.bdate_range(start="2019-01-01", periods=n_days)

    df = pd.DataFrame({
        "Date":   dates,
        "Open":   open_.round(2),
        "High":   high.round(2),
        "Low":    low.round(2),
        "Close":  close.round(2),
        "Volume": volume,
        "Ticker": ticker,
    })
    return df


def ensure_sample_data(path: str = "data/raw/stock_data.csv") -> str:
    """Create sample CSV if no raw data exists."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = generate_sample_stock_data()
        df.to_csv(path, index=False)
        logger.info(f"Sample data generated → {path}")
    return path


# ─────────────────────────────────────────────
# EDA Visualisations
# ─────────────────────────────────────────────

def plot_price_history(df: pd.DataFrame, col: str = "Close",
                       ticker: str = "", save: bool = True) -> None:
    """Full-width close price time-series with volume bar."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7),
                                   gridspec_kw={"height_ratios": [3, 1]},
                                   sharex=True)

    ax1.plot(df["Date"], df[col], color="#1565C0", linewidth=1.4)
    ax1.fill_between(df["Date"], df[col], alpha=0.08, color="#1565C0")
    ax1.set_title(f"{ticker} – {col} Price History", fontweight="bold")
    ax1.set_ylabel(f"{col} Price (USD)")

    ax2.bar(df["Date"], df["Volume"], color="#90A4AE", width=1, alpha=0.8)
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("Date")
    fig.autofmt_xdate()
    plt.tight_layout()
    _save(fig, "price_history.png", save)


def plot_rolling_stats(df: pd.DataFrame, col: str = "Close",
                       windows: list = [7, 30, 90], save: bool = True) -> None:
    """Rolling mean and standard deviation overlay."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax1.plot(df["Date"], df[col], label="Close", color="#37474F", linewidth=1.0, alpha=0.7)
    colors = ["#EF5350", "#42A5F5", "#66BB6A"]
    for w, c in zip(windows, colors):
        ax1.plot(df["Date"], df[col].rolling(w).mean(),
                 label=f"SMA-{w}", color=c, linewidth=1.3)
    ax1.set_title("Price with Rolling Moving Averages", fontweight="bold")
    ax1.set_ylabel("Price")
    ax1.legend()

    ax2.plot(df["Date"], df[col].rolling(30).std(),
             color="#AB47BC", linewidth=1.3)
    ax2.set_title("30-Day Rolling Volatility (Std Dev of Price)", fontweight="bold")
    ax2.set_ylabel("Std Dev")
    ax2.set_xlabel("Date")
    fig.autofmt_xdate()
    plt.tight_layout()
    _save(fig, "rolling_stats.png", save)


def plot_correlation_heatmap(df: pd.DataFrame,
                             cols: list = None, save: bool = True) -> None:
    """Correlation heatmap for OHLCV + derived columns."""
    if cols is None:
        numeric = df.select_dtypes(include=np.number)
        # Keep only columns with reasonable variance
        cols = [c for c in numeric.columns
                if numeric[c].std() > 1e-6 and c not in ["Year"]][:20]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.4, ax=ax, annot_kws={"size": 7})
    ax.set_title("Feature Correlation Heatmap", fontweight="bold", fontsize=13)
    plt.tight_layout()
    _save(fig, "correlation_heatmap.png", save)


def plot_return_distribution(df: pd.DataFrame, save: bool = True) -> None:
    """Histogram + KDE of daily returns with normal overlay."""
    fig, ax = plt.subplots(figsize=(10, 5))
    returns = df["Daily_Return"].dropna()
    sns.histplot(returns, bins=60, kde=True, stat="density",
                 color="#5C6BC0", ax=ax, edgecolor="none", alpha=0.7)

    from scipy.stats import norm
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 200)
    ax.plot(x, norm.pdf(x, mu, sigma), "r--", linewidth=1.8, label="Normal fit")
    ax.axvline(0, color="black", linestyle=":", linewidth=1.2)
    ax.set_title("Daily Return Distribution", fontweight="bold")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Density")
    ax.legend()
    stats_text = f"μ={mu:.4f}  σ={sigma:.4f}\nSkew={returns.skew():.2f}  Kurt={returns.kurtosis():.2f}"
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.7))
    plt.tight_layout()
    _save(fig, "return_distribution.png", save)


def plot_bollinger_bands_interactive(df: pd.DataFrame,
                                     tail_days: int = 252) -> go.Figure:
    """Interactive Plotly candlestick + Bollinger Bands chart."""
    d = df.tail(tail_days).copy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3],
                        subplot_titles=["Price & Bollinger Bands", "Volume"])

    fig.add_trace(go.Candlestick(
        x=d["Date"], open=d["Open"], high=d["High"],
        low=d["Low"], close=d["Close"], name="OHLC",
    ), row=1, col=1)

    if "BB_Upper" in d.columns:
        fig.add_trace(go.Scatter(x=d["Date"], y=d["BB_Upper"],
                                 name="BB Upper", line=dict(color="gray", dash="dot"),
                                 showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=d["Date"], y=d["BB_Lower"],
                                 name="BB Lower", line=dict(color="gray", dash="dot"),
                                 fill="tonexty", fillcolor="rgba(128,128,128,0.1)",
                                 showlegend=True), row=1, col=1)

    fig.add_trace(go.Bar(x=d["Date"], y=d["Volume"],
                         name="Volume", marker_color="#90A4AE"), row=2, col=1)

    fig.update_layout(
        title="Interactive Candlestick Chart with Bollinger Bands",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=650,
    )
    return fig


def plot_technical_indicators_interactive(df: pd.DataFrame,
                                           tail_days: int = 252) -> go.Figure:
    """RSI and MACD panel chart."""
    d = df.tail(tail_days).copy()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=["Close Price", "RSI", "MACD"])

    fig.add_trace(go.Scatter(x=d["Date"], y=d["Close"],
                             name="Close", line=dict(color="#1565C0")), row=1, col=1)

    if "RSI" in d.columns:
        fig.add_trace(go.Scatter(x=d["Date"], y=d["RSI"],
                                 name="RSI", line=dict(color="#9C27B0")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red",   row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

    if "MACD" in d.columns:
        fig.add_trace(go.Scatter(x=d["Date"], y=d["MACD"],
                                 name="MACD",   line=dict(color="#F57C00")), row=3, col=1)
        fig.add_trace(go.Scatter(x=d["Date"], y=d["MACD_Signal"],
                                 name="Signal", line=dict(color="#EF5350")), row=3, col=1)
        colors = ["#4CAF50" if v >= 0 else "#F44336"
                  for v in d["MACD_Hist"]]
        fig.add_trace(go.Bar(x=d["Date"], y=d["MACD_Hist"],
                             name="Histogram", marker_color=colors), row=3, col=1)

    fig.update_layout(
        template="plotly_white", height=750,
        title="Technical Indicators Dashboard",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        hovermode="x unified",
    )
    return fig


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _save(fig: plt.Figure, filename: str, save: bool) -> None:
    if save:
        path = os.path.join(PLOT_DIR, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {path}")
    plt.close(fig)
