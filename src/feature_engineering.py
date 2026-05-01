"""
feature_engineering.py
-----------------------
Creates all technical indicators, lag features, and time-based features
required for ML modelling of stock price data.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 1. Lag Features
# ─────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame, col: str = "Close", lags: int = 10) -> pd.DataFrame:
    """
    Create lag features Close(t-1) … Close(t-lags).

    WHY: Tree-based models (RF, XGB) have no notion of time order.
    Explicitly supplying past values teaches them the auto-regressive
    structure of price series.
    """
    for lag in range(1, lags + 1):
        df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    logger.info(f"Added {lags} lag features for '{col}'.")
    return df


# ─────────────────────────────────────────────
# 2. Moving Averages
# ─────────────────────────────────────────────

def add_moving_averages(df: pd.DataFrame, col: str = "Close",
                        windows: list = [7, 14, 30]) -> pd.DataFrame:
    """
    Simple Moving Average (SMA) for multiple windows.

    WHY: SMAs smooth noise and reveal underlying trend direction.
    Price vs SMA relationships are classic trading signals.
    """
    for w in windows:
        df[f"SMA_{w}"] = df[col].rolling(window=w, min_periods=1).mean()
    logger.info(f"Added SMA for windows {windows}.")
    return df


def add_ema(df: pd.DataFrame, col: str = "Close",
            spans: list = [12, 26]) -> pd.DataFrame:
    """
    Exponential Moving Average (EMA) – weights recent prices more.

    WHY: EMA reacts faster to price changes than SMA, important for
    MACD calculation and trend-following strategies.
    """
    for span in spans:
        df[f"EMA_{span}"] = df[col].ewm(span=span, adjust=False).mean()
    logger.info(f"Added EMA for spans {spans}.")
    return df


# ─────────────────────────────────────────────
# 3. RSI
# ─────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, col: str = "Close", period: int = 14) -> pd.DataFrame:
    """
    Relative Strength Index (RSI) — momentum oscillator [0, 100].

    Interpretation:
      RSI > 70  → overbought (potential sell)
      RSI < 30  → oversold  (potential buy)

    WHY: Captures momentum regime; helps models understand whether a
    trend is exhausted.
    """
    delta = df[col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)   # neutral RSI for initial NaN rows
    logger.info("Added RSI(14).")
    return df


# ─────────────────────────────────────────────
# 4. MACD
# ─────────────────────────────────────────────

def add_macd(df: pd.DataFrame, col: str = "Close",
             fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence (MACD).

    Components:
      MACD Line   = EMA(fast) - EMA(slow)
      Signal Line = EMA(MACD, signal)
      Histogram   = MACD - Signal

    WHY: Measures trend momentum and direction shifts; MACD crossovers
    are among the most widely used trading signals.
    """
    ema_fast = df[col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[col].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    logger.info("Added MACD(12,26,9).")
    return df


# ─────────────────────────────────────────────
# 5. Bollinger Bands
# ─────────────────────────────────────────────

def add_bollinger_bands(df: pd.DataFrame, col: str = "Close",
                        window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands: SMA ± k * rolling standard deviation.

    Components:
      BB_Upper = SMA + k*σ
      BB_Lower = SMA - k*σ
      BB_Width = (Upper - Lower) / Middle  [normalised volatility]
      BB_%B    = (Close - Lower) / (Upper - Lower)  [position in band]

    WHY: Encodes volatility context; %B tells models where the price
    sits within the recent volatility range.
    """
    mid = df[col].rolling(window=window, min_periods=1).mean()
    std = df[col].rolling(window=window, min_periods=1).std()

    df["BB_Upper"] = mid + num_std * std
    df["BB_Lower"] = mid - num_std * std
    df["BB_Mid"]   = mid
    df["BB_Width"]  = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]
    df["BB_PctB"]   = (df[col] - df["BB_Lower"]) / (
        df["BB_Upper"] - df["BB_Lower"]
    ).replace(0, np.nan)
    logger.info("Added Bollinger Bands(20, 2σ).")
    return df


# ─────────────────────────────────────────────
# 6. Volume Features
# ─────────────────────────────────────────────

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volume-derived signals.

    Features:
      Volume_MA7  – 7-day rolling average volume
      Volume_Ratio – today's volume vs 7-day average (>1 = unusual activity)

    WHY: Volume confirms price moves; spikes often precede breakouts.
    """
    df["Volume_MA7"]   = df["Volume"].rolling(7, min_periods=1).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA7"].replace(0, np.nan)
    logger.info("Added volume features.")
    return df


# ─────────────────────────────────────────────
# 7. Volatility
# ─────────────────────────────────────────────

def add_volatility(df: pd.DataFrame, windows: list = [7, 14, 30]) -> pd.DataFrame:
    """
    Rolling standard deviation of daily returns.

    WHY: Regime detection – models trained on volatile periods behave
    differently from calm periods; volatility is itself predictive.
    """
    for w in windows:
        df[f"Volatility_{w}"] = df["Daily_Return"].rolling(w, min_periods=2).std()
    logger.info(f"Added rolling volatility for windows {windows}.")
    return df


# ─────────────────────────────────────────────
# 8. Time Features
# ─────────────────────────────────────────────

def add_time_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Calendar-based features extracted from the Date column.

    Features:
      DayOfWeek  – 0 (Mon) … 4 (Fri); markets are closed on weekends
      Month      – seasonal patterns (e.g. January Effect)
      Quarter    – earnings season alignment
      Year       – long-term trend encoding
      IsMonday / IsFriday – first/last day of week effects

    WHY: Stock returns exhibit documented calendar anomalies.
    """
    df["DayOfWeek"] = df[date_col].dt.dayofweek
    df["Month"]     = df[date_col].dt.month
    df["Quarter"]   = df[date_col].dt.quarter
    df["Year"]      = df[date_col].dt.year
    df["IsMonday"]  = (df["DayOfWeek"] == 0).astype(int)
    df["IsFriday"]  = (df["DayOfWeek"] == 4).astype(int)
    logger.info("Added time features.")
    return df


# ─────────────────────────────────────────────
# 9. Target Variable
# ─────────────────────────────────────────────

def add_target(df: pd.DataFrame, col: str = "Close") -> pd.DataFrame:
    """
    Define the prediction target.

    Target_Close  – next day's closing price  (regression)
    Target_Dir    – 1 if price goes UP, 0 if DOWN (classification)

    WHY: Using next-day price as target encodes a genuine forecast task;
    using same-day data would cause data leakage.
    """
    df["Target_Close"] = df[col].shift(-1)
    df["Target_Dir"]   = (df["Target_Close"] > df[col]).astype(int)
    return df


# ─────────────────────────────────────────────
# 10. Master Pipeline
# ─────────────────────────────────────────────

def build_features(df: pd.DataFrame, lag_periods: int = 10) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline and return a clean,
    model-ready DataFrame (NaN rows from rolling windows are dropped).
    """
    df = df.copy()
    df = add_lag_features(df, lags=lag_periods)
    df = add_moving_averages(df)
    df = add_ema(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_volume_features(df)
    df = add_volatility(df)
    df = add_time_features(df)
    df = add_target(df)

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    logger.info(
        f"Feature engineering complete. {before - len(df)} rows dropped "
        f"(rolling window warm-up + target shift). Final shape: {df.shape}"
    )
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return model input feature columns (excludes Date, Ticker, Targets)."""
    exclude = {"Date", "Ticker", "Target_Close", "Target_Dir",
               "Open", "High", "Low", "Close", "Volume",
               "Daily_Return", "Log_Return"}
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    import sys
    from data_preprocessing import preprocess
    raw = sys.argv[1] if len(sys.argv) > 1 else "data/raw/stock_data.csv"
    df_clean = preprocess(raw, save_path="data/processed/stock_data_clean.csv")
    df_feat  = build_features(df_clean)
    feat_cols = get_feature_columns(df_feat)
    print(f"\nFeature columns ({len(feat_cols)}):\n", feat_cols)
    df_feat.to_csv("data/processed/stock_data_features.csv", index=False)
    print("\nSaved: data/processed/stock_data_features.csv")
