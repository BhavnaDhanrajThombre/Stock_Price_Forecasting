"""
data_preprocessing.py
---------------------
Handles loading, cleaning, and saving of raw stock price data.
Designed to be robust to varying CSV schemas.
"""

import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name aliases – maps common variants to canonical names
# ---------------------------------------------------------------------------
COLUMN_ALIASES = {
    "date": "Date",
    "time": "Date",
    "timestamp": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adj close": "Adj_Close",
    "adj_close": "Adj_Close",
    "adjusted close": "Adj_Close",
    "volume": "Volume",
    "ticker": "Ticker",
    "symbol": "Ticker",
    "stock": "Ticker",
    "name": "Ticker",
}

REQUIRED_COLS = ["Date", "Open", "High", "Low", "Close", "Volume"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load a raw CSV file and return a DataFrame."""
    logger.info(f"Loading raw data from: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Raw shape: {df.shape}")
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names using COLUMN_ALIASES."""
    rename_map = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[key]
    df = df.rename(columns=rename_map)
    logger.info(f"Columns after normalization: {df.columns.tolist()}")
    return df


def parse_dates(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Parse date column to datetime and sort chronologically."""
    df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
    df = df.sort_values(date_col).reset_index(drop=True)
    logger.info(f"Date range: {df[date_col].min()} → {df[date_col].max()}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy:
    - OHLCV numeric columns: forward-fill then back-fill (carry last known price)
    - Any remaining NaNs: drop the row
    """
    before = len(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    df = df.dropna()
    after = len(df)
    logger.info(f"Missing value handling: {before - after} rows dropped.")
    return df


def remove_duplicates(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Remove duplicate rows based on Date (and Ticker if present)."""
    subset = [date_col]
    if "Ticker" in df.columns:
        subset.append("Ticker")
    before = len(df)
    df = df.drop_duplicates(subset=subset).reset_index(drop=True)
    after = len(df)
    logger.info(f"Removed {before - after} duplicate rows.")
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """Raise an error if any required column is missing."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required columns missing after normalization: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def add_basic_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add lightweight derived columns useful across all modules."""
    df["Daily_Return"] = df["Close"].pct_change()          # % daily return
    df["HL_Spread"] = df["High"] - df["Low"]               # intraday range
    df["OC_Spread"] = df["Close"] - df["Open"]             # open-to-close move
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def preprocess(filepath: str, save_path: str = None) -> pd.DataFrame:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    filepath  : path to raw CSV
    save_path : if provided, saves cleaned CSV here

    Returns
    -------
    Cleaned DataFrame
    """
    df = load_raw_data(filepath)
    df = normalize_columns(df)
    validate_columns(df)
    df = parse_dates(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = add_basic_derived_columns(df)

    logger.info(f"Preprocessing complete. Final shape: {df.shape}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Cleaned data saved to: {save_path}")

    return df


# ---------------------------------------------------------------------------
# Quick dataset summary (used in EDA notebook)
# ---------------------------------------------------------------------------

def dataset_summary(df: pd.DataFrame) -> dict:
    """Return a dictionary of summary statistics for display."""
    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "date_start": df["Date"].min(),
        "date_end": df["Date"].max(),
        "trading_days": len(df),
        "missing_values": df.isnull().sum().sum(),
        "close_min": df["Close"].min(),
        "close_max": df["Close"].max(),
        "close_mean": df["Close"].mean(),
        "avg_daily_volume": df["Volume"].mean(),
    }
    return summary


if __name__ == "__main__":
    import sys
    raw_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/stock_data.csv"
    save_path = "data/processed/stock_data_clean.csv"
    df = preprocess(raw_path, save_path)
    print(df.head())
    print("\nSummary:")
    for k, v in dataset_summary(df).items():
        print(f"  {k}: {v}")
