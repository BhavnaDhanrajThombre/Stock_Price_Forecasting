"""
model_training.py
-----------------
Trains multiple forecasting models with a proper time-based train/test split.

Models
------
1. Linear Regression   (baseline)
2. Random Forest       (ensemble, handles non-linearity)
3. XGBoost             (gradient boosting, state-of-the-art on tabular data)
4. ARIMA               (classical time-series benchmark)
5. LSTM                (deep learning, optional – skipped gracefully if TF absent)

WHY time-based split?
---------------------
Random splits would allow future data to leak into training, making
performance metrics unrealistically optimistic. In production, a model
always trains on the past and predicts the future – we replicate that here.
"""

import os
import logging
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. Train / Test Split  (time-based, no shuffle)
# ─────────────────────────────────────────────

def time_based_split(df: pd.DataFrame, feature_cols: list,
                     target_col: str = "Target_Close",
                     train_ratio: float = 0.80):
    """
    Split dataset chronologically.

    Returns
    -------
    X_train, X_test, y_train, y_test, train_df, test_df
    """
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]

    X_train = train[feature_cols].values
    X_test  = test[feature_cols].values
    y_train = train[target_col].values
    y_test  = test[target_col].values

    logger.info(
        f"Train: {len(train)} rows | Test: {len(test)} rows "
        f"({100*(1-train_ratio):.0f}% held-out)"
    )
    return X_train, X_test, y_train, y_test, train, test


def scale_features(X_train: np.ndarray, X_test: np.ndarray):
    """Fit StandardScaler on train, apply to both splits."""
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    return X_train_sc, X_test_sc, scaler


# ─────────────────────────────────────────────
# 2. Individual Model Trainers
# ─────────────────────────────────────────────

def train_linear_regression(X_train, y_train):
    """Baseline linear model – fast, interpretable."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    logger.info("Linear Regression trained.")
    return model


def train_random_forest(X_train, y_train,
                        n_estimators: int = 200,
                        max_depth: int = 10,
                        random_state: int = 42):
    """
    Random Forest Regressor.

    - 200 trees for stability
    - max_depth=10 prevents overfitting on financial data
    - n_jobs=-1 uses all CPU cores
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info(f"Random Forest trained ({n_estimators} trees).")
    return model


def train_xgboost(X_train, y_train,
                  n_estimators: int = 300,
                  learning_rate: float = 0.05,
                  max_depth: int = 6,
                  random_state: int = 42):
    """
    XGBoost Regressor.

    - Lower learning rate (0.05) with more trees → better generalisation
    - subsample & colsample_bytree add regularisation
    """
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        eval_metric="rmse",
        verbosity=0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train)],
              verbose=False)
    logger.info(f"XGBoost trained ({n_estimators} estimators).")
    return model


def train_arima(series: pd.Series, order: tuple = (5, 1, 0)):
    """
    ARIMA(p, d, q) time-series baseline.

    - Uses only the Close price (univariate)
    - p=5: 5 autoregressive lags
    - d=1: first differencing for stationarity
    - q=0: no moving-average term (keeps it simple)

    Returns the fitted ARIMA result object.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(series, order=order)
        result = model.fit()
        logger.info(f"ARIMA{order} trained.")
        return result
    except Exception as e:
        logger.warning(f"ARIMA training failed: {e}")
        return None


def train_lstm(X_train, y_train, X_test, y_test,
               epochs: int = 50, batch_size: int = 32):
    """
    LSTM neural network for sequential pattern learning.

    Architecture:
      Input → LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(1)

    WHY LSTM: Unlike tree models, LSTM natively captures temporal
    dependencies through its gating mechanism.

    Returns (model, history) or (None, None) if TensorFlow is not available.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping

        # Reshape for LSTM: (samples, timesteps=1, features)
        X_tr = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_te = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(1, X_train.shape[1])),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")

        es = EarlyStopping(monitor="val_loss", patience=10,
                           restore_best_weights=True)
        history = model.fit(
            X_tr, y_train,
            validation_data=(X_te, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es],
            verbose=0,
        )
        logger.info(f"LSTM trained ({epochs} max epochs, early-stopping active).")
        return model, history

    except ImportError:
        logger.warning("TensorFlow not installed – LSTM skipped.")
        return None, None


# ─────────────────────────────────────────────
# 3. Prediction Helpers
# ─────────────────────────────────────────────

def predict_arima_steps(arima_result, n_steps: int) -> np.ndarray:
    """Generate n_steps ahead ARIMA forecasts."""
    if arima_result is None:
        return np.array([np.nan] * n_steps)
    forecast = arima_result.forecast(steps=n_steps)
    return forecast.values


def predict_lstm(model, X_test: np.ndarray) -> np.ndarray:
    """Run LSTM inference; handles None model gracefully."""
    if model is None:
        return np.array([np.nan] * len(X_test))
    X_re = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    return model.predict(X_re, verbose=0).flatten()


# ─────────────────────────────────────────────
# 4. Model Persistence
# ─────────────────────────────────────────────

def save_model(model, path: str) -> None:
    """Pickle a sklearn / XGB model."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved → {path}")


def load_model(path: str):
    """Load a pickled model."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded ← {path}")
    return model


# ─────────────────────────────────────────────
# 5. Master Training Pipeline
# ─────────────────────────────────────────────

def train_all_models(df: pd.DataFrame, feature_cols: list,
                     models_dir: str = "models/",
                     train_ratio: float = 0.80) -> dict:
    """
    Train all models and return a results dictionary.

    Returns
    -------
    {
        "splits": (X_train, X_test, y_train, y_test, train_df, test_df),
        "scaler": StandardScaler,
        "models": {
            "LinearRegression": {...},
            "RandomForest": {...},
            "XGBoost": {...},
            "ARIMA": {...},
            "LSTM": {...},
        }
    }
    """
    X_train, X_test, y_train, y_test, train_df, test_df = time_based_split(
        df, feature_cols, train_ratio=train_ratio
    )
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    results = {
        "splits": (X_train_sc, X_test_sc, y_train, y_test, train_df, test_df),
        "scaler": scaler,
        "models": {},
    }

    # ── Linear Regression ──────────────────────
    lr = train_linear_regression(X_train_sc, y_train)
    results["models"]["LinearRegression"] = {
        "model": lr,
        "y_pred": lr.predict(X_test_sc),
    }
    save_model(lr, os.path.join(models_dir, "linear_regression.pkl"))

    # ── Random Forest ──────────────────────────
    rf = train_random_forest(X_train_sc, y_train)
    results["models"]["RandomForest"] = {
        "model": rf,
        "y_pred": rf.predict(X_test_sc),
        "feature_importance": pd.Series(rf.feature_importances_, index=feature_cols),
    }
    save_model(rf, os.path.join(models_dir, "random_forest.pkl"))

    # ── XGBoost ────────────────────────────────
    xgb_model = train_xgboost(X_train_sc, y_train)
    results["models"]["XGBoost"] = {
        "model": xgb_model,
        "y_pred": xgb_model.predict(X_test_sc),
        "feature_importance": pd.Series(
            xgb_model.feature_importances_, index=feature_cols
        ),
    }
    save_model(xgb_model, os.path.join(models_dir, "xgboost.pkl"))

    # ── ARIMA ──────────────────────────────────
    train_close = train_df["Close"]
    arima_result = train_arima(train_close)
    arima_pred = predict_arima_steps(arima_result, n_steps=len(test_df))
    results["models"]["ARIMA"] = {
        "model": arima_result,
        "y_pred": arima_pred,
    }

    # ── LSTM ───────────────────────────────────
    lstm_model, lstm_history = train_lstm(X_train_sc, y_train, X_test_sc, y_test)
    results["models"]["LSTM"] = {
        "model": lstm_model,
        "history": lstm_history,
        "y_pred": predict_lstm(lstm_model, X_test_sc),
    }

    logger.info("All models trained successfully.")
    return results
