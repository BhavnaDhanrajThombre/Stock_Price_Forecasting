"""
evaluation.py
-------------
Computes evaluation metrics, builds the comparison table, generates
Actual vs Predicted plots, residual plots, and a Buy/Sell signal engine.
"""

import os
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

PLOT_DIR = "outputs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. Core Metrics
# ─────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    model_name: str = "Model") -> dict:
    """
    Compute MAE, RMSE, MAPE for a single model.

    MAPE: Mean Absolute Percentage Error – scale-independent,
    interpretable as "on average we are X% off".
    """
    # Guard: remove NaN predictions (e.g. ARIMA length mismatch)
    mask = ~np.isnan(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    mae  = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) /
                          np.where(y_true_clean != 0, y_true_clean, np.nan))) * 100

    metrics = {"Model": model_name, "MAE": round(mae, 4),
               "RMSE": round(rmse, 4), "MAPE (%)": round(mape, 4)}
    logger.info(f"{model_name}: MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%")
    return metrics


def build_comparison_table(results: dict, y_test: np.ndarray) -> pd.DataFrame:
    """Build a sorted model comparison DataFrame."""
    rows = []
    for name, info in results["models"].items():
        pred = info.get("y_pred")
        if pred is not None:
            rows.append(compute_metrics(y_test, np.asarray(pred), name))
    df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    df.index += 1   # rank starts at 1
    return df


# ─────────────────────────────────────────────
# 2. Actual vs Predicted – Matplotlib
# ─────────────────────────────────────────────

def plot_actual_vs_predicted(test_df: pd.DataFrame, results: dict,
                             y_test: np.ndarray, save: bool = True) -> None:
    """Create a multi-panel Actual vs Predicted comparison figure."""
    model_names = [n for n in results["models"] if results["models"][n]["y_pred"] is not None]
    n = len(model_names)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    dates = test_df["Date"].values if "Date" in test_df.columns else np.arange(len(y_test))

    for ax, name in zip(axes, model_names):
        pred = np.asarray(results["models"][name]["y_pred"])
        ax.plot(dates, y_test, label="Actual",    color="#2196F3", linewidth=1.5)
        ax.plot(dates, pred,  label="Predicted",  color="#FF5722", linewidth=1.2,
                linestyle="--", alpha=0.85)
        m = compute_metrics(y_test, pred, name)
        ax.set_title(f"{name}  |  MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}"
                     f"  MAPE={m['MAPE (%)']:.2f}%", fontsize=11, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)
        ax.set_ylabel("Close Price", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle("Actual vs Predicted – All Models", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "actual_vs_predicted.png")
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {path}")
    plt.close()


# ─────────────────────────────────────────────
# 3. Residual Plots
# ─────────────────────────────────────────────

def plot_residuals(results: dict, y_test: np.ndarray, save: bool = True) -> None:
    """
    Residual distribution plot for each model.
    Ideal residuals are centered at zero with low spread.
    """
    model_names = list(results["models"].keys())
    fig, axes = plt.subplots(1, len(model_names),
                             figsize=(4 * len(model_names), 4))
    if len(model_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        pred = np.asarray(results["models"][name].get("y_pred", []))
        if len(pred) == 0 or np.all(np.isnan(pred)):
            ax.set_title(f"{name}\n(N/A)", fontsize=10)
            continue
        residuals = y_test[:len(pred)] - pred
        ax.hist(residuals, bins=30, color="#7E57C2", edgecolor="white", alpha=0.85)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
        ax.set_title(f"{name}\nResiduals", fontsize=10, fontweight="bold")
        ax.set_xlabel("Error (Actual − Predicted)")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Residual Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "residual_distributions.png")
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {path}")
    plt.close()


# ─────────────────────────────────────────────
# 4. Feature Importance
# ─────────────────────────────────────────────

def plot_feature_importance(results: dict, top_n: int = 20,
                            save: bool = True) -> None:
    """Bar charts of feature importance for tree-based models."""
    tree_models = ["RandomForest", "XGBoost"]
    for name in tree_models:
        if name not in results["models"]:
            continue
        fi = results["models"][name].get("feature_importance")
        if fi is None:
            continue
        top_fi = fi.nlargest(top_n)
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = sns.color_palette("viridis", len(top_fi))[::-1]
        top_fi[::-1].plot(kind="barh", ax=ax, color=colors, edgecolor="none")
        ax.set_title(f"{name} – Top {top_n} Feature Importances",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance Score")
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(PLOT_DIR, f"feature_importance_{name.lower()}.png")
        if save:
            plt.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved: {path}")
        plt.close()


# ─────────────────────────────────────────────
# 5. Interactive Plotly Chart
# ─────────────────────────────────────────────

def plotly_predictions(test_df: pd.DataFrame, results: dict,
                       y_test: np.ndarray, save: bool = True) -> go.Figure:
    """Interactive Plotly figure: actual + all model predictions."""
    fig = go.Figure()
    dates = test_df["Date"].values if "Date" in test_df.columns else np.arange(len(y_test))

    fig.add_trace(go.Scatter(
        x=dates, y=y_test,
        name="Actual", line=dict(color="#2196F3", width=2),
        mode="lines",
    ))

    colors = ["#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]
    for (name, info), color in zip(results["models"].items(), colors):
        pred = np.asarray(info.get("y_pred", []))
        if len(pred) == 0 or np.all(np.isnan(pred)):
            continue
        fig.add_trace(go.Scatter(
            x=dates[:len(pred)], y=pred,
            name=name, line=dict(color=color, width=1.5, dash="dash"),
            mode="lines", opacity=0.85,
        ))

    fig.update_layout(
        title="📈 Stock Price Forecast – Model Comparison",
        xaxis_title="Date",
        yaxis_title="Close Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )
    if save:
        path = os.path.join(PLOT_DIR, "interactive_predictions.html")
        fig.write_html(path)
        logger.info(f"Saved interactive chart: {path}")
    return fig


# ─────────────────────────────────────────────
# 6. Buy / Sell Signal Engine
# ─────────────────────────────────────────────

def generate_signals(test_df: pd.DataFrame, best_pred: np.ndarray,
                     threshold_pct: float = 0.005) -> pd.DataFrame:
    """
    Decision-support layer based on model forecasts.

    Logic
    -----
    predicted_return = (predicted_next_close − current_close) / current_close

    if predicted_return >  threshold_pct → BUY
    if predicted_return < -threshold_pct → SELL
    else                                 → HOLD

    threshold_pct (default 0.5%) filters noise; adjust to your risk appetite.

    ⚠ DISCLAIMER: These signals are purely model-based and do NOT constitute
    financial advice. Past performance does not guarantee future results.
    """
    df = test_df.copy().reset_index(drop=True)
    n = min(len(df), len(best_pred))
    df = df.iloc[:n].copy()
    df["Predicted_Close"] = best_pred[:n]
    df["Predicted_Return"] = (df["Predicted_Close"] - df["Close"]) / df["Close"]

    def _signal(r):
        if r > threshold_pct:
            return "BUY"
        elif r < -threshold_pct:
            return "SELL"
        return "HOLD"

    df["Signal"] = df["Predicted_Return"].apply(_signal)

    buy_count  = (df["Signal"] == "BUY").sum()
    sell_count = (df["Signal"] == "SELL").sum()
    hold_count = (df["Signal"] == "HOLD").sum()
    logger.info(f"Signals → BUY:{buy_count}  SELL:{sell_count}  HOLD:{hold_count}")
    return df


def plot_signals(signal_df: pd.DataFrame, save: bool = True) -> None:
    """Overlay Buy/Sell/Hold signals on a price chart."""
    fig, ax = plt.subplots(figsize=(14, 6))
    dates = signal_df["Date"] if "Date" in signal_df.columns else signal_df.index

    ax.plot(dates, signal_df["Close"], color="#37474F", linewidth=1.5,
            label="Close Price", zorder=2)
    ax.plot(dates, signal_df["Predicted_Close"], color="#0288D1",
            linewidth=1.2, linestyle="--", label="Predicted", alpha=0.8, zorder=2)

    buy_mask  = signal_df["Signal"] == "BUY"
    sell_mask = signal_df["Signal"] == "SELL"
    hold_mask = signal_df["Signal"] == "HOLD"

    ax.scatter(dates[buy_mask],  signal_df["Close"][buy_mask],
               marker="^", color="#4CAF50", s=60, label="BUY",  zorder=5)
    ax.scatter(dates[sell_mask], signal_df["Close"][sell_mask],
               marker="v", color="#F44336", s=60, label="SELL", zorder=5)
    ax.scatter(dates[hold_mask], signal_df["Close"][hold_mask],
               marker="o", color="#FFC107", s=20, label="HOLD", zorder=3, alpha=0.5)

    ax.set_title("Buy / Sell / Hold Signals", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()

    path = os.path.join(PLOT_DIR, "buy_sell_signals.png")
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {path}")
    plt.close()


# ─────────────────────────────────────────────
# 7. Report Writer
# ─────────────────────────────────────────────

def save_report(comparison_df: pd.DataFrame,
                path: str = "outputs/reports/model_comparison.csv") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    comparison_df.to_csv(path)
    logger.info(f"Model comparison saved: {path}")


# ─────────────────────────────────────────────
# 8. Master Evaluation Runner
# ─────────────────────────────────────────────

def evaluate_all(results: dict, best_model_name: str = "XGBoost") -> None:
    """Run full evaluation pipeline."""
    _, X_test, y_train, y_test, train_df, test_df = results["splits"]

    # Metrics table
    comp_df = build_comparison_table(results, y_test)
    print("\n" + "="*60)
    print("MODEL COMPARISON TABLE")
    print("="*60)
    print(comp_df.to_string())
    save_report(comp_df)

    # Plots
    plot_actual_vs_predicted(test_df, results, y_test)
    plot_residuals(results, y_test)
    plot_feature_importance(results)
    plotly_predictions(test_df, results, y_test)

    # Signals using best model
    if best_model_name in results["models"]:
        best_pred = np.asarray(results["models"][best_model_name]["y_pred"])
        sig_df = generate_signals(test_df, best_pred)
        plot_signals(sig_df)
        sig_df.to_csv("outputs/reports/signals.csv", index=False)
        logger.info("Signal report saved: outputs/reports/signals.csv")
