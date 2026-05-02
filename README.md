# Stock Price Forecasting & Analytics


## Problem Statement
Stock markets generate vast amounts of price and volume data daily. The challenge is to transform this raw data into actionable forecasts and decision-support signals. This project builds a complete ML pipeline that:

Cleans and analyses historical OHLCV stock data

Engineers meaningful technical and statistical features

Trains and compares multiple ML models (Linear Regression, Random Forest, XGBoost, ARIMA, LSTM)

Generates Buy / Sell / Hold signals based on predicted price direction

Presents everything through an interactive Streamlit dashboard


## Dataset
The project accepts any CSV with OHLCV columns. Column names are normalised automatically (case-insensitive, common aliases supported).

Minimum required columns:
Column Description Date Trading date (any parseable format)Open Opening price High Intraday high Low Intraday low Close Closing price Volume Trading volume

Optional: Ticker / Symbol / Stock for multi-stock datasets.

If no CSV is provided, the dashboard auto-generates realistic simulated data using Geometric Brownian Motion.


## Tech Stack

Category Library Data Processing pandas, numpy

Visualisation matplotlib, seaborn, plotly 

Machine Learning scikit-learn, xgboost 

Time Series statsmodels (ARIMA)

Deep Learning tensorflow (LSTM – optional)

Dashboard streamlit 

Statistics scipy


## Feature Engineering
Feature Group Features Created Lag Features Close(t-1) … Close(t-10) Simple MASMA-7, SMA-14, SMA-30 Exponential MAEMA-12, EMA-26 MomentumRSI(14), MACD(12,26,9), MACD Signal, MACD Histogram Volatility Bollinger Bands (Upper/Lower/Width/%B), Rolling Vol (7/14/30)Volume Volume MA-7, Volume Ratio Calendar Day of Week, Month, Quarter, Year, Is Monday, Is Friday Targets Next-Day Close (regression), Direction Up/Down (classification)


## Models Used
Model Type  

Linear Regression Baseline Fast, interpretable 

Random Forest Ensemble Handles non-linearity, robust to noise 

XGBoost Gradient Boosting Best performance on tabular dataARIMA(5,1,0)Classical TS Univariate benchmark LSTM Deep Learning Captures sequential patterns (optional)

Training strategy: Chronological 80/20 split. No random shuffling. 

Future data never leaks into training.


## Evaluation Metrics

Metric Formula Interpretation MAE mean :

(actual − predictedRMSE√mean((actual − predicted)²)Penalises large errors moreMAPEmean(error / actual))


## Sample Results

(Results will vary with your dataset. Below are representative numbers from demo data.)

Rank Model MAERMSEMAPE (%)1 

XGBoost 0.821.140.712

Random Forest 1.051.470.923

Linear Regression 2.313.102.044

ARIMA 3.454.823.21

Key insight: XGBoost consistently outperforms because financial data contains non-linear interactions between technical indicators that tree ensembles capture well.


## Business Signals

predicted_return = (predicted_next_close − current_close) / current_close

BUY  → predicted_return >  +0.5%

SELL → predicted_return < −0.5%

HOLD → otherwise


## Installation & Setup

1. Clone the repository

bashgit clone https://github.com/BhavnaDhanrajThombre/Stock_Price_Forecasting.git

cd stock_forecasting_project


2. Create virtual environment

bashpython -m venv venv

source venv/bin/activate          # Linux / macOS

venv\Scripts\activate             # Windows


3. Install dependencies

bashpip install -r requirements.txt


## How to Run

Option A – Streamlit Dashboard

bash streamlit run app.py

Opens at http://localhost:8501


Option B – Python Scripts directly

bash


## Preprocess + feature engineering

python src/data_preprocessing.py data/raw/stock_data.csv

python src/feature_engineering.py


## Full pipeline (edit paths inside)

python src/model_training.py

## Adding Your Own Data

Place your CSV in data/raw/ (e.g. data/raw/AAPL.csv)

Ensure it has at minimum: Date, Open, High, Low, Close, Volume

In the dashboard: select Upload CSV and upload your file

In notebooks: update RAW_PATH = '../data/raw/AAPL.csv'


## Key Files Reference

File Purpose src/data_preprocessing.

pyClean, validate, normalise raw CSV src/feature_engineering.

pyCreate 40+ technical & statistical features src/model_training.

pyTrain LR / RF / XGB / ARIMA / LSTM src/evaluation.

pyMetrics, plots, signal generation src/utils.

pyEDA charts, sample data generator app.

pyFull interactive Streamlit app



License
MIT License – free to use for educational and non-commercial purposes.
