📈 Stock Price Forecasting & Analytics

A production-style end-to-end ML project for predicting stock prices and generating trading signals.
Built as a Data Analyst internship portfolio project.


📌 Problem Statement
Stock markets generate vast amounts of price and volume data daily. The challenge is to transform this raw data into actionable forecasts and decision-support signals. This project builds a complete ML pipeline that:

Cleans and analyses historical OHLCV stock data
Engineers meaningful technical and statistical features
Trains and compares multiple ML models (Linear Regression, Random Forest, XGBoost, ARIMA, LSTM)
Generates Buy / Sell / Hold signals based on predicted price direction
Presents everything through an interactive Streamlit dashboard


📂 Project Structure
stock_forecasting_project/
│
├── data/
│   ├── raw/                    ← Place your CSV here
│   └── processed/              ← Auto-generated cleaned & feature datasets
│
├── notebooks/
│   ├── 01_eda.ipynb            ← Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb       ← Training, evaluation, signals
│
├── src/
│   ├── data_preprocessing.py   ← Cleaning, normalisation, validation
│   ├── feature_engineering.py  ← Technical indicators, lag & time features
│   ├── model_training.py       ← All model definitions and training pipelines
│   ├── evaluation.py           ← Metrics, plots, signal engine
│   └── utils.py                ← EDA visualisations, sample data generator
│
├── models/                     ← Serialised trained models (pickle)
├── outputs/
│   ├── plots/                  ← All saved charts (PNG + HTML)
│   └── reports/                ← model_comparison.csv, signals.csv
│
├── dashboard/
│   └── app.py                  ← Streamlit dashboard
│
├── requirements.txt
└── README.md

📊 Dataset
The project accepts any CSV with OHLCV columns. Column names are normalised automatically (case-insensitive, common aliases supported).
Minimum required columns:
ColumnDescriptionDateTrading date (any parseable format)OpenOpening priceHighIntraday highLowIntraday lowCloseClosing priceVolumeTrading volume
Optional: Ticker / Symbol / Stock for multi-stock datasets.
If no CSV is provided, the dashboard auto-generates realistic simulated data using Geometric Brownian Motion.

🧰 Tech Stack
CategoryLibraryData Processingpandas, numpyVisualisationmatplotlib, seaborn, plotlyMachine Learningscikit-learn, xgboostTime Seriesstatsmodels (ARIMA)Deep Learningtensorflow (LSTM – optional)DashboardstreamlitStatisticsscipy

⚙️ Feature Engineering
Feature GroupFeatures CreatedLag FeaturesClose(t-1) … Close(t-10)Simple MASMA-7, SMA-14, SMA-30Exponential MAEMA-12, EMA-26MomentumRSI(14), MACD(12,26,9), MACD Signal, MACD HistogramVolatilityBollinger Bands (Upper/Lower/Width/%B), Rolling Vol (7/14/30)VolumeVolume MA-7, Volume RatioCalendarDay of Week, Month, Quarter, Year, IsMonday, IsFridayTargetsNext-Day Close (regression), Direction Up/Down (classification)

🤖 Models Used
ModelTypeNotesLinear RegressionBaselineFast, interpretableRandom ForestEnsembleHandles non-linearity, robust to noiseXGBoostGradient BoostingBest performance on tabular dataARIMA(5,1,0)Classical TSUnivariate benchmarkLSTMDeep LearningCaptures sequential patterns (optional)
Training strategy: Chronological 80/20 split. No random shuffling. Future data never leaks into training.

📏 Evaluation Metrics
MetricFormulaInterpretationMAEmean(actual − predictedRMSE√mean((actual − predicted)²)Penalises large errors moreMAPEmean(error / actual

📈 Sample Results

(Results will vary with your dataset. Below are representative numbers from demo data.)

RankModelMAERMSEMAPE (%)1XGBoost0.821.140.712Random Forest1.051.470.923Linear Regression2.313.102.044ARIMA3.454.823.21
Key insight: XGBoost consistently outperforms because financial data contains non-linear interactions between technical indicators that tree ensembles capture well.

💡 Business Signals
predicted_return = (predicted_next_close − current_close) / current_close

BUY  → predicted_return >  +0.5%
SELL → predicted_return < −0.5%
HOLD → otherwise
⚠️ Disclaimer: Signals are model-generated for educational purposes only. They do not constitute financial advice.

🖥️ Screenshots
SectionDescriptionShow ImageHistorical price + volumeShow ImageMoving averages & volatilityShow ImageModel forecasts vs actualShow ImageBuy/Sell/Hold signal overlay
(Screenshots auto-generated after running the notebooks.)

🚀 Installation & Setup
1. Clone the repository
bashgit clone https://github.com/yourusername/stock_forecasting_project.git
cd stock_forecasting_project
2. Create virtual environment
bashpython -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows
3. Install dependencies
bashpip install -r requirements.txt

TensorFlow is optional. If installation fails, remove it from requirements.txt – the LSTM model will be skipped gracefully.


▶️ How to Run
Option A – Jupyter Notebooks (recommended for learning)
bashjupyter notebook
Run notebooks in order:

notebooks/01_eda.ipynb
notebooks/02_feature_engineering.ipynb
notebooks/03_modeling.ipynb

Option B – Streamlit Dashboard
bashstreamlit run dashboard/app.py
Opens at http://localhost:8501
Option C – Python Scripts directly
bash# Preprocess + feature engineering
python src/data_preprocessing.py data/raw/stock_data.csv
python src/feature_engineering.py

# Full pipeline (edit paths inside)
python src/model_training.py

📁 Adding Your Own Data

Place your CSV in data/raw/ (e.g. data/raw/AAPL.csv)
Ensure it has at minimum: Date, Open, High, Low, Close, Volume
In the dashboard: select Upload CSV and upload your file
In notebooks: update RAW_PATH = '../data/raw/AAPL.csv'


🗂️ Key Files Reference
FilePurposesrc/data_preprocessing.pyClean, validate, normalise raw CSVsrc/feature_engineering.pyCreate 40+ technical & statistical featuressrc/model_training.pyTrain LR / RF / XGB / ARIMA / LSTMsrc/evaluation.pyMetrics, plots, signal generationsrc/utils.pyEDA charts, sample data generatordashboard/app.pyFull interactive Streamlit app

📬 Author
Built as a portfolio project demonstrating:

End-to-end ML pipeline design
Financial domain knowledge (technical indicators)
Production-grade code structure
Data storytelling and visualisation


📄 License
MIT License – free to use for educational and non-commercial purposes.
