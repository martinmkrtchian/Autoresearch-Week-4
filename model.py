import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


def build_dataset():
    """
    Loads and engineers features using yfinance.
    Returns final_df ready for modeling.
    """
    import yfinance as yf

    stocks = ["XOM", "CVX", "BP", "CAT", "SHEL", "COP", "CSUAY", "PBR", "ENB", "MITSY", "ITOCY"]

    data = []
    for stock in stocks:
        df = yf.download(stock, start="2018-01-01", end="2024-01-01").reset_index()
        df["Stock"] = stock
        df = df.droplevel(axis=1, level=1)
        data.append(df)
    df = pd.concat(data)

    # Feature engineering
    df["Return_1d"] = df["Close"].pct_change()
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["Volatility"] = df["Return_1d"].rolling(10).std()
    df["Intraday_return"] = df["Close"] / df["Open"] - 1
    df["Return_prev_5d"] = df["Close"].shift(1) / df["Close"].shift(5) - 1
    df["Return_prev_10d"] = df["Close"].shift(1) / df["Close"].shift(10) - 1
    df["Close_yesterday"] = df.groupby("Stock")["Close"].shift(1)

    # RSI-14
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # Calendar features
    df["DayOfWeek"] = pd.to_datetime(df["Date"]).dt.dayofweek
    df["Month"] = pd.to_datetime(df["Date"]).dt.month

    # Bollinger Band width (20-day)
    bb_ma = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_width"] = (2 * bb_std) / bb_ma

    # Oil price (WTI Crude)
    oil = yf.download("CL=F", start="2018-01-01", end="2024-01-01").reset_index()
    oil = oil.droplevel(axis=1, level=1)
    final_df = df.merge(oil, on="Date")
    final_df.rename(columns={
        "Close_x": "Close",
        "Open_x": "Open",
        "High_x": "High",
        "Low_x": "Low",
        "Volume_x": "Volume",
        "Close_y": "Oil_Close",
        "Open_y": "Oil_Open",
        "High_y": "Oil_High",
        "Low_y": "Oil_Low",
        "Volume_y": "Oil_volume"
    }, inplace=True)

    # Volume ratio: today's volume vs 20-day average
    final_df["Volume_ratio"] = final_df["Volume"] / final_df["Volume"].rolling(20).mean()

    # Rolling 10-day correlation between stock return and oil return
    oil_return = final_df["Oil_Close"].pct_change()
    final_df["Stock_Oil_Corr10"] = (
        final_df["Return_1d"].rolling(10).corr(oil_return)
    )

    # Lagged intraday returns (t-1, t-2, t-3)
    final_df["Intraday_lag1"] = final_df.groupby("Stock")["Intraday_return"].shift(1)
    final_df["Intraday_lag2"] = final_df.groupby("Stock")["Intraday_return"].shift(2)
    final_df["Intraday_lag3"] = final_df.groupby("Stock")["Intraday_return"].shift(3)

    # MACD signal (12-day EMA - 26-day EMA) and signal line (9-day EMA of MACD)
    ema12 = final_df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = final_df["Close"].ewm(span=26, adjust=False).mean()
    final_df["MACD"] = ema12 - ema26
    final_df["MACD_signal"] = final_df["MACD"].ewm(span=9, adjust=False).mean()
    final_df["MACD_hist"] = final_df["MACD"] - final_df["MACD_signal"]

    # Price-to-MA ratios: where is Close relative to its moving averages
    final_df["Close_to_MA5"] = final_df["Close"] / final_df["MA_5"] - 1
    final_df["Close_to_MA10"] = final_df["Close"] / final_df["MA_10"] - 1

    return final_df


def get_features_and_target(final_df: pd.DataFrame):
    """
    Takes the merged dataframe and returns X, y.
    Modify feature_cols during AutoResearch iterations.
    """
    df = final_df.copy()

    # Binary target: 1 if intraday return > 0, else 0
    df["Target"] = (df["Intraday_return"] > 0).astype(int)

    feature_cols = [
        "Return_1d", "MA_5", "MA_10", "Volatility",
        "Return_prev_5d", "Return_prev_10d", "Close_yesterday",
        "Oil_Close", "Oil_Open", "Oil_High", "Oil_Low", "Oil_volume",
        "Open", "High", "Low", "Volume",
        "RSI_14", "DayOfWeek", "Month", "BB_width", "Stock_Oil_Corr10",
        "Volume_ratio", "Intraday_lag1", "Intraday_lag2", "Intraday_lag3",
        "MACD", "MACD_signal", "MACD_hist",
        "Close_to_MA5", "Close_to_MA10",
    ]

    df = df.dropna(subset=feature_cols + ["Target"])

    X = df[feature_cols]
    y = df["Target"]

    return X, y


def build_model():
    """
    Returns the classifier to be used.
    Modify this function during AutoResearch iterations.
    """
    model = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    return model
