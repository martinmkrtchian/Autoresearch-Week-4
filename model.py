import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

training_data = pd.read_csv('training_data.csv')

feature_cols = [
        "Return_prev_5d", "Return_prev_10d", "Close_yesterday", 'Volatility',
        "Oil_Open", "Oil_volume", 'Return_prev_5d_oil', 'Return_prev_10d_oil',
        "Open", "Volume",
        "DayOfWeek", "Month"
    ]

X = training_data[feature_cols]
y = training_data[['Target']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
