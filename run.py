"""
run.py — Train model, evaluate ROC-AUC, and log experiment results.

Runtime budget: 5 minutes per experiment. Exceeding this logs a TIMEOUT.

Usage:
    python run.py "baseline"
"""

import sys
import os
import json
import signal
import datetime
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from model import build_model

LOG_FILE       = "experiments.json"
RUNTIME_BUDGET = 5 * 60  # 5 minutes in seconds


# ── Timeout handling (Unix) ──────────────────────────────────────────────────
class TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutError("Runtime budget of 5 minutes exceeded.")

def set_timeout(seconds):
    """Register SIGALRM timeout (Unix only). No-op on Windows."""
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(seconds)

def clear_timeout():
    if hasattr(signal, "SIGALRM"):
        signal.alarm(0)
# ─────────────────────────────────────────────────────────────────────────────


def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    return []


def save_log(log):
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


def main():
    description = sys.argv[1] if len(sys.argv) > 1 else "unnamed"

    print(f"\n{'='*60}")
    print(f"Experiment : {description}")
    print(f"Budget     : {RUNTIME_BUDGET // 60} minutes")
    print(f"{'='*60}")

    log = load_log()
    start_time = time.time()
    set_timeout(RUNTIME_BUDGET)

    try:
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
        
      # 4. Train
        print("Training model...")
        model = build_model()
        model.fit(X_train, y_train)

        # 5. Evaluate
        train_proba = model.predict_proba(X_train)[:, 1]
        val_proba   = model.predict_proba(X_test)[:, 1]

        thr = 0.5
        train_auc = roc_auc_score(y_train, train_proba>thr)
        val_auc   = roc_auc_score(y_test,   val_proba>thr)

        elapsed = time.time() - start_time
        clear_timeout()

        print(f"\nTrain ROC-AUC : {train_auc:.4f}")
        print(f"Val   ROC-AUC : {val_auc:.4f}")
        print(f"Runtime       : {elapsed:.1f}s / {RUNTIME_BUDGET}s budget")

        # 6. Compare & log
        prev_best = max(
            (e["val_roc_auc"] for e in log if not e.get("timeout") and e["val_roc_auc"] is not None),
            default=0
        )
        kept   = bool(val_auc >= prev_best)
        status = "KEPT ✓" if kept else "DISCARDED ✗"

        print(f"Previous best : {prev_best:.4f}")
        print(f"Status        : {status}")

        entry = {
            "timestamp":     datetime.datetime.now().isoformat(),
            "description":   description,
            "train_roc_auc": round(train_auc, 4),
            "val_roc_auc":   round(val_auc, 4),
            "runtime_s":     round(elapsed, 1),
            "kept":          kept,
            "timeout":       False,
        }

    except TimeoutError:
        clear_timeout()
        elapsed = time.time() - start_time
        print(f"\n⚠ TIMEOUT: experiment exceeded {RUNTIME_BUDGET // 60}-minute budget "
              f"({elapsed:.0f}s elapsed). Change DISCARDED — revert model.py.")
        entry = {
            "timestamp":     datetime.datetime.now().isoformat(),
            "description":   description,
            "train_roc_auc": None,
            "val_roc_auc":   None,
            "runtime_s":     round(elapsed, 1),
            "kept":          False,
            "timeout":       True,
        }

    log.append(entry)
    save_log(log)
    print(f"\nResults saved to '{LOG_FILE}'")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
