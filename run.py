"""
run.py — Train model, evaluate ROC-AUC, and log experiment results.

Data split strategy:
  - Full dataset split time-aware into 80% TRAIN POOL and 20% HOLDOUT TEST.
  - The holdout test set is saved to disk and NEVER used during AutoResearch.
  - Within the 80% train pool, a further 80/20 time-aware split produces
    the TRAIN and VALIDATION sets used during the AutoResearch loop.
  - After all iterations, run evaluate_test.py to score the best model on
    the holdout test set.

Runtime budget: 5 minutes per experiment. Exceeding this logs a TIMEOUT.

Usage:
    python run.py "baseline"
    python run.py "added RSI feature"
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

from model import build_dataset, get_features_and_target, build_model

LOG_FILE       = "experiments.json"
HOLDOUT_FILE   = "holdout_test.pkl"
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


def prepare_splits(X, y):
    """
    Two-stage time-aware split:

      Full data (100%)
        └── Train pool  (first 80%)  ← used during AutoResearch
              ├── Train  (first 80% of train pool = 64% of full)
              └── Val    (last  20% of train pool = 16% of full)
        └── Holdout test (last  20%) ← saved once, never touched again

    Returns X_train, X_val, y_train, y_val.
    Saves holdout to HOLDOUT_FILE on first call; reloads on subsequent calls.
    """
    holdout_idx = int(len(X) * 0.80)
    X_pool, y_pool = X.iloc[:holdout_idx], y.iloc[:holdout_idx]
    X_test, y_test = X.iloc[holdout_idx:], y.iloc[holdout_idx:]

    # Save holdout once — never overwrite
    if not os.path.exists(HOLDOUT_FILE):
        pd.concat([X_test, y_test], axis=1).to_pickle(HOLDOUT_FILE)
        print(f"Holdout test set saved to '{HOLDOUT_FILE}' "
              f"({len(X_test)} rows). Do NOT use during AutoResearch.")
    else:
        print(f"Holdout test set already saved ({len(X_test)} rows held out).")

    # Val split within train pool
    val_idx = int(len(X_pool) * 0.80)
    X_train, X_val = X_pool.iloc[:val_idx], X_pool.iloc[val_idx:]
    y_train, y_val = y_pool.iloc[:val_idx], y_pool.iloc[val_idx:]

    return X_train, X_val, y_train, y_val


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
        # 1. Load data
        print("Loading data via yfinance...")
        final_df = build_dataset()

        # 2. Features & target
        X, y = get_features_and_target(final_df)
        print(f"Full dataset  : {X.shape[0]} rows | "
              f"Target balance: {y.mean():.3f} positive rate")

        # 3. Two-stage split
        X_train, X_val, y_train, y_val = prepare_splits(X, y)
        print(f"Train rows    : {len(X_train)} | "
              f"Val rows: {len(X_val)} | "
              f"Holdout rows: {len(X) - len(X_train) - len(X_val)}")

        # 4. Train
        print("Training model...")
        model = build_model()
        model.fit(X_train, y_train)

        # 5. Evaluate
        train_proba = model.predict_proba(X_train)[:, 1]
        val_proba   = model.predict_proba(X_val)[:, 1]

        train_auc = roc_auc_score(y_train, train_proba)
        val_auc   = roc_auc_score(y_val,   val_proba)

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
