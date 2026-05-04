# AutoResearch: Energy Stock Intraday Return Classification

## Objective
Classify whether each day's **intraday return** for energy stocks will be **positive (1) or negative (0)**.
- Target: `Intraday_return = Close/Open - 1` → binary label (`1` if > 0, else `0`)
- Optimization metric: **Validation ROC-AUC**

## Tickers
XOM, CVX, BP, CAT, SHEL, COP, CSUAY, PBR, ENB, MITSY, ITOCY

## Data Source
- Fetched via `yfinance` (2018-01-01 to 2024-01-01)
- Oil prices (CL=F) merged as external feature

## Data Split Strategy
The dataset is split **once**, time-aware, before any AutoResearch begins:

```
Full data (100%)
  ├── Train pool (first 80%)   ← AutoResearch uses only this
  │     ├── Train (64% of full)
  │     └── Val   (16% of full)   ← ROC-AUC optimized here
  └── Holdout test (last 20%)  ← saved to holdout_test.pkl, NEVER touched
```

The holdout is written to disk on the first `run.py` call and must not be
used until all iterations are complete.

## Runtime Budget
Each experiment has a **5-minute wall-clock budget**. If exceeded:
- The run is logged as `timeout: true`
- The change is automatically discarded
- You must revert `model.py` to the previous version

## Files
| File | Purpose |
|------|---------|
| `model.py` | Data loading, feature engineering, model definition — **edited each iteration** |
| `run.py` | Trains model, evaluates ROC-AUC, enforces 5-min budget, logs to `experiments.json` |
| `prepare.py` | Reads `experiments.json`, generates `performance.png` |
| `evaluate_test.py` | Final step only — scores best model on holdout test set |

---

## AutoResearch Loop Instructions

Read `program.md` for your instructions, then read `model.py`.

Run `python run.py "baseline"` to establish the baseline ROC-AUC.

Then enter the AutoResearch loop:

1. Propose one modification to `model.py` (e.g., different estimator,
   feature engineering, hyperparameter change, class balancing, new features).
2. Edit `model.py` with your change.
3. Run: `python run.py "<short description of what you changed>"`
4. Compare the new `val_roc_auc` to the current best.
   - If improved: **KEEP** the change, note the new best.
   - If worse or TIMEOUT: **REVERT** `model.py` to the previous version.
5. Repeat from step 1. Try **at least 6 different ideas**.

After all iterations:
1. Run `python prepare.py` → generates `performance.png`
2. Run `python evaluate_test.py` → scores best model on holdout test set
3. Print a summary table of all experiments (kept / discarded / timeout)

---

## Suggested Ideas to Try
1)Switch estimator to GradientBoostingClassifier (after leakage fix)

2)Add safe lagged features: volatility_lag1, oil_return_1d, oil_return_5d

3)Add calendar features: Holiday flag, days since last earnings

4)Try class_weight="balanced" to handle class imbalance

5)Add safe rolling correlation: Between lagged stock and lagged oil returns only

6)Feature selection: Drop low-importance features (use permutation importance)

7)Hyperparameter tuning:

  n_estimators: [100, 200, 400]
  max_depth: [3, 5, 7]
  learning_rate: [0.01, 0.05, 0.1]



8)Price momentum indicator: (Close_t-1 / Close_t-5) - 1


9)Track leakage impact: Run baseline with cleaned features first, measure performance drop
