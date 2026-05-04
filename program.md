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
1. Switch estimator to GradientBoostingClassifier or XGBClassifier
2. Add RSI (14-day) as a feature
3. Add lagged intraday returns (t-1, t-2, t-3 days)
4. Add day-of-week and month as calendar features
5. Try `class_weight="balanced"` to handle class imbalance
6. Add rolling correlation between stock return and oil return
7. Feature selection: drop low-importance features
8. Tune hyperparameters: increase `n_estimators`, adjust `max_depth`
9. Add MACD signal as a feature
10. Add Bollinger Band width as a volatility feature
