# fraud-threshold-cost-policy (FD05)

**Question:** Which threshold minimizes expected loss under an alert budget?

## Portfolio

| ID | Repository | Focus |
|---|---|---|
| FD01 | [fraud-logistic-baseline](https://github.com/Patotricks15/fraud-logistic-baseline) | Logistic baseline |
| FD02 | [fraud-cost-sensitive-xgboost](https://github.com/Patotricks15/fraud-cost-sensitive-xgboost) | Cost-sensitive XGBoost |
| FD03 | [fraud-unsupervised-anomaly](https://github.com/Patotricks15/fraud-unsupervised-anomaly) | Isolation Forest and LOF |
| FD04 | [fraud-velocity-graph-features](https://github.com/Patotricks15/fraud-velocity-graph-features) | Velocity and graph features (blocked) |
| **FD05** | **fraud-threshold-cost-policy** (this repo) | Cost-based threshold policy |
| FD06 | [fraud-drift-monitoring](https://github.com/Patotricks15/fraud-drift-monitoring) | Drift monitoring |

## Data

- Input is `data/fd02_scores.csv`, exported by the FD02 repo (`fraud-cost-sensitive-xgboost`) from its Kaggle run. It holds validation and test rows only, never training rows.
- Columns: `split` (`val` or `test`), `Time`, `Amount`, `Class`, `score` (uncalibrated XGBoost fraud probability from FD02).
- The underlying data is Credit Card Fraud Detection, Kaggle `mlg-ulb/creditcardfraud` (real anonymized transactions, September 2013). The raw file was downloaded through the Kaggle API to a local folder outside the repos, at the user's explicit request. It is not committed. License per Kaggle API: DbCL-1.0. This repo uses only the FD02 score export.
- `Time` spans about two days, so monthly breakdowns are impossible. Alert rates are reported per 6-hour window instead.
- See [data/README.md](data/README.md) for how to copy the FD02 export into place.

## Cost assumptions

These assumptions were written before any threshold tuning code was run. They are policy inputs, not measured facts.

- **Cost of a missed fraud:** `c_FN` = the transaction `Amount` of that fraud (actual loss proxy, per row). Recovery, chargebacks and fees are ignored.
- **Cost of a false alert:** `c_FP` = a fixed review cost of **2.0 currency units** per alerted legitimate transaction. This value is an assumption to be replaced by the business team.
- **Alert budget:** at most **1%** of transactions in the window may be alerted. This is a policy choice.
- **Alert rule:** a transaction is alerted when its calibrated probability is greater than or equal to the threshold `t`.
- **Expected cost at threshold `t`:** `C(t) = FP(t) * c_FP + sum(Amount of frauds with score < t)`, where `FP(t)` is the number of legitimate transactions alerted at `t`.
- **Cost curve:** `C(t)` is plotted over the validation grid (see `reports/figures/`).

## Method

- **Calibration:** two calibrators are fitted directly on the uncalibrated FD02 score: isotonic regression (`IsotonicRegression`, increasing, clipped to [0, 1]) and sigmoid (logistic regression on the logit of the score). Both use **validation rows only**. The method is chosen by validation Brier score only. An earlier version wrapped the score in `CalibratedClassifierCV`, which inverted the mapping on the real scores (frauds got low probabilities). That version was replaced, and a regression test now checks the direction.
- **Threshold selection:** on validation only, over a grid of the unique calibrated probabilities plus the edges 0 and 1. The threshold minimizing `C(t)` is chosen among those with validation alert rate `<= 1%`. If no threshold meets the budget, the code raises an error.
- **Test report (once):** the chosen threshold is applied to the test window. Reported: expected cost, alert rate, confusion matrix, recall of frauds by amount and by count, and alert rate by 6-hour window (labeled by window start).
- **Naive comparison:** the same test metrics at the default threshold `t = 0.5` on calibrated probabilities, to show the effect of the policy.
- **Calibration check:** Brier score on test before and after calibration, with a reliability diagram.

## Results

Real run on the FD02 export (Kaggle data). Validation: 42,718 rows, 56 frauds. Test: 42,721 rows, 52 frauds, 6.0 h span.

Calibration chosen on validation: **isotonic** (validation Brier 0.000270 vs 0.000309 for sigmoid). Chosen threshold on validation: **t = 0.0789** (validation alert rate 0.22%, validation expected cost 234.16).

| Quantity (test window) | Policy (`t = 0.0789`) | Naive (`t = 0.5`) |
|---|---|---|
| Expected cost | **2,466.40** | **2,387.16** |
| Alert rate (budget 1%) | 0.20% | 0.11% |
| Alerts (FP) | 86 (47 FP) | 45 (7 FP) |
| Frauds caught (TP / FN) | 39 / 13 | 38 / 14 |
| Recall by amount | 0.615 | 0.615 |
| Recall by count | 0.750 | 0.731 |
| Missed fraud amount | 2,372.40 | 2,373.16 |

Brier on test: raw FD02 score 0.000453; calibrated (isotonic) 0.000394.

Test alert rate by window: one 6-hour window (starting at Time 151,200 s), 86 alerts out of 42,721 transactions.

## Decision

Under the decision rule above (adopt the policy only if its test expected cost is lower than the naive `t = 0.5` cost), **the policy is not adopted**. The validation-tuned threshold catches one more fraud (39 vs 38), but that fraud's amount is worth only about 0.76 in recovered loss, while the 40 extra false alerts add about 80 in review cost. The naive `t = 0.5` threshold is the better choice on this window under the stated costs. The naive threshold also stays within the 1% budget.

This result depends on the assumed costs. If a false alert costs more than about 2.0 per alert, the naive threshold wins by more; if a missed fraud is worth substantially more than its amount, the lower threshold may win.

## Limitations

- The cost parameters (`c_FP = 2.0` and `c_FN = Amount`) are assumed. Results change if the business team sets different costs.
- The 1% alert budget is a policy choice, not an optimum.
- The test window spans about six hours (one alert-rate window). Threshold stability over time cannot be shown.
- The calibration method is chosen by Brier score on the same validation rows used to calibrate. The Brier gap between methods is small, so the choice is not reliable on this data.
- Only 56 validation frauds and 52 test frauds exist. The threshold choice rests on very few positives, so the test difference (about 80 currency units) is within noise.
- Scores come from FD02 and inherit its features and any leakage it has. This repo does not re-check them.

## Reproduce

```bash
pip install -e ".[dev]"
pytest
# Copy the FD02 export into data/ (see data/README.md), or set FD02_SCORES_PATH.
export FD02_SCORES_PATH=/path/to/fd02_scores.csv   # optional if data/fd02_scores.csv exists
make notebook
```

Outputs are written to `reports/` (tables) and `reports/figures/` (figures). Both are git-ignored until a real run is committed deliberately:

- `reports/threshold_selection.csv`: validation grid (threshold, alert rate, FP, missed-fraud amount, expected cost, feasibility).
- `reports/policy_test_report.json`: chosen threshold, test metrics, naive comparison, Brier before and after calibration.
- `reports/figures/fd05_cost_curve_val.png`, `fd05_reliability_test.png`, `fd05_alert_rate_window_test.png`.

## Layout

```text
src/fraud_threshold_policy/   data validation, metrics, costs, calibration, policy, windows, plots
notebooks/                    jupytext source (.py) and paired notebook (.ipynb)
tests/                        hand-computed cost, validation-only selection, budget, calibration, windows
reports/                      generated tables (ignored) and figures (ignored)
data/                         FD02 export (ignored), with README
```
