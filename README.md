# fraud-threshold-cost-policy (FD05)

**Question:** Which threshold minimizes expected loss under an alert budget?

## Data

- Input is `data/fd02_scores.csv`, exported by the FD02 repo (`fraud-cost-sensitive-xgboost`) from its Kaggle run. It holds validation and test rows only, never training rows.
- Columns: `split` (`val` or `test`), `Time`, `Amount`, `Class`, `score` (uncalibrated XGBoost fraud probability from FD02).
- The underlying data is Credit Card Fraud Detection, Kaggle `mlg-ulb/creditcardfraud` (real anonymized transactions, September 2013). The raw Kaggle file is not used here and is not downloaded to the workstation.
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

- **Calibration:** the uncalibrated FD02 score is wrapped in a score-passthrough estimator (`ScoreEstimator`, returns `[1 - s, s]`) and then `FrozenEstimator` + `CalibratedClassifierCV`. Calibrators are fitted on **validation rows only**. The method (`isotonic` or `sigmoid`) is chosen by validation Brier score only.
- **Threshold selection:** on validation only, over a grid of the unique calibrated probabilities plus the edges 0 and 1. The threshold minimizing `C(t)` is chosen among those with validation alert rate `<= 1%`. If no threshold meets the budget, the code raises an error.
- **Test report (once):** the chosen threshold is applied to the test window. Reported: expected cost, alert rate, confusion matrix, recall of frauds by amount and by count, and alert rate by 6-hour window (labeled by window start).
- **Naive comparison:** the same test metrics at the default threshold `t = 0.5` on calibrated probabilities, to show the effect of the policy.
- **Calibration check:** Brier score on test before and after calibration, with a reliability diagram.

## Results

Not yet produced. The FD02 export is not available on this workstation, so no numbers are reported. After the FD02 Kaggle run and the FD05 notebook run, fill in these tables from `reports/threshold_selection.csv` and `reports/policy_test_report.json`:

| Quantity (test window) | Policy (chosen `t`) | Naive (`t = 0.5`) |
|---|---|---|
| Threshold | pending | 0.5 |
| Expected cost | pending | pending |
| Alert rate | pending | pending |
| Recall by amount | pending | pending |
| Recall by count | pending | pending |

| Calibration (test) | Brier |
|---|---|
| Raw FD02 score | pending |
| Calibrated | pending |

## Decision

Pending the Kaggle run. The policy is adopted only if the test expected cost under the stated assumptions is lower than the naive `t = 0.5` cost and the alert rate stays within the budget.

## Limitations

- The cost parameters (`c_FP = 2.0` and `c_FN = Amount`) are assumed. Results change if the business team sets different costs.
- The 1% alert budget is a policy choice, not an optimum.
- The test window spans about two days. Threshold stability over time cannot be shown.
- Calibration uses validation rows only, but the calibration method is chosen by Brier score on those same rows. Isotonic regression can overfit a small validation set, which would make the chosen method look better than it is on test.
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
