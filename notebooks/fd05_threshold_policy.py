# %% [markdown]
# # FD05: Cost-based threshold policy under an alert budget
# Source for the FD05 notebook. Convert with `jupytext --to ipynb` and run it after copying the FD02 export to `data/fd02_scores.csv`, or set `FD02_SCORES_PATH`.
# Input: FD02 validation and test scores (uncalibrated XGBoost probabilities). Costs and the 1% alert budget are stated in the README under "Cost assumptions".
# Calibration and threshold selection use validation rows only. The test window is reported once.

# %%
import json
from pathlib import Path

from fraud_threshold_policy import plots
from fraud_threshold_policy.calibration import calibrated_probabilities, select_calibration
from fraud_threshold_policy.costs import ALERT_BUDGET, FP_COST
from fraud_threshold_policy.data import AMOUNT, SCORE, SPLIT, TARGET, TIME, load_scores, resolve_scores_path
from fraud_threshold_policy.metrics import brier
from fraud_threshold_policy.policy import evaluate_test, select_threshold
from fraud_threshold_policy.windows import WINDOW_SECONDS, alert_rate_by_window

IN_NOTEBOOKS = Path.cwd().name == "notebooks"
FIG_DIR = Path("../reports/figures") if IN_NOTEBOOKS else Path("reports/figures")
REPORT_DIR = Path("../reports") if IN_NOTEBOOKS else Path("reports")
DATASET = "Credit Card Fraud (FD02 export)"
MODEL = "FD02 XGBoost"

# %%
df = load_scores(resolve_scores_path())
val = df[df[SPLIT] == "val"]
test = df[df[SPLIT] == "test"]
for name, part in (("val", val), ("test", test)):
    hours = (part[TIME].max() - part[TIME].min()) / 3600
    print(f"{name}: {len(part):,} rows, {int(part[TARGET].sum())} frauds, span {hours:.1f} h")

# %% [markdown]
# ## Cost assumptions (from README, before tuning)

# %%
print(f"c_FP per alerted legitimate transaction: {FP_COST}")
print(f"c_FN per missed fraud: its Amount")
print(f"alert budget (share of transactions alerted): {ALERT_BUDGET:.2%}")

# %% [markdown]
# ## Calibrate on validation only

# %%
calibrator, method, val_briers = select_calibration(val[SCORE], val[TARGET])
print(f"validation Brier by method: {val_briers}")
print(f"chosen calibration: {method}")

# %% [markdown]
# ## Choose the threshold on validation only

# %%
val_probs = calibrated_probabilities(calibrator, val[SCORE])
choice = select_threshold(val[TARGET], val[AMOUNT], val_probs, budget=ALERT_BUDGET, fp_cost=FP_COST)
choice.grid.to_csv(REPORT_DIR / "threshold_selection.csv", index=False)
print(f"chosen threshold: {choice.threshold:.6f}")
print(f"validation expected cost: {choice.expected_cost:,.2f}, alert rate: {choice.alert_rate:.4%}")

# %%
plots.plot_cost_curve(
    choice.grid,
    choice.threshold,
    f"Expected cost vs threshold, {DATASET}, validation split, {MODEL} calibrated ({method})",
    FIG_DIR / "fd05_cost_curve_val.png",
)

# %% [markdown]
# ## Test window (reported once)

# %%
test_raw = test[SCORE].to_numpy()
test_probs = calibrated_probabilities(calibrator, test_raw)
brier_raw = brier(test[TARGET], test_raw)
brier_cal = brier(test[TARGET], test_probs)
print(f"test Brier, raw score: {brier_raw:.6f}; calibrated ({method}): {brier_cal:.6f}")

results = evaluate_test(test[TARGET], test[AMOUNT], test_probs, choice.threshold, ALERT_BUDGET, FP_COST)
for label, metrics in results.items():
    print(label, metrics)

# %%
plots.plot_reliability(
    test[TARGET],
    {"raw FD02 score": test_raw, f"calibrated ({method})": test_probs},
    f"Reliability diagram, {DATASET}, test split, {MODEL}",
    FIG_DIR / "fd05_reliability_test.png",
)

# %%
windows = alert_rate_by_window(test[TIME], test_probs, choice.threshold, WINDOW_SECONDS)
print(windows)
plots.plot_alert_rate_by_window(
    windows,
    ALERT_BUDGET,
    f"Alert rate per 6-hour window, {DATASET}, test split, {MODEL} calibrated ({method})",
    FIG_DIR / "fd05_alert_rate_window_test.png",
)

# %%
report = {
    "model": f"{MODEL}, calibrated",
    "calibration_method": method,
    "calibration_validation_brier": val_briers,
    "brier_test_raw": brier_raw,
    "brier_test_calibrated": brier_cal,
    "chosen_threshold": choice.threshold,
    "validation_expected_cost": choice.expected_cost,
    "validation_alert_rate": choice.alert_rate,
    "test": results,
    "cost_assumptions": {
        "fp_cost": FP_COST,
        "missed_fraud_cost": "Amount",
        "alert_budget": ALERT_BUDGET,
    },
}
(REPORT_DIR / "policy_test_report.json").write_text(json.dumps(report, indent=2))
print(f"saved {REPORT_DIR / 'policy_test_report.json'}")
