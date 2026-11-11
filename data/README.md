# Data

Data is not committed. The input to this repo is an export from FD02, never the raw Kaggle file.

| Field | Value |
|---|---|
| Input file | `data/fd02_scores.csv` (or any path set in `FD02_SCORES_PATH`) |
| Produced by | FD02 repo `fraud-cost-sensitive-xgboost`, notebook run on Kaggle |
| Upstream dataset | Kaggle `mlg-ulb/creditcardfraud` (Database Open License), read only inside the FD02 Kaggle run |
| Retrieval | Not downloaded to the workstation. The FD02 export is copied in by hand |

## How to get the input

1. Run the FD02 notebook on Kaggle (attach `mlg-ulb/creditcardfraud`, Run All).
2. From the FD02 Kaggle notebook's Output tab, download `fd02_scores.csv`.
3. Copy it to `data/fd02_scores.csv` in this repo, or set `FD02_SCORES_PATH` to its location.

## Schema

| Column | Type | Meaning |
|---|---|---|
| `split` | text | `val` or `test` only. FD02 exports validation and test rows; training rows are not exported |
| `Time` | number | Seconds from the first transaction (about two days in total) |
| `Amount` | number | Transaction amount, used as the cost of a missed fraud |
| `Class` | 0/1 | 1 = fraud |
| `score` | number in [0, 1] | Uncalibrated FD02 XGBoost fraud probability |

The loader raises an error if a column is missing, a split value is unexpected, a score falls outside [0, 1], or either split is empty.
