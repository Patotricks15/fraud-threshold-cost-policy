# Data

Data is not committed. The input to this repo is an export from FD02, never the raw Kaggle file.

| Field | Value |
|---|---|
| Input file | `data/fd02_scores.csv` (or any path set in `FD02_SCORES_PATH`) |
| Produced by | FD02 repo `fraud-cost-sensitive-xgboost`, notebook run on Kaggle |
| Upstream dataset | Kaggle `mlg-ulb/creditcardfraud` (Database Open License), read only inside the FD02 Kaggle run |
| Retrieval | Local copy downloaded with the Kaggle API to a folder outside the repos, at the user's explicit request (not committed). The FD05 notebook reads the FD02 export copied into `data/` |

## How to get the input

1. Run the FD02 notebook (`FRAUD_DATA_PATH` set to the local copy of `creditcard.csv`, or the Kaggle path inside Kaggle).
2. Copy `fd02_scores.csv` from the FD02 repo's `data/` folder to `data/fd02_scores.csv` in this repo, or set `FD02_SCORES_PATH` to its location.

## Schema

| Column | Type | Meaning |
|---|---|---|
| `split` | text | `val` or `test` only. FD02 exports validation and test rows; training rows are not exported |
| `Time` | number | Seconds from the first transaction (about two days in total) |
| `Amount` | number | Transaction amount, used as the cost of a missed fraud |
| `Class` | 0/1 | 1 = fraud |
| `score` | number in [0, 1] | Uncalibrated FD02 XGBoost fraud probability |

The loader raises an error if a column is missing, a split value is unexpected, a score falls outside [0, 1], or either split is empty.
