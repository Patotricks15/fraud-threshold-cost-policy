.PHONY: install test notebook reproduce

install:
	pip install -e ".[dev]"

test:
	pytest

notebook:
	jupytext --to ipynb notebooks/fd05_threshold_policy.py
	jupyter nbconvert --to notebook --execute --inplace notebooks/fd05_threshold_policy.ipynb

reproduce: test notebook
