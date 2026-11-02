# Regenerate every measured result of the injection analysis (SYNTHETIC).
# `make results`          re-collect all raw runs, re-run every analysis, rebuild docs/RESULTS-injection.md
# `make results-offline`  re-run the analyses from the cached raw runs and rebuild docs/RESULTS-injection.md
# Without make: python scripts/reproduce_results.py [--offline]
PY ?= python

.PHONY: results results-offline

results:
	$(PY) scripts/reproduce_results.py

results-offline:
	$(PY) scripts/reproduce_results.py --offline
