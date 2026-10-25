# Changelog

## 0.1.0 (unreleased)

- First version: a 3x3 lattice of 32-dimensional density-matrix cells, a read-only Yang Observer, a bounded Yin Calibrator with an optional sinusoid, transactional all-or-nothing updates, a circuit breaker and a command-line entry point. Started from a pasted specification and code, committed unchanged first.
- Changes to the pasted code: the `Generator` type annotation in `core.py` named a class that does not exist (`np.Generator`); the observer now works on read-only views; calibration actions log the lattice parameters in effect instead of zeros; the summary of a zero-step run has the full shape (the CLI raised `KeyError`); `run()` no longer resets the step counter while keeping the logs; `EngineConfig` validates its settings and rejects a sinusoid at or above the sampling limit; invalid CLI settings are usage errors; the `--quiet` help text is accurate; docstrings that misdescribed dephasing, coherence, the stability score and the calibrator were corrected.
- Not changed, documented under "Known limits" in the README: the dynamics, the guard thresholds and the default parameters. The default settings latch the breaker after about 8 steps.
- AGPL-3.0-only with the full license text, CITATION.cff, .zenodo.json.
- 63 tests; CI on Python 3.10 to 3.13.
