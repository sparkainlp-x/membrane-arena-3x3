# Changelog

## Unreleased

- Added an opt-in Crystallize injection engine, `membrane_arena_3x3/crystallize_injection.py` (SYNTHETIC classical software). `MembraneArena3x3WithInjection` tries up to 3 variant proposals (smaller `dt`, softened mirror) on a rejected step, validates them with the unmodified `_validate`, and keeps a calibration ledger (Brier score, `100 (1 - 4 Brier)` points, 0.8 cost per variant). It uses its own seeded random stream, `max_attempts=0` is identical to stock, and the stock engine, guards, thresholds, breaker and CLI are unchanged.
- It does not cure the arena: in the measured sweep (seeds 42, 0, 1, 2, 3) every injected run still latched the breaker, after 9 to 12 steps instead of 8 or 9. See the README section "Crystallize injection (opt-in)" and `docs/injection_results.md`.
- Added `scripts/injection_sweep.py` and 15 tests (9 supplied contract tests, 6 added properties). 78 tests in total.

## 0.1.0 (unreleased)

- First version: a 3x3 lattice of 32-dimensional density-matrix cells, a read-only Yang Observer, a bounded Yin Calibrator with an optional sinusoid, transactional all-or-nothing updates, a circuit breaker and a command-line entry point. Started from a pasted specification and code, committed unchanged first.
- Changes to the pasted code: the `Generator` type annotation in `core.py` named a class that does not exist (`np.Generator`); the observer now works on read-only views; calibration actions log the lattice parameters in effect instead of zeros; the summary of a zero-step run has the full shape (the CLI raised `KeyError`); `run()` no longer resets the step counter while keeping the logs; `EngineConfig` validates its settings and rejects a sinusoid at or above the sampling limit; invalid CLI settings are usage errors; the `--quiet` help text is accurate; docstrings that misdescribed dephasing, coherence, the stability score and the calibrator were corrected.
- Not changed, documented under "Known limits" in the README: the dynamics, the guard thresholds and the default parameters. The default settings latch the breaker after about 8 steps.
- AGPL-3.0-only with the full license text, CITATION.cff, .zenodo.json.
- 63 tests; CI on Python 3.10 to 3.13.
