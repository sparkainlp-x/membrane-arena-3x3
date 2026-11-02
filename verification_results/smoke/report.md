# Scientific verification smoke/full run

All results are synthetic classical software measurements. This is not an NV hardware simulator, physical memory model, or quantum-channel claim.

- Profile/split: `smoke` / `all`
- Seeds: 2
- Engine ablation runs: 10
- Equal accepted-time trajectory comparisons: 7
- Wave-memory rows: 192
- Held-out executed: `False`; tuning: `none`

## Wave-memory outcomes

| intervention | trials | decoding errors | error rate |
|---|---:|---:|---:|
| coherent_reversible_manipulation | 48 | 0 | 0.000000 |
| detector_only_cancellation_then_removal | 48 | 0 | 0.000000 |
| dissipative_reset_reference | 48 | 48 | 1.000000 |
| no_intervention | 48 | 0 | 0.000000 |

### Delay/crosstalk strata

| intervention | delay | crosstalk | trials | error rate | mean condition number |
|---|---:|---:|---:|---:|---:|
| coherent_reversible_manipulation | 1 | 0.000 | 12 | 0.000000 | 1.050563 |
| coherent_reversible_manipulation | 1 | 0.350 | 12 | 0.000000 | 1.555220 |
| coherent_reversible_manipulation | 4 | 0.000 | 12 | 0.000000 | 1.097417 |
| coherent_reversible_manipulation | 4 | 0.350 | 12 | 0.000000 | 1.589013 |
| detector_only_cancellation_then_removal | 1 | 0.000 | 12 | 0.000000 | 1.050563 |
| detector_only_cancellation_then_removal | 1 | 0.350 | 12 | 0.000000 | 1.555220 |
| detector_only_cancellation_then_removal | 4 | 0.000 | 12 | 0.000000 | 1.097417 |
| detector_only_cancellation_then_removal | 4 | 0.350 | 12 | 0.000000 | 1.589013 |
| dissipative_reset_reference | 1 | 0.000 | 12 | 1.000000 | 1.050563 |
| dissipative_reset_reference | 1 | 0.350 | 12 | 1.000000 | 1.555220 |
| dissipative_reset_reference | 4 | 0.000 | 12 | 1.000000 | 1.097417 |
| dissipative_reset_reference | 4 | 0.350 | 12 | 1.000000 | 1.589013 |
| no_intervention | 1 | 0.000 | 12 | 0.000000 | 1.050563 |
| no_intervention | 1 | 0.350 | 12 | 0.000000 | 1.555220 |
| no_intervention | 4 | 0.000 | 12 | 0.000000 | 1.097417 |
| no_intervention | 4 | 0.350 | 12 | 0.000000 | 1.589013 |

The signed integer alphabet has canonical zero, and nonzero benchmark inputs have matched energy before the dissipative-reset reference. Noise variance is normalized so expected total noise power equals input energy divided by `10^(SNR_dB/10)`. Channel drift, crosstalk, conditioning, delay, and errors are reported in `raw_results.csv`.

Detector-side inverse cancellation is an oracle benchmark intervention; it changes only the decoded observation, not the stored input. The coherent arm applies a unitary and its inverse. The dissipative reset replaces every input by zero and is expected to erase nonzero messages; it does not model a unitary reset pulse.

## Engine outcomes and limitations

### Engine ablation outcomes

| arm | target reached / runs | mean accepted time | mean attempts | estimated matrix operations | guard causes |
|---|---:|---:|---:|---:|---|
| exact_operators | 2 / 2 | 0.100000 | 2.00 | 9216 | `{}` |
| no_coupling | 0 / 2 | 0.037500 | 21.50 | 99072 | `{"divergence_above_maximum": 41}` |
| no_dephasing | 0 / 2 | 0.025000 | 22.00 | 101376 | `{"divergence_above_maximum": 42}` |
| no_mirror | 2 / 2 | 0.100000 | 2.00 | 9216 | `{}` |
| reference | 0 / 2 | 0.037500 | 21.50 | 99072 | `{"divergence_above_maximum": 41}` |

### Equal-time trajectory differences

| arm | matched timepoints | mean error | maximum error |
|---|---:|---:|---:|
| exact_operators | 1 | 0.059550125 | 0.059550125 |
| no_coupling | 2 | 8.8587579e-05 | 0.0001147827 |
| no_dephasing | 1 | 0.026327503 | 0.026327503 |
| no_mirror | 1 | 0.026339227 | 0.026339227 |
| reference | 2 | 0 | 0 |

`raw_results.csv` reports accepted time, proposal/repair counts, guard causes, a deterministic matrix-operation cost proxy, and trajectory error only at matched accepted times. Missing common times are not interpolated. The operation count is a proxy, not measured wall time.

Seed-level pairing is descriptive; the seed count alone does not establish independence or statistical significance.
Paired effects and percentile intervals are descriptive at the seed-level experimental unit; no significance or independence claim is made. Smoke results are development-only and not confirmatory.
This artifact is a smoke run only; full development, validation, held-out, and physical-validation runs were not run.

The stock retry policy remains the historical model-changing policy. A `dt_only` option is available for numerical backtracking; mirror and coupling changes are model interventions, not accuracy improvements. The default drive retains legacy `step_index * nominal_dt` phase; `accepted_time` is an explicit alternate.

PR #10 remains an open dependency. No mirror-reference implementation was copied or wired into this branch. Its proposed map is an exact mirror-only map, not an exact combined solver or physical quantum channel. Other operators still require timestep-convergence analysis.

Historical reproduction uses `retry_policy=legacy_model_changing`, `drive_time_mode=legacy_step_index`, fixed per-step mirror strength, the stock `operator_mode=legacy`, and the default seed stream `numpy.default_rng(seed)`. These settings are recorded in `configuration.json`; changing them does not reproduce historical runs.

## Reproduction and hashes

```bash
/usr/bin/python scripts/scientific_verification.py --config verification_configs/smoke.json --out-dir verification_results/smoke --split all
```

- Configuration SHA-256: `5447b0b47f68b3b89bd3d8bbb1bd6118a9cad890bd798e0e6fe1ae8db9123322`
- Protocol SHA-256: `bc029d1bb48b38edbe32972adb06cba6210e4ef0f1a4d3aba1b7f485f3c2db40`

Runtime and timestamps are intentionally absent; deterministic numerical comparisons exclude runtime/environment metadata and use the exact raw seed/configuration. See `manifest.json` and `checksums.sha256`.

## Scope / unresolved validation

This run does not establish controller effectiveness beyond the stated toy outcomes, threshold robustness, physical validity, atomic addressing, zero-power memory, thousands of independent modes, gravity, biology, or consciousness. Numerical validity failures remain fail-closed. The mirror integration and hardware validation remain unrun.
