# Scientific verification protocol

Status: implementation protocol and synthetic software benchmark. It is not a
physical-validation protocol. No atomic-addressing, zero-power memory, thousands
of independent modes, gravity, biological, medical, or consciousness claims are
made.

## What is being measured

The verification runner contains two distinct experiments:

1. **Engine ablations.** Each arm starts from a seeded identical 3x3 state.
   The `reference`, `no_mirror`, `no_coupling`, `no_dephasing`, and
   `exact_operators` arms are compared only at identical accepted simulation
   times. Time-series errors are not interpolated. The runner reports proposals,
   repairs, accepted time, guard causes, and a deterministic dense-matrix
   operation-count proxy. It reports no wall-clock claim.
2. **Synthetic wave-memory channel.** Four to sixteen channels carry finite
   signed integer symbols from `{-2,-1,0,1,2}`. Zero has one canonical encoding:
   exactly `0j`. Nonzero words use a fixed composition of magnitudes one and two,
   so inputs have matched total energy. Magnitude encodes amplitude and sign
   encodes phase. Decoding selects the Euclidean-nearest member of a finite,
   explicitly recorded codebook.

The channel is `y = Hx + n`. For crosstalk parameter `c` and channel count `N`,
`C_ii = 1`, `C_ij = c/(N-1)` for `i != j`, and
`H = diag(g_k exp(i phi_k)) C`; the realized gains and phases, input, noise,
received vector, finite codebook, and its hash are included in the raw rows.
For input energy `E` and configured
`SNR_dB`, independent circular complex Gaussian noise is generated with
`E[||n||^2] = E / 10^(SNR_dB/10)`; each real/imaginary component has variance
`E / (2 N 10^(SNR_dB/10))` for `N` channels. Message, drift, codebook, and
noise use independent `SeedSequence` streams listed in `manifest.json`.
Retention is phenomenological: the outcome is decoding reliability as delay
increases, not a physical T1/T2 measurement.

The four paired channel arms are:

- `no_intervention`: decode the raw channel output;
- `detector_only_cancellation_then_removal`: apply the known channel inverse
  only to the detector observation. The stored input is unchanged; this is an
  oracle benchmark, not an available physical controller;
- `coherent_reversible_manipulation`: apply a unitary transform and its inverse
  around channel compensation; it does not reset or erase arbitrary inputs;
- `dissipative_reset_reference`: replace every stored input with zero. This is
  a separately labeled destructive reference, not a unitary pulse.

Matched inputs, drift and noise are paired across arms. Conditioning, delay,
phase/gain drift, and decoding errors are retained in `raw_results.csv`.

## Retry and time semantics

The public defaults remain historical. The engine default is
`retry_policy=legacy_model_changing`, which retries with dt factors
`(0.5, 0.25, 0.125)`, mirror factors `(0.5, 0, 0)`, and coupling factors
`(0.5, 0.25, 0)`. This reproduces the existing acceptance behavior. Any retry
that changes mirror strength or coupling is a **model intervention**, not an
accuracy improvement. `retry_policy=dt_only` is an explicit alternative that
changes only dt and labels a retry as numerical backtracking.

Only committed updates advance `accepted_time`: a stock acceptance adds stock
dt, an accepted repair adds that candidate's dt, and rejection adds zero. The
default `drive_time_mode=legacy_step_index` preserves the old sinusoid phase
`attempt_index * nominal_dt`, including after rejection. Select
`drive_time_mode=accepted_time` to use accumulated committed time instead; this
is opt-in and has separate regression coverage. Every proposal logs its
effective dt, mirror, coupling, dephasing, drive time, decision, metrics (when
safe to compute), thresholds, all guard violations, and offending cells/pairs.
The step-level final reason is stored independently from the reason for each
proposal. Invalid candidates fail closed before observer eigendecompositions.
Both configured stock bounds and each proposal's effective parameters are
checked.

For historical legacy reproduction, retain `operator_mode=legacy`, fixed
per-step `mirror_strength`, `retry_policy=legacy_model_changing`,
`drive_time_mode=legacy_step_index`, the configured guards, and the original
seed stream (`numpy.default_rng(seed)`, shared by initialization and lattice
dynamics). Exact operator mode changes legacy Euler/dephasing behavior and is
not the historical model.

## Mirror reference dependency (PR #10)

PR #10 (`Add mirror flow reference implementation`) is open and is not merged.
This branch does not copy its `mirror_reference.py` implementation or silently
activate a rate-based mirror. The current legacy mirror is a fixed per-step
power filter, so its physical effect depends on dt. Once PR #10 is reviewed and
merged, a separate integration should opt into fixed rate
`q=exp(alpha*dt)` with `alpha=log1p(s)/reference_dt` for legacy matching at one
reference timestep. The PR's map is an **exact mirror-only map**, not an exact
combined solver or physical quantum channel.

Integration prerequisites are tested conditionally in
`tests/test_mirror_reference_integration.py`: legacy-strength matching,
composition, degenerate/rank-deficient states, finite-difference RHS and entropy
derivatives, entropy nonincrease, invalid-input rejection, and overflow/
roundoff behavior use the PR module if available. Without it these integration
tests are skipped with the dependency named; PR #10's own tests and the
integration contract have not been executed from this branch. Before claiming
convergence, separately measure dt dependence of unitary, dephasing,
interaction, calibration, projection, and their operator splitting.

## Experimental design and held-out discipline

`verification_configs/smoke.json` is a small development-only CI/manual
configuration. `verification_configs/full.json` assigns seeds 0–19 to
development, 20–49 to validation, and 50–99 to held-out evaluation. All arms
within a seed share the initialized state and paired wave-memory trials use the
same inputs, drift, and noise.

The full runner refuses an all-splits invocation. Run development and
validation first, archive their `manifest.json` files, then use the exact
configuration, protocol, and source-bundle SHA-256 values from those manifests
when invoking `--split held_out`. The held-out command fails if any of the
three hashes changed, including source code. Do not tune a protocol,
configuration, or implementation after this freeze, and do not tune on
held-out results. The output configuration and manifest record the exact input
hashes, source hashes, commit/dirty state, Python/NumPy versions, seed-stream
definitions, and complete rerun command.

Paired effects use seed-level paired means as the experimental unit; percentile
intervals are descriptive. A seed count alone does not establish independent
replicates or significance. Smoke outcomes are development-only. No p-values or
causal claims are generated.

## Separate validity questions

- **Numerical correctness:** density validity, transaction atomicity, attempt
  ledger completeness, accepted-time accounting, exact-seed reproduction, and
  analytical controls are code/test questions. Failure remains fail-closed.
- **Controller effectiveness:** accepted updates and guard causes describe this
  controller in this synthetic model only; they do not establish physical
  control.
- **Threshold sensitivity:** the smoke/full configurations hold guard
  thresholds fixed. No threshold tuning or robustness result is claimed here.
- **Physical validation:** requires an independently specified and validated
  physical model and relevant measurements. It is not done by this repository.

## Commands and artifacts

Smoke:

```bash
python scripts/scientific_verification.py \
  --config verification_configs/smoke.json \
  --out-dir verification_results/smoke
```

Full development and validation (opt-in; run each split separately):

```bash
python scripts/scientific_verification.py --config verification_configs/full.json --out-dir verification_results/full-development --split development
python scripts/scientific_verification.py --config verification_configs/full.json --out-dir verification_results/full-validation --split validation
```

Only after freezing all three hashes in the development/validation manifests:

```bash
python scripts/scientific_verification.py --config verification_configs/full.json --out-dir verification_results/full-held-out --split held_out --expected-config-sha256 <frozen-configuration-sha256> --expected-protocol-sha256 <frozen-protocol-sha256> --expected-source-sha256 <frozen-source-bundle-sha256>
```

Each invocation writes `configuration.json`, `manifest.json`,
`raw_results.csv`, `summary.json`, `report.md`, and `checksums.sha256`.
Numerical outputs contain no timestamp/runtime fields. Reproduction is compared
within declared NumPy tolerances; environment/version metadata and checksums
are provenance, not numerical equality criteria. No full benchmark, held-out
run, physical validation, or PR #10 integration is claimed until performed.

## Integration/merge prerequisites

1. Review and merge/reject PR #10 before a future rate-mirror integration; do
   not duplicate its implementation.
2. Review open PR #8's injection-scoring change and PR #7's injection-results
   document separately; neither is a dependency of this protocol.
3. Inspect smoke artifacts and repeat the deterministic numerical run in the
   declared supported Python/NumPy environment.
4. Review full development/validation results and freeze hashes before any
   held-out run.
5. Treat physical validation and any threshold/controller effectiveness claim
   as unresolved until separately preregistered and measured.
