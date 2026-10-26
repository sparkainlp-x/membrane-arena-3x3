# Pre-registered analysis protocol: Crystallize injection (SYNTHETIC)

Status: written and committed before any confirmatory run. SYNTHETIC classical
NumPy simulation; nothing here is a physical claim. Author: Jean-François
Brisson (sole author).

This protocol fixes the questions, data, metrics, tests and decision rules for
the comparison of the stock arena with the opt-in injection
(`MembraneArena3x3WithInjection`) and its commit policies. The results go to
`docs/RESULTS-injection.md`. Anything done differently from this text is listed
under "Deviations" in that results file.

## 1. Scope and what is not tested

- Tested: the merged injection engine at commit 70dbad6 (main after PR #1),
  with the per-attempt ledger as merged.
- Not tested: any change to the published model, the guards, the thresholds or
  the breaker. The injection never changes them, and a property test pins that.
- The commit probability `p` changes only the ledger, never whether a step is
  accepted. The dynamics of a run are therefore identical under every commit
  policy, and the policies are compared by re-scoring one set of runs (section 4).
  A test that the re-scoring equals the engine's own ledger ships with the
  analysis code.

## 2. Regimes

| regime | coupling | dephasing | mirror_strength | dt | first_factor | mirror_variant |
|---|---|---|---|---|---|---|
| stock | 0.02 | 0.5 | 0.10 | 0.05 | 0.5 | 0.0 |
| intermediate | 0.02695 | 0.39355 | 0.13625 | 0.03140 | 0.37775 | 0.0 |
| evolved | 0.0339 | 0.2871 | 0.1725 | 0.0128 | 0.2555 | 0.0 |

- "stock" is the engine default. "evolved" is the genome supplied with the
  design note and is used as given; it was not re-evolved.
- "intermediate" is the arithmetic midpoint of the other two, fixed by
  definition (`experiments/regimes.py`), not tuned.
- Every run requests 200 steps. The breaker normally ends a run well before that.

Pilot (seeds 900 to 909, disjoint from every set below, run only to check that
the regimes run and are not degenerate; `experiments/pilot_intermediate_regime.py`):

| regime | mean steps | mean accepted | injection pass rate | breaker open |
|---|---|---|---|---|
| stock | 10.7 | 5.7 | 24/74 = 0.32 | 10/10 |
| intermediate | 15.8 | 10.7 | 67/118 = 0.57 | 10/10 |
| evolved | 64.0 | 58.8 | 561/613 = 0.92 | 10/10 |

The pilot cannot change any regime or hypothesis below.

## 3. Seeds

| set | seeds | use |
|---|---|---|
| dev | 1000 to 1099 (100) | code checks and the dev-tuned fixed commit; never used for a confirmatory test |
| confirm | 2000 to 2099 (100) | all confirmatory tests, run once |
| tuning | 42, 0, 1, 2, 3 | the seeds on which the fixed 0.35 default was chosen; reported separately and labelled in-sample |

Dev and confirm are disjoint from each other, from the tuning seeds and from
the pilot. 200 seeds are used per regime in total.

## 4. Policies (commit rules for the first variant of an event)

An event is one rejected step on which the injection ran. Its outcome is 1 if
any variant passed, else 0. Revisions inside an event follow the merged engine
(halfway toward 0.5 after a miss).

- F35: fixed 0.35 (the merged default).
- F75: fixed 0.75.
- A35: adaptive, `(passes + 4 * 0.35) / (events + 4)` over earlier events of the run.
- A75: adaptive from 0.75, same smoothing.
- FT: fixed at the pooled injection pass rate of the dev set for that regime,
  rounded to two decimals and clipped to [0.01, 0.99]. This is the honest
  tuned-in-advance comparator: its value comes from dev seeds only.

## 5. Metrics

Per run (the unit of analysis):

- steps executed, accepted steps, number of events, number of passes, whether
  the breaker was open at the end, and the breaker's rejection reasons.
- Ledger score of each policy, as merged (sum over every variant tried of
  `100 * (1 - 4 * (p - truth)^2)`; cost is tracked separately and not subtracted).
- Mean first-forecast Brier: over the events of the run, the mean of
  `(p_first - outcome)^2`, where `p_first` is the policy's commit for the first
  variant of the event. Runs with zero events are excluded from Brier
  comparisons and their count is reported.

Pooled over the confirm set, per regime and policy:

- Expected calibration error (ECE) of the first forecasts, 5 equal-width bins on
  [0, 1], count-weighted. Reliability tables (bin mean forecast, observed rate,
  count). Confidence intervals by cluster bootstrap over runs.

## 6. Confirmatory hypotheses (family of 12)

All on the confirm seeds, paired by seed, two-sided tests.

| id | regime | comparison | metric | prior expectation |
|---|---|---|---|---|
| C1 to C3 | stock, intermediate, evolved | injection minus no injection | accepted steps | positive |
| C4 to C6 | stock, intermediate, evolved | F35 minus F75 | mean first-forecast Brier | negative in stock, positive in evolved, small in intermediate |
| C7 to C9 | stock, intermediate, evolved | A35 minus F35 | mean first-forecast Brier | positive or none in stock, negative in intermediate and evolved |
| C10 to C12 | stock, intermediate, evolved | A35 minus FT | mean first-forecast Brier | positive or none: adaptive has a burn-in cost, FT knows the rate |

The prior expectations are recorded so that a surprise is visible. They do not
change the two-sided tests.

Test and error control:

- Paired sign-flip permutation test on the per-seed differences, 100,000
  random sign flips, generator seed 20261026; p = (hits + 1) / (100,000 + 1).
- Holm step-down correction over the 12 p-values, family-wise level 0.05.
- Effect sizes: the mean paired difference and the standardised paired effect
  dz (mean difference over the standard deviation of the differences), each with
  a 95% percentile bootstrap interval, 10,000 resamples of seeds, generator seed
  20261025. The intervals are not adjusted for multiplicity and are for
  magnitude, not for the decision.

Descriptive, not tested: the fraction of runs ending with the breaker open in
each regime (Wilson 95% interval), steps executed, and the first rejection
reasons. The claim "the injection does not cure the breaker" is judged from
this fraction, not from a test.

## 7. Stopping rule

Fixed sample: 100 confirm seeds per regime, one analysis, no interim looks, no
optional stopping, no extra seeds added after seeing results. If a bug is found
after the confirmatory run, it is fixed, the whole run is repeated on the same
seeds, and both outcomes are reported under "Deviations".

## 8. Secondary and exploratory analyses (no multiplicity control, labelled as such)

- The tuning seeds: ledger scores of all policies, in-sample.
- Ledger scores of each policy on the confirm seeds, both under the merged
  per-attempt scoring and under the two-forecast scoring used by the design
  note (average of the first and last commit of an event, minus 0.8 per variant).
- Sensitivity of the adaptive rule to its pseudo-count prior (0, 1, 4, 16;
  prior 0 uses `first_commit` only while there are no events).
- Negative controls: outcomes replaced by fair coin flips (a fixed `p` has
  expected points `-400 * (p - 0.5)^2`, at most 0); outcomes shuffled across
  events within a regime (this keeps the base rate, so it shows how much of a
  score is base-rate calibration and not event-level skill).
- The true phase damping experiment, the term-by-term attribution of residual
  growth and the reconciliation with the design note are separate scripts with
  their own descriptions; they are not part of the confirmatory family.

## 9. Known limits fixed in advance

- All regimes are synthetic parameter sets of one toy model. Nothing here says
  anything about a physical system.
- The forecasts do not use event features, so the ledger measures calibration
  to a base rate. It does not measure event-level discrimination.
- The pass rate depends on the regime, so a fixed default is right only near
  its own regime's rate.
- Results are for this NumPy and these seeds on one machine.

---

## Amendment A1 (erratum), dated 2026-10-25 (America/Toronto), written after the confirmatory run

The text above is unchanged. This section records what changed afterwards and
how both versions are reported.

**What the pre-registration tested.** Section 1 names the engine at commit
70dbad6 "with the per-attempt ledger as merged", and section 5 defines the ledger
score "as merged": the sum over every variant tried of `100 * (1 - 4 * (p -
truth)^2)`, cost not subtracted. The confirmatory run and its outputs
(experiments/results at commit c54d54b) were produced with that engine. Those
files are kept unchanged in `experiments/results/pre_alignment/`.

**What changed.** After the confirmatory run the author supplied the original
`crystallize_injection.py`. The merged module was aligned to it on two points
only, in the pull request that carries this amendment:

1. *Scoring.* The engine now scores exactly two forecasts per event: event score
   = 0.5 points(first commit) + 0.5 points(last commit) - 0.8 per variant tried,
   with truth = whether any variant passed (the author's original). It no longer
   scores every variant tried.
2. *Random stream.* The jitter stream is `default_rng(seed + 10000007)` with one
   `uniform(0.8, 1.2)` draw per attempt and a variant factor of `min(factor *
   jitter, 0.99)`. The merged engine used `default_rng([seed, salt])`. This
   changes the jitter values, hence which variants pass, hence the run
   dynamics (steps, accepted steps, events). It does not change the guards,
   thresholds, breaker or any hypothesis.

The commit-policy rules (section 4), the regimes, the seed sets, the confirmatory
family C1 to C12, the tests, the Holm correction and the bootstrap settings are
unchanged.

**What was re-run.** In line with section 7 ("the whole run is repeated on the
same seeds, and both outcomes are reported"), the dev, confirm and tuning sets
were collected again with the aligned engine and the same analysis code, and the
analysis was run once more. Outputs: `experiments/results/prereg_results.json`,
`prereg_tables.md`, `event_structure.md`, `runs_*.jsonl`. The side-by-side
comparison of the two analyses, test by test, is
`experiments/results/alignment_comparison.md`
(generated by `experiments/compare_alignment.py`).

**Which ledger scoring is primary.** The confirmatory family C1 to C12 uses
accepted steps and the mean first-forecast Brier score, which do not depend on
the ledger scoring rule, so no confirmatory decision depends on it. The ledger
score is a descriptive and exploratory metric (sections 5 and 8). It is reported
under both scorings, side by side: per-attempt (the original merged version) and
two-forecast (the author's original). The primary descriptive ledger is the
**two-forecast scoring**. This is confirmed in the following limited sense: on
2026-10-25 the author instructed that the engine score exactly two forecasts per
event, as the original module does, and not every variant tried. That
instruction is the basis for this choice. It is not an approval of the text of
this amendment, which the author has not reviewed. (An earlier draft of this
amendment recorded the choice as "pending the author's confirmation"; that
wording was replaced before this amendment was merged.)

**What was not changed.** No hypothesis, metric definition, seed, test or
decision rule was altered. The pre-alignment results were not edited or removed.
The results file named in the introduction, `docs/RESULTS-injection.md`, is now
written, generated by `experiments/build_results.py` from the files listed
above (`make results` regenerates every number in it); see
[RESULTS-injection.md](RESULTS-injection.md).

**Outcome of the comparison** (all numbers are in
`experiments/results/alignment_comparison.md`): the direction and the Holm
decision of all 12 confirmatory tests are the same before and after
alignment; the mean differences change by at most 0.24 accepted steps (C1 to C3) and at most
0.0025 in the first-forecast Brier difference (C4 to C12). This is a statement
about these seeds and this toy model only.
