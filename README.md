# Membrane Arena 3x3 + Yin-Yang Shield

**SYNTHETIC classical software simulation.** This repository models a 3x3 lattice
of 32-dimensional density-matrix “holographic universe” cells, with two independent
shield membranes:

- **Yang Observer Shield** — read-only measurement and stability telemetry.
- **Yin Calibrator Shield** — bounded global feedback, including an optional
  7.83 Hz “Schumann-mode” modulation.

It is **not** a physical holographic-universe simulator, quantum computer,
consciousness model, medical device, or control system. All results are synthetic.

## Model

Each cell is a density matrix:

```text
rho[i,j] in C^32 x C^32
```

with `rho = rho^dagger`, `rho >= 0`, and `trace(rho) = 1`.

The nine cells are arranged as:

```text
A1: M11 M12 M13
A2: M21 M22 M23
A3: M31 M32 M33
```

The observer computes coherence, divergence, residual, purity, and entropy.
The calibrator may adjust global feedback within hard bounds, but only after a
transactional safety check.

## Stability target

The system aims for:

- low residual error,
- high purity,
- bounded—not zero—cross-cell divergence,
- no invalid density matrix,
- no partial update after a failed transaction.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
membrane-arena-3x3 --steps 200 --seed 42 --json results.json
```

With Schumann-mode calibration:

```bash
membrane-arena-3x3 --steps 200 --seed 42 \
  --schumann --schumann-hz 7.83 --schumann-amplitude 0.02 \
  --json results.json
```

## Tests

```bash
pytest -q
```

## Safety rules

- The Observer cannot mutate membrane states.
- The Calibrator has bounded authority and is logged for every action.
- Any failed residual, positivity, trace, Hermiticity, symmetry, or divergence check
  rejects the complete global update.
- The circuit breaker latches after repeated failures.

## License

AGPL-3.0-or-later.
