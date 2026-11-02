"""Small synthetic wave-memory channel benchmark; not a hardware simulator."""

from __future__ import annotations

import hashlib
import json

import numpy as np

ALPHABET = (-2, -1, 0, 1, 2)


def _complex_pairs(values: np.ndarray) -> list[list[float]]:
    return [[float(value.real), float(value.imag)] for value in values]


def encode_symbols(symbols: list[int] | tuple[int, ...] | np.ndarray) -> np.ndarray:
    """Encode signed integers as amplitude and phase; integer zero maps to 0j."""
    values = np.asarray(symbols)
    if (
        values.ndim != 1
        or not 4 <= values.size <= 16
        or not np.issubdtype(values.dtype, np.integer)
    ):
        raise ValueError("symbols must be an integer vector of 4 to 16 channels")
    if np.any(~np.isin(values, ALPHABET)):
        raise ValueError("symbols must be in the finite alphabet -2..2")
    amplitudes = np.abs(values).astype(float) / 2.0
    phases = np.where(values < 0, np.pi, 0.0)
    encoded = amplitudes * np.exp(1j * phases)
    encoded[values == 0] = 0.0j
    return encoded


def decode_nearest(
    received: np.ndarray, codewords: np.ndarray
) -> tuple[int, float]:
    """Return the first Euclidean-nearest codeword index and its distance."""
    received = np.asarray(received, dtype=np.complex128)
    codewords = np.asarray(codewords, dtype=np.complex128)
    if codewords.ndim != 2 or received.shape != (codewords.shape[1],):
        raise ValueError("received vector and codeword dimensions must agree")
    if not np.all(np.isfinite(received)) or not np.all(np.isfinite(codewords)):
        raise ValueError("decoder inputs must be finite")
    distances = np.linalg.norm(codewords - received[None, :], axis=1)
    index = int(np.argmin(distances))
    return index, float(distances[index])


def _codebook(channels: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Build a bounded alphabet codebook with matched-energy nonzero words."""
    if not 4 <= channels <= 16:
        raise ValueError("channels must be between 4 and 16")
    if not 2 <= size <= min(256, 1 + 4 * channels * (channels - 1)):
        raise ValueError("codebook_size exceeds the finite matched-energy alphabet")
    words = [(0,) * channels]
    seen = {words[0]}
    while len(words) < size:
        locations = rng.choice(channels, size=2, replace=False)
        magnitudes = rng.permutation([1, 2])
        values = [0] * channels
        for location, magnitude in zip(locations, magnitudes):
            values[int(location)] = int(magnitude * rng.choice([-1, 1]))
        word = tuple(values)
        if word not in seen:
            seen.add(word)
            words.append(word)
    return np.stack([encode_symbols(word) for word in words])


def run_wave_memory(
    seed: int,
    config: dict,
) -> list[dict]:
    """Run paired, seeded channel trials at matched input energy."""
    channels = int(config["channels"])
    trials = int(config["trials"])
    delays = [int(value) for value in config["delays"]]
    crosstalk_levels = [float(value) for value in config["crosstalk_levels"]]
    if trials < 1 or any(delay < 1 for delay in delays):
        raise ValueError("trials and delays must be positive")
    if any(not 0.0 <= value < 1.0 for value in crosstalk_levels):
        raise ValueError("crosstalk levels must be in [0, 1)")
    snr_db = float(config["snr_db"])
    if not np.isfinite(snr_db) or not -300.0 <= snr_db <= 300.0:
        raise ValueError("snr_db must be finite and in [-300, 300]")
    phase_sd = float(config["phase_drift_std"])
    gain_sd = float(config["gain_drift_std"])
    max_delay_scale = np.sqrt(max(delays))
    if not np.isfinite(phase_sd) or phase_sd < 0 or phase_sd * max_delay_scale > 20:
        raise ValueError("phase_drift_std must be finite, nonnegative, and bounded")
    if not np.isfinite(gain_sd) or gain_sd < 0 or gain_sd * max_delay_scale > 20:
        raise ValueError("gain_drift_std must be finite, nonnegative, and bounded")

    codebook_rng = np.random.default_rng(np.random.SeedSequence([seed, 401]))
    codebook = _codebook(channels, int(config["codebook_size"]), codebook_rng)
    message_rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    codebook_symbols = np.rint(2 * codebook.real).astype(int).tolist()
    codebook_hash = hashlib.sha256(
        json.dumps(codebook_symbols, separators=(",", ":")).encode()
    ).hexdigest()
    dft = np.exp(
        -2j * np.pi * np.outer(np.arange(channels), np.arange(channels)) / channels
    ) / np.sqrt(channels)
    rows = []

    for cross_index, cross in enumerate(crosstalk_levels):
        base_channel = np.eye(channels, dtype=complex)
        base_channel += (cross / (channels - 1)) * (
            np.ones((channels, channels), dtype=complex) - np.eye(channels)
        )
        for trial in range(trials):
            codeword_index = int(message_rng.integers(1, len(codebook)))
            symbols = [
                int(value)
                for value in np.rint(
                    2 * codebook[codeword_index].real
                ).astype(int)
            ]
            encoded = codebook[codeword_index]
            input_energy = float(np.vdot(encoded, encoded).real)
            if not np.isclose(input_energy, 1.25, atol=1e-12, rtol=0):
                raise RuntimeError("nonzero codewords must have matched energy")

            for delay_index, delay in enumerate(delays):
                stream_key = [seed, trial, cross_index, delay_index]
                drift_rng = np.random.default_rng(
                    np.random.SeedSequence([*stream_key, 201])
                )
                noise_rng = np.random.default_rng(
                    np.random.SeedSequence([*stream_key, 301])
                )
                scale = np.sqrt(delay)
                gains = np.exp(drift_rng.normal(0.0, gain_sd * scale, channels))
                phases = drift_rng.normal(0.0, phase_sd * scale, channels)
                channel = np.diag(gains * np.exp(1j * phases)) @ base_channel
                if not np.all(np.isfinite(channel)):
                    raise FloatingPointError("channel drift produced nonfinite values")
                condition = float(np.linalg.cond(channel))
                # E[||n||^2] = E[||x||^2] / 10^(SNR_dB/10).
                noise_variance = input_energy / (10 ** (snr_db / 10))
                sigma = np.sqrt(noise_variance / (2 * channels))
                noise = sigma * (
                    noise_rng.normal(size=channels)
                    + 1j * noise_rng.normal(size=channels)
                )
                received = channel @ encoded + noise
                recovered_channel = np.linalg.solve(channel, received)
                recovered_coherent = dft.conj().T @ np.linalg.solve(
                    channel, channel @ (dft @ encoded) + noise
                )
                reset_received = channel @ np.zeros(channels, complex) + noise

                rows.append({
                    "experiment": "wave_channel_observation",
                    "seed": seed,
                    "trial": trial,
                    "delay": delay,
                    "channels": channels,
                    "crosstalk": cross,
                    "condition_number": condition,
                    "phase_drift_std": phase_sd * scale,
                    "gain_drift_std": gain_sd * scale,
                    "phase_drift_radians": phases.tolist(),
                    "gain_factors": gains.tolist(),
                    "snr_db": snr_db,
                    "noise_variance_total_expected": noise_variance,
                    "input_energy": input_energy,
                    "input_symbols": symbols,
                    "input_codeword": codeword_index,
                    "codebook_symbols": codebook_symbols,
                    "codebook_sha256": codebook_hash,
                    "channel_input": _complex_pairs(encoded),
                    "noise_vector": _complex_pairs(noise),
                    "received_vector_y": _complex_pairs(received),
                })
                observations = {
                    "no_intervention": received,
                    "detector_only_cancellation_then_removal": recovered_channel,
                    "coherent_reversible_manipulation": recovered_coherent,
                    "dissipative_reset_reference": reset_received,
                }
                for intervention, observation in observations.items():
                    decoded_index, distance = decode_nearest(observation, codebook)
                    rows.append({
                        "experiment": "wave_memory",
                        "seed": seed,
                        "trial": trial,
                        "delay": delay,
                        "channels": channels,
                        "crosstalk": cross,
                        "condition_number": condition,
                        "phase_drift_std": phase_sd * scale,
                        "gain_drift_std": gain_sd * scale,
                        "snr_db": snr_db,
                        "noise_variance_total_expected": noise_variance,
                        "input_energy": input_energy,
                        "stored_energy_after_intervention": (
                            0.0
                            if intervention == "dissipative_reset_reference"
                            else input_energy
                        ),
                        "input_symbols": symbols,
                        "input_codeword": codeword_index,
                        "codebook_sha256": codebook_hash,
                        "decoded_codeword": decoded_index,
                        "decoding_error": decoded_index != codeword_index,
                        "nearest_distance": distance,
                        "intervention": intervention,
                        "detector_only_state_unchanged": (
                            intervention == "detector_only_cancellation_then_removal"
                        ),
                        "decoder_observation": _complex_pairs(observation),
                    })
    return rows


def zero_codeword_controls(channels: int) -> dict:
    """Expose the exact canonical-zero mapping for protocol/control tests."""
    symbols = (0,) * channels
    encoded = encode_symbols(symbols)
    if not np.array_equal(encoded, np.zeros(channels, dtype=complex)):
        raise RuntimeError("canonical zero must encode to exactly zero")
    index, distance = decode_nearest(encoded, encoded.reshape(1, -1))
    return {"symbols": list(symbols), "encoded": encoded, "index": index, "distance": distance}
