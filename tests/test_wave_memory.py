from itertools import product

import numpy as np
import pytest

from membrane_arena_3x3.wave_memory import (
    ALPHABET,
    decode_nearest,
    encode_symbols,
    run_wave_memory,
    zero_codeword_controls,
)


def small_config(**overrides):
    return {
        "channels": 4,
        "codebook_size": 16,
        "trials": 4,
        "delays": [1, 3],
        "crosstalk_levels": [0.0, 0.4],
        "phase_drift_std": 0.02,
        "gain_drift_std": 0.01,
        "snr_db": 20.0,
        **overrides,
    }


def test_finite_integer_encoding_is_injective_and_zero_is_canonical():
    encoded = [
        encode_symbols(symbols)
        for symbols in product(ALPHABET, repeat=4)
    ]
    assert len({value.tobytes() for value in encoded}) == 5**4
    controls = zero_codeword_controls(4)
    assert controls["encoded"].tolist() == [0j] * 4
    assert controls["distance"] == 0.0
    assert controls["index"] == 0


def test_amplitude_saturation_and_signed_phase_are_explicit():
    encoded = encode_symbols([-2, -1, 0, 1, 2])
    assert np.abs(encoded).tolist() == [1.0, 0.5, 0.0, 0.5, 1.0]
    np.testing.assert_allclose(encoded[[0, 1]], [-1.0, -0.5], atol=1e-15)
    np.testing.assert_allclose(encoded[[3, 4]], [0.5, 1.0], atol=1e-15)


def test_seeded_trials_are_paired_at_matched_energy_and_include_delays():
    config = small_config()
    rows = run_wave_memory(9, config)
    assert rows == run_wave_memory(9, config)
    results = [row for row in rows if row["experiment"] == "wave_memory"]
    assert {row["delay"] for row in results} == {1, 3}
    assert {row["intervention"] for row in results} == {
        "no_intervention",
        "detector_only_cancellation_then_removal",
        "coherent_reversible_manipulation",
        "dissipative_reset_reference",
    }
    for row in results:
        if row["intervention"] == "dissipative_reset_reference":
            assert row["stored_energy_after_intervention"] == 0.0
        else:
            assert row["input_energy"] == pytest.approx(1.25)
            assert row["stored_energy_after_intervention"] == pytest.approx(1.25)
        assert row["noise_variance_total_expected"] == pytest.approx(
            row["input_energy"] / 100
        )
    detector_rows = [
        row for row in results
        if row["intervention"] == "detector_only_cancellation_then_removal"
    ]
    assert all(row["detector_only_state_unchanged"] for row in detector_rows)
    assert all(row["condition_number"] >= 1.0 for row in rows)


def test_raw_channel_row_reconstructs_y_equals_hx_plus_n():
    rows = run_wave_memory(5, small_config(
        trials=1, delays=[1], crosstalk_levels=[0.4]
    ))
    row = next(item for item in rows
               if item["experiment"] == "wave_channel_observation")
    channels = row["channels"]
    x = np.array([complex(*value) for value in row["channel_input"]])
    noise = np.array([complex(*value) for value in row["noise_vector"]])
    received = np.array([complex(*value) for value in row["received_vector_y"]])
    diagonal = np.array(row["gain_factors"]) * np.exp(
        1j * np.array(row["phase_drift_radians"])
    )
    cross = row["crosstalk"]
    mixing = np.eye(channels) + (cross / (channels - 1)) * (
        np.ones((channels, channels)) - np.eye(channels)
    )
    np.testing.assert_allclose(received, np.diag(diagonal) @ mixing @ x + noise)
    assert len(row["codebook_symbols"]) == 16
    assert len(row["codebook_sha256"]) == 64
    assert not np.allclose(row["gain_factors"], 1.0)
    assert not np.allclose(row["phase_drift_radians"], 0.0)


def test_nearest_codeword_uses_euclidean_distance():
    words = np.array([[0, 0], [1, 0], [0, 1]], dtype=complex)
    assert decode_nearest(np.array([0.8, 0.1]), words)[0] == 1


def test_stressed_drift_and_noise_produce_measured_decoding_errors():
    rows = run_wave_memory(42, small_config(
        codebook_size=24,
        trials=10,
        delays=[1, 20],
        crosstalk_levels=[0.95],
        phase_drift_std=0.4,
        gain_drift_std=0.2,
        snr_db=0,
    ))
    ordinary = [
        row for row in rows
        if row["experiment"] == "wave_memory"
        and row["intervention"] == "no_intervention"
    ]
    assert any(row["decoding_error"] for row in ordinary)


def test_dissipative_reset_loses_nonzero_messages():
    rows = run_wave_memory(1, small_config(
        trials=3, delays=[1], crosstalk_levels=[0], snr_db=100
    ))
    reset = [row for row in rows if row["experiment"] == "wave_memory"
             if row["intervention"] == "dissipative_reset_reference"]
    assert all(row["decoding_error"] for row in reset)


@pytest.mark.parametrize(
    "symbols",
    [[0.0, 1, 0, 0], [3, 0, 0, 0], [0, 0, 0]],
)
def test_invalid_symbol_vectors_are_rejected(symbols):
    with pytest.raises(ValueError):
        encode_symbols(symbols)


def test_channel_config_bounds_and_decoder_finiteness():
    with pytest.raises(ValueError):
        run_wave_memory(0, small_config(channels=3))
    with pytest.raises(ValueError):
        run_wave_memory(0, small_config(snr_db=1000))
    with pytest.raises(ValueError):
        decode_nearest(np.array([np.nan]), np.zeros((1, 1), complex))
