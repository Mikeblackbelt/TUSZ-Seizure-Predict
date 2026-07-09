 
import numpy as np
import pytest
 
from pipeline.resampling import resample_eeg, rescale_sample_index

def make_sine(freq_hz: float, fs: float, duration_s: float, n_channels: int = 1) -> np.ndarray:
    """Generate a clean sine wave, optionally tiled across channels."""
    t = np.arange(0, duration_s, 1.0 / fs)
    sig = np.sin(2 * np.pi * freq_hz * t)
    if n_channels == 1:
        return sig
    return np.tile(sig, (n_channels, 1))
 
def dominant_freq(sig: np.ndarray, fs: float) -> float:
    """Return the frequency with peak magnitude in the FFT of a 1D signal."""
    n = len(sig)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(sig))
    return freqs[np.argmax(mag)]
 
def test_length_downsample_256_to_250():
    orig_fs, target_fs, duration = 256.0, 250.0, 10.0
    sig = make_sine(10, orig_fs, duration)
    out = resample_eeg(sig, orig_fs, target_fs)
    expected_len = round(len(sig) * target_fs / orig_fs)
    # polyphase resampling can be off by a sample or two depending on ratio
    assert abs(len(out) - expected_len) <= 2

def test_length_upsample_250_to_256():
    orig_fs, target_fs, duration = 250.0, 256.0, 10.0
    sig = make_sine(10, orig_fs, duration)
    out = resample_eeg(sig, orig_fs, target_fs)
    expected_len = round(len(sig) * target_fs / orig_fs)
    assert abs(len(out) - expected_len) <= 2

def test_length_downsample_512_to_250():
    orig_fs, target_fs, duration = 512.0, 250.0, 10.0
    sig = make_sine(10, orig_fs, duration)
    out = resample_eeg(sig, orig_fs, target_fs)
    expected_len = round(len(sig) * target_fs / orig_fs)
    assert abs(len(out) - expected_len) <= 2

def test_multichannel_shape_preserved():
    orig_fs, target_fs, duration = 256.0, 250.0, 5.0
    sig = make_sine(10, orig_fs, duration, n_channels=19)
    out = resample_eeg(sig, orig_fs, target_fs)
    expected_len = round(sig.shape[-1] * target_fs / orig_fs)
    assert out.shape[0] == 19
    assert abs(out.shape[-1] - expected_len) <= 2

def test_identity_when_rates_match():
    sig = make_sine(10, 250.0, 5.0)
    out = resample_eeg(sig, 250.0, 250.0)
    np.testing.assert_array_equal(sig, out)


def test_preserves_dominant_frequency_downsample():
    orig_fs, target_fs = 256.0, 250.0
    sig = make_sine(10, orig_fs, duration_s=20.0)
    out = resample_eeg(sig, orig_fs, target_fs)
    peak = dominant_freq(out, target_fs)
    assert abs(peak - 10) < 0.5

def test_preserves_dominant_frequency_upsample():
    orig_fs, target_fs = 250.0, 256.0
    sig = make_sine(10, orig_fs, duration_s=20.0)
    out = resample_eeg(sig, orig_fs, target_fs)
    peak = dominant_freq(out, target_fs)
    assert abs(peak - 10) < 0.5

def test_high_freq_content_attenuated_not_aliased():
    """
    A tone above the new Nyquist frequency should be killed by the
    anti-aliasing filter rather than folded down into the passband as a
    fake low-frequency component. We check this by comparing the total
    output energy of an above-Nyquist tone against a below-Nyquist tone
    of the same input amplitude. If the filter is doing its job, the
    above-Nyquist tone's surviving energy should be much smaller.
    """
    orig_fs, target_fs = 512.0, 250.0
    duration = 20.0

    below_nyquist_tone = 10.0   # comfortably passes through
    above_nyquist_tone = 200.0  # above new Nyquist (125 Hz), should be killed

    sig_pass = make_sine(below_nyquist_tone, orig_fs, duration)
    sig_block = make_sine(above_nyquist_tone, orig_fs, duration)

    out_pass = resample_eeg(sig_pass, orig_fs, target_fs)
    out_block = resample_eeg(sig_block, orig_fs, target_fs)

    energy_pass = np.sum(out_pass ** 2)
    energy_block = np.sum(out_block ** 2)

    assert energy_block < 0.1 * energy_pass

def test_zero_orig_fs_raises():
    sig = make_sine(10, 250.0, 1.0)
    with pytest.raises(ValueError):
        resample_eeg(sig, 0, 250.0)

def test_negative_target_fs_raises():
    sig = make_sine(10, 250.0, 1.0)
    with pytest.raises(ValueError):
        resample_eeg(sig, 250.0, -250.0)

def test_short_signal_does_not_crash():
    # A handful of samples - resample_poly should still run without error
    sig = np.array([0.1, 0.2, 0.15, -0.1, -0.2, 0.05])
    out = resample_eeg(sig, 256.0, 250.0)
    assert out.ndim == 1
    assert len(out) > 0

def test_output_dtype_is_float():
    sig = make_sine(10, 256.0, 2.0)
    out = resample_eeg(sig, 256.0, 250.0)
    assert np.issubdtype(out.dtype, np.floating)

def test_downsample_index_scales_down():
    # should map to sample 2500 at 250 Hz.
    assert rescale_sample_index(2560, 256.0, 250.0) == 2500

def test_upsample_index_scales_up():
    assert rescale_sample_index(2500, 250.0, 256.0) == 2560

def test_identity_when_rates_match():
    assert rescale_sample_index(1234, 250.0, 250.0) == 1234

def test_roundtrip_close_to_original():
    """Downsampling then upsampling an index should land close to start."""
    orig_idx = 5000
    down = rescale_sample_index(orig_idx, 256.0, 250.0)
    back = rescale_sample_index(down, 250.0, 256.0)
    assert abs(back - orig_idx) <= 1
