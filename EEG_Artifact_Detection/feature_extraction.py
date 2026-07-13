import numpy as np
import pywt
from scipy.signal import welch, butter, filtfilt

# def wavelet_transform(signal, wavelet='db4', level=4):
#     coeffs = pywt.wavedec(signal, wavelet, level=level)
#     approx = coeffs[0]
#     details = coeffs[1:]
#     return np.concatenate([approx.flatten()])

def power_spectral_density(signal, fs=256):
    """
    Compute the power spectral density of a signal.
    
    Parameters:
    	signal: The input signal.
    	fs (int): The sampling frequency in hertz.
    
    Returns:
    	np.ndarray: The power spectral density values.
    """
    freqs, psd = welch(signal, fs=fs)
    return psd

def design_lowpass_filter(cutoff, fs=256, order=4):
    """
    Designs a digital Butterworth low-pass filter.
    
    Parameters:
    	cutoff (float): Cutoff frequency in hertz.
    	fs (float): Sampling frequency in hertz.
    	order (int): Filter order.
    
    Returns:
    	tuple: Numerator and denominator filter coefficients.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(b, a, signal):
    """Apply a low-pass filter to a signal without introducing phase shifts.
    
    Parameters:
        b: Numerator coefficients of the filter.
        a: Denominator coefficients of the filter.
        signal: Signal to filter.
    
    Returns:
        The filtered signal.
    """
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def decimate_signal(signal, decimation_factor=16):
    """
    Downsample a signal by selecting samples at a fixed interval.
    
    Parameters:
        decimation_factor (int): Number of samples between selected values.
    
    Returns:
        The downsampled signal.
    """
    return signal[::decimation_factor]

def wavelet_4th_level_approximation_simulation(signal, level=4):
    """
    Approximate a wavelet decomposition level with low-pass signal features.
    
    Parameters:
        signal: The input signal.
        level (int): The approximation level used to determine the filter cutoff and decimation factor.
    
    Returns:
        A flattened array containing the low-frequency approximation of the signal.
    """
    fs = 256
    cutoff = fs / (2 ** (level + 1))
    b, a = design_lowpass_filter(cutoff, fs=fs)
    filtered_signal = apply_lowpass_filter(b, a, signal)
    decimated_signal = decimate_signal(filtered_signal, decimation_factor=2**level)
    return decimated_signal.flatten()

def extract_features(eeg_signals):
    """
    Builds feature vectors for a collection of EEG signals.
    
    Parameters:
    	eeg_signals: EEG signals to process.
    
    Returns:
    	NumPy array containing one concatenated low-frequency approximation and power spectral density feature vector per signal.
    """
    features = []
    for signal in eeg_signals:
        low_pass_info = wavelet_4th_level_approximation_simulation(signal)
        psd = power_spectral_density(signal)
        feature_vector = np.concatenate([low_pass_info, psd])
        features.append(feature_vector)
    return np.array(features)