import numpy as np
from joblib import Memory
from sklearn.preprocessing import scale
from scipy.signal import resample

# Cache feature computations
dmemory = Memory(location='cache', verbose=0)

@dmemory.cache
def fft_features(data: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute log-FFT magnitudes for frequency bins 1–47 Hz for each channel.

    Args:
        data: (n_channels, n_samples) EEG segment
        eps: small constant to avoid log10(0)
    Returns:
        fft_feats: (n_channels, 47) array of log10 magnitudes
    """
    fft = np.fft.rfft(data, axis=1)[:, 1:48]
    return np.log10(np.abs(fft) + eps)


def freq_corr_feats(fft_data: np.ndarray) -> np.ndarray:
    """
    Compute frequency-domain correlation features.

    Args:
        fft_data: (n_channels, n_bins) output of fft_features
    Returns:
        1D array: upper-triangle correlations + sorted eigenvalues
    """
    scaled = scale(fft_data, axis=0)
    corr = np.corrcoef(scaled)
    iu = np.triu_indices_from(corr, k=1)
    eigs = np.sort(np.abs(np.linalg.eigvals(corr)))
    return np.concatenate([corr[iu], eigs])


def time_corr_feats(data: np.ndarray, fs: int) -> np.ndarray:
    """
    Compute time-domain correlation features over a fixed-length window.

    Args:
        data: (n_channels, n_samples) EEG segment
        fs: target number of samples (sampling rate × window length)
    Returns:
        1D array: upper-triangle correlations + sorted eigenvalues
    """
    # Resample to exactly fs samples if needed
    data_rs = resample(data, fs, axis=1) if data.shape[1] != fs else data
    scaled = scale(data_rs, axis=0)
    corr = np.corrcoef(scaled)
    iu = np.triu_indices_from(corr, k=1)
    eigs = np.sort(np.abs(np.linalg.eigvals(corr)))
    return np.concatenate([corr[iu], eigs])


def extract_features(data: np.ndarray, fs: int) -> np.ndarray:
    """
    Combine all features into a single vector with NaNs/infs clamped.

    Args:
        data: (n_channels, n_samples) EEG segment
        fs: sample count for time correlation
    Returns:
        1D float32 feature vector
    """
    fft_out = fft_features(data)
    feats = np.concatenate([
        fft_out.ravel(),
        freq_corr_feats(fft_out),
        time_corr_feats(data, fs)
    ]).astype(np.float32)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
