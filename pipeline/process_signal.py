import numpy as np

from util import handle_logs

logger = handle_logs.get_logger("epoching", "applog")


def segment_into_epochs(data: np.ndarray, fs: float, epoch_duration: float = 1.0, drop_last: bool = True):
    """
    Split a continuous (n_channels, n_samples) array into fixed-duration,
    non-overlapping epochs.

    Parameters:
        data (np.ndarray): Signal array of shape (n_channels, n_samples),
            already loaded/concatenated/resampled/bipolar-converted/
            artifact-masked upstream.
        fs (float): Sampling rate of `data`, in Hz.
        epoch_duration (float): Epoch length in seconds. Default 1.0.
        drop_last (bool): If True (default), discard a trailing partial
            epoch that doesn't fill a full `epoch_duration` window. If
            False, raise instead of silently dropping data.

    Returns:
        np.ndarray of shape (n_epochs, n_channels, epoch_samples)
    """
    if data.ndim != 2:
        raise ValueError(f"Expected data of shape (n_channels, n_samples), got shape {data.shape}")
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    if epoch_duration <= 0:
        raise ValueError(f"epoch_duration must be positive, got {epoch_duration}")

    n_channels, n_samples = data.shape
    epoch_samples = int(round(epoch_duration * fs))

    if epoch_samples <= 0:
        raise ValueError(
            f"epoch_duration={epoch_duration} at fs={fs} rounds to {epoch_samples} samples/epoch"
        )

    n_epochs = n_samples // epoch_samples
    remainder = n_samples - (n_epochs * epoch_samples)

    if remainder and not drop_last:
        raise ValueError(
            f"{n_samples} samples not evenly divisible by epoch_samples={epoch_samples} "
            f"(remainder={remainder}); pass drop_last=True to discard the trailing partial epoch"
        )

    if remainder:
        logger.debug(f"Dropping trailing {remainder} samples ({remainder / fs:.3f}s) that don't fill a full epoch")

    trimmed = data[:, : n_epochs * epoch_samples]
    epochs = trimmed.reshape(n_channels, n_epochs, epoch_samples).transpose(1, 0, 2)

    logger.info(
        f"Segmented ({n_channels}, {n_samples}) @ {fs}Hz into "
        f"{n_epochs} epochs of {epoch_duration}s ({epoch_samples} samples each)"
    )

    return epochs


if __name__ == "__main__":
    fs = 250
    data = np.random.randn(17, fs * 10).astype(np.float64)  # 10 seconds, 17 channels

    epochs = segment_into_epochs(data, fs, epoch_duration=1)
    print(f"Input shape: {data.shape}")
    print(f"Output shape: {epochs.shape}")
    assert epochs.shape == (10, 17, fs)

    # Uneven case
    data2 = np.random.randn(17, fs * 10 + 37).astype(np.float64)
    epochs2 = segment_into_epochs(data2, fs, epoch_duration=1)
    assert epochs2.shape == (10, 17, fs)
    print("Self-test passed.")