from pathlib import Path
from random import seed
import numpy as np
from numpy import array, genfromtxt
from scipy.stats import zscore
from torch.utils.data import Dataset
import scipy.io as sio
import os
import argparse
from utils import *

seed(2408)
np.random.seed(1305)

class DataNoiseCombiner:
    def __init__(self, config):
        """Initialize the combiner with the configured datasets and generate the output splits.
        
        Parameters:
            config: Configuration containing the data path and dataset processing settings.
        """
        self.config = config
        self.data_clean = self.load_samples(os.path.join(config.datapath, "filtered80Hz_EEG_all_epochs.mat"), 0)
        self.data_eog = self.load_samples(os.path.join(config.datapath, "filtered80Hz_EOG_all_epochs.mat"), 1)
        self.data_emg = self.load_samples(os.path.join(config.datapath, "filtered80Hz_EMG_all_epochs.mat"), 2)
        self.clean_indices = self.shuffle_indices(len(self.data_clean[0]))
        self.eog_indices = self.shuffle_indices(len(self.data_eog[0]))
        self.emg_indices = self.shuffle_indices(len(self.data_emg[0]))
        self.process_and_save_data()

    def load_samples(self, path, label=None):
        """
        Load waveform samples from a supported file and optionally assign labels.
        
        Parameters:
            path (str or pathlib.Path): Path to a `.mat`, `.csv`, or `.npy` file.
            label (int, optional): Label assigned to every loaded sample. Labels `1` and `2`
                cause samples to be repeated or truncated to match the clean dataset length.
        
        Returns:
            tuple: The waveform array and an array containing the assigned label for each
                sample, or `None` when no label is provided.
        
        Raises:
            ValueError: If the file has an unsupported extension.
        """
        path = Path(path)
        data = sio.loadmat(path) if path.suffix == ".mat" else (
            genfromtxt(path, delimiter=",") if path.suffix == ".csv" else (
                np.load(path) if path.suffix == ".npy" else None))
        if data is None:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        X = next(value for value in data.values() if isinstance(value, np.ndarray)) if path.suffix == ".mat" else data
        X = np.repeat(X, np.ceil(len(self.data_clean[0]) / len(X)), axis=0)[: len(self.data_clean[0]), :] if label in [1, 2] else X
        return X, array([label] * len(X)) if label is not None else None

    @staticmethod
    def shuffle_indices(length):
        """
        Create a randomized ordering of integer indices from zero through length minus one.
        
        Parameters:
        	length (int): Number of indices to generate.
        
        Returns:
        	np.ndarray: Shuffled integer indices.
        """
        indices = np.arange(length)
        np.random.shuffle(indices)
        return indices

    def split_indices(self, indices, test_size, val_size):
        """
        Split shuffled indices into test, validation, and training subsets.
        
        Parameters:
        	indices (array-like): Indices to partition.
        	test_size (float): Fraction of indices assigned to the test subset.
        	val_size (float): Fraction of indices assigned to the validation subset.
        
        Returns:
        	tuple: Test, validation, and training index subsets, in that order.
        """
        test_size, val_size = int(test_size * len(indices)), int(val_size * len(indices))
        return indices[:test_size], indices[test_size:test_size + val_size], indices[test_size + val_size:]

    def save_data(self, X, y, data_type, snr_type=None):
        """
        Save feature and label arrays to the configured data directory.
        
        Parameters:
        	X (numpy.ndarray): Feature array to z-score along axis 1 before saving.
        	y (numpy.ndarray): Label array to save unchanged.
        	data_type (str): Subdirectory identifying the dataset split.
        	snr_type (str, optional): Subdirectory identifying the SNR condition.
        """
        directory = Path(self.config.datapath) / data_type / (snr_type or "")
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "X.npy", zscore(X, axis=1))
        np.save(directory / "Y.npy", y)

    def combine_and_save(self, clean_indices, noise_indices, data_clean, data_noise, snr):
        """
        Combine selected clean and noise waveforms at the specified signal-to-noise ratio.
        
        Parameters:
            clean_indices: Indices selecting clean waveforms.
            noise_indices: Indices selecting noise waveforms and their labels.
            data_clean: Clean waveform data and labels.
            data_noise: Noise waveform data and labels.
            snr: Signal-to-noise ratio used for waveform combination.
        
        Returns:
            A tuple containing the combined waveforms and their labels.
        """
        combined_data = combine_waveforms((data_clean[0][clean_indices], data_clean[0][clean_indices]),
                                          (data_noise[0][noise_indices], data_noise[1][noise_indices]), snr_db=snr)
        return combined_data[0], combined_data[1]

    def process_and_save_data(self):
        """
        Generate and save test, validation, and training datasets from clean EEG, EOG, and EMG samples.
        
        Test datasets are generated across the configured SNR range, while validation and training datasets use the corresponding splits without an explicit SNR.
        """
        clean_test_indices, clean_val_indices, clean_training_indices = self.split_indices(self.clean_indices, self.config.test_size, self.config.val_size)
        eog_test_indices, eog_val_indices, eog_training_indices = self.split_indices(self.eog_indices, self.config.test_size, self.config.val_size)
        emg_test_indices, emg_val_indices, emg_training_indices = self.split_indices(self.emg_indices, self.config.test_size, self.config.val_size)

        for snr in np.arange(self.config.lower_snr, self.config.higher_snr, 0.5):
            X_eog, y_eog = self.combine_and_save(clean_test_indices, eog_test_indices, self.data_clean, self.data_eog, snr)
            X_emg, y_emg = self.combine_and_save(clean_test_indices, emg_test_indices, self.data_clean, self.data_emg, snr)
            X_clean, y_clean = self.data_clean[0][clean_test_indices], self.data_clean[1][clean_test_indices]
            X = np.concatenate((X_eog, X_emg, X_clean), axis=0)
            y = np.concatenate((y_eog, y_emg, y_clean), axis=0)
            self.save_data(X, y, "test", f"snr {snr}")

        subsets = [("val", clean_val_indices, eog_val_indices, emg_val_indices),
         ("train", clean_training_indices, eog_training_indices, emg_training_indices)]
        for subset_name, clean_indices, eog_indices, emg_indices in subsets:
            X_eog, y_eog = self.combine_and_save(clean_indices, eog_indices, self.data_clean, self.data_eog, None)
            X_emg, y_emg = self.combine_and_save(clean_indices, emg_indices, self.data_clean, self.data_emg, None)
            X_clean, y_clean = self.data_clean[0][clean_indices], self.data_clean[1][clean_indices]
            self.save_data(np.concatenate((X_eog, X_emg, X_clean), axis=0),
                           np.concatenate((y_eog, y_emg, y_clean), axis=0), subset_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='./data')
    parser.add_argument('--lower_snr', type=float, default=-7)
    parser.add_argument('--higher_snr', type=float, default=4.5)
    parser.add_argument('--test_size', type=float, default=0.25)
    parser.add_argument('--val_size', type=float, default=0.2)
    args = parser.parse_args()
    DataNoiseCombiner(args)