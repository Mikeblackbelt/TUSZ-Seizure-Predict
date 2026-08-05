from torch.utils.data import Dataset, random_split
from feature_extraction import extract_features
from pathlib import Path
import termcolor
import numpy as np
from typing import Tuple
import argparse


class EEGDataset(Dataset):
    def __init__(self, data_dir: Path, raw: bool = False):
        """
        Initialize the dataset from EEG samples and labels stored in a directory.

        Parameters:
            data_dir (Path): Directory containing the `X.npy` feature data and `Y.npy` labels.
            raw (bool): If True, skip hand-crafted feature extraction and pass X.npy through
                as-is (cast to float32). Use this for models that learn directly from signal
                (CNN, SincNet, ArtifactDetectionCNN_MultiChannel) -- extract_features() only
                operates on 1D per-window signals and is meaningless (and wasteful) as input
                to a convolutional model that's meant to learn its own filters. X.npy may be
                2D (n_windows, win_len) for single-channel data or 3D
                (n_windows, n_channels, win_len) for multi-channel data; both pass through
                unchanged when raw=True.
        """
        try:
            X = np.load(data_dir / "X.npy")
            y = np.load(data_dir / "Y.npy")
        except FileNotFoundError:
            print(termcolor.colored("Data files not found. Please ensure the data files are in the correct directory.", "red"))
            exit(1)

        self.features = X.astype(np.float32) if raw else extract_features(X)
        self.labels = y.flatten()

    def __len__(self):
        """Return the number of samples in the dataset.
        
        Returns:
            int: The number of extracted feature samples.
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        Return the feature and label for a dataset sample.
        
        Parameters:
            idx: The index of the sample to retrieve.
        
        Returns:
            tuple: The sample's features and label.
        """
        return self.features[idx], self.labels[idx]