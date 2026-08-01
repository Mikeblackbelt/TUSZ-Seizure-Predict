import os
import time
import datetime
import logging
import pickle
from pathlib import Path
from contextlib import contextmanager
import termcolor
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.decomposition import PCA, IncrementalPCA, FastICA
from sklearn.preprocessing import StandardScaler
from models import ArtifactDetectionNN,ArtifactDetectionCNN,ConvNet
from dataset import EEGDataset
from datanoise_combiner import DataNoiseCombiner
from utils import calculate_metrics, setup_logging, EarlyStopping
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from models import ArtifactDetectionNN, ArtifactDetectionCNN, ConvNet, FocalLoss

run_datetime = datetime.datetime.now()
plt.rcParams.update({'font.size': 14})


def _fmt_secs(seconds):
    """Format a duration in seconds as H:MM:SS (or M:SS if under an hour)."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def _iter_chunks(n_rows, chunk_size):
    """
    Yield (start, end) index pairs covering ``n_rows`` in steps of ``chunk_size``.

    Used to fit/transform large feature arrays (e.g. 7e7 EEG windows) in bounded-memory
    pieces instead of requiring the whole array to be duplicated in memory at once.
    """
    for start in range(0, n_rows, chunk_size):
        yield start, min(start + chunk_size, n_rows)


class MLPTrainer:
    def __init__(self, config):
        """
        Initialize the trainer with the provided configuration and prepare its training and evaluation components.
        
        Parameters:
        	config: Configuration containing dataset paths, model settings, preprocessing options, and training parameters.
        """
        self.config = config
        # Rows processed per chunk during scaling/PCA fit+transform, so a 7e7-window
        # dataset never needs a second full-size copy in memory at once. Override via
        # config.preprocess_chunk_size if needed.
        self.chunk_size = getattr(config, 'preprocess_chunk_size', 500_000)
        self.device = self._setup_device()
        self._setup_directories()
        self._setup_logging()
        self._init_data_combiner()
        self._load_datasets()
        self._setup_preprocessing()
        self._init_model()
        self._init_training_components()
        self._init_metrics()
        self._epoch_durations = []
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
    # ------------------------------------------------------------------ #
    # Logging helpers
    # ------------------------------------------------------------------ #

    @contextmanager
    def _log_step(self, name, extra=""):
        """
        Log the start and end of a long-running step (preprocessing, model loading, etc.)
        along with its wall-clock duration, so slow steps are visible in the logs/console
        instead of appearing to hang.

        Parameters:
            name (str): Short human-readable name of the step (e.g. 'Fitting StandardScaler').
            extra (str): Optional extra context to include in the start log line (e.g. shape).
        """
        start_msg = f"[START] {name}" + (f" ({extra})" if extra else "")
        logging.info(start_msg)
        print(termcolor.colored(start_msg, 'yellow'))
        t0 = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - t0
            end_msg = f"[DONE]  {name} — took {_fmt_secs(elapsed)}"
            logging.info(end_msg)
            print(termcolor.colored(end_msg, 'yellow'))

    def _setup_device(self):
        """
        Select the available computation device for model operations.
        
        Returns:
            torch.device: CUDA when available; otherwise, CPU.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')
        return device

    def _setup_directories(self):
        """
        Create the directories used to store preprocessing artifacts, results, and confusion matrices.
        """
        os.makedirs(self.config.save_path, exist_ok=True)
        os.makedirs(self.config.outputpath, exist_ok=True)
        os.makedirs(Path(self.config.outputpath) / Path('cnf_matrices'), exist_ok=True)

    def _setup_logging(self):
        """Configure application logging using the configured log file and log level."""
        setup_logging(self.config.log_file, self.config.log_level)
        logging.info(f"=== Run started {run_datetime.isoformat(timespec='seconds')} ===")
        logging.info(f"Config: mode={self.config.mode}, model={self.config.model}, "
                     f"epochs={self.config.num_epochs}, batch_size={self.config.batch_size}, "
                     f"pca={self.config.pca}, ica={self.config.ica}")

    def _init_data_combiner(self):
        """Initialize the data noise combiner using the trainer configuration."""
        with self._log_step("Combining/generating noised data"):
            DataNoiseCombiner(self.config)

    def _load_datasets(self):
        """
        Load the training, validation, and SNR-specific test datasets from the configured data directory.
        """
        with self._log_step("Loading train dataset"):
            self.train_dataset = EEGDataset(Path(self.config.datapath) / "train", raw=(self.config.model in ('CNN', 'SincNet')))
        logging.info(f"Train dataset: {len(self.train_dataset):,} windows")

        with self._log_step("Loading validation dataset"):
            self.val_dataset = EEGDataset(Path(self.config.datapath) / "val", raw=(self.config.model in ('CNN', 'SincNet')))
        logging.info(f"Validation dataset: {len(self.val_dataset):,} windows")

        with self._log_step("Loading test datasets (all SNRs)"):
            self.test_datasets = self._load_test_datasets(Path(self.config.datapath) / "test", raw=(self.config.model in ('CNN', 'SincNet')))
        total_test = sum(len(d) for d in self.test_datasets.values())
        logging.info(f"Test datasets: {len(self.test_datasets)} SNR levels, "
                     f"{total_test:,} windows total")

    def _load_test_datasets(self, test_dir, raw=False):
        """
        Load test datasets keyed by SNR strings extracted from subdirectory names.
        
        Parameters:
        	test_dir (Path): Directory containing one subdirectory for each test SNR.
        	raw (bool): Whether to use raw features without preprocessing.
        
        Returns:
        	dict: Mapping of SNR strings to their corresponding EEG datasets.
        """
        test_datasets = {}
        snr_dirs = [d for d in test_dir.iterdir() if d.is_dir()]
        for snr_dir in tqdm(snr_dirs, desc="Loading test SNR folders", unit="snr"):
            snr_value = snr_dir.name.split(' ')[-1]
            test_datasets[snr_value] = EEGDataset(snr_dir, raw=raw)
            logging.info(f"  SNR {snr_value}: {len(test_datasets[snr_value]):,} windows")
        return test_datasets

    def _setup_preprocessing(self):
        """Prepare training and test features using the configured preprocessing artifacts."""
        if self.config.mode == 'train':
            self._preprocess_data()
        self._load_preprocessing()

    def _init_model(self):
        """Initialize the configured artifact-detection model on the selected device."""
        feature_size = next(iter(self.test_datasets.values())).features.shape[1]
        print(f'Feature shape: {feature_size}')
        logging.info(f"Feature size: {feature_size}")
        if self.config.model == 'MLP':
            self.model = ArtifactDetectionNN(feature_size).to(self.device)
        elif self.config.model == 'CNN':
            self.model = ArtifactDetectionCNN(feature_size).to(self.device)
        elif self.config.model == 'SincNet':
            self.model = ConvNet(sr=256,min_band_hz=1,kernel_mult=3.903).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        logging.info(f"Model: {self.config.model}, {n_params:,} parameters")

    def _init_training_components(self):
        counts = np.bincount(self.train_dataset.labels.astype(int), minlength=3).astype(np.float64)
        weights = np.sqrt(counts.sum() / (len(counts) * counts))
        weights = weights / weights.mean()
        class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        self.criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.early_stopping = EarlyStopping(patience=self.config.patience, min_delta=0)
        self.train_loader, self.val_loader = self._split_dataset()
        logging.info(f"Class counts: {counts.tolist()}, class weights: {weights.tolist()}")
        logging.info(f"Train batches/epoch: {len(self.train_loader):,}, "
                    f"Val batches/epoch: {len(self.val_loader):,}")

    def _init_metrics(self):
        """
        Initialize training and validation metric histories and the best validation loss.
        """
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')

    def _preprocess_data(self):
        """
        Preprocess the training and validation features and save the fitted preprocessors.
        
        Standard scaling is always applied. PCA is additionally applied when enabled in the
        configuration, retaining the configured preprocessing state for later evaluation.

        Both fit and transform are done in chunks of ``self.chunk_size`` rows so that, for
        large datasets (tens of millions of windows), only one chunk plus the output array
        needs to be resident in memory at a time instead of several full-size copies.
        """
        n_windows, n_features = self.train_dataset.features.shape
        logging.info(f"Preprocessing {n_windows:,} training windows x {n_features} features "
                     f"(chunk size {self.chunk_size:,})")

        with self._log_step("Fitting + applying StandardScaler", extra=f"{n_windows:,} windows"):
            self.train_dataset.features, scaler = self._scale_data(self.train_dataset.features)
        self._save_preprocessor(scaler, 'scaler.pkl')

        with self._log_step("Scaling validation set", extra=f"{len(self.val_dataset):,} windows"):
            self.val_dataset.features = self._transform_in_chunks(scaler, self.val_dataset.features)

        if self.config.pca:
            with self._log_step("Fitting + applying IncrementalPCA (95% variance)",
                                 extra=f"{n_windows:,} windows"):
                self.train_dataset.features, pca = self._apply_pca(self.train_dataset.features)
            logging.info(f"PCA reduced features {n_features} -> {pca.n_components_}")
            self._save_preprocessor(pca, 'pca.pkl')
            with self._log_step("Applying PCA to validation set"):
                self.val_dataset.features = self._transform_in_chunks(pca, self.val_dataset.features)
        # if self.config.ica:
        #     self.train_dataset.features, ica = self._apply_ica(self.train_dataset.features)
        #     self._save_preprocessor(ica, 'ica.pkl')
        #     self.val_dataset.features = ica.transform(self.val_dataset.features)


    def _scale_data(self, features):
        """
        Fit a StandardScaler and transform features in bounded-memory chunks.

        The scaler's running mean/variance are accumulated chunk-by-chunk via
        ``partial_fit`` (mathematically equivalent to fitting on the full array at once),
        then each chunk is transformed and written directly into a preallocated output
        array so the raw and scaled data are never both fully duplicated in memory.

        Parameters:
            features: Feature data to scale.

        Returns:
            tuple: The scaled feature data and the fitted standard scaler.
        """
        n_rows = features.shape[0]
        scaler = StandardScaler()

        for start, end in tqdm(list(_iter_chunks(n_rows, self.chunk_size)),
                                desc="Fitting StandardScaler", unit="chunk"):
            scaler.partial_fit(features[start:end])

        scaled_features = self._transform_in_chunks(scaler, features)
        return scaled_features, scaler

    def _transform_in_chunks(self, transformer, features):
        """
        Apply an already-fitted transformer's ``.transform`` chunk-by-chunk.

        Parameters:
            transformer: A fitted StandardScaler, PCA, or IncrementalPCA instance.
            features: Feature array to transform.

        Returns:
            np.ndarray: The transformed features, assembled from per-chunk results.
        """
        n_rows = features.shape[0]
        out_dim = getattr(transformer, 'n_components_', features.shape[1])
        out = np.empty((n_rows, out_dim), dtype=np.float32)

        for start, end in tqdm(list(_iter_chunks(n_rows, self.chunk_size)),
                                desc=f"Applying {type(transformer).__name__}", unit="chunk"):
            out[start:end] = transformer.transform(features[start:end])
        return out

    # def _apply_ica(self, features):
    #     ica = FastICA(n_components=80, random_state=10)
    #     ica_features = ica.fit_transform(features)
    #     return ica_features, ica

    def _apply_pca(self, features):
        """
        Fit an IncrementalPCA to the features, retaining ~95% of the variance, using
        bounded-memory chunked passes instead of a single full-array ``fit_transform``.

        IncrementalPCA requires a fixed integer component count up front (unlike
        ``PCA(n_components=0.95)``), so the target count is first estimated by fitting a
        regular PCA on a subsample, then IncrementalPCA is fit/applied over the full
        dataset in chunks.

        Parameters:
            features: The feature data to transform.

        Returns:
            tuple: The transformed features and fitted IncrementalPCA transformer.
        """
        n_rows, n_features = features.shape
        n_components = self._estimate_pca_components(features, target_variance=0.95)

        pca = IncrementalPCA(n_components=n_components, batch_size=self.chunk_size)
        for start, end in tqdm(list(_iter_chunks(n_rows, self.chunk_size)),
                                desc="Fitting IncrementalPCA", unit="chunk"):
            chunk = features[start:end]
            # IncrementalPCA requires each partial_fit batch to have at least
            # n_components rows; skip a too-small trailing remainder (negligible
            # impact on the running fit given the size of the full dataset).
            if chunk.shape[0] >= n_components:
                pca.partial_fit(chunk)

        pca_features = self._transform_in_chunks(pca, features)
        return pca_features, pca

    def _estimate_pca_components(self, features, target_variance=0.95, max_sample=200_000):
        """
        Estimate how many principal components are needed to reach a target explained
        variance ratio, using a random subsample so the estimate itself stays cheap on
        very large datasets.

        Parameters:
            features: Full feature array to sample from.
            target_variance (float): Desired cumulative explained variance ratio.
            max_sample (int): Maximum number of rows to use for the estimate.

        Returns:
            int: Number of components needed to reach ``target_variance``.
        """
        n_rows = features.shape[0]
        sample_size = min(max_sample, n_rows)
        with self._log_step(f"Estimating PCA components for {target_variance:.0%} variance",
                             extra=f"sampling {sample_size:,} of {n_rows:,} windows"):
            rng = np.random.default_rng(seed=0)
            idx = rng.choice(n_rows, size=sample_size, replace=False)
            idx.sort()  # sorted indexing is faster/safer for most array-like backends
            sample_pca = PCA(n_components=target_variance)
            sample_pca.fit(features[idx])
            n_components = sample_pca.n_components_
        logging.info(f"Estimated {n_components} components reach {target_variance:.0%} variance "
                     f"(from a {sample_size:,}-window sample)")
        return n_components

    def _save_preprocessor(self, preprocessor, filename):
        """
        Save a preprocessing object to the configured save directory.
        
        Parameters:
            preprocessor: The preprocessing object to serialize.
            filename (str): The output filename.
        """
        with open(os.path.join(self.config.save_path, filename), 'wb') as f:
            pickle.dump(preprocessor, f)
        logging.info(f"Saved preprocessor: {filename}")

    def _load_preprocessing(self):
        """
        Apply the saved preprocessing artifacts to all test datasets.
        
        The scaler is always applied. PCA and ICA transformations are applied when enabled in the configuration.
        """
        for snr, test_dataset in tqdm(self.test_datasets.items(), desc="Preprocessing test SNRs", unit="snr"):
            t0 = time.time()
            scaler = self._load_preprocessor('scaler.pkl')
            test_dataset.features = self._transform_in_chunks(scaler, test_dataset.features)
            if self.config.pca:
                pca = self._load_preprocessor('pca.pkl')
                test_dataset.features = self._transform_in_chunks(pca, test_dataset.features)
            if self.config.ica:
                ica = self._load_preprocessor('ica.pkl')
                test_dataset.features = ica.transform(test_dataset.features)
            logging.info(f"  SNR {snr}: preprocessed {len(test_dataset):,} windows "
                         f"in {_fmt_secs(time.time() - t0)}")


    def _load_preprocessor(self, filename):
        """
        Load a serialized preprocessing object from the configured save directory.
        
        Parameters:
            filename (str): Name of the serialized preprocessor file.
        
        Returns:
            object: The deserialized preprocessing object.
        """
        with open(os.path.join(self.config.save_path, filename), 'rb') as f:
            return pickle.load(f)

    def _split_dataset(self):
        """
        Create data loaders for the training and validation datasets.
        
        Returns:
            tuple: A training data loader with shuffling enabled and a validation data loader with shuffling disabled.
        """
        train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.config.batch_size, shuffle=False)
        return train_loader, val_loader

    def train_one_epoch(self, epoch):
        """
        Train the model for one epoch and record its training metrics.
        
        Parameters:
        	epoch (int): The current training epoch number.
        """
        self.model.train()
        running_loss, all_labels, all_preds = 0.0, [], []
        n_seen = 0
        t0 = time.time()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs} [train]",
                    unit="batch", leave=False)
        for batch_features, batch_labels in pbar:
            batch_features, batch_labels = batch_features.to(self.device), batch_labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(batch_features.float())
            loss = self.criterion(outputs, batch_labels.long())
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            n_seen += batch_labels.size(0)
            all_labels.extend(batch_labels.cpu().numpy())
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())

            elapsed = time.time() - t0
            windows_per_sec = n_seen / elapsed if elapsed > 0 else 0.0
            pbar.set_postfix(loss=f"{loss.item():.4f}", win_s=f"{windows_per_sec:,.0f}/s")

        self._log_epoch_metrics(epoch, running_loss, all_labels, all_preds, 'Training',
                                 duration=time.time() - t0)

    def validate_one_epoch(self, epoch):
        """
        Evaluate the model on the validation dataset and save a checkpoint when validation loss improves.
        
        Parameters:
        	epoch (int): Training epoch used when recording validation metrics.
        """
        self.model.eval()
        val_loss, all_val_labels, all_val_preds = 0.0, [], []
        t0 = time.time()

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs} [val]",
                    unit="batch", leave=False)
        with torch.no_grad():
            for val_features, val_labels in pbar:
                val_features, val_labels = val_features.to(self.device), val_labels.to(self.device)
                val_outputs = self.model(val_features.float())
                loss = self.criterion(val_outputs, val_labels.long())
                val_loss += loss.item()
                all_val_labels.extend(val_labels.cpu().numpy())
                _, val_preds = torch.max(val_outputs, 1)
                all_val_preds.extend(val_preds.cpu().numpy())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        self._log_epoch_metrics(epoch, val_loss, all_val_labels, all_val_preds, 'Validation',
                                 duration=time.time() - t0)

        if val_loss < self.best_val_loss:
            improvement = self.best_val_loss - val_loss if self.best_val_loss != float('inf') else 0.0
            self.best_val_loss = val_loss
            self._save_checkpoint()
            if improvement:
                logging.info(f"  New best val loss (improved by {improvement:.4f})")

    def _log_epoch_metrics(self, epoch, running_loss, all_labels, all_preds, phase, duration=None):
        """
        Record and display loss and classification metrics for a training or validation epoch.
        
        Parameters:
        	epoch (int): Zero-based epoch index.
        	running_loss (float): Total loss accumulated during the epoch.
        	all_labels (array-like): True class labels for the epoch.
        	all_preds (array-like): Predicted class labels for the epoch.
        	phase (str): Epoch phase, either ``'Training'`` or ``'Validation'``.
        	duration (float): Optional wall-clock seconds this phase took, appended to the log line.
        """
        avg_loss = running_loss / len(self.train_loader if phase == 'Training' else self.val_loader)
        acc, f1, precision, recall = calculate_metrics(all_labels, all_preds)
        duration_str = f", Time: {_fmt_secs(duration)}" if duration is not None else ""
        metrics_log = (f"[{phase}] Epoch {epoch + 1}/{self.config.num_epochs}, Loss: {avg_loss:.4f}, "
                       f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
                       f"{duration_str}")
        logging.info(metrics_log)
        print(termcolor.colored(metrics_log, 'green' if phase == 'Training' else 'blue'))

        if phase == 'Training':
            self.train_losses.append(avg_loss)
            self.train_accuracies.append(acc)
        else:
            self.val_losses.append(avg_loss)
            self.val_accuracies.append(acc)

    def _save_checkpoint(self):
        """Save the current model as the best-model checkpoint."""
        checkpoint_path = os.path.join(self.config.save_path, 'best_model.pth')
        torch.save(self.model, checkpoint_path)
        logging.info(f"Model checkpoint saved at {checkpoint_path}")

    def test(self):
        """
        Evaluate the best model across all test datasets and plot accuracy by SNR.
        
        The test datasets are processed in ascending numerical SNR order, and each
        dataset's evaluation results contribute to the final SNR accuracy plot.
        """
        test_accuracies, snr_values = [], []

        def sort_key(item):
            name = str(item[0])
            try:
                return (0, float(name))
            except (TypeError, ValueError):
                return (1, name)

        self.test_datasets = dict(sorted(self.test_datasets.items(), key=sort_key))
        self._load_best_model()

        with self._log_step(f"Testing across {len(self.test_datasets)} SNR levels"):
            for snr_value, test_dataset in tqdm(self.test_datasets.items(), desc="Testing SNRs", unit="snr"):
                test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
                self._evaluate_test_set(test_loader, snr_value, test_accuracies, snr_values)

        self._plot_test_results(snr_values, test_accuracies)

    def _load_best_model(self):
        """
        Load the best saved model checkpoint and move it to the configured device.
        """
        import torch
        from models import ArtifactDetectionNN
        
        # Allow custom model class (required for PyTorch >= 2.6)
        torch.serialization.add_safe_globals([ArtifactDetectionNN])
        
        model_path = os.path.join(self.config.save_path, 'best_model.pth')
        with self._log_step("Loading best model checkpoint"):
            self.model = torch.load(model_path, weights_only=False)
            self.model.to(self.device)
        print(f"[INFO] Successfully loaded best model from {model_path}")

    def _evaluate_test_set(self, test_loader, snr_value, test_accuracies, snr_values):
        """
        Evaluate the model on a test dataset and record its classification metrics.
        
        Parameters:
            test_loader: DataLoader providing test features and labels.
            snr_value: Signal-to-noise ratio associated with the test dataset.
            test_accuracies: List to receive the computed test accuracy.
            snr_values: List to receive the corresponding SNR value.
        """
        self.model.eval()
        test_loss, correct, total = 0.0, 0, 0
        all_test_labels, all_test_preds = [], []
        t0 = time.time()

        with torch.no_grad():
            for test_features, test_labels in test_loader:
                test_features, test_labels = test_features.to(self.device), test_labels.to(self.device)
                test_outputs = self.model(test_features.float())
                loss = self.criterion(test_outputs, test_labels.long())
                test_loss += loss.item()
                _, test_preds = torch.max(test_outputs, 1)
                all_test_labels.extend(test_labels.cpu().numpy())
                all_test_preds.extend(test_preds.cpu().numpy())
                total += test_labels.size(0)
                correct += (test_preds == test_labels).sum().item()

        self._log_test_metrics(test_loader, test_loss, snr_value, all_test_labels, all_test_preds,
                                test_accuracies, snr_values, duration=time.time() - t0)
        self._plot_confusion_matrix(all_test_labels, all_test_preds, snr_value)


    def _log_test_metrics(self, test_loader, test_loss, snr_value, all_test_labels, all_test_preds,
                           test_accuracies, snr_values, duration=None):
        """
        Record classification metrics for a test dataset and save the results for its SNR value.
        
        Parameters:
        	test_loader: The data loader used to determine the average test loss.
        	test_loss: The accumulated test loss.
        	snr_value: The signal-to-noise ratio associated with the test dataset.
        	all_test_labels: The ground-truth class labels.
        	all_test_preds: The predicted class labels.
        	test_accuracies: List to which the test accuracy is appended.
        	snr_values: List to which the SNR value is appended.
        	duration (float): Optional wall-clock seconds this evaluation took.
        """
        test_acc, test_f1, test_precision, test_recall = calculate_metrics(all_test_labels, all_test_preds)
        test_accuracies.append(test_acc)
        snr_values.append(snr_value)
        avg_test_loss = test_loss / len(test_loader)
        duration_str = f", Time: {_fmt_secs(duration)}" if duration is not None else ""
        metrics_log = (f"[Test] SNR: {snr_value}, Loss: {avg_test_loss:.4f}, Accuracy: {test_acc:.4f}, "
                       f"F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}"
                       f"{duration_str}")
        logging.info(metrics_log)
        print(metrics_log)
        self._save_test_results(snr_value, test_acc, test_f1, test_precision, test_recall)

    def _plot_confusion_matrix(self, test_labels, test_preds, snr):
        """
        Plot and save a confusion matrix for predictions at a specified signal-to-noise ratio.
        
        Parameters:
        	test_labels: Ground-truth class labels.
        	test_preds: Predicted class labels.
        	snr: Signal-to-noise ratio associated with the test results.
        """
        cm = confusion_matrix(test_labels, test_preds, labels=[0,1,2])
        plt.figure(figsize=(10, 7))
        class_names = ['EEG', 'EOG', 'EMG']
        sns.heatmap(cm, annot=True, fmt='g', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix, SNR: {snr}dB')
        plt.savefig(os.path.join(Path(self.config.outputpath) / Path('cnf_matrices'), f'confusion_matrix_{snr}.png'))
        plt.close()

    def _save_test_results(self, snr_value, test_acc, test_f1, test_precision, test_recall):
        """
        Append test metrics for an SNR value to its results CSV file.
        
        Parameters:
        	snr_value: The signal-to-noise ratio associated with the test results.
        	test_acc: The test accuracy.
        	test_f1: The test F1 score.
        	test_precision: The test precision.
        	test_recall: The test recall.
        """
        res_path = os.path.join(self.config.outputpath, f'results_{snr_value}.csv')
        with open(res_path, 'a') as f:
            if os.stat(res_path).st_size == 0:
                f.write('SNR,Accuracy,F1,Precision,Recall\n')
            f.write(f'{snr_value},{test_acc},{test_f1},{test_precision},{test_recall}\n')

    def _plot_test_results(self, snr_values, test_accuracies):
        """
        Plot test accuracy by signal-to-noise ratio and save the resulting figure.
        
        Parameters:
            snr_values: Signal-to-noise ratio values for the horizontal axis.
            test_accuracies: Test accuracy corresponding to each signal-to-noise ratio.
        """
        plt.figure(figsize=(15, 5))
        plt.plot(snr_values, test_accuracies, marker='o', color='b')
        plt.xlabel('SNR [dB]')
        plt.xticks(snr_values)
        plt.ylabel('Test accuracy')
        plt.yticks(np.arange(0.6, 1.05, 0.05))
        # plt.title('Relationship between SNR and classification accuracy')
        plt.grid(True)
        plt.savefig(os.path.join(self.config.outputpath, 'snr_accuracy.png'))
        if not self.config.no_plot:
            plt.show(block=False)

    def plot_metrics(self):
        """
        Plot training and validation loss and accuracy curves, saving the combined figure to the configured output directory.
        """
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch',fontweight='bold')
        plt.ylabel('Loss',fontweight='bold')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Training Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch',fontweight='bold')
        plt.ylabel('Accuracy',fontweight='bold')
        plt.legend()
        plt.savefig(os.path.join(self.config.outputpath, f'combined_curves.png'))
        if not self.config.no_plot:
            plt.show(block=False)

    def run(self):
        """
        Run the configured training or testing workflow.
        
        Training mode runs training, generates training metrics plots, and evaluates the test datasets. Test mode evaluates the test datasets using the saved model.
        """
        run_t0 = time.time()
        if self.config.mode == 'train':
            self._train()
            self.plot_metrics()
            self.test()
        elif self.config.mode == 'test':
            self.test()
        logging.info(f"=== Run finished in {_fmt_secs(time.time() - run_t0)} ===")

    def _train(self):
        """Run the training loop for the configured number of epochs, stopping early when validation loss no longer improves."""
        train_t0 = time.time()
        for epoch in range(self.config.num_epochs):
            epoch_t0 = time.time()

            self.train_one_epoch(epoch)
            self.validate_one_epoch(epoch)
            self.scheduler.step(self.val_losses[-1])

            epoch_duration = time.time() - epoch_t0
            self._epoch_durations.append(epoch_duration)
            avg_epoch_time = sum(self._epoch_durations) / len(self._epoch_durations)
            remaining_epochs = self.config.num_epochs - (epoch + 1)
            eta = avg_epoch_time * remaining_epochs
            logging.info(f"Epoch {epoch + 1}/{self.config.num_epochs} complete in "
                         f"{_fmt_secs(epoch_duration)} (avg {_fmt_secs(avg_epoch_time)}/epoch, "
                         f"ETA {_fmt_secs(eta)})")
            print(termcolor.colored(
                f"--- Epoch {epoch + 1}/{self.config.num_epochs} done in {_fmt_secs(epoch_duration)} "
                f"| ETA {_fmt_secs(eta)} ---", 'cyan'))

            if self.early_stopping(self.val_losses[-1]):
                logging.info(f"Early stopping at epoch {epoch + 1}")
                print("Early stopping")
                break

        logging.info(f"Training loop total time: {_fmt_secs(time.time() - train_t0)}")