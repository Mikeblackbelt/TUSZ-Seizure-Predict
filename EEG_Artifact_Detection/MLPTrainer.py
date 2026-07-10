import os
import datetime
import logging
import pickle
from pathlib import Path
import termcolor
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from models import ArtifactDetectionNN,ArtifactDetectionCNN,ConvNet
from dataset import EEGDataset
from datanoise_combiner import DataNoiseCombiner
from utils import calculate_metrics, setup_logging, EarlyStopping
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

run_datetime = datetime.datetime.now()
plt.rcParams.update({'font.size': 14})


class MLPTrainer:
    def __init__(self, config):
        """Initialize the trainer with the provided configuration and runtime components.
        
        Parameters:
        	config: Configuration containing dataset paths, model settings, preprocessing options, and training parameters.
        """
        self.config = config
        self.device = self._setup_device()
        self._setup_directories()
        self._setup_logging()
        self._init_data_combiner()
        self._load_datasets()
        self._setup_preprocessing()
        self._init_model()
        self._init_training_components()
        self._init_metrics()

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

    def _init_data_combiner(self):
        """Initialize the data noise combiner using the trainer configuration."""
        DataNoiseCombiner(self.config)

    def _load_datasets(self):
        """
        Load the training, validation, and SNR-specific test datasets from the configured data directory.
        """
        self.train_dataset = EEGDataset(Path(self.config.datapath) / "train")
        self.val_dataset = EEGDataset(Path(self.config.datapath) / "val")
        self.test_datasets = self._load_test_datasets(Path(self.config.datapath) / "test")

    def _load_test_datasets(self, test_dir):
        """
        Load test datasets keyed by the SNR value extracted from each subdirectory name.
        
        Parameters:
        	test_dir (Path): Directory containing one subdirectory per test SNR.
        
        Returns:
        	dict: Mapping of SNR values to their corresponding EEG datasets.
        """
        test_datasets = {}
        for snr_dir in test_dir.iterdir():
            if snr_dir.is_dir():
                snr_value = snr_dir.name.split(' ')[-1]
                test_datasets[snr_value] = EEGDataset(snr_dir)
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
        if self.config.model == 'MLP':
            self.model = ArtifactDetectionNN(feature_size).to(self.device)
        elif self.config.model == 'CNN':
            self.model = ArtifactDetectionCNN(feature_size).to(self.device)
        elif self.config.model == 'SincNet':
            self.model = ConvNet(sr=256,min_band_hz=1,kernel_mult=3.903).to(self.device)

    def _init_training_components(self):
        """Initialize the loss function, optimizer, early-stopping handler, and training data loaders."""
        self.criterion = CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.early_stopping = EarlyStopping(patience=20, min_delta=0)
        self.train_loader, self.val_loader = self._split_dataset()

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
        Preprocesses the training and validation features and saves the fitted preprocessors.
        
        Standard scaling is always applied. Principal component analysis is also applied when enabled in the configuration.
        """
        self.train_dataset.features, scaler = self._scale_data(self.train_dataset.features)
        self._save_preprocessor(scaler, 'scaler.pkl')
        self.val_dataset.features = scaler.transform(self.val_dataset.features)
        if self.config.pca:
            self.train_dataset.features, pca = self._apply_pca(self.train_dataset.features)
            self._save_preprocessor(pca, 'pca.pkl')
            self.val_dataset.features = pca.transform(self.val_dataset.features)
        # if self.config.ica:
        #     self.train_dataset.features, ica = self._apply_ica(self.train_dataset.features)
        #     self._save_preprocessor(ica, 'ica.pkl')
        #     self.val_dataset.features = ica.transform(self.val_dataset.features)


    def _scale_data(self, features):
        """
        Scale feature data using a fitted standard scaler.
        
        Parameters:
            features: The feature data to fit and transform.
        
        Returns:
            tuple: The scaled feature data and the fitted scaler.
        """
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        return scaled_features, scaler

    # def _apply_ica(self, features):
    #     ica = FastICA(n_components=80, random_state=10)
    #     ica_features = ica.fit_transform(features)
    #     return ica_features, ica

    def _apply_pca(self, features):
        """
        Fit PCA to the features while retaining 95% of the variance.
        
        Parameters:
            features: The feature data to transform.
        
        Returns:
            tuple: The transformed features and fitted PCA transformer.
        """
        pca = PCA(n_components=0.95)
        pca_features = pca.fit_transform(features)
        return pca_features, pca

    def _save_preprocessor(self, preprocessor, filename):
        """
        Save a preprocessing object to the configured save directory.
        
        Parameters:
            preprocessor: The preprocessing object to serialize.
            filename (str): The output filename.
        """
        with open(os.path.join(self.config.save_path, filename), 'wb') as f:
            pickle.dump(preprocessor, f)

    def _load_preprocessing(self):
        """
        Apply the saved preprocessing artifacts to all test datasets.
        
        The scaler is always applied. PCA and ICA transformations are applied when enabled in the configuration.
        """
        for snr, test_dataset in self.test_datasets.items():
            scaler = self._load_preprocessor('scaler.pkl')
            test_dataset.features = scaler.transform(test_dataset.features)
            if self.config.pca:
                pca = self._load_preprocessor('pca.pkl')
                test_dataset.features = pca.transform(test_dataset.features)
            if self.config.ica:
                ica = self._load_preprocessor('ica.pkl')
                test_dataset.features = ica.transform(test_dataset.features)


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
        """Train the model for one epoch and record its training metrics.
        
        Parameters:
        	epoch (int): The current training epoch number.
        """
        self.model.train()
        running_loss, all_labels, all_preds = 0.0, [], []

        for batch_features, batch_labels in self.train_loader:
            batch_features, batch_labels = batch_features.to(self.device), batch_labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(batch_features.float())
            loss = self.criterion(outputs, batch_labels.long())
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            all_labels.extend(batch_labels.cpu().numpy())
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())

        self._log_epoch_metrics(epoch, running_loss, all_labels, all_preds, 'Training')

    def validate_one_epoch(self, epoch):
        """
        Evaluate the model on the validation dataset for one epoch and save improved checkpoints.
        
        Parameters:
        	epoch (int): The current training epoch.
        """
        self.model.eval()
        val_loss, all_val_labels, all_val_preds = 0.0, [], []

        with torch.no_grad():
            for val_features, val_labels in self.val_loader:
                val_features, val_labels = val_features.to(self.device), val_labels.to(self.device)
                val_outputs = self.model(val_features.float())
                loss = self.criterion(val_outputs, val_labels.long())
                val_loss += loss.item()
                all_val_labels.extend(val_labels.cpu().numpy())
                _, val_preds = torch.max(val_outputs, 1)
                all_val_preds.extend(val_preds.cpu().numpy())

        self._log_epoch_metrics(epoch, val_loss, all_val_labels, all_val_preds, 'Validation')

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._save_checkpoint()

    def _log_epoch_metrics(self, epoch, running_loss, all_labels, all_preds, phase):
        """
        Record and display loss and classification metrics for a training or validation epoch.
        
        Parameters:
        	epoch (int): Zero-based epoch index.
        	running_loss (float): Total loss accumulated during the epoch.
        	all_labels (array-like): True class labels for the epoch.
        	all_preds (array-like): Predicted class labels for the epoch.
        	phase (str): Epoch phase, either ``'Training'`` or ``'Validation'``.
        """
        avg_loss = running_loss / len(self.train_loader if phase == 'Training' else self.val_loader)
        acc, f1, precision, recall = calculate_metrics(all_labels, all_preds)
        metrics_log = (f"[{phase}] Epoch {epoch + 1}/{self.config.num_epochs}, Loss: {avg_loss:.4f}, "
                       f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
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

        self.test_datasets = dict(sorted(self.test_datasets.items(), key=lambda x: float(x[0])))
        for snr_value, test_dataset in self.test_datasets.items():
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
            self._load_best_model()
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
        self.model = torch.load(model_path, weights_only=False)
        self.model.to(self.device)
        print(f"[INFO] Successfully loaded best model from {model_path}")

    def _evaluate_test_set(self, test_loader, snr_value, test_accuracies, snr_values):
        """
        Evaluate the model on a test dataset and record its classification results.
        
        Parameters:
        	test_loader: DataLoader providing test features and labels.
        	snr_value: Signal-to-noise ratio associated with the test dataset.
        	test_accuracies: List to which the test accuracy is appended.
        	snr_values: List to which the corresponding SNR value is appended.
        """
        self.model.eval()
        test_loss, correct, total = 0.0, 0, 0
        all_test_labels, all_test_preds = [], []

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

        self._log_test_metrics(test_loader, test_loss, snr_value, all_test_labels, all_test_preds, test_accuracies, snr_values)
        self._plot_confusion_matrix(all_test_labels, all_test_preds, snr_value)


    def _log_test_metrics(self, test_loader, test_loss, snr_value, all_test_labels, all_test_preds, test_accuracies, snr_values):
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
        """
        test_acc, test_f1, test_precision, test_recall = calculate_metrics(all_test_labels, all_test_preds)
        test_accuracies.append(test_acc)
        snr_values.append(snr_value)
        avg_test_loss = test_loss / len(test_loader)
        metrics_log = (f"[Test] SNR: {snr_value}, Loss: {avg_test_loss:.4f}, Accuracy: {test_acc:.4f}, "
                       f"F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
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
        Plot test accuracy against SNR and save the resulting figure.
        
        Parameters:
            snr_values: SNR values represented on the horizontal axis.
            test_accuracies: Test accuracy corresponding to each SNR value.
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
            plt.show()

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
            plt.show()

    def run(self):
        """
        Run the configured training or testing workflow.
        
        Training mode runs training, generates training metrics plots, and evaluates the test datasets. Test mode evaluates the test datasets using the saved model.
        """
        if self.config.mode == 'train':
            self._train()
            self.plot_metrics()
            self.test()
        elif self.config.mode == 'test':
            self.test()

    def _train(self):
        """Run the training loop for the configured number of epochs, stopping early when validation loss no longer improves."""
        for epoch in tqdm(range(self.config.num_epochs)):
            self.train_one_epoch(epoch)
            self.validate_one_epoch(epoch)
            if self.early_stopping(self.val_losses[-1]):
                logging.info("Early stopping")
                print("Early stopping")
                break