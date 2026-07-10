import os
import sys
import numpy as np
import torch
import pickle
from scipy.signal import resample_poly

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, "EEG_Artifact_Detection"))

from EEG_Artifact_Detection.feature_extraction import extract_features
from EEG_Artifact_Detection.models import ArtifactDetectionNN


class ArtifactDetector:
    def __init__(self, ckpt_dir=None, device="cpu"):
        self.device = torch.device(device)
        
        if ckpt_dir is None:
            ckpt_dir = os.path.join("EEG_Artifact_Detection", "checkpoints")
        
        self.ckpt_dir = ckpt_dir
        print(f"[INFO] Using checkpoints: {self.ckpt_dir}")

        self._load_model()
        self._load_preprocessors()

    def _load_model(self):
        model_path = os.path.join(self.ckpt_dir, "best_model.pth")
        print(f"[INFO] Loading model from {model_path}")
        
        torch.serialization.add_safe_globals([ArtifactDetectionNN])
        import EEG_Artifact_Detection.models as models_mod
        torch.serialization.add_safe_globals([models_mod])

        self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.to(self.device)
        self.model.eval()
        print("[SUCCESS] Model loaded")

    def _load_preprocessors(self):
        scaler_path = os.path.join(self.ckpt_dir, "scaler.pkl")
        pca_path = os.path.join(self.ckpt_dir, "pca.pkl")

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        print("[SUCCESS] Scaler loaded")

        if os.path.exists(pca_path):
            with open(pca_path, "rb") as f:
                self.pca = pickle.load(f)
            print("[SUCCESS] PCA loaded")
            self.use_pca = True
        else:
            print("[WARNING] pca.pkl not found → Running without PCA")
            self.pca = None
            self.use_pca = False

    def _resample_to_256(self, ch, fs_in):
        if fs_in == 256:
            return np.asarray(ch, dtype=np.float64)
        return resample_poly(ch, 256, fs_in)

    def _segment_into_windows(self, sig):
        sig = np.asarray(sig, dtype=np.float64)
        n = (len(sig) // 512) * 512
        if n == 0:
            return np.empty((0, 512))
        return sig[:n].reshape(-1, 512)

    def predict_channel(self, channel, fs_in):
        ch256 = self._resample_to_256(channel, fs_in)
        windows = self._segment_into_windows(ch256)
        if len(windows) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        feats = extract_features(windows)
        feats = self.scaler.transform(feats)
        
        if self.use_pca and self.pca is not None:
            feats = self.pca.transform(feats)

        with torch.no_grad():
            x = torch.tensor(feats, dtype=torch.float32, device=self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict_segment(self, eeg_data, fs_in):
        eeg_data = np.asarray(eeg_data)
        if eeg_data.ndim == 1:
            eeg_data = eeg_data.reshape(1, -1)

        all_probs = [self.predict_channel(ch, fs_in) for ch in eeg_data]
        flat_probs = np.vstack([p for p in all_probs if len(p) > 0]) if all_probs else np.zeros((0, 3))
        artifact_frac = flat_probs[:, 1:].sum(axis=1).mean() if len(flat_probs) > 0 else 0.0

        return {
            "per_channel_probs": all_probs,
            "artifact_fraction": float(artifact_frac),
            "total_windows": len(flat_probs)
        }


if __name__ == "__main__":
    print("\n=== Testing ArtifactDetector ===")
    detector = ArtifactDetector()
    
    eeg_signal = np.random.randn(2, 5120).astype(np.float64)
    fs_original = 512
    
    print("Running prediction on test signal...")
    results = detector.predict_segment(eeg_signal, fs_original)
    
    print(f"Artifact Fraction: {results['artifact_fraction']:.4f}")
    print(f"Total windows: {results['total_windows']}")
    print("Success!")