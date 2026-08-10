import numpy as np
import os
import glob
import math
from scipy.stats import kurtosis, skew

n_channels = 22
sample_rate = 256
window_size = 64

processed_data_dir = "processed_data"
feature_output_dir = "processed_data/features"

def det_entropy(channel_data):
    z = np.abs(channel_data)
    entropy = 0
    for i in range(len(channel_data)):
        entropy += z[i] * math.log(z[i], 2)
    return -entropy

def gen_time_domain_features(data):
    features = []
    for channel in range(data.shape[0]):
        channel_data = np.abs(data[channel])

        features.append(np.mean(channel_data))
        features.append(np.var(channel_data))
        features.append(skew(channel_data))
        features.append(kurtosis(channel_data))
        features.append((math.sqrt(np.var(channel_data)) // np.mean(channel_data)))
        features.append(np.mean(np.abs(channel_data - np.mean(channel_data))))
        features.append(np.sqrt(np.mean(channel_data ** 2)))
        features.append(det_entropy(channel_data))

    return np.array(features)

def save_features(features, file):
    out_path = os.path.join(feature_output_dir, os.path.basename(file))
    np.save(out_path, features)

if __name__ == "__main__":
    files = glob.glob(os.path.join(processed_data_dir, "*.npy"))
    for file in files:
        print("Processing:", file)
        try:
            data = np.load(file)  # shape (channels, samples)
            features = gen_time_domain_features(data)
            save_features(features, file)
        except Exception as e:
            print("Failed:", file)
            print(e)
