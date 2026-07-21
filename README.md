# Tuh-Preprocess

An EEG seizure prediction/categorization preprocessing pipeline built on the [TUSZ (Temple University Seizure) corpus](https://isip.piconepress.com/projects/tuh_eeg/). The repo has two processes that plug into each other:

1. **Annotation pipeline** (`pipeline/preictal_segment.py`) - scans a TUSZ dataset directory, builds a master CSV of seizure events, and generates **preictal**, **postictal**, and **consecutive** windows around each event. At the point of writing (7/20/2026), the master csv is outdated as it fails to consider that sessions are segmented into files. This will be fixed in the upcoming commits. 
2. **Signal pipeline** (`pipeline/process_signal.py`, `pipeline/resampling.py`, `filters/`, `pipeline/artifact_detection.py`, `pipeline/artifact_masking.py`) - loads and standardizes raw `.edf` recordings, resamples and filters them, and detects/masks non-cerebral artifacts (EOG/EMG) using a classifier trained in the standalone `EEG_Artifact_Detection/` module.

## Attribution

- The EOG/EMG artifact-classification approach (`EEG_Artifact_Detection/`, used by `pipeline/artifact_detection.py`) follows Hossein Enshaei, Pariya Jebreili, and Sayed Mahmoud Sakhaei, *"Real-time Noise Detection and Classification in Single-Channel EEG: A Lightweight Machine Learning Approach for EMG, White Noise, and EOG Artifacts"* (Babol Noshirvani University of Technology).
- The `SincNet` model option (`EEG_Artifact_Detection/models.py`) is adapted from Francesco Paissan's implementation of SincNet, based on Mirco Ravanelli and Yoshua Bengio, *"Speaker Recognition from Raw Waveform with SincNet"*.
- The adaptive notch filter (`filters/adaptive_filters.py`) uses the methods of Mahdi, M., & Baghdadi, A. (2026).

## Requirements

- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
pip install -r EEG_Artifact_Detection/requirements.txt   # only needed to (re)train the artifact classifier
```

## 1. Annotation pipeline

### What this does

1. Walks a TUSZ dataset directory and reads every `.csv` annotation file (each row is a labeled time segment (e.g. `fnsz`, `gnsz`, `bckg`) tied to a `.edf` recording and a specific `channel`).
2. Filters those rows down to a set of seizure tags you select.
3. Builds a single **master CSV** combining all selected events across the dataset, tagged with which TUSZ split (`train` / `dev` / `eval`) each row belongs to.
4. For every event row, generates a matching **preictal row** (`p{label}`) - the window of time *before* the seizure starts.
5. For every event row, generates either:
   - a **postictal row** (`q{label}`) - the window of time *after* the seizure ends, if the gap to the next seizure on the same channel is large enough, or
   - a **consecutive row** (`c{label1}{label2}`, or `c{label}2` if both seizures share a label) - spanning the gap to the next seizure, if that gap is too small to fit both a postictal and preictal window.
6. Resolves any remaining overlaps between generated windows sharing identical time boundaries on the same channel, keeping only the highest-priority row.
7. Writes the combined result (ictal + preictal + postictal/consecutive rows) back out as a single CSV, ready for downstream feature extraction / model training.

### Usage

The pipeline is driven directly through `pipeline/preictal_segment.py`'s functions - scan for tags, build the master file, then layer on preictal/postictal/consecutive tags:

```python
from pipeline import preictal_segment

tags = preictal_segment.get_unique_tags("<path-to-tusz-dataset>")

master_df = preictal_segment.make_master_file(
    "<path-to-tusz-dataset>",
    output_path="master_full.csv",
    allow_tag=tags,          # or a subset of tags
)
master_df = preictal_segment.add_preictal_tags(
    master_df, start_cutoff=300, max_duration=600,
)
master_df = preictal_segment.add_postictal_and_consecutive(
    master_df, postictal_length=300, preictal_length=600,
)
master_df.to_csv("master_full.csv", index=False)
```

`add_postictal_and_consecutive` calls `resolve_overlaps` automatically, so no separate call is needed. Before running against real data, validate the input path with `util.verify_data.validate_input(data_path)` (see [Data validation](#data-validation-utilverify_datapy) below).

### Pipeline internals (`pipeline/preictal_segment.py`)

#### `get_unique_tags(dataset_path)`

Recursively scans `dataset_path` for `.csv` annotation files and returns the set of every unique value in the `label` column across the whole dataset. Used to populate the tag-selection prompt and as the default filter (every tag) if none is explicitly provided.

#### `get_split(path)`

Infers which TUSZ split (`train`, `dev`, or `eval`) a file belongs to by checking the path's directory components (case-insensitive), matching TUSZ's native layout (e.g. `.../edf/train/01_tcp_ar/...`). Returns `"unknown"` if none of the three match, and logs a warning when that happens.

#### `make_master_file(dataset_path, output_path="master.csv", allow_tag=None)`

Walks `dataset_path`, and for every `.csv` annotation file with a matching `.edf` recording:

- Parses the CSV and filters rows down to `allow_tag` (defaults to every tag found via `get_unique_tags` if not provided).
- Tags each row with its `split` (via `get_split`) and a `status` of `-1` (not applicable - status only carries meaning for generated rows, see below).
- Skips files with no matching `.edf`, no rows in `allow_tag`, or that fail to parse, logging a count of each at the end of the scan.

Writes the combined result to `output_path` and returns it as a DataFrame. Returns `None` (and logs a warning) if no records were found.

**Output columns:** `edf_path`, `csv_path`, `split`, `channel`, `start_time`, `stop_time`, `label`, `confidence`, `status`

`status == -1` always identifies an original TUSZ row, never a generated one.

#### `add_preictal_tags(master_df, start_cutoff, max_duration)`

For every row in `master_df`, generates a corresponding preictal row labeled `p{original_label}` (e.g. `fnsz` → `pfnsz`), representing the window of time *before* the seizure that a model should learn to recognize as a warning sign.

The window is computed as:

```
raw_end   = ictal_start - start_cutoff
raw_start = raw_end - max_duration
```

- `start_cutoff` - gap (in seconds) between the seizure's onset and the end of the preictal window. Use this to avoid labeling time immediately adjacent to seizure onset as "preictal," since EEG artifacts there can already resemble ictal activity.
- `max_duration` - the maximum length (in seconds) of the preictal window itself.

Both values are clamped to never go negative, and a `status` column on every preictal row records what (if anything) got trimmed:

| Status | Meaning |
|---|---|
| `-1` | Not a preictal row (original ictal/background row) |
| `0` | Nothing trimmed - full window of `max_duration` seconds applied |
| `1` | `raw_start` was trimmed to `0` - window shortened (still ends at `raw_end`) |
| `2` | `raw_end` was trimmed to `0` - window collapsed entirely to `[0, 0]` (e.g. seizure starts before `start_cutoff` has elapsed) |

The returned DataFrame contains both the original rows and the new preictal rows, sorted by `split`, `edf_path`, then `start_time`.

#### `add_postictal_and_consecutive(master_df, postictal_length, preictal_length)`

For every original (non-preictal) row, grouped by `(edf_path, channel)` and walked in `start_time` order, decides between two outcomes based on the gap to the *next* seizure on that same channel:

```
gap = next_seizure.start_time - current_seizure.stop_time
```

- **If `gap < (postictal_length + preictal_length)`**: the two seizures are treated as consecutive - too close together for a full postictal window on the first and a full preictal window on the second to both fit without colliding. A single **consecutive row** is generated instead, labeled `c{label1}{label2}` (or `c{label}2` if both seizures share the same label), spanning from the first seizure's `stop_time` to `next.start_time - preictal_length`. If that computed end would fall at or before the start (i.e. the reserved preictal gap alone doesn't fit), the row is collapsed to zero-length and marked `status=2`; otherwise `status=0`. The loop then advances past both seizures (`i += 2`), since the second seizure's own postictal/consecutive decision is already accounted for by this row.
- **Otherwise**: a standard **postictal row** is generated, labeled `q{label}`, spanning from `stop_time` to `stop_time + postictal_length`, with `status=0`.
- **The last seizure in a channel's timeline** (no next seizure to compare against) always gets a standard postictal row.

After generating these rows, `resolve_overlaps` is called automatically on the combined DataFrame before it's returned, so callers don't need to invoke it separately.

The returned DataFrame contains the original rows, the new preictal rows (if already present), and the new postictal/consecutive rows, sorted by `split`, `edf_path`, `channel`, then `start_time`.

#### `resolve_overlaps(df)`

Assigns each row a priority based on its label prefix:

```
consecutive (c)  >  preictal (p)  >  original ictal  >  postictal (q)
```

Rows are sorted by `(edf_path, channel, start_time, priority)` and then deduplicated on exact `(edf_path, channel, start_time, stop_time)` matches, keeping only the highest-priority row in each group. This means `resolve_overlaps` removes rows that share **identical time boundaries** with a higher-priority row on the same channel - it does not clip or trim rows whose time ranges only partially overlap with different start/stop values; such partial overlaps are left in the output as-is.

### `pipeline/session_index.py`

`index_sessions(dataset_path)` walks a TUSZ split and groups every `.edf`/`.csv`/`.csv_bi` file by session, keyed by a composed ID like `trn_p001_s001_2015_ar1` (split prefix + patient ID + session ID + montage type, parsed from folder names like `01_tcp_ar`). Each entry records the split, patient/session IDs, montage type, and the sorted list of `.edf`/`.csv`/`.csv_bi` paths belonging to that session. Logs and skips directories where the split can't be determined or the directory depth doesn't match TUSZ's `<split>/<patient>/<session>/<recording>` layout.

## 2. Signal pipeline

### Loading and channel standardization (`pipeline/process_signal.py`)

- `load_edf(path)` - loads a `.edf` recording via MNE (`preload=False`) and returns it alongside a metadata DataFrame (path, channel names, sampling frequency, sample count, duration).
- `split_into_epochs(edf_path, epoch_duration=1)` - loads a recording and segments it into fixed-length, non-overlapping MNE `Epochs`.
- `standardize_channel_name(ch)` - strips the `EEG` prefix and `-LE`/`-REF` reference suffixes from a raw TUSZ channel name (e.g. `EEG FP1-REF` → `FP1`).
- `standardize_channels_names(raw, metadata)` - applies `standardize_channel_name` across a recording, drops non-electrode channels (e.g. `PHOTIC PH`), and updates the metadata's channel list to match.
- `drop_channels(raw, metadata, desired_order=standard_channels)` - standardizes channel names, then keeps and reorders only the 19 standard 10-20 channels (`FP1, FP2, F7, F3, FZ, F4, F8, T3, C3, CZ, C4, T4, T5, P3, PZ, P4, T6, O1, O2`). Returns `None` (and logs a warning) if any of those channels is missing from the recording; logs when extra channels are dropped.

### Resampling (`pipeline/resampling.py`)

- `resample_eeg(data, orig_fs, target_fs)` - polyphase-resamples `(n_channels, n_samples)` or `(n_samples,)` EEG data to a target sampling rate (`scipy.signal.resample_poly`, rational up/down factors from `Fraction(target_fs/orig_fs).limit_denominator(1000)`). No-ops if the rates already match.
- `rescale_sample_index(sample_idx, orig_fs, target_fs)` - converts a single sample index between sampling rates.

### Filtering (`filters/`)

- **`simple_filters.bandpass_filter_interval`** - applies a zero-phase Butterworth bandpass (default 0.5-40 Hz) to selected channels over a specific time interval, with a padded window around the interval to reduce edge transients (`sosfiltfilt`).
- **`adaptive_filters.detect_noise_frequencies`** / **`apply_notch_filter`** - identifies line-noise-like frequencies (high power, low cross-channel variation in the normalized Welch PSD, following Mahdi & Baghdadi (2026)) and notch-filters them out with MNE's `spectrum_fit` method.

### Artifact detection and masking

`pipeline/artifact_detection.py` (`ArtifactDetector`) loads the trained model/scaler/PCA from `EEG_Artifact_Detection/checkpoints/` (see [EEG_Artifact_Detection](#3-eeg_artifact_detection-training-the-artifact-classifier) below) and classifies EEG signal windows as clean/EOG/EMG:

- `predict_channel(channel, fs_in)` - resamples a channel to 256 Hz, segments it into contiguous 512-sample (2 s) windows, extracts features, and returns per-window class probabilities.
- `predict_segment(eeg_data, fs_in)` - runs `predict_channel` across every channel in a segment and returns per-channel probabilities, the mean artifact probability, and the total window count.

`pipeline/artifact_masking.py` turns those predictions into a native-rate boolean mask (`build_artifact_mask`, recomputing the window length in native samples since the detector's windows are fixed at 2 s regardless of the original sampling rate) and offers two ways to act on it:

- `apply_zero_masking` - zeros out flagged samples.
- `apply_interpolation_masking` - replaces flagged samples with a per-channel linear interpolation across the surrounding clean samples; channels that are fully flagged are left unchanged and reported separately.

## 3. `EEG_Artifact_Detection/` - training the artifact classifier

A standalone module (with its own `requirements.txt`) that trains the 3-class (clean EEG / EOG / EMG) classifier consumed by `pipeline/artifact_detection.py`.

1. Loads clean EEG epochs alongside EOG and EMG noise epochs, already bandpass-filtered to 0-80 Hz.
2. Synthetically combines clean EEG with EOG/EMG noise at a range of SNRs to build labeled 3-class data (`DataNoiseCombiner`), split into `train`/`val`/per-SNR `test` sets and z-scored per epoch.
3. Extracts a feature vector per epoch: a level-4 wavelet-style low-frequency approximation concatenated with the epoch's power spectral density.
4. Trains a classifier - `MLP` (`ArtifactDetectionNN`), 1D `CNN` (`ArtifactDetectionCNN`), or `SincNet` (`ConvNet` with a learnable `SincConv_fast` filter bank) - with optional PCA/ICA, early stopping, and checkpointing.
5. Evaluates the best checkpoint across every SNR level, logging accuracy/F1/precision/recall per SNR and saving accuracy-vs-SNR and per-SNR confusion-matrix plots.

```bash
cd EEG_Artifact_Detection
python main.py --model MLP --pca --ica   # or --model CNN / --model SincNet
```

Checkpoints (`best_model.pth`) and preprocessors (`scaler.pkl`, `pca.pkl`) are written to `checkpoints/`, which is exactly where `pipeline/artifact_detection.py` looks for them by default. See `EEG_Artifact_Detection/README.md` for the full argument reference, data layout, and internals of that module.

## Data validation (`util/verify_data.py`)

Call `validate_input(data_path)` before pointing the annotation pipeline at real data:

1. Confirms `data_path` exists (`verify_data_path`).
2. Confirms it's a non-empty directory (`verify_data_integrity`).
3. Lists every file in the directory for a quick visual sanity check (`list_files_glob`).
4. Asks for interactive `y`/`n` confirmation before the run proceeds.

## Logging and config (`util/handle_logs.py`)

- `get_logger(name, log_pseudo=None, level=logging.DEBUG)` returns a standard `logging.Logger` that always writes to stdout and, if `log_pseudo` is given, also to a file - either a direct path (if it contains a `/`/`\` or ends in `.log`) or a key looked up in `app_path.json`. Loggers are keyed by `name`; re-fetching the same name returns the same configured logger rather than duplicating handlers.
- `load_config()` / `save_config(config)` read and write `app_path.json`, which stores defaults like the log path across runs.

## Testing

The full suite lives in `testing/`. Run it with:

```bash
python -m pytest testing/ -v
```
or simply
```bash
pytest
```

| Directory / file | Covers |
|---|---|
| `testing/test_logging.py` | Logger creation and file output |
| `testing/testing_segmentation/` | `get_unique_tags`, `get_split`, `make_master_file`, `add_preictal_tags`, `add_postictal_and_consecutive`, `resolve_overlaps` |
| `testing/testing_filters/` | Bandpass and adaptive-notch filtering |
| `testing/testing_loading/` | EDF loading and channel standardization |
| `testing/testing_pipeline/` | Signal-processing pipeline components (resampling, artifact detection/masking) |

Shared fixtures (`write_csv`, `write_edf`, `dataset_dir`, and related helpers) live in `testing/helpers.py`.

## Project structure

```
.
├── app_path.json                     # saved defaults (e.g. log path)
├── pipeline/
│   ├── preictal_segment.py           # master-file / preictal / postictal / overlap logic
│   ├── session_index.py              # groups TUSZ files by session
│   ├── process_signal.py             # EDF loading + channel standardization
│   ├── resampling.py                 # polyphase resampling
│   ├── artifact_detection.py         # ArtifactDetector - loads the trained classifier for inference
│   └── artifact_masking.py           # turns detector output into a mask; zero/interpolation masking
├── filters/
│   ├── simple_filters.py             # bandpass filtering
│   └── adaptive_filters.py           # adaptive notch filtering (Mahdi & Baghdadi, 2026)
├── EEG_Artifact_Detection/           # standalone training module for the artifact classifier
│   └── README.md                     # full docs for this module
├── util/
│   ├── handle_logs.py                # shared logger factory + app_path.json config
│   └── verify_data.py                # input path validation
├── testing/
│   ├── helpers.py                    # shared pytest fixtures
│   ├── test_logging.py
│   ├── testing_segmentation/
│   ├── testing_filters/
│   ├── testing_loading/
│   └── testing_pipeline/
└── requirements.txt
```
