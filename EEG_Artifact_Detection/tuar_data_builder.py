"""TUAR dataset builder helpers for EEG_Artifact_Detection."""
import csv
import difflib
import logging
import os
import re
import tempfile
from fractions import Fraction

import numpy as np
from scipy.signal import firwin, filtfilt, resample_poly

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = (".edf", ".npy")
TARGET_FS = 256.0
WINDOW_SEC = 2.0
WINDOW_DTYPE = np.float32

CLASS_CLEAN = 0
CLASS_EOG = 1
CLASS_EMG = 2

ARTIFACT_LABEL_TO_CLASS = {
    "eyem": CLASS_EOG,
    "chew": CLASS_EMG,
    "musc": CLASS_EMG,
    "shiv": CLASS_EMG,
    "elec": CLASS_EMG,
    "elpp": CLASS_EMG,
}

_CHANNEL_PREFIX_RE = re.compile(r"^(EEG|EKG|ECG)\s+", re.IGNORECASE)
_CHANNEL_SUFFIX_RE = re.compile(r"-(REF|LE|AR|AVG|CZ)\d*$", re.IGNORECASE)
_MONTAGE_LINE_RE = re.compile(
    r"^\s*montage\s*=\s*\d+\s*,\s*([^:]+?)\s*:\s*(.+?)\s*--\s*(.+?)\s*$",
    re.IGNORECASE,
)
_montage_file_index = None

_PATIENT_ID_RE = re.compile(r"^[0-9a-z]{8}$")


def load_eeg_file(path, fallback_fs):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".edf":
        import mne

        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        data = raw.get_data().astype(WINDOW_DTYPE)
        fs = float(raw.info["sfreq"])
        return data, fs, list(raw.ch_names)
    if ext == ".npy":
        data = np.load(path).astype(WINDOW_DTYPE)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        sidecar = path + ".fs.txt"
        if os.path.exists(sidecar):
            with open(sidecar, encoding="utf-8") as f:
                fs = float(f.read().strip())
        else:
            fs = fallback_fs
        return data, fs, None
    raise ValueError(f"Unsupported file extension for {path!r}: {ext}")


def load_annotations(path):
    csv_path = os.path.splitext(path)[0] + ".csv"
    if not os.path.exists(csv_path):
        return {}, None

    with open(csv_path, newline="", encoding="utf-8") as f:
        raw_lines = f.readlines()

    montage_file = None
    for line in raw_lines:
        if line.strip().startswith("#") and "montage_file" in line.lower():
            _, _, value = line.partition("=")
            montage_file = value.strip()
            break

    rows = [r for r in csv.reader(raw_lines) if r and not r[0].strip().startswith("#")]
    if len(rows) < 2:
        return {}, montage_file

    header = [h.strip().lower() for h in rows[0]]
    required = ("channel", "start_time", "stop_time", "label")
    if not all(col in header for col in required):
        logger.warning("Annotation CSV %s missing required columns %s", csv_path, required)
        return {}, montage_file
    col = {name: header.index(name) for name in required}

    entries = {}
    for row in rows[1:]:
        if len(row) <= max(col.values()):
            continue
        try:
            start = float(row[col["start_time"]])
            stop = float(row[col["stop_time"]])
        except ValueError:
            continue
        ch_name = row[col["channel"]].strip()
        label = row[col["label"]].strip()
        entries.setdefault(ch_name, []).append((start, stop, label))
    return entries, montage_file


def normalize_channel_name(name):
    n = name.strip().upper()
    n = _CHANNEL_PREFIX_RE.sub("", n)
    n = _CHANNEL_SUFFIX_RE.sub("", n)
    return re.sub(r"\s+", "", n)


def match_annotation_channels(channel_names, annotation_channel_names):
    norm_to_annot = {}
    for annot_name in annotation_channel_names:
        norm_to_annot.setdefault(normalize_channel_name(annot_name), annot_name)
    norm_annot_keys = list(norm_to_annot.keys())

    mapping = {}
    for ch_name in channel_names:
        norm_ch = normalize_channel_name(ch_name)
        if norm_ch in norm_to_annot:
            mapping[ch_name] = norm_to_annot[norm_ch]
            continue
        close = difflib.get_close_matches(norm_ch, norm_annot_keys, n=1, cutoff=0.8)
        mapping[ch_name] = norm_to_annot[close[0]] if close else None
    return mapping


def _index_montage_files(search_root):
    global _montage_file_index
    _montage_file_index = {}
    for root, _dirs, files in os.walk(search_root):
        for f in files:
            if f.lower().endswith(".txt"):
                _montage_file_index.setdefault(f.lower(), os.path.join(root, f))


def find_montage_file(montage_filename, search_root, montage_dir=None):
    if not montage_filename:
        return None
    basename = os.path.basename(montage_filename).lower()
    if montage_dir:
        candidate = os.path.join(montage_dir, os.path.basename(montage_filename))
        if os.path.exists(candidate):
            return candidate
    global _montage_file_index
    if _montage_file_index is None:
        _index_montage_files(search_root)
    return _montage_file_index.get(basename)


def parse_montage_file(path):
    derivations = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _MONTAGE_LINE_RE.match(line)
            if m:
                name, ch_a, ch_b = m.groups()
                derivations.append((name.strip(), ch_a.strip(), ch_b.strip()))
    return derivations


def guess_bipolar_split(name):
    parts = name.strip().split("-", 1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        return None
    return parts[0].strip(), parts[1].strip()


def build_derived_bipolar_channels(data, channel_names, derivations):
    norm_lookup = {normalize_channel_name(n): i for i, n in enumerate(channel_names)}
    norm_keys = list(norm_lookup.keys())

    def resolve(raw_name):
        norm = normalize_channel_name(raw_name)
        if norm in norm_lookup:
            return norm_lookup[norm]
        close = difflib.get_close_matches(norm, norm_keys, n=1, cutoff=0.8)
        return norm_lookup[close[0]] if close else None

    derived = {}
    for name, ch_a_raw, ch_b_raw in derivations:
        idx_a, idx_b = resolve(ch_a_raw), resolve(ch_b_raw)
        if idx_a is None or idx_b is None:
            continue
        derived[name] = data[idx_a] - data[idx_b]
    return derived


def find_all_files(folder):
    matches = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(SUPPORTED_EXTS):
                matches.append(os.path.join(root, f))
    return sorted(matches)


def resample_channel(signal, orig_fs, target_fs=TARGET_FS):
    if abs(orig_fs - target_fs) < 1e-6:
        return np.asarray(signal, dtype=WINDOW_DTYPE)
    frac = Fraction(target_fs / orig_fs).limit_denominator(1000)
    return np.asarray(resample_poly(signal, frac.numerator, frac.denominator), dtype=WINDOW_DTYPE)


def bandpass_filter(signal, lowcut=1, highcut=80, fs=TARGET_FS, filter_length=101, pad_length=100):
    nyq = 0.5 * fs
    taps = firwin(filter_length, [lowcut / nyq, highcut / nyq], window="hann", pass_zero=False)
    padded = np.pad(np.asarray(signal, dtype=WINDOW_DTYPE), pad_length, mode="edge")
    filtered = filtfilt(taps, 1.0, padded)
    return np.asarray(filtered[pad_length:-pad_length], dtype=WINDOW_DTYPE)


def classes_in_label(label):
    label_l = label.strip().lower()
    if label_l in ("", "bckg"):
        return set()
    hit = set()
    for token, cls in ARTIFACT_LABEL_TO_CLASS.items():
        if token in label_l:
            hit.add(cls)
    if not hit:
        logger.debug("Unrecognized TUAR label %r -- ignoring for training labels", label)
    return hit


def coverage_masks_per_class(events, fs, n_samples):
    masks = {CLASS_EOG: np.zeros(n_samples, dtype=bool), CLASS_EMG: np.zeros(n_samples, dtype=bool)}
    for start_sec, stop_sec, label in events:
        classes = classes_in_label(label)
        if not classes:
            continue
        start_idx = max(0, min(n_samples, int(round(start_sec * fs))))
        stop_idx = max(0, min(n_samples, int(round(stop_sec * fs))))
        if stop_idx <= start_idx:
            continue
        for c in classes:
            masks[c][start_idx:stop_idx] = True
    return masks


def window_and_label(signal, class_masks, win_len, label_threshold):
    n_samples = len(signal)
    n_windows = n_samples // win_len
    X = np.empty((n_windows, win_len), dtype=WINDOW_DTYPE)
    y = np.empty(n_windows, dtype=np.int64)

    for i in range(n_windows):
        start, end = i * win_len, (i + 1) * win_len
        seg = signal[start:end]
        cov_eog = class_masks[CLASS_EOG][start:end].mean()
        cov_emg = class_masks[CLASS_EMG][start:end].mean()

        if cov_eog < label_threshold and cov_emg < label_threshold:
            label = CLASS_CLEAN
        elif cov_eog >= cov_emg:
            label = CLASS_EOG
        else:
            label = CLASS_EMG

        std = seg.std()
        X[i] = (seg - seg.mean()) / std if std > 1e-12 else np.zeros_like(seg, dtype=WINDOW_DTYPE)
        y[i] = label

    return X, y


def process_file(path, fallback_fs, montage_dir, win_len, label_threshold):
    try:
        data, fs, channel_names = load_eeg_file(path, fallback_fs)
    except Exception as exc:
        logger.warning("Skipping %s (failed to load: %s)", path, exc)
        return None

    n_channels, _n_samples_orig = data.shape
    if channel_names is None:
        channel_names = [f"ch{ch}" for ch in range(n_channels)]

    annotations, montage_filename = load_annotations(path)
    channel_to_annot_name = match_annotation_channels(channel_names, list(annotations.keys()))
    matched_annot_names = {v for v in channel_to_annot_name.values() if v is not None}
    n_direct = len(matched_annot_names)

    X_parts, y_parts = [], []

    for ch in range(n_channels):
        annot_name = channel_to_annot_name.get(channel_names[ch])
        events = annotations.get(annot_name, []) if annot_name else []
        resampled = resample_channel(data[ch], fs, TARGET_FS)
        filtered = bandpass_filter(resampled, fs=TARGET_FS)
        class_masks = coverage_masks_per_class(events, TARGET_FS, len(filtered))
        X_ch, y_ch = window_and_label(filtered, class_masks, win_len, label_threshold)
        X_parts.append(X_ch)
        y_parts.append(y_ch)

    unmatched = [name for name in annotations if name not in matched_annot_names]
    n_bipolar_derived = 0
    if unmatched:
        derivations = []
        if montage_filename:
            montage_path = find_montage_file(montage_filename, os.path.dirname(path), montage_dir)
            if montage_path:
                wanted = set(unmatched)
                derivations = [d for d in parse_montage_file(montage_path) if d[0] in wanted]
        if not derivations:
            for name in unmatched:
                split = guess_bipolar_split(name)
                if split:
                    derivations.append((name, split[0], split[1]))
        derived_signals = build_derived_bipolar_channels(data, channel_names, derivations)
        n_bipolar_derived = len(derived_signals)
        for name, sig in derived_signals.items():
            resampled = resample_channel(sig, fs, TARGET_FS)
            filtered = bandpass_filter(resampled, fs=TARGET_FS)
            events = annotations.get(name, [])
            class_masks = coverage_masks_per_class(events, TARGET_FS, len(filtered))
            X_ch, y_ch = window_and_label(filtered, class_masks, win_len, label_threshold)
            X_parts.append(X_ch)
            y_parts.append(y_ch)

    n_unmatched = len(unmatched) - n_bipolar_derived
    match_stats = {"direct": n_direct, "bipolar_derived": n_bipolar_derived, "unmatched": n_unmatched}

    if not X_parts:
        return None
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0), match_stats


def get_patient_id(path):
    for part in path.split(os.sep):
        if _PATIENT_ID_RE.match(part):
            return part
    return os.path.splitext(os.path.basename(path))[0]


def split_files_by_patient(files, val_frac, test_frac, seed):
    by_patient = {}
    for f in files:
        by_patient.setdefault(get_patient_id(f), []).append(f)

    patients = list(by_patient.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(patients)

    n_val = max(1, int(round(val_frac * len(patients)))) if len(patients) > 2 else 0
    n_test = max(1, int(round(test_frac * len(patients)))) if len(patients) > 2 else 0

    val_patients = set(patients[:n_val])
    test_patients = set(patients[n_val:n_val + n_test])
    train_patients = set(patients[n_val + n_test:])

    train_files = [f for p in train_patients for f in by_patient[p]]
    val_files = [f for p in val_patients for f in by_patient[p]]
    test_files = [f for p in test_patients for f in by_patient[p]]
    return train_files, val_files, test_files


def build_split(files, fallback_fs, montage_dir, win_len, label_threshold, split_name, max_windows=2000):
    totals = {"direct": 0, "bipolar_derived": 0, "unmatched": 0}
    n_windows = 0
    if max_windows is not None and max_windows < 0:
        max_windows = None
    temp_dir = tempfile.mkdtemp(prefix=f"{split_name}_", dir=tempfile.gettempdir())
    x_path = os.path.join(temp_dir, "X.dat")
    y_path = os.path.join(temp_dir, "Y.dat")

    try:
        X_mem = np.memmap(x_path, dtype=WINDOW_DTYPE, mode="w+", shape=(1, win_len))
        y_mem = np.memmap(y_path, dtype=np.int64, mode="w+", shape=(1,))
    except Exception:
        if os.path.exists(x_path):
            os.remove(x_path)
        if os.path.exists(y_path):
            os.remove(y_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
        raise

    for i, path in enumerate(files, 1):
        logger.info("[%s %d/%d] %s", split_name, i, len(files), path)
        result = process_file(path, fallback_fs, montage_dir, win_len, label_threshold)
        if result is None:
            continue
        X, y, match_stats = result
        if X.size == 0:
            continue
        if max_windows is not None and n_windows >= max_windows:
            break
        remaining = None if max_windows is None else max_windows - n_windows
        if remaining is not None and len(y) > remaining:
            X = X[:remaining]
            y = y[:remaining]
        X_mem = np.memmap(x_path, dtype=WINDOW_DTYPE, mode="r+", shape=(n_windows + len(X), win_len))
        y_mem = np.memmap(y_path, dtype=np.int64, mode="r+", shape=(n_windows + len(y),))
        X_mem[n_windows:n_windows + len(X)] = X
        y_mem[n_windows:n_windows + len(y)] = y
        n_windows += len(y)
        for k in totals:
            totals[k] += match_stats[k]

    try:
        if n_windows == 0:
            logger.warning("%s split produced zero windows", split_name)
            X_mem.flush()
            y_mem.flush()
            return np.empty((0, win_len)), np.empty((0,), dtype=np.int64), totals

        X_all = np.memmap(x_path, dtype=WINDOW_DTYPE, mode="r", shape=(n_windows, win_len))
        y_all = np.memmap(y_path, dtype=np.int64, mode="r", shape=(n_windows,))

        counts = {c: int((y_all == c).sum()) for c in (CLASS_CLEAN, CLASS_EOG, CLASS_EMG)}
    finally:
        for mem in (X_mem, y_mem):
            if mem is not None:
                try:
                    mem.flush()
                except Exception:
                    pass
        for path in (x_path, y_path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
        try:
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception:
            pass
    logger.info(
        "%s: %d windows -- clean=%d eog=%d emg=%d",
        split_name,
        len(y_all),
        counts[CLASS_CLEAN],
        counts[CLASS_EOG],
        counts[CLASS_EMG],
    )
    n_seen = totals["direct"] + totals["bipolar_derived"] + totals["unmatched"]
    if n_seen:
        logger.info(
            "%s: channel match -- direct=%d bipolar_derived=%d unmatched=%d (rate=%.4f)",
            split_name,
            totals["direct"],
            totals["bipolar_derived"],
            totals["unmatched"],
            totals["unmatched"] / n_seen,
        )

    return X_all, y_all, totals


def save_split(datapath, subdir, X, y):
    out_dir = os.path.join(datapath, subdir)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "Y.npy"), y)
    logger.info("Wrote %s (%d windows)", out_dir, len(y))


def build_tuar_dataset(tuar_path, out_datapath, fallback_fs, montage_dir, label_threshold, val_frac, test_frac, seed, max_windows=2000):
    files = find_all_files(tuar_path)
    logger.info("Found %d file(s) under %s", len(files), tuar_path)

    train_files, val_files, test_files = split_files_by_patient(files, val_frac, test_frac, seed)
    logger.info(
        "Patient-level split: %d train files, %d val files, %d test files",
        len(train_files),
        len(val_files),
        len(test_files),
    )

    win_len = int(round(WINDOW_SEC * TARGET_FS))
    X_train, y_train, _ = build_split(train_files, fallback_fs, montage_dir, win_len, label_threshold, "train", max_windows=max_windows)
    X_val, y_val, _ = build_split(val_files, fallback_fs, montage_dir, win_len, label_threshold, "val", max_windows=max_windows)
    X_test, y_test, _ = build_split(test_files, fallback_fs, montage_dir, win_len, label_threshold, "test", max_windows=max_windows)

    save_split(out_datapath, "train", X_train, y_train)
    save_split(out_datapath, "val", X_val, y_val)
    save_split(out_datapath, os.path.join("test", "0"), X_test, y_test)

    return {
        "files": files,
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files,
        "train_counts": len(y_train),
        "val_counts": len(y_val),
        "test_counts": len(y_test),
    }
