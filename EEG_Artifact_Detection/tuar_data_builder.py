"""TUAR dataset builder helpers for EEG_Artifact_Detection.

This version builds MULTI-CHANNEL windows: instead of treating each channel as an
independent single-channel sample (discarding cross-channel/spatial information),
it aligns each file to a shared "canonical" channel set and produces windows shaped
(n_channels, win_len). The canonical channel set is auto-discovered by scanning EDF
headers across the corpus rather than hardcoded, so it adapts to whatever montage(s)
your TUAR files actually use.

BREAKING CHANGE from the single-channel version: X.npy is now 3D
(n_windows, n_channels, win_len) instead of 2D (n_windows, win_len). Any existing
data/train, data/val, data/test built with the old pipeline must be rebuilt --
they are not shape-compatible with this version. dataset.py and models.py were
updated alongside this file to consume the new shape (see EEGDataset(raw=True)
and ArtifactDetectionCNN_MultiChannel).

Only .edf files are usable with multi-channel building: .npy files carry no channel
names, so there's nothing to align them against. They are skipped with a warning.
"""
import csv
import difflib
import logging
import os
import re
import tempfile
from collections import Counter
from fractions import Fraction

import numpy as np
from scipy.signal import firwin, filtfilt, resample_poly
from tqdm import tqdm

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


def load_eeg_file(path, fallback_fs, preload=True):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".edf":
        import mne

        raw = mne.io.read_raw_edf(path, preload=preload, verbose=False)
        if not preload:
            return None, float(raw.info["sfreq"]), list(raw.ch_names)
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


# --------------------------------------------------------------------------- #
# Canonical channel discovery
# --------------------------------------------------------------------------- #

def discover_canonical_channels(files, min_presence_frac=0.6, max_channels=19, scan_limit=None):
    """
    Scan EDF headers (no signal data loaded -- preload=False, so this is cheap even
    across a large corpus) to find a fixed channel set common across files, instead
    of hardcoding a montage. Windows are later built against this discovered set so
    every file's window has the same (n_channels, win_len) shape and channel order.

    Parameters:
        files: Candidate file paths (from find_all_files). .npy files are skipped --
            they carry no channel names, so they can't be aligned to a canonical set.
        min_presence_frac (float): A channel must appear in at least this fraction of
            scanned EDF files to be included. Lower this if your corpus uses several
            inconsistent montages and too few channels survive; raise it if you want
            a stricter, more universally-present set.
        max_channels (int): Upper bound on how many channels to keep, most-common first.
        scan_limit (int or None): If set, only scan the first N eligible files rather
            than the whole corpus -- a representative sample is normally enough and
            keeps this step fast on very large corpora.

    Returns:
        list[str]: Normalized channel names (see normalize_channel_name), ordered by
        descending presence count, capped at max_channels.

    Raises:
        RuntimeError: If no EDF headers could be read at all.
    """
    import mne

    edf_files = [f for f in files if os.path.splitext(f)[1].lower() == ".edf"]
    if scan_limit is not None:
        edf_files = edf_files[:scan_limit]

    counts = Counter()
    n_scanned = 0
    for path in tqdm(edf_files, desc="Scanning channel headers", unit="file"):
        try:
            raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
        except Exception as exc:
            logger.debug("Could not read header for %s: %s", path, exc)
            continue
        norm_names = {normalize_channel_name(n) for n in raw.ch_names}
        counts.update(norm_names)
        n_scanned += 1

    if n_scanned == 0:
        raise RuntimeError(
            "No readable EDF headers found under the given files -- cannot discover "
            "canonical channels. Check --tuar-path points at a folder containing .edf files."
        )

    min_count = max(1, int(round(min_presence_frac * n_scanned)))
    candidates = [(name, c) for name, c in counts.items() if c >= min_count]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    canonical = [name for name, _ in candidates[:max_channels]]

    logger.info(
        "Scanned %d/%d EDF headers; %d distinct channel names seen; %d present in "
        ">= %.0f%% of scanned files",
        n_scanned, len(edf_files), len(counts), len(candidates), min_presence_frac * 100,
    )
    logger.info("Canonical channel set (%d channels): %s", len(canonical), canonical)
    if len(canonical) < 4:
        logger.warning(
            "Only %d canonical channels found (threshold %.0f%%) -- your corpus may use "
            "several inconsistent montages. Consider lowering --min-channel-frac.",
            len(canonical), min_presence_frac * 100,
        )
    return canonical


# --------------------------------------------------------------------------- #
# Multi-channel windowing
# --------------------------------------------------------------------------- #

def window_and_label_multichannel(signal_matrix, class_masks_per_channel, win_len, label_threshold):
    """
    Window an aligned multi-channel signal and label each window.

    Parameters:
        signal_matrix (np.ndarray): Shape (n_channels, n_samples), already resampled,
            filtered, and in canonical-channel order (see process_file_multichannel).
        class_masks_per_channel (list[dict]): One {CLASS_EOG: bool_array, CLASS_EMG:
            bool_array} per channel, same length/order as signal_matrix's first axis.
        win_len (int): Window length in samples.
        label_threshold (float): Minimum per-class coverage fraction (see below) for a
            window to be labeled that class.

    A window's artifact coverage is taken as the MAX across channels, not the mean --
    artifacts like eye movements are often localized to a few channels (e.g. frontal),
    so requiring the whole multi-channel window to average past the threshold would
    systematically under-label real, channel-localized artifacts.

    Returns:
        tuple: (X, y) where X has shape (n_windows, n_channels, win_len) and y has
        shape (n_windows,).
    """
    n_channels, n_samples = signal_matrix.shape
    n_windows = n_samples // win_len
    X = np.empty((n_windows, n_channels, win_len), dtype=WINDOW_DTYPE)
    y = np.empty(n_windows, dtype=np.int64)

    for i in range(n_windows):
        start, end = i * win_len, (i + 1) * win_len

        cov_eog = max(m[CLASS_EOG][start:end].mean() for m in class_masks_per_channel)
        cov_emg = max(m[CLASS_EMG][start:end].mean() for m in class_masks_per_channel)

        if cov_eog < label_threshold and cov_emg < label_threshold:
            label = CLASS_CLEAN
        elif cov_eog >= cov_emg:
            label = CLASS_EOG
        else:
            label = CLASS_EMG

        seg = signal_matrix[:, start:end]
        mean = seg.mean(axis=1, keepdims=True)
        std = seg.std(axis=1, keepdims=True)
        X[i] = np.divide(seg - mean, std, out=np.zeros_like(seg), where=std > 1e-12)
        y[i] = label

    return X, y


def process_file_multichannel(path, fallback_fs, montage_dir, win_len, label_threshold, canonical_channels):
    """
    Build an aligned (n_channels, n_samples) matrix for one file against
    `canonical_channels` (from discover_canonical_channels), then window and label it.

    Any canonical channel missing from this particular file is zero-filled -- this is
    a simplification, not a correction. If your corpus has substantial per-file channel
    variation, prefer raising min_presence_frac / shrinking canonical_channels over
    relying on zero-fill for a large fraction of channels (see discover_canonical_channels
    docstring). .npy files are skipped: without channel names there's nothing to align.
    """
    if os.path.splitext(path)[1].lower() != ".edf":
        logger.debug("Skipping %s: multi-channel builder requires named EDF channels", path)
        return None
    try:
        data, fs, channel_names = load_eeg_file(path, fallback_fs)
    except Exception as exc:
        logger.warning("Skipping %s (failed to load: %s)", path, exc)
        return None

    norm_lookup = {}
    for i, name in enumerate(channel_names):
        norm_lookup.setdefault(normalize_channel_name(name), i)

    n_present = sum(1 for c in canonical_channels if c in norm_lookup)
    if n_present == 0:
        return None

    annotations, montage_filename = load_annotations(path)
    channel_to_annot = match_annotation_channels(channel_names, list(annotations.keys()))

    filtered_channels = []
    masks_per_channel = []
    lengths = []
    for canon_name in canonical_channels:
        idx = norm_lookup.get(canon_name)
        if idx is None:
            filtered_channels.append(None)
            masks_per_channel.append(None)
            continue
        resampled = resample_channel(data[idx], fs, TARGET_FS)
        filtered = bandpass_filter(resampled, fs=TARGET_FS)
        filtered_channels.append(filtered)
        lengths.append(len(filtered))
        annot_name = channel_to_annot.get(channel_names[idx])
        events = annotations.get(annot_name, []) if annot_name else []
        masks_per_channel.append(coverage_masks_per_class(events, TARGET_FS, len(filtered)))

    if not lengths:
        return None
    n_samples = min(lengths)

    n_channels = len(canonical_channels)
    signal_matrix = np.zeros((n_channels, n_samples), dtype=WINDOW_DTYPE)
    aligned_masks = []
    for i, filt in enumerate(filtered_channels):
        if filt is not None:
            signal_matrix[i] = filt[:n_samples]
            aligned_masks.append({k: v[:n_samples] for k, v in masks_per_channel[i].items()})
        else:
            aligned_masks.append({
                CLASS_EOG: np.zeros(n_samples, dtype=bool),
                CLASS_EMG: np.zeros(n_samples, dtype=bool),
            })

    X, y = window_and_label_multichannel(signal_matrix, aligned_masks, win_len, label_threshold)
    match_stats = {"direct": n_present, "bipolar_derived": 0, "unmatched": n_channels - n_present}
    return X, y, match_stats


def _save_array_chunked(out_path, src, chunk_rows=200_000):
    """
    Write `src` (a memmap or ndarray) to a .npy file at `out_path` in row-chunks,
    instead of via a single np.save()/tofile() call.

    np.save() on a large array eventually calls ndarray.tofile(), which issues one big
    OS-level write. On Windows, very large single writes (multi-channel splits at
    scale can easily exceed a gigabyte) can silently truncate -- the CRT/OS write layer
    caps out mid-call rather than erroring cleanly or looping to finish, producing
    "OSError: N requested and M written" with M well short of N. Writing in bounded
    chunks via a destination memmap sidesteps this: each chunk assignment is a normal,
    much smaller memory copy rather than one giant write syscall.

    Parameters:
        out_path (str): Destination .npy file path.
        src: Source array (typically a np.memmap) to copy from.
        chunk_rows (int): Number of rows (axis 0) to copy per iteration.
    """
    dest = np.lib.format.open_memmap(out_path, mode="w+", dtype=src.dtype, shape=src.shape)
    n_rows = src.shape[0]
    for start in range(0, n_rows, chunk_rows):
        end = min(start + chunk_rows, n_rows)
        dest[start:end] = src[start:end]
    dest.flush()
    del dest


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


def build_split(files, fallback_fs, montage_dir, win_len, label_threshold, split_name, out_dir,
                 canonical_channels, max_windows=2000, seed=0, tmp_dir=None):
    """Walk `files` (shuffled first, so results don't depend on find_all_files's sort
    order) and accumulate multi-channel windows up to a PER-CLASS cap, not a raw total
    cap. A flat total cap lets whichever files are processed first exhaust the quota
    before rarer classes (e.g. EOG, a small share of all TUAR windows) ever show up --
    this stops as soon as every class hits its share of max_windows, or all files are
    exhausted.

    canonical_channels: list of normalized channel names from discover_canonical_channels,
    fixing the channel axis for every window in this split (and, since it should be
    computed once across the whole corpus, across all splits too).

    out_dir: final destination directory for this split's X.npy/Y.npy. Written via
    np.save on the accumulation memmap directly (streamed, not a full in-RAM copy) --
    do not also call save_split on the result, this function writes the files itself.

    tmp_dir: where the working memmap files are written while accumulating. Defaults
    to the OS temp dir (tempfile.gettempdir()) if not given -- on Windows that's
    usually %LOCALAPPDATA%\\Temp on the C: drive, which can silently fill up during a
    large/unbounded build even if --out-datapath points somewhere else with plenty of
    room. Pass a directory on a drive with real free space if you're building a large
    corpus.

    The whole accumulation + save is wrapped in one try/finally: a crash partway
    through (disk full, Ctrl+C, anything) still removes the temp memmap files instead
    of leaving multi-GB orphans behind in tmp_dir. Returns a small dict of counts/stats
    only -- never the full window array -- to avoid ever holding a whole split in RAM.
    """
    n_channels = len(canonical_channels)
    totals = {"direct": 0, "bipolar_derived": 0, "unmatched": 0}
    if max_windows is not None and max_windows < 0:
        max_windows = None
    per_class_cap = None if max_windows is None else max(1, max_windows // 3)
    class_counts = {CLASS_CLEAN: 0, CLASS_EOG: 0, CLASS_EMG: 0}
    n_windows = 0

    shuffled_files = list(files)
    np.random.RandomState(seed).shuffle(shuffled_files)

    tmp_root = tmp_dir or tempfile.gettempdir()
    os.makedirs(tmp_root, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix=f"{split_name}_", dir=tmp_root)
    x_path = os.path.join(temp_dir, "X.dat")
    y_path = os.path.join(temp_dir, "Y.dat")
    X_mem = None
    y_mem = None
    X_view = None
    y_view = None

    try:
        X_mem = np.memmap(x_path, dtype=WINDOW_DTYPE, mode="w+", shape=(1, n_channels, win_len))
        y_mem = np.memmap(y_path, dtype=np.int64, mode="w+", shape=(1,))

        for i, path in enumerate(shuffled_files, 1):
            if per_class_cap is not None and all(class_counts[c] >= per_class_cap for c in class_counts):
                logger.info("%s: every class reached its cap (%d each) after %d/%d files -- stopping early",
                            split_name, per_class_cap, i - 1, len(shuffled_files))
                break
            logger.info("[%s %d/%d] %s", split_name, i, len(shuffled_files), path)
            result = process_file_multichannel(path, fallback_fs, montage_dir, win_len, label_threshold,
                                                canonical_channels)
            if result is None:
                continue
            X, y, match_stats = result
            if X.size == 0:
                continue

            if per_class_cap is not None:
                keep_mask = np.zeros(len(y), dtype=bool)
                for idx, label in enumerate(y):
                    label = int(label)
                    if class_counts[label] < per_class_cap:
                        keep_mask[idx] = True
                        class_counts[label] += 1
                if not keep_mask.any():
                    continue
                X, y = X[keep_mask], y[keep_mask]
            else:
                for label in (CLASS_CLEAN, CLASS_EOG, CLASS_EMG):
                    class_counts[label] += int((y == label).sum())

            X_mem = np.memmap(x_path, dtype=WINDOW_DTYPE, mode="r+", shape=(n_windows + len(X), n_channels, win_len))
            y_mem = np.memmap(y_path, dtype=np.int64, mode="r+", shape=(n_windows + len(y),))
            X_mem[n_windows:n_windows + len(X)] = X
            y_mem[n_windows:n_windows + len(y)] = y
            n_windows += len(y)
            for k in totals:
                totals[k] += match_stats[k]

        if n_windows == 0:
            logger.warning("%s split produced zero windows", split_name)
            return {"n_windows": 0, "counts": {CLASS_CLEAN: 0, CLASS_EOG: 0, CLASS_EMG: 0}, "totals": totals}

        # Stream directly to the final .npy location via np.save on the memmap itself.
        # np.save writes a memmap by reading through it (tofile-style, backed by the OS
        # page cache) rather than requiring a second full-size allocation -- this avoids
        # ever holding the whole split in RAM at once, which is what caused the earlier
        # MemoryError from np.array(memmap). Only after this completes do we delete the
        # temp files, in `finally` below.
        os.makedirs(out_dir, exist_ok=True)
        X_view = np.memmap(x_path, dtype=WINDOW_DTYPE, mode="r", shape=(n_windows, n_channels, win_len))
        y_view = np.memmap(y_path, dtype=np.int64, mode="r", shape=(n_windows,))
        _save_array_chunked(os.path.join(out_dir, "X.npy"), X_view)
        _save_array_chunked(os.path.join(out_dir, "Y.npy"), y_view)

        counts = {c: int((y_view == c).sum()) for c in (CLASS_CLEAN, CLASS_EOG, CLASS_EMG)}
        logger.info(
            "%s: %d windows x %d channels -- clean=%d eog=%d emg=%d",
            split_name, n_windows, n_channels, counts[CLASS_CLEAN], counts[CLASS_EOG], counts[CLASS_EMG],
        )
        n_seen = totals["direct"] + totals["bipolar_derived"] + totals["unmatched"]
        if n_seen:
            logger.info(
                "%s: canonical channel coverage -- present=%d missing(zero-filled)=%d (rate=%.4f missing)",
                split_name, totals["direct"], totals["unmatched"],
                totals["unmatched"] / n_seen,
            )
        logger.info("Wrote %s (%d windows)", out_dir, n_windows)

        return {"n_windows": n_windows, "counts": counts, "totals": totals}

    finally:
        for mem in (X_mem, y_mem, X_view, y_view):
            if mem is not None:
                try:
                    mem.flush()
                except Exception:
                    pass
        del X_mem, y_mem, X_view, y_view
        for p in (x_path, y_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception as exc:
                logger.warning("Could not remove temp file %s: %s", p, exc)
        try:
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as exc:
            logger.warning("Could not remove temp dir %s: %s", temp_dir, exc)


def save_split(datapath, subdir, X, y):
    """Standalone helper for saving an already-in-memory (X, y) pair -- NOT used by
    build_tuar_dataset anymore, since build_split now streams its output directly to
    out_dir to avoid ever materializing a whole split in RAM. Kept for other callers
    that already have small enough arrays in hand."""
    out_dir = os.path.join(datapath, subdir)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "Y.npy"), y)
    logger.info("Wrote %s (%d windows)", out_dir, len(y))


def build_tuar_dataset(tuar_path, out_datapath, fallback_fs, montage_dir, label_threshold, val_frac, test_frac,
                        seed, max_windows=2000, tmp_dir=None, min_channel_frac=0.6, max_channels=19,
                        channel_scan_limit=None):
    files = find_all_files(tuar_path)
    logger.info("Found %d file(s) under %s", len(files), tuar_path)

    canonical_channels = discover_canonical_channels(
        files, min_presence_frac=min_channel_frac, max_channels=max_channels, scan_limit=channel_scan_limit,
    )

    train_files, val_files, test_files = split_files_by_patient(files, val_frac, test_frac, seed)
    logger.info(
        "Patient-level split: %d train files, %d val files, %d test files",
        len(train_files),
        len(val_files),
        len(test_files),
    )

    win_len = int(round(WINDOW_SEC * TARGET_FS))
    train_stats = build_split(train_files, fallback_fs, montage_dir, win_len, label_threshold, "train",
                               out_dir=os.path.join(out_datapath, "train"), canonical_channels=canonical_channels,
                               max_windows=max_windows, seed=seed, tmp_dir=tmp_dir)
    val_stats = build_split(val_files, fallback_fs, montage_dir, win_len, label_threshold, "val",
                             out_dir=os.path.join(out_datapath, "val"), canonical_channels=canonical_channels,
                             max_windows=max_windows, seed=seed, tmp_dir=tmp_dir)
    test_stats = build_split(test_files, fallback_fs, montage_dir, win_len, label_threshold, "test",
                              out_dir=os.path.join(out_datapath, "test", "0"), canonical_channels=canonical_channels,
                              max_windows=max_windows, seed=seed, tmp_dir=tmp_dir)

    return {
        "files": files,
        "canonical_channels": canonical_channels,
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files,
        "train_counts": train_stats["n_windows"],
        "val_counts": val_stats["n_windows"],
        "test_counts": test_stats["n_windows"],
    }