"""
Validate ArtifactDetector against ground-truth labels drawn from TUAR and
TUSZ annotation CSVs.

Ground truth per file/channel is built directly from the CSV's
start_time/stop_time columns: any interval whose label is not "bckg" marks
those samples as a true artifact. The detector's own flagged-sample mask
(from apply_zero_masking) is the prediction. Comparing the two sample-by-
sample over every processed file/channel gives a single aggregated 2x2
confusion matrix:

    [[TP, FP],
     [FN, TN]]

TP = samples correctly flagged as artifact
FP = samples flagged as artifact but not labeled as one
FN = samples labeled as artifact but not flagged
TN = samples correctly left unflagged

Only 100 files total are processed (across --artifact-path and
--seizure-path combined, by default split proportionally to how many files
exist in each), keeping runtime bounded. Use --n-total to change the cap.

Usage:
    python validate_artifact_detector.py \
        --artifact-path /data/tuh_eeg_artifact/v3.0.1 \
        --seizure-path  /data/tuh_eeg_seizure/v2.0.3/edf/dev \
        --out validation_report.json
"""
import argparse
import csv
import difflib
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.artifact_detection import ArtifactDetector
from pipeline.artifact_masking import apply_zero_masking
from util import handle_logs

logger = handle_logs.get_logger("validate_artifact_detector", "logs/app.log")

SUPPORTED_EXTS = (".edf", ".npy")
DEFAULT_N_TOTAL = 100


# --------------------------------------------------------------------------
# File / annotation loading
# --------------------------------------------------------------------------

def load_eeg_file(path, fallback_fs):
    """Load an EDF or NPY file -> (data[(n_channels, n_samples)], fs, channel_names|None)."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".edf":
        import mne
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        data = raw.get_data().astype(np.float64)
        fs = float(raw.info["sfreq"])
        return data, fs, list(raw.ch_names)

    if ext == ".npy":
        data = np.load(path).astype(np.float64)
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
    """Load {channel_name: [(start_sec, stop_sec, label), ...]} from a sidecar CSV, plus montage_file."""
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


_CHANNEL_PREFIX_RE = re.compile(r"^(EEG|EKG|ECG)\s+", re.IGNORECASE)
_CHANNEL_SUFFIX_RE = re.compile(r"-(REF|LE|AR|AVG|CZ)\d*$", re.IGNORECASE)


def normalize_channel_name(name):
    n = name.strip().upper()
    n = _CHANNEL_PREFIX_RE.sub("", n)
    n = _CHANNEL_SUFFIX_RE.sub("", n)
    return re.sub(r"\s+", "", n)


def match_annotation_channels(channel_names, annotation_channel_names):
    """Map recording channel names -> best-matching annotation channel name (or None)."""
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


# --- Bipolar montage derivation (TUAR/TUSZ label CSVs sometimes annotate a
# derived bipolar montage rather than the raw monopolar recording channels) ---
_MONTAGE_LINE_RE = re.compile(
    r"^\s*montage\s*=\s*\d+\s*,\s*([^:]+?)\s*:\s*(.+?)\s*--\s*(.+?)\s*$", re.IGNORECASE
)
_montage_file_index = None


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
        derived[name] = (idx_a, idx_b, data[idx_a] - data[idx_b])
    return derived


def find_all_files(folder):
    matches = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(SUPPORTED_EXTS):
                matches.append(os.path.join(root, f))
    return sorted(matches)


# --------------------------------------------------------------------------
# Ground truth (from CSV times) vs. predicted mask -> confusion counts
# --------------------------------------------------------------------------

def ground_truth_mask_for_channel(events, fs, n_samples):
    """Bool array over samples: True wherever a non-'bckg' CSV interval covers that sample."""
    mask = np.zeros(n_samples, dtype=bool)
    for start_sec, stop_sec, label in events:
        if label.strip().lower() == "bckg":
            continue
        start_idx = max(0, min(n_samples, int(round(start_sec * fs))))
        stop_idx = max(0, min(n_samples, int(round(stop_sec * fs))))
        if stop_idx > start_idx:
            mask[start_idx:stop_idx] = True
    return mask


def confusion_counts(pred_mask, truth_mask):
    """Return (tp, fp, fn, tn) counts from two same-shape bool arrays."""
    tp = int(np.count_nonzero(pred_mask & truth_mask))
    fp = int(np.count_nonzero(pred_mask & ~truth_mask))
    fn = int(np.count_nonzero(~pred_mask & truth_mask))
    tn = int(np.count_nonzero(~pred_mask & ~truth_mask))
    return tp, fp, fn, tn


def process_file(detector, path, fallback_fs, montage_dir=None):
    """Run the detector on one file and return (tp, fp, fn, tn, match_stats), or None on failure.

    match_stats = {"direct": n, "bipolar_derived": n, "unmatched": n} counting how each
    annotation channel in the CSV was (or wasn't) resolved to a signal to compare against.
    """
    try:
        data, fs, channel_names = load_eeg_file(path, fallback_fs)
    except Exception as exc:
        logger.warning("Skipping %s (failed to load: %s)", path, exc)
        return None

    n_channels, n_samples = data.shape
    if channel_names is None:
        channel_names = [f"ch{ch}" for ch in range(n_channels)]

    annotations, montage_filename = load_annotations(path)
    channel_to_annot_name = match_annotation_channels(channel_names, list(annotations.keys()))

    det_result = detector.predict_segment(data, fs)
    _zero_masked, zero_mask = apply_zero_masking(data.copy(), det_result, fs)

    tp = fp = fn = tn = 0
    matched_annot_names = {v for v in channel_to_annot_name.values() if v is not None}
    n_direct = len(matched_annot_names)

    for ch in range(n_channels):
        annot_name = channel_to_annot_name.get(channel_names[ch])
        ch_events = annotations.get(annot_name, []) if annot_name else []
        truth_mask = ground_truth_mask_for_channel(ch_events, fs, n_samples)
        c_tp, c_fp, c_fn, c_tn = confusion_counts(zero_mask[ch], truth_mask)
        tp += c_tp; fp += c_fp; fn += c_fn; tn += c_tn

    # Bipolar montage matching: annotation channels (e.g. "FP1-F7") that never matched a
    # raw recording channel directly are resolved by deriving that bipolar pair from two
    # monopolar channels -- first via an explicit montage definition file if the CSV
    # references one, falling back to splitting the channel name on "-" if not.
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
        for name, (idx_a, idx_b, _sig) in derived_signals.items():
            derived_pred_mask = zero_mask[idx_a] | zero_mask[idx_b]
            truth_mask = ground_truth_mask_for_channel(annotations.get(name, []), fs, n_samples)
            c_tp, c_fp, c_fn, c_tn = confusion_counts(derived_pred_mask, truth_mask)
            tp += c_tp; fp += c_fp; fn += c_fn; tn += c_tn

    n_unmatched = len(unmatched) - n_bipolar_derived
    match_stats = {"direct": n_direct, "bipolar_derived": n_bipolar_derived, "unmatched": n_unmatched}
    if n_unmatched:
        still_unmatched = [name for name in unmatched if name not in derived_signals] if unmatched else []
        logger.warning(
            "%s: %d annotation channel(s) never resolved to a signal (ground truth silently "
            "dropped for these) -- %s",
            path, n_unmatched, still_unmatched[:5],
        )

    return tp, fp, fn, tn, match_stats


def print_matrix(matrix):
    tp, fp = matrix[0]
    fn, tn = matrix[1]
    total = tp + fp + fn + tn
    print("\n" + "=" * 50)
    print("ARTIFACT DETECTOR CONFUSION MATRIX (sample-level)")
    print("=" * 50)
    print(f"                 pred:artifact   pred:clean")
    print(f"  true:artifact  {tp:>14d}   {fn:>10d}")
    print(f"  true:clean     {fp:>14d}   {tn:>10d}")
    if total:
        print(f"\n  accuracy: {(tp + tn) / total:.4f}")
        if tp + fn:
            print(f"  recall (sensitivity): {tp / (tp + fn):.4f}")
        if tp + fp:
            print(f"  precision: {tp / (tp + fp):.4f}")
    print("=" * 50)


def gather_capped_files(artifact_path, seizure_path, n_total, seed):
    """Collect files from both corpora, capped at n_total, split proportionally to availability."""
    artifact_files = find_all_files(artifact_path)
    seizure_files = find_all_files(seizure_path)
    total_available = len(artifact_files) + len(seizure_files)

    if total_available <= n_total:
        return artifact_files, seizure_files

    rng = np.random.RandomState(seed)
    n_artifact = round(n_total * len(artifact_files) / total_available) if total_available else 0
    n_seizure = n_total - n_artifact

    n_artifact = min(n_artifact, len(artifact_files))
    n_seizure = min(n_seizure, len(seizure_files))
    # If one side is short, give the remainder to the other side.
    leftover = n_total - (n_artifact + n_seizure)
    if leftover > 0:
        if len(artifact_files) - n_artifact >= leftover:
            n_artifact += leftover
        else:
            n_seizure += min(leftover, len(seizure_files) - n_seizure)

    picked_artifact = list(rng.choice(artifact_files, size=n_artifact, replace=False)) if n_artifact else []
    picked_seizure = list(rng.choice(seizure_files, size=n_seizure, replace=False)) if n_seizure else []
    return picked_artifact, picked_seizure


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact-path", required=True, help="Root folder of the TUAR corpus (walked recursively)")
    parser.add_argument("--seizure-path", required=True, help="Root folder of the TUSZ dev/seizure corpus (walked recursively)")
    parser.add_argument("--fallback-fs", type=float, default=256.0, help="Fallback sampling rate for .npy files with no <file>.fs.txt sidecar")
    parser.add_argument("--ckpt-dir", default=None, help="ArtifactDetector checkpoint dir override")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--montage-dir", default=None, help="Directory to look for montage definition .txt files first")
    parser.add_argument("--n-total", type=int, default=DEFAULT_N_TOTAL, help="Total number of files to test, across both corpora combined (default: 100)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used when sampling down to --n-total files")
    parser.add_argument("--out", default="validation_report.json", help="Where to write the JSON summary (matrix + per-file counts)")
    args = parser.parse_args()

    detector = ArtifactDetector(ckpt_dir=args.ckpt_dir, device=args.device)

    artifact_files, seizure_files = gather_capped_files(
        args.artifact_path, args.seizure_path, args.n_total, args.seed
    )
    all_files = [(p, "artifact") for p in artifact_files] + [(p, "seizure") for p in seizure_files]
    logger.info("Testing %d file(s) total (%d artifact, %d seizure)", len(all_files), len(artifact_files), len(seizure_files))

    total_tp = total_fp = total_fn = total_tn = 0
    total_direct = total_bipolar_derived = total_unmatched = 0
    per_file = []

    for i, (path, source) in enumerate(all_files, 1):
        logger.info("[%d/%d] (%s) %s", i, len(all_files), source, path)
        result = process_file(detector, path, args.fallback_fs, montage_dir=args.montage_dir)
        if result is None:
            continue
        tp, fp, fn, tn, match_stats = result
        total_tp += tp; total_fp += fp; total_fn += fn; total_tn += tn
        total_direct += match_stats["direct"]
        total_bipolar_derived += match_stats["bipolar_derived"]
        total_unmatched += match_stats["unmatched"]
        per_file.append({
            "path": path, "source": source, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "channel_match": match_stats,
        })

    matrix = [[total_tp, total_fp], [total_fn, total_tn]]
    print_matrix(matrix)

    n_annot_channels_seen = total_direct + total_bipolar_derived + total_unmatched
    print("\nANNOTATION CHANNEL MATCHING (across all files)")
    print(f"  direct match           : {total_direct}")
    print(f"  bipolar-derived match  : {total_bipolar_derived}")
    print(f"  unmatched (dropped)    : {total_unmatched}")
    if n_annot_channels_seen:
        print(f"  unmatched rate         : {total_unmatched / n_annot_channels_seen:.4f}")
        if total_unmatched / n_annot_channels_seen > 0.05:
            print("  -> WARNING: a meaningful fraction of annotation channels never resolved to a "
                  "signal. Ground truth is silently False for those, which inflates false positives "
                  "and can tank precision.")

    report = {
        "n_files_tested": len(per_file),
        "matrix": matrix,
        "channel_match_totals": {
            "direct": total_direct,
            "bipolar_derived": total_bipolar_derived,
            "unmatched": total_unmatched,
        },
        "per_file": per_file,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote %s", args.out)

    return matrix


if __name__ == "__main__":
    main()