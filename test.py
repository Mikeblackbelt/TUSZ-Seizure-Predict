"""
This is an AI generated testing file which is not in use in the main codebase. It is used to test the functionality of the artifact detector and its interpolation.
"""
import argparse
import difflib
import json
import os
import random
import re
import sys
from collections import Counter

import numpy as np

# Make sibling modules importable regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.artifact_detection import ArtifactDetector
from pipeline.artifact_masking import apply_zero_masking, apply_interpolation_masking
from util import handle_logs

logger = handle_logs.get_logger("compare_masking", "logs/app.log")

SUPPORTED_EXTS = (".edf", ".npy")


def load_eeg_file(path, fallback_fs):
    """
    Load an EEG file and provide its samples, sampling frequency, and channel names.
    
    Parameters:
        path (str): Path to an EDF or NPY file.
        fallback_fs (float): Sampling frequency used for NPY files without an adjacent
            `.fs.txt` sidecar.
    
    Returns:
        tuple: A `(data, fs, channel_names)` tuple. `data` has shape
            `(n_channels, n_samples)`, and `channel_names` is `None` for NPY files.
    
    Raises:
        ValueError: If the file extension is unsupported.
    """
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
    """
    Load annotation intervals and optional montage metadata from a sidecar CSV file.
    
    Malformed data rows are skipped. Missing or invalid annotation files produce empty
    entries while preserving any parsed montage filename.
    
    Returns:
        tuple: A mapping of channel names to `(start_time, stop_time, label)` tuples
        and the referenced montage filename, or `None` when unavailable.
    """
    csv_path = os.path.splitext(path)[0] + ".csv"
    if not os.path.exists(csv_path):
        logger.debug("No annotation CSV at %s", csv_path)
        return {}, None

    import csv
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
        logger.warning("Annotation CSV %s has no data rows after stripping comments", csv_path)
        return {}, montage_file

    header = [h.strip().lower() for h in rows[0]]
    required = ("channel", "start_time", "stop_time", "label")
    if not all(col in header for col in required):
        logger.warning(
            "Annotation CSV %s header %s is missing one of %s -- skipping", csv_path, header, required
        )
        return {}, montage_file
    col = {name: header.index(name) for name in required}

    entries = {}
    label_counts = Counter()
    skipped_rows = 0
    for row in rows[1:]:
        if len(row) <= max(col.values()):
            skipped_rows += 1
            continue
        try:
            start = float(row[col["start_time"]])
            stop = float(row[col["stop_time"]])
        except ValueError:
            skipped_rows += 1
            continue
        ch_name = row[col["channel"]].strip()
        label = row[col["label"]].strip()
        entries.setdefault(ch_name, []).append((start, stop, label))
        label_counts[label] += 1

    non_bckg_count = sum(c for lbl, c in label_counts.items() if lbl.lower() != "bckg")
    logger.info(
        "Loaded %s: %d rows across %d channels, labels=%s (%d non-bckg)%s",
        csv_path, sum(label_counts.values()), len(entries), dict(label_counts),
        non_bckg_count, f", skipped {skipped_rows} malformed rows" if skipped_rows else "",
    )
    return entries, montage_file


# EDF channel names often carry extra noise the CSV label file doesn't
# (e.g. "EEG FP1-F7-REF" vs "FP1-F7"). Strip that before comparing.
_CHANNEL_PREFIX_RE = re.compile(r"^(EEG|EKG|ECG)\s+", re.IGNORECASE)
_CHANNEL_SUFFIX_RE = re.compile(r"-(REF|LE|AR|AVG|CZ)\d*$", re.IGNORECASE)


def normalize_channel_name(name):
    """Normalize a channel name for matching by removing common prefixes, reference suffixes, and whitespace.
    
    Parameters:
    	name (str): Channel name to normalize.
    
    Returns:
    	str: Canonicalized channel name in uppercase."""
    n = name.strip().upper()
    n = _CHANNEL_PREFIX_RE.sub("", n)
    n = _CHANNEL_SUFFIX_RE.sub("", n)
    n = re.sub(r"\s+", "", n)
    return n


def match_annotation_channels(channel_names, annotation_channel_names):
    """
    Map recording channels to their best-matching annotation channel names.
    
    Parameters:
    	channel_names (iterable): Recording channel names to match.
    	annotation_channel_names (iterable): Channel names from the annotation CSV.
    
    Returns:
    	tuple: A mapping from each recording channel name to its matched annotation channel name or `None`, and a list of annotation channel names that were not matched.
    """
    norm_to_annot = {}
    for annot_name in annotation_channel_names:
        norm_to_annot.setdefault(normalize_channel_name(annot_name), annot_name)
    norm_annot_keys = list(norm_to_annot.keys())

    mapping = {}
    used_annot_norms = set()
    for ch_name in channel_names:
        norm_ch = normalize_channel_name(ch_name)
        if norm_ch in norm_to_annot:
            mapping[ch_name] = norm_to_annot[norm_ch]
            used_annot_norms.add(norm_ch)
            continue
        close = difflib.get_close_matches(norm_ch, norm_annot_keys, n=1, cutoff=0.8)
        if close:
            mapping[ch_name] = norm_to_annot[close[0]]
            used_annot_norms.add(close[0])
        else:
            mapping[ch_name] = None

    unmatched_annotation_channels = [
        norm_to_annot[k] for k in norm_annot_keys if k not in used_annot_norms
    ]
    return mapping, unmatched_annotation_channels


# --- Bipolar montage support -------------------------------------------------
# TUSZ-style label CSVs annotate a *derived bipolar montage* (e.g. "FP1-F7" =
# EEG FP1-REF minus EEG F7-REF), not the raw monopolar recording channels. The
# montage_file named in the CSV header defines exactly which pairs to subtract.
# Lines in that file look like:
#     montage = 0, FP1-F7: EEG FP1-REF -- EEG F7-REF
_MONTAGE_LINE_RE = re.compile(
    r"^\s*montage\s*=\s*\d+\s*,\s*([^:]+?)\s*:\s*(.+?)\s*--\s*(.+?)\s*$", re.IGNORECASE
)

_montage_file_index = None  # lazily built: {basename.lower(): full_path}


def _index_montage_files(search_root):
    """Index montage definition files below a search directory for later lookup.
    
    Parameters:
    	search_root (str): Directory to search recursively for `.txt` montage files.
    """
    global _montage_file_index
    _montage_file_index = {}
    for root, _dirs, files in os.walk(search_root):
        for f in files:
            if f.lower().endswith(".txt"):
                _montage_file_index.setdefault(f.lower(), os.path.join(root, f))


def find_montage_file(montage_filename, search_root, montage_dir=None):
    """Locate a montage definition file by basename, searching montage_dir (if given)
    then search_root recursively (indexed once and cached across calls)."""
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
    """Parse a TUSZ-style montage definition file into [(bipolar_name, ch_a_raw, ch_b_raw), ...]."""
    derivations = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _MONTAGE_LINE_RE.match(line)
            if m:
                name, ch_a, ch_b = m.groups()
                derivations.append((name.strip(), ch_a.strip(), ch_b.strip()))
    return derivations


def guess_bipolar_split(name):
    """
    Infer the two source electrode names from a bipolar channel label.
    
    Parameters:
        name (str): Bipolar channel label containing two electrode names separated
            by the first hyphen.
    
    Returns:
        tuple[str, str] | None: The two electrode names, or None if the label does
            not contain two non-empty parts.
    """
    parts = name.strip().split("-", 1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        return None
    return parts[0].strip(), parts[1].strip()


def build_derived_bipolar_channels(data, channel_names, derivations):
    """
    Build derivable bipolar signals from monopolar EEG channels.
    
    Parameters:
        data (numpy.ndarray): Monopolar channel data with shape `(n_channels, n_samples)`.
        channel_names (Sequence[str]): Names corresponding to the rows of `data`.
        derivations (Iterable[tuple]): Bipolar definitions as `(bipolar_name, source_a, source_b)`.
    
    Returns:
        tuple: A mapping of bipolar names to `(source_a_index, source_b_index, signal)` tuples, and a list of bipolar names whose source channels could not be resolved.
    """
    norm_lookup = {normalize_channel_name(n): i for i, n in enumerate(channel_names)}
    norm_keys = list(norm_lookup.keys())

    def resolve(raw_name):
        """
        Resolve a raw channel name to its matching channel index.
        
        Parameters:
            raw_name (str): Electrode name to match.
        
        Returns:
            int or None: Matching channel index, or `None` if no channel matches.
        """
        norm = normalize_channel_name(raw_name)
        if norm in norm_lookup:
            return norm_lookup[norm]
        close = difflib.get_close_matches(norm, norm_keys, n=1, cutoff=0.8)
        return norm_lookup[close[0]] if close else None

    derived_signals = {}
    unresolved = []
    for name, ch_a_raw, ch_b_raw in derivations:
        idx_a = resolve(ch_a_raw)
        idx_b = resolve(ch_b_raw)
        if idx_a is None or idx_b is None:
            unresolved.append(name)
            continue
        derived_signals[name] = (idx_a, idx_b, data[idx_a] - data[idx_b])
    return derived_signals, unresolved


def find_all_files(folder):
    """Recursively find all supported files under folder, returned as paths relative to folder."""
    matches = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(SUPPORTED_EXTS):
                full = os.path.join(root, f)
                matches.append(os.path.relpath(full, folder))
    return sorted(matches)


def pick_random_files(folder, n, seed=None):
    """
    Select up to a specified number of supported files from a folder recursively.
    
    Parameters:
    	folder (str): Root folder to search.
    	n (int): Maximum number of files to select.
    	seed (int | None): Seed for reproducible selection.
    
    Returns:
    	list[str]: Selected file paths relative to `folder`.
    
    Raises:
    	SystemExit: If no supported files are found.
    """
    files = find_all_files(folder)
    if not files:
        raise SystemExit(f"No {SUPPORTED_EXTS} files found under {folder!r} (searched recursively)")
    rng = random.Random(seed)
    return rng.sample(files, min(n, len(files)))


def downsample_for_plot(y, max_points):
    """Decimate a 1D array for display only -- doesn't touch the underlying data/analysis."""
    n = len(y)
    if n <= max_points:
        return np.arange(n), y
    step = int(np.ceil(n / max_points))
    idx = np.arange(0, n, step)
    return idx, y[idx]


def mask_to_spans(mask_row):
    """
    Convert a one-dimensional boolean mask into contiguous index spans.
    
    Parameters:
    	mask_row: Boolean mask whose true regions should be identified.
    
    Returns:
    	A list of [start, end] pairs, with end indices exclusive.
    """
    spans = []
    in_span = False
    start = 0
    for i, v in enumerate(mask_row):
        if v and not in_span:
            start = i
            in_span = True
        elif not v and in_span:
            spans.append([start, i])
            in_span = False
    if in_span:
        spans.append([start, len(mask_row)])
    return spans


def interpolate_1d_with_mask(signal, mask):
    """Same rule as apply_interpolation_masking, for a single 1D signal + boolean mask.
    Returns (interpolated, fully_flagged)."""
    if not mask.any():
        return signal.copy(), False
    all_idx = np.arange(len(signal))
    clean_idx = all_idx[~mask]
    if len(clean_idx) == 0:
        return signal.copy(), True
    flagged_idx = all_idx[mask]
    out = signal.copy()
    out[flagged_idx] = np.interp(flagged_idx, clean_idx, signal[clean_idx])
    return out, False


def build_event_spans(events, fs, n_samples, mask):
    """
    Convert time-based event annotations into sample-index spans with artifact overlap statistics.
    
    Parameters:
        events (iterable): Event tuples containing start time, stop time, and label in seconds.
        fs (float): Sampling frequency in samples per second.
        n_samples (int): Number of samples used to clamp span boundaries.
        mask (array-like): Boolean artifact mask aligned with the samples.
    
    Returns:
        tuple: A tuple containing the event spans, the number of non-background events overlapping the mask, and the total number of non-background events. Each span is represented as [start index, stop index, label, overlap percentage, is_background].
    """
    spans = []
    n_overlapping = 0
    n_non_bckg = 0
    for start_sec, stop_sec, label in events:
        start_idx = max(0, min(n_samples, int(round(start_sec * fs))))
        stop_idx = max(0, min(n_samples, int(round(stop_sec * fs))))
        if stop_idx <= start_idx:
            continue
        is_bckg = label.strip().lower() == "bckg"
        overlap_frac = float(mask[start_idx:stop_idx].mean())
        spans.append([start_idx, stop_idx, label, round(overlap_frac * 100, 1), is_bckg])
        if not is_bckg:
            n_non_bckg += 1
            if overlap_frac > 0:
                n_overlapping += 1
    return spans, n_overlapping, n_non_bckg


def process_file(detector, folder, fname, fallback_fs, max_points, montage_dir=None):
    """
    Process one EEG file, compare zero masking with interpolation masking, and prepare report data.
    
    Parameters:
    	detector: Artifact detector used to identify artifact intervals.
    	folder (str): Root directory containing the EEG file.
    	fname (str): Relative path of the EEG file within `folder`.
    	fallback_fs: Sampling rate used when the input file does not provide one.
    	max_points (int): Maximum number of samples retained for each plotted trace.
    	montage_dir (str, optional): Directory to search for referenced montage files.
    
    Returns:
    	dict: File metadata, artifact statistics, annotation overlap statistics, unresolved annotation channels, and per-channel plot data.
    """
    path = os.path.join(folder, fname)
    data, fs, channel_names = load_eeg_file(path, fallback_fs)
    n_channels, n_samples = data.shape
    if channel_names is None:
        channel_names = [f"ch{ch}" for ch in range(n_channels)]

    annotations, montage_filename = load_annotations(path)  # {channel_name: [(start_sec, stop_sec, label), ...]}
    channel_to_annot_name, unmatched_annotation_channels = match_annotation_channels(
        channel_names, list(annotations.keys())
    )
    if annotations:
        n_matched_directly = sum(1 for v in channel_to_annot_name.values() if v is not None)
        logger.info(
            "%s: %d/%d recording channels matched an annotation channel directly; "
            "%d annotation channel(s) unmatched: %s",
            fname, n_matched_directly, n_channels, len(unmatched_annotation_channels),
            unmatched_annotation_channels,
        )

    # Two independent copies so zero-masking and interpolation-masking
    # never touch the same array or each other.
    copy_for_zero = data.copy()
    copy_for_interp = data.copy()

    det_result = detector.predict_segment(data, fs)

    zero_masked, zero_mask = apply_zero_masking(copy_for_zero, det_result, fs)
    interp_masked, interp_mask, fully_flagged = apply_interpolation_masking(
        copy_for_interp, det_result, fs
    )

    n_total_events = 0
    n_non_bckg_events = 0
    n_events_overlapping_artifact = 0
    channels = []
    for ch in range(n_channels):
        idx, orig_d = downsample_for_plot(data[ch], max_points)
        _, zero_d = downsample_for_plot(zero_masked[ch], max_points)
        _, interp_d = downsample_for_plot(interp_masked[ch], max_points)

        annot_name = channel_to_annot_name.get(channel_names[ch])
        matched_fuzzily = bool(
            annot_name and normalize_channel_name(annot_name) != normalize_channel_name(channel_names[ch])
        )
        ch_events = annotations.get(annot_name, []) if annot_name else []
        event_spans, n_overlap, n_non_bckg = build_event_spans(ch_events, fs, n_samples, zero_mask[ch])
        n_total_events += len(event_spans)
        n_non_bckg_events += n_non_bckg
        n_events_overlapping_artifact += n_overlap

        channels.append({
            "name": channel_names[ch],
            "x": idx.tolist(),
            "original": orig_d.tolist(),
            "zero_masked": zero_d.tolist(),
            "interp_masked": interp_d.tolist(),
            "artifact_spans": mask_to_spans(zero_mask[ch]),  # same spans for both maskers
            "event_spans": event_spans,  # all labeled intervals from the CSV sidecar, incl. "bckg"
            "fully_flagged": ch in fully_flagged,
            "annot_match": annot_name,       # which CSV channel name this was matched to (None if no annotations at all)
            "annot_fuzzy": matched_fuzzily,  # True if the match wasn't an exact (normalized) name match
            "is_derived": False,
        })

    # --- Bipolar-derived channels ------------------------------------------------
    # Any annotation channel name that never matched a raw recording channel
    # (e.g. "FP1-F7" against monopolar "FP1-REF"/"F7-REF" channels) may be a
    # bipolar montage derivation rather than a naming mismatch.
    #   1) If the CSV header names a montage file and we can find + parse it,
    #      use its exact source_a -- source_b definitions (most reliable: it
    #      also fixes the sign/order).
    #   2) Otherwise, guess the split from the label text itself ("FP1-F7" ->
    #      "FP1" minus "F7"). Works for standard two-electrode names but can't
    #      confirm sign/order the way a montage file can.
    montage_file_found = None
    derivation_source = None  # "montage_file" | "guessed_from_label" | None
    guessed_channels = []      # names derived via guessing, for a UI caveat
    if unmatched_annotation_channels:
        derivations = []

        if montage_filename:
            montage_path = find_montage_file(montage_filename, folder, montage_dir)
            if montage_path:
                montage_file_found = montage_path
                wanted = set(unmatched_annotation_channels)
                derivations = [d for d in parse_montage_file(montage_path) if d[0] in wanted]
                if derivations:
                    derivation_source = "montage_file"
                    logger.info("%s: deriving %d channel(s) from montage file %s", fname, len(derivations), montage_path)
            else:
                logger.warning(
                    "%s: CSV references montage_file=%s but it wasn't found under %s (pass --montage-dir)",
                    fname, montage_filename, folder,
                )

        if not derivations:
            for bipolar_name in unmatched_annotation_channels:
                split = guess_bipolar_split(bipolar_name)
                if split:
                    derivations.append((bipolar_name, split[0], split[1]))
            if derivations:
                derivation_source = "guessed_from_label"
                guessed_channels = [d[0] for d in derivations]
                logger.info("%s: guessing %d bipolar channel(s) from label text (no montage file)", fname, len(derivations))

        if derivations:
            derived_signals, _resolved_unresolved = build_derived_bipolar_channels(
                data, channel_names, derivations
            )
            still_unmatched = []
            for bipolar_name in unmatched_annotation_channels:
                if bipolar_name not in derived_signals:
                    still_unmatched.append(bipolar_name)
                    continue
                idx_a, idx_b, derived_original = derived_signals[bipolar_name]
                derived_mask = zero_mask[idx_a] | zero_mask[idx_b]
                derived_zero = derived_original.copy()
                derived_zero[derived_mask] = 0.0
                derived_interp, derived_fully_flagged = interpolate_1d_with_mask(derived_original, derived_mask)

                d_idx, d_orig = downsample_for_plot(derived_original, max_points)
                _, d_zero = downsample_for_plot(derived_zero, max_points)
                _, d_interp = downsample_for_plot(derived_interp, max_points)

                events, n_overlap, n_non_bckg = build_event_spans(
                    annotations.get(bipolar_name, []), fs, n_samples, derived_mask
                )
                n_total_events += len(events)
                n_non_bckg_events += n_non_bckg
                n_events_overlapping_artifact += n_overlap

                channels.append({
                    "name": bipolar_name,
                    "x": d_idx.tolist(),
                    "original": d_orig.tolist(),
                    "zero_masked": d_zero.tolist(),
                    "interp_masked": d_interp.tolist(),
                    "artifact_spans": mask_to_spans(derived_mask),
                    "event_spans": events,
                    "fully_flagged": derived_fully_flagged,
                    "annot_match": bipolar_name,
                    "annot_fuzzy": False,
                    "is_derived": True,
                    "derivation_guessed": bipolar_name in guessed_channels,
                    "source_channels": [channel_names[idx_a], channel_names[idx_b]],
                })
            unmatched_annotation_channels = still_unmatched
            if unmatched_annotation_channels:
                logger.warning(
                    "%s: %d annotation channel(s) still unresolved after derivation attempt: %s",
                    fname, len(unmatched_annotation_channels), unmatched_annotation_channels,
                )

    artifact_pct = 100.0 * zero_mask.mean()

    if not annotations:
        logger.info("%s: no annotation CSV -- 0 labeled events (nothing to compare against)", fname)
    elif n_non_bckg_events == 0:
        logger.info(
            "%s: annotation CSV loaded, %d total labeled interval(s), all 'bckg' -- 0 non-bckg events "
            "(this may be correct: most TUSZ recordings are seizure-free)",
            fname, n_total_events,
        )
    else:
        logger.info(
            "%s: %d total labeled interval(s) (%d non-bckg), %d non-bckg overlapping an artifact-flagged window",
            fname, n_total_events, n_non_bckg_events, n_events_overlapping_artifact,
        )

    return {
        "name": fname,
        "fs": fs,
        "n_channels": n_channels,
        "n_samples": n_samples,
        "artifact_fraction": det_result["artifact_fraction"],
        "artifact_pct_of_samples": artifact_pct,
        "total_windows": det_result["total_windows"],
        "fully_flagged_channels": fully_flagged,
        "has_annotations": bool(annotations),
        "n_total_events": n_total_events,
        "n_non_bckg_events": n_non_bckg_events,
        "n_events_overlapping_artifact": n_events_overlapping_artifact,
        "unmatched_annotation_channels": unmatched_annotation_channels,
        "montage_filename": montage_filename,
        "montage_file_found": montage_file_found,
        "derivation_source": derivation_source,
        "channels": channels,
    }


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Artifact Masking Comparison</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.32.0/plotly.min.js"></script>
<style>
  :root {
    --bg: #0f1115;
    --panel: #171a21;
    --border: #262a35;
    --text: #e6e8ec;
    --muted: #9aa1ac;
    --accent: #5b9bd5;
    --orig: #7f8ea3;
    --zero: #e0655b;
    --interp: #52b788;
    --event: #f2b134;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  header {
    padding: 20px 28px;
    border-bottom: 1px solid var(--border);
  }
  header h1 { margin: 0 0 4px 0; font-size: 20px; }
  header p { margin: 0; color: var(--muted); font-size: 13px; }
  .controls {
    display: flex;
    gap: 16px;
    padding: 16px 28px;
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
    align-items: center;
  }
  .controls label { font-size: 12px; color: var(--muted); display: block; margin-bottom: 4px; }
  select {
    background: var(--panel);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 13px;
  }
  .stats {
    display: flex;
    gap: 24px;
    padding: 14px 28px;
    flex-wrap: wrap;
    border-bottom: 1px solid var(--border);
    background: var(--panel);
  }
  .stat { min-width: 120px; }
  .stat .label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
  .stat .value { font-size: 18px; margin-top: 2px; }
  .flag-warning { color: var(--zero); font-size: 12px; margin-top: 6px; }
  main { padding: 20px 28px 40px; }
  .legend { display: flex; gap: 18px; margin-bottom: 8px; font-size: 12px; color: var(--muted); }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
  #plot { width: 100%; height: 720px; }
</style>
</head>
<body>
<header>
  <h1>Artifact Masking Comparison</h1>
  <p id="subtitle"></p>
</header>
<div class="controls">
  <div>
    <label for="fileSelect">File</label>
    <select id="fileSelect"></select>
  </div>
  <div>
    <label for="channelSelect">Channel</label>
    <select id="channelSelect"></select>
  </div>
</div>
<div class="stats" id="stats"></div>
<main>
  <div class="legend">
    <span><span class="swatch" style="background:var(--orig)"></span>original</span>
    <span><span class="swatch" style="background:var(--zero)"></span>zero-masked</span>
    <span><span class="swatch" style="background:var(--interp)"></span>interpolated</span>
    <span><span class="swatch" style="background:rgba(224,101,91,0.18)"></span>flagged artifact window</span>
    <span><span class="swatch" style="background:rgba(127,142,163,0.25)"></span>bckg (background, unshaded label)</span>
    <span><span class="swatch" style="background:rgba(242,177,52,0.30)"></span>non-bckg labeled event</span>
    <span style="border:2px solid #e0655b; border-radius:3px; padding:0 4px;">red border = event overlaps an artifact-flagged window</span>
  </div>
  <div id="plot"></div>
</main>

<script>
const DATA = __DATA_JSON__;

const fileSelect = document.getElementById('fileSelect');
const channelSelect = document.getElementById('channelSelect');
const statsEl = document.getElementById('stats');
const subtitle = document.getElementById('subtitle');

subtitle.textContent = `${DATA.length} file(s) sampled -- generated ${new Date().toLocaleString()}`;

DATA.forEach((f, i) => {
  const opt = document.createElement('option');
  opt.value = i;
  opt.textContent = `${f.name} (${f.n_channels} ch, fs=${f.fs})`;
  fileSelect.appendChild(opt);
});

function populateChannels(fileIdx) {
  channelSelect.innerHTML = '';
  const f = DATA[fileIdx];

  function makeOption(c, i) {
    const opt = document.createElement('option');
    opt.value = i;
    let label = c.is_derived ? `${c.name}  (derived: ${c.source_channels.join(' − ')}${c.derivation_guessed ? ', guessed' : ''})` : `Ch ${i}: ${c.name}`;
    if (c.annot_fuzzy) label += ` (matched to CSV "${c.annot_match}")`;
    const nonBckg = c.event_spans.filter(e => !e[4]);
    if (nonBckg.length) {
      const overlapping = nonBckg.filter(e => e[3] > 0).length;
      label += ` (${nonBckg.length} non-bckg event${nonBckg.length > 1 ? 's' : ''}${overlapping ? `, ${overlapping} overlap artifact mask` : ''})`;
    }
    if (c.fully_flagged) label += ' [fully flagged -- untouched by interp]';
    opt.textContent = label;
    return opt;
  }

  const rawIdxs = f.channels.map((c, i) => i).filter(i => !f.channels[i].is_derived);
  const derivedIdxs = f.channels.map((c, i) => i).filter(i => f.channels[i].is_derived);

  if (derivedIdxs.length) {
    const rawGroup = document.createElement('optgroup');
    rawGroup.label = 'Recording channels';
    rawIdxs.forEach(i => rawGroup.appendChild(makeOption(f.channels[i], i)));
    channelSelect.appendChild(rawGroup);

    const derivedGroup = document.createElement('optgroup');
    derivedGroup.label = f.derivation_source === 'guessed_from_label'
      ? 'Derived bipolar channels (guessed from label name -- no montage file)'
      : 'Derived bipolar montage channels';
    derivedIdxs.forEach(i => derivedGroup.appendChild(makeOption(f.channels[i], i)));
    channelSelect.appendChild(derivedGroup);
  } else {
    rawIdxs.forEach(i => channelSelect.appendChild(makeOption(f.channels[i], i)));
  }
}

function renderStats(fileIdx) {
  const f = DATA[fileIdx];
  let montageNote = '';
  if (f.derivation_source === 'montage_file') {
    montageNote = `<div class="stat"><div class="label">Bipolar channels</div><div class="value">derived from montage file (${f.montage_filename})</div></div>`;
  } else if (f.derivation_source === 'guessed_from_label') {
    montageNote = `<div class="stat"><div class="label">Bipolar channels</div><div class="value flag-warning">derived by guessing from label text -- no montage file${f.montage_filename ? ` (${f.montage_filename} not found)` : ''}</div></div>`;
  } else if (f.montage_filename && !f.montage_file_found) {
    montageNote = `<div class="stat"><div class="label">Montage file</div><div class="value flag-warning">${f.montage_filename} (not found -- pass --montage-dir)</div></div>`;
  }
  statsEl.innerHTML = `
    <div class="stat"><div class="label">Artifact fraction</div><div class="value">${(f.artifact_fraction * 100).toFixed(1)}%</div></div>
    <div class="stat"><div class="label">Samples flagged</div><div class="value">${f.artifact_pct_of_samples.toFixed(1)}%</div></div>
    <div class="stat"><div class="label">Total windows</div><div class="value">${f.total_windows}</div></div>
    <div class="stat"><div class="label">Samples / channels</div><div class="value">${f.n_samples} / ${f.n_channels}</div></div>
    <div class="stat"><div class="label">Labeled events (total / non-bckg)</div><div class="value">${f.has_annotations ? `${f.n_total_events} / ${f.n_non_bckg_events}` : '(no CSV found)'}</div></div>
    ${f.n_events_overlapping_artifact ? `<div class="stat"><div class="label">Non-bckg events overlapping artifact mask</div><div class="value flag-warning">${f.n_events_overlapping_artifact} / ${f.n_non_bckg_events}</div></div>` : ''}
    ${montageNote}
    ${f.unmatched_annotation_channels.length ? `<div class="stat"><div class="label">Unmatched CSV channels</div><div class="value flag-warning">${f.unmatched_annotation_channels.join(', ')}</div></div>` : ''}
    ${f.fully_flagged_channels.length ? `<div class="stat"><div class="label">Fully-flagged channels</div><div class="value flag-warning">${f.fully_flagged_channels.join(', ')}</div></div>` : ''}
  `;
}

function renderPlot(fileIdx, chIdx) {
  const c = DATA[fileIdx].channels[chIdx];

  const artifactShapes = c.artifact_spans.map(([s, e]) => ({
    type: 'rect', xref: 'x', yref: 'paper',
    x0: s, x1: e, y0: 0, y1: 1,
    fillcolor: 'rgba(224,101,91,0.18)', line: { width: 0 }
  }));
  // "bckg" spans get a subtle neutral shade -- present so full label coverage is visible,
  // but deliberately unobtrusive so the non-bckg events (seizure/artifact/etc.) still stand out.
  const bckgShapes = c.event_spans.filter(e => e[4]).map(([s, e]) => ({
    type: 'rect', xref: 'x', yref: 'paper',
    x0: s, x1: e, y0: 0, y1: 1,
    fillcolor: 'rgba(127,142,163,0.08)', line: { width: 0 }
  }));
  const eventShapes = c.event_spans.filter(e => !e[4]).map(([s, e, _label, overlapPct]) => ({
    type: 'rect', xref: 'x', yref: 'paper',
    x0: s, x1: e, y0: 0, y1: 1,
    fillcolor: 'rgba(242,177,52,0.30)',
    line: { color: overlapPct > 0 ? 'rgba(224,101,91,0.95)' : 'rgba(242,177,52,0.8)', width: overlapPct > 0 ? 2 : 1 }
  }));
  const shapes = [...bckgShapes, ...artifactShapes, ...eventShapes];

  // Label text sits just above the top subplot, centered on each non-bckg event span
  // (bckg spans aren't labeled individually -- there are usually many, and they're not the point).
  const annotations = c.event_spans.filter(e => !e[4]).map(([s, e, label, overlapPct]) => ({
    x: (s + e) / 2, xref: 'x', y: 1.0, yref: 'y domain',
    text: overlapPct > 0 ? `${label} (${overlapPct}% masked as artifact)` : label,
    showarrow: false, yanchor: 'bottom',
    font: { color: overlapPct > 0 ? '#e0655b' : '#f2b134', size: 11 },
  }));

  const traces = [
    { x: c.x, y: c.original, name: 'original', mode: 'lines',
      line: { color: '#7f8ea3', width: 1 }, xaxis: 'x', yaxis: 'y' },
    { x: c.x, y: c.zero_masked, name: 'zero-masked', mode: 'lines',
      line: { color: '#e0655b', width: 1 }, xaxis: 'x2', yaxis: 'y2' },
    { x: c.x, y: c.interp_masked, name: 'interpolated', mode: 'lines',
      line: { color: '#52b788', width: 1 }, xaxis: 'x3', yaxis: 'y3' },
  ];

  const layout = {
    grid: { rows: 3, columns: 1, pattern: 'independent', roworder: 'top to bottom' },
    paper_bgcolor: '#0f1115', plot_bgcolor: '#0f1115',
    font: { color: '#e6e8ec' },
    margin: { t: 40, r: 20, l: 50, b: 40 },
    showlegend: true,
    legend: { orientation: 'h', y: 1.12 },
    shapes: [...shapes, ...shapes.map(s => ({...s, yref: 'y2 domain'})), ...shapes.map(s => ({...s, yref: 'y3 domain'}))],
    annotations: annotations,
    xaxis: { anchor: 'y', matches: 'x3', showticklabels: false, gridcolor: '#262a35' },
    xaxis2: { anchor: 'y2', matches: 'x3', showticklabels: false, gridcolor: '#262a35' },
    xaxis3: { anchor: 'y3', title: 'sample index (decimated for display)', gridcolor: '#262a35' },
    yaxis: { title: 'orig', gridcolor: '#262a35' },
    yaxis2: { title: 'zero', gridcolor: '#262a35' },
    yaxis3: { title: 'interp', gridcolor: '#262a35' },
  };

  Plotly.newPlot('plot', traces, layout, { responsive: true, displaylogo: false });
}

function onFileChange() {
  const fileIdx = parseInt(fileSelect.value, 10);
  populateChannels(fileIdx);
  renderStats(fileIdx);
  renderPlot(fileIdx, 0);
}

fileSelect.addEventListener('change', onFileChange);
channelSelect.addEventListener('change', () => {
  renderPlot(parseInt(fileSelect.value, 10), parseInt(channelSelect.value, 10));
});

onFileChange();
</script>
</body>
</html>
"""


def build_html_report(results, out_path):
    """
    Write an interactive HTML report containing the provided results.
    
    Parameters:
    	results (object): Report data to serialize as JSON.
    	out_path (str or os.PathLike): Destination path for the generated HTML file.
    """
    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(results))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    """Sample EEG files, compare masking strategies, and generate an HTML report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="Folder containing .edf / .npy EEG files (searched recursively)")
    parser.add_argument("--n", type=int, default=5, help="Number of random files to sample (default 5)")
    parser.add_argument("--fs", type=float, default=256.0, help="Fallback sampling rate for .npy files with no <file>.fs.txt sidecar (default 256)")
    parser.add_argument("--out", default="masking_comparison.html", help="Output HTML path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible file sampling")
    parser.add_argument("--max-points", type=int, default=4000, help="Max points per channel/trace in the plot (display-only decimation)")
    parser.add_argument("--ckpt-dir", default=None, help="ArtifactDetector checkpoint dir override")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--montage-dir", default=None, help="Directory to look for montage definition .txt files first (e.g. 02_tcp_le_montage.txt), before searching the input folder recursively")
    args = parser.parse_args()

    chosen = pick_random_files(args.folder, args.n, seed=args.seed)
    logger.info("Sampled %d file(s): %s", len(chosen), chosen)

    detector = ArtifactDetector(ckpt_dir=args.ckpt_dir, device=args.device)

    results = []
    for fname in chosen:
        logger.info("Processing %s ...", fname)
        results.append(process_file(detector, args.folder, fname, args.fs, args.max_points, montage_dir=args.montage_dir))

    build_html_report(results, args.out)

    n_with_csv = sum(1 for r in results if r["has_annotations"])
    total_events = sum(r["n_total_events"] for r in results)
    total_non_bckg = sum(r["n_non_bckg_events"] for r in results)
    logger.info(
        "Wrote %s -- %d/%d sampled file(s) had an annotation CSV, %d total labeled interval(s) "
        "(%d non-bckg) across the run",
        args.out, n_with_csv, len(results), total_events, total_non_bckg,
    )
    if n_with_csv and total_non_bckg == 0:
        logger.info(
            "No non-bckg events in any sampled file. Check the per-file logs above for label distributions -- "
            "if they all show only 'bckg', that's likely genuine (TUSZ has far more seizure-free than seizure "
            "recordings); try a larger --n or point at files you know contain seizures."
        )


if __name__ == "__main__":
    main()