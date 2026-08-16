import os
import pandas as pd
from util import handle_logs

logger = handle_logs.get_logger("make_master_file", "applog")

SPLITS = ("train", "dev", "eval")

# Labels that do NOT count as reliable annotation coverage when deriving
# background windows -- i.e. a span carrying only one of these labels is
# treated the same as a span with no label at all. "bckg" is TUSZ's own
# background tag, but per project findings it does not reliably indicate
# artifact-free background (it's an artifact-flagged label, not a clean
# negative-class signal), so it must not be trusted as coverage.
DEFAULT_UNRELIABLE_LABELS = ("bckg",)


def get_split(path):
    """
    Determine the dataset split for a path.
    
    Parameters:
        path (str): File or directory path to inspect.
    
    Returns:
        str: The first matching split name from `train`, `dev`, or `eval`, or `"unknown"` if none is found.
    """
    parts_lower = [p.lower() for p in os.path.normpath(path).split(os.sep)]
    for split in SPLITS:
        if split in parts_lower:
            return split
    logger.warning(f"Could not determine split for path: {path}")
    return "unknown"


def get_unique_tags(dataset_path):
    """
    Scan CSV files under a dataset directory and collect unique label values.
    
    Parameters:
    	dataset_path (str): Root directory to search recursively.
    
    Returns:
    	set: Unique values found in the ``label`` column across readable CSV files.
    """
    logger.info(f"Scanning for unique tags in {dataset_path}")
    tags = set()
    csv_count = 0
    failed_count = 0

    for root, dirs, files in os.walk(dataset_path):
        for csv_file in [f for f in files if f.endswith(".csv")]:
            csv_path = os.path.join(root, csv_file)
            try:
                df = pd.read_csv(csv_path, comment="#")
                df.columns = df.columns.str.strip()
                if "label" not in df.columns:
                    logger.warning(f"Skipping {csv_path} - no 'label' column found")
                    failed_count += 1
                    continue
                new_tags = set(df["label"].unique())
                tags.update(new_tags)
                csv_count += 1
                logger.debug(f"Parsed {csv_path} - found tags: {new_tags}")
            except KeyError as e:
                logger.warning(f"Skipping {csv_path} - missing column {e}")
                failed_count += 1
            except Exception as e:
                logger.error(f"Failed to parse {csv_path}: {e}")
                failed_count += 1

    logger.info(f"Scanned {csv_count} valid CSV files, skipped {failed_count} - unique tags found: {tags}")
    return tags


def make_master_file(dataset_path, output_path="master.csv", allow_tag=None):
    """
    Build a master CSV from annotation files in a TUSZ-style dataset.
    
    Parameters:
    	dataset_path: Root directory containing annotation CSV files and matching EDF files.
    	output_path: Destination path for the generated master CSV.
    	allow_tag: Collection of labels to keep. If omitted, all labels found in the dataset are included.
    
    Returns:
    	A DataFrame containing the combined master rows, or None if no valid records are found.
    """
    logger.info(f"Building master file from {dataset_path}")
    records = []
    skipped_no_edf = 0
    skipped_no_allowed_tags = 0
    skipped_parse_error = 0

    if allow_tag is None:
        logger.info("No allow_tag provided - scanning for all unique tags")
        allow_tag = get_unique_tags(dataset_path)
        logger.info(f"Using all tags: {allow_tag}")
    else:
        logger.info(f"Filtering to tags: {allow_tag}")

    for root, dirs, files in os.walk(dataset_path):
        csv_files = [f for f in files if f.endswith(".csv")]

        for csv_file in csv_files:
            csv_path = os.path.join(root, csv_file)
            edf_path = os.path.join(root, csv_file.replace(".csv", ".edf"))

            if not os.path.exists(edf_path):
                logger.warning(f"No matching .edf for {csv_path}, skipping")
                skipped_no_edf += 1
                continue

            try:
                df = pd.read_csv(csv_path, comment="#")
                df.columns = df.columns.str.strip()

                # Validate required columns
                required_cols = ["label", "start_time", "stop_time", "channel"]
                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                    logger.error(f"Skipping {csv_path} - missing columns: {missing}")
                    skipped_parse_error += 1
                    continue

                filtered = df[df["label"].isin(allow_tag)].copy()

                if filtered.empty:
                    logger.debug(f"No allowed tags in {csv_path}, skipping")
                    skipped_no_allowed_tags += 1
                    continue

                filtered["edf_path"] = edf_path
                filtered["csv_path"] = csv_path
                filtered["split"] = get_split(edf_path)
                # -1 = not applicable; status only means something for preictal rows
                filtered["status"] = -1
                records.append(filtered)
                logger.debug(f"Added {len(filtered)} rows from {csv_path}")

            except Exception as e:
                logger.error(f"Failed to parse {csv_path}: {e}")
                skipped_parse_error += 1

    logger.info(
        f"Scan complete - "
        f"skipped {skipped_no_edf} (no EDF), "
        f"{skipped_no_allowed_tags} (no allowed tags), "
        f"{skipped_parse_error} (parse errors)"
    )

    if not records:
        logger.error("No records found - master file not written")
        return None

    master = pd.concat(records, ignore_index=True)

    cols = ["edf_path", "csv_path", "split", "channel", "start_time", "stop_time", "label", "confidence", "status"]
    cols = [c for c in cols if c in master.columns]
    master = master[cols]

    master.to_csv(output_path, index=False)
    logger.info(f"Master file written to {output_path} with {len(master)} rows")
    return master


def add_preictal_tags(df: pd.DataFrame, start_cutoff: float, max_duration: float) -> pd.DataFrame:
    """
    Adds preictal windows ('p{type}') prior to seizure onset.
    
    Status codes:
      -1 : Original seizure row
       1 : Valid preictal window
       0 : Collapsed window (insufficient baseline or start_time < 0)
    """
    rows = []
    
    for idx, row in df.iterrows():
        # Retain original seizure row
        rows.append(row.to_dict())
        
        pre_row = row.copy().to_dict()
        pre_row["label"] = f"p{row['label']}"
        
        # Calculate raw times
        raw_stop = row["start_time"] - start_cutoff
        raw_start = raw_stop - max_duration
        
        # Binary check: If raw_start < 0 or raw_stop <= 0, collapse window to 0 with status 0
        if raw_start < 0 or raw_stop <= 0:
            pre_row["start_time"] = 0.0
            pre_row["stop_time"] = 0.0
            pre_row["status"] = 0  # Collapsed / omitted
        else:
            pre_row["start_time"] = raw_start
            pre_row["stop_time"] = raw_stop
            pre_row["status"] = 1  # Valid window
            
        rows.append(pre_row)
        
    result_df = pd.DataFrame(rows)
    
    # Sort by split -> edf_path -> channel -> start_time
    if not result_df.empty and all(col in result_df.columns for col in ["split", "edf_path", "channel", "start_time"]):
        result_df = result_df.sort_values(
            by=["split", "edf_path", "channel", "start_time"], 
            ascending=[True, True, True, True]
        ).reset_index(drop=True)
        
    return result_df


def add_exclusion_intervals(
    master_df: pd.DataFrame, 
    postictal_time: float, 
    sph: float, 
    sop: float
) -> pd.DataFrame:
    """
    Adds exclusion intervals ('x{type}') immediately after seizures.
    """
    rows = []
    
    for idx, row in master_df.iterrows():
        rows.append(row.to_dict())
        
        # Skip creating exclusion tags for already-tagged generated rows
        if str(row["label"]).startswith(("p", "q", "c", "x")):
            continue
            
        excl_row = row.copy().to_dict()
        excl_row["label"] = f"x{row['label']}"
        
        excl_start = row["stop_time"]
        excl_stop = excl_start + postictal_time
        
        excl_row["start_time"] = excl_start
        excl_row["stop_time"] = excl_stop
        excl_row["status"] = 1  # Valid exclusion interval
        
        rows.append(excl_row)
        
    result_df = pd.DataFrame(rows)
    
    if not result_df.empty and all(col in result_df.columns for col in ["split", "edf_path", "channel", "start_time"]):
        result_df = result_df.sort_values(
            by=["split", "edf_path", "channel", "start_time"], 
            ascending=[True, True, True, True]
        ).reset_index(drop=True)
        
    return result_df


def add_postictal_and_consecutive(
    master_df: pd.DataFrame,
    postictal_time: float,
    preictal_duration: float,
) -> pd.DataFrame:
    """
    Adds q* postictal windows and c* consecutive windows for closely spaced seizures.
    """
    if master_df.empty:
        return master_df.copy()

    generated = []
    original_rows = master_df[~master_df["label"].astype(str).str.startswith(("p", "q", "c", "x"))]

    for (edf_path, channel), group in original_rows.groupby(["edf_path", "channel"], sort=False):
        group = group.sort_values(by=["start_time"]).reset_index(drop=True)
        idx = 0

        while idx < len(group):
            current = group.loc[idx].to_dict()
            next_row = group.loc[idx + 1].to_dict() if idx + 1 < len(group) else None

            if next_row is None:
                q_row = current.copy()
                q_row["label"] = f"q{current['label']}"
                q_row["start_time"] = current["stop_time"]
                q_row["stop_time"] = q_row["start_time"] + postictal_time
                q_row["status"] = 1
                generated.append(q_row)
                idx += 1
                continue

            gap = next_row["start_time"] - current["stop_time"]
            if gap < (postictal_time + preictal_duration):
                if current["label"] == next_row["label"]:
                    c_label = f"{current['label']}2"
                else:
                    c_label = f"{current['label']}{next_row['label']}"

                c_row = current.copy()
                c_row["label"] = f"c{c_label}"
                c_row["start_time"] = current["stop_time"]
                c_row["stop_time"] = next_row["start_time"] - preictal_duration

                if c_row["stop_time"] <= c_row["start_time"]:
                    c_row["stop_time"] = c_row["start_time"]
                    c_row["status"] = 2
                else:
                    c_row["status"] = 1

                generated.append(c_row)
                idx += 1
            else:
                q_row = current.copy()
                q_row["label"] = f"q{current['label']}"
                q_row["start_time"] = current["stop_time"]
                q_row["stop_time"] = q_row["start_time"] + postictal_time
                q_row["status"] = 1
                generated.append(q_row)
                idx += 1

    if not generated:
        return master_df.copy()

    result_df = pd.concat([master_df.copy(), pd.DataFrame(generated)], ignore_index=True)
    if all(col in result_df.columns for col in ["split", "edf_path", "channel", "start_time"]):
        result_df = result_df.sort_values(
            by=["split", "edf_path", "channel", "start_time"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)

    return result_df

def resolve_overlaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the highest priority annotation when overlaps occur on the same (edf_path, channel).
    Priority: exclusion (x*) > consecutive (c*) > preictal (p*) > original ictal > postictal (q*)
    """
    if df.empty:
        return df

    def _get_label_priority(label):
        lbl = str(label)
        if lbl.startswith('x'):
            return 6
        if lbl.startswith('c'):
            return 5
        if lbl.startswith('p'):
            return 4
        if lbl.startswith('q'):
            return 1
        # Derived background windows (see add_background_tags) are the
        # lowest priority -- they fill gaps with no reliable label, so any
        # real annotation (including the original "bckg" tag) should win
        # if boundaries ever coincide exactly.
        if lbl.startswith('bg'):
            return 0
        return 2
    
    df = df.copy()
    df['priority'] = df['label'].apply(_get_label_priority)
    
    df = df.sort_values(
        by=['edf_path', 'channel', 'start_time', 'priority'],
        ascending=[True, True, True, False]
    )
    
    df = df.drop_duplicates(
        subset=['edf_path', 'channel', 'start_time', 'stop_time'], 
        keep='first'
    )
    
    df = df.drop(columns=['priority'])
    
    logger.info(f"Overlap resolution complete. Final row count: {len(df)}")
    return df


def _get_edf_duration_seconds(edf_path, cache):
    """
    Header-only lookup (no data preload) of an .edf recording's duration in
    seconds, memoized in `cache` since multiple channel groups share the
    same file.
    """
    if edf_path in cache:
        return cache[edf_path]

    # Imported lazily so this module stays importable (e.g. for unit tests
    # on the tag-generation logic) in environments without mne installed.
    from pipeline.recording_info import get_recording_info

    try:
        info = get_recording_info(edf_path)
        duration = float(info["n_times"]) / float(info["sfreq"])
    except Exception as e:
        logger.error(f"Could not read duration for {edf_path}: {e}")
        duration = None

    cache[edf_path] = duration
    return duration


def add_background_tags(
    master_df: pd.DataFrame,
    window_duration: float,
    stride: float = None,
    min_gap: float = None,
    label: str = "bg",
    unreliable_labels=DEFAULT_UNRELIABLE_LABELS,
    duration_cache: dict = None,
) -> pd.DataFrame:
    """
    Derive explicit background/negative-class windows from the stretches of
    each recording that carry NO reliable label at all -- not from the
    "bckg" tag, which is an artifact-flagged label and is not trustworthy as
    a clean negative-class signal (any label in `unreliable_labels` is
    treated as if that span were unlabeled).

    For every (edf_path, channel) group of ORIGINAL rows (status == -1,
    excluding any previously generated p*/q*/c*/x* rows):
      1. Reads the recording's true duration from its .edf header (not just
         the span covered by annotations), so background before the first
         event / after the last event is captured too, not only gaps
         between events.
      2. Computes the complement of the "reliable" (non-unreliable-labeled)
         intervals over [0, duration].
      3. Chops every gap >= min_gap into consecutive `window_duration`
         second windows (stride `stride`, default = window_duration i.e.
         non-overlapping) -- so a single long gap becomes MANY background
         training rows proportional to its length, the same way preictal
         windows are already one-per-event, instead of collapsing to a
         single row regardless of duration.

    Parameters:
        master_df: DataFrame as returned by make_master_file (+ any tags
            already added -- this only reads original rows regardless of
            what's already been layered on).
        window_duration: Length, in seconds, of each generated background
            row -- should match (or be a multiple of) the model's window
            length.
        stride: Step, in seconds, between consecutive generated windows
            within a gap. Defaults to `window_duration` (non-overlapping).
        min_gap: Minimum gap length, in seconds, worth tagging at all.
            Defaults to `window_duration` (shorter gaps can't fit a window).
        label: Label to assign generated rows (default "bg"). Kept distinct
            from "bckg" so downstream code can tell true derived background
            apart from the original (unreliable) tag.
        unreliable_labels: Labels that do NOT block a background window
            from being generated over their span (default: ("bckg",)).
        duration_cache: Optional dict to memoize edf duration lookups
            across multiple calls.

    Returns:
        master_df with the generated background rows appended (status=1).
        Original rows, including any "bckg" rows already in master_df, are
        left untouched -- this only ADDS new "bg" rows, it doesn't remove
        or relabel anything.
    """
    if master_df.empty:
        return master_df.copy()

    stride = stride if stride is not None else window_duration
    min_gap = min_gap if min_gap is not None else window_duration
    if window_duration <= 0 or stride <= 0:
        raise ValueError(f"window_duration and stride must be positive, got {window_duration}, {stride}")

    cache = duration_cache if duration_cache is not None else {}

    original_rows = master_df[
        ~master_df["label"].astype(str).str.startswith(("p", "q", "c", "x", "bg"))
    ]

    generated = []
    skipped_no_duration = 0

    for (edf_path, channel), group in original_rows.groupby(["edf_path", "channel"], sort=False):
        duration = _get_edf_duration_seconds(edf_path, cache)
        if duration is None or duration <= 0:
            skipped_no_duration += 1
            continue

        group = group.sort_values(by=["start_time"])
        reliable = group[~group["label"].astype(str).isin(unreliable_labels)]

        # Merge reliable intervals, clipped to [0, duration], to find the complement.
        covered = []
        for _, row in reliable.sort_values(by=["start_time"]).iterrows():
            s = max(0.0, float(row["start_time"]))
            e = min(duration, float(row["stop_time"]))
            if e <= s:
                continue
            if covered and s <= covered[-1][1]:
                covered[-1] = (covered[-1][0], max(covered[-1][1], e))
            else:
                covered.append((s, e))

        gaps = []
        cursor = 0.0
        for s, e in covered:
            if s > cursor:
                gaps.append((cursor, s))
            cursor = max(cursor, e)
        if cursor < duration:
            gaps.append((cursor, duration))

        # Template row for columns we need to carry forward (split, csv_path, ...)
        template = group.iloc[0].to_dict()

        for gap_start, gap_end in gaps:
            if (gap_end - gap_start) < min_gap:
                continue
            t = gap_start
            while t + window_duration <= gap_end:
                bg_row = dict(template)
                bg_row["label"] = label
                bg_row["start_time"] = t
                bg_row["stop_time"] = t + window_duration
                bg_row["status"] = 1
                if "confidence" in bg_row:
                    bg_row["confidence"] = 1.0
                generated.append(bg_row)
                t += stride

    logger.info(
        f"add_background_tags: generated {len(generated)} '{label}' rows "
        f"({skipped_no_duration} (edf_path, channel) groups skipped - no readable duration)"
    )

    if not generated:
        return master_df.copy()

    result_df = pd.concat([master_df.copy(), pd.DataFrame(generated)], ignore_index=True)
    if all(col in result_df.columns for col in ["split", "edf_path", "channel", "start_time"]):
        result_df = result_df.sort_values(
            by=["split", "edf_path", "channel", "start_time"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)

    return result_df