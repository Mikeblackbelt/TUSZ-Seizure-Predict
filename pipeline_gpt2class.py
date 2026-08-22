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

# Bare TUSZ seizure-type labels (the "original ictal" rows before any
# p*/q*/c*/x* marker gets prefixed onto them). Kept in sync with
# TUSZ-Conformer-Prediction/dataset/label_utils.py's KNOWN_SEIZURE_TYPES --
# if you add a type there, add it here too, and vice versa.
#
# FIX: two of these ("cpsz" complex-partial, "cnsz" clonic) start with the
# same letter "c" that this pipeline also uses as a PREFIX to mark
# consecutive-seizure rows (e.g. "cgnszfnsz"). Several functions below used
# to identify "original" ictal rows with a raw
# `label.startswith(("p","q","c","x"))` check, which can't distinguish
# "cpsz"/"cnsz" (a bare seizure type that happens to start with 'c') from a
# genuine "c"-prefixed consecutive marker. That silently treated real
# cpsz/cnsz seizure rows as if they were derived/marker rows: their spans
# were excluded from "reliable" coverage when deriving background windows
# (so synthetic "bg" rows could get stamped directly on top of real
# cpsz/cnsz seizure activity), they never received a postictal q* tag or
# consecutive-seizure handling, and they were mis-prioritized during overlap
# resolution. Use KNOWN_SEIZURE_TYPES membership (an exact allow-list check)
# instead of prefix parsing wherever "is this an original/bare ictal row?"
# is being decided -- that's unambiguous regardless of which letter a type
# name happens to start with.
KNOWN_SEIZURE_TYPES = {
    "fnsz", "gnsz", "cpsz", "spsz", "tnsz", "tcsz",
    "absz", "mysz", "cnsz", "atsz", "seiz", "sz", "nesz",
}


def _is_original_ictal_label(label) -> bool:
    """True if *label* is a bare seizure-type tag (no p/q/c/x marker
    prefix yet) -- i.e. it is a row as it originally appeared in the raw
    TUSZ annotation CSVs, before this pipeline added any derived rows.

    Uses exact membership in KNOWN_SEIZURE_TYPES rather than prefix
    stripping, so real types that happen to start with the same letter as
    a marker prefix (namely "cpsz"/"cnsz" vs the "c" consecutive-marker
    prefix) are never misclassified.
    """
    return str(label) in KNOWN_SEIZURE_TYPES


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
                filtered["is_valid"] = True
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

    cols = ["edf_path", "csv_path", "split", "channel", "start_time", "stop_time", "label", "confidence", "is_valid", "status"]
    cols = [c for c in cols if c in master.columns]
    master = master[cols]

    master.to_csv(output_path, index=False)
    logger.info(f"Master file written to {output_path} with {len(master)} rows")
    return master

def add_interictal_tags(master_df, recording_durations, min_interictal_length=0):
    """
    Add interictal (background) rows for gaps between existing labeled
    windows, per (edf_path, channel).

    .. deprecated::
        add_interictal_tags() is deprecated. Standardize entirely on
        add_background_tags() for fixed-length 'bg' negative-class generation.

    Parameters:
        master_df (pd.DataFrame): master annotations, after preictal/
            postictal/consecutive tagging.
        recording_durations (dict): edf_path -> total recording duration (s).
        min_interictal_length (float): minimum gap length to keep.

    Returns:
        pd.DataFrame with interictal rows added.
    """
    import warnings
    warnings.warn(
        "add_interictal_tags() is deprecated and will be retired in a future release. "
        "Use add_background_tags() instead, which standardizes negative-class "
        "generation on fixed-length 'bg' windows.",
        DeprecationWarning,
        stacklevel=2,
    )
    if master_df is None or master_df.empty:
        raise ValueError("master_df cannot be None or empty")

    new_rows = []
    for (edf_path, channel), group in master_df.groupby(["edf_path", "channel"]):
        duration = recording_durations.get(edf_path)
        if duration is None:
            logger.warning(f"No duration for {edf_path} - skipping interictal for this group")
            continue

        intervals = group.sort_values("start_time")[["start_time", "stop_time"]].values.tolist()

        merged = [] 
        for start, stop in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], stop)
            else:
                merged.append([start, stop])

        template = group.iloc[0].to_dict()
        cursor = 0.0
        for start, stop in merged:
            gap = start - cursor
            if gap > 0 and gap >= min_interictal_length:
                new_rows.append({**template, "label": "interictal",
                                "start_time": cursor, "stop_time": start, "is_valid": True})
            cursor = max(cursor, stop)

        trailing_gap = duration - cursor
        if trailing_gap > 0 and trailing_gap >= min_interictal_length:
            new_rows.append({**template, "label": "interictal",
                            "start_time": cursor, "stop_time": duration, "is_valid": True})
            
    if not new_rows:
        logger.warning("No interictal rows generated")
        return master_df.copy()

    result = pd.concat([master_df, pd.DataFrame(new_rows)], ignore_index=True)
    return result.sort_values(["split", "edf_path", "channel", "start_time"]).reset_index(drop=True)


def add_preictal_tags(master_df, sph, sop, postictal_time=None):
    """
    Add preictal window rows to a master annotations DataFrame.

    Definitions (per user correction - previously swapped):
      - SOP (Seizure Occurrence Period): the buffer/safety-clearance gap
        kept immediately before seizure onset. No preictal data is drawn
        from this gap.
      - SPH (Seizure Prediction Horizon): the actual preictal window length
        that gets extracted and analyzed, sitting before the SOP buffer.

    A preictal window for seizure `j` is exactly `[t_j(start) - sop - sph,
    t_j(start) - sop]` - always this exact length, SPH+SOP. There is no
    partial/trimmed case: a seizure's preictal example is either the full
    required length (is_valid=True) or entirely dropped (is_valid=False). This is a hard all-or-nothing rule - no window
    is ever shortened to fit available space. A dropped window has [0, j_start]
    on a Gate 1 failure or [i_end, j_start] on a Gate 2
    failure, so downstream logic (e.g. add_interictal_tags) correctly
    treats that stretch as unusable/occupied time rather than as clean,
    samplable background.

    Two independent hard gates determine whether a seizure's preictal
    window is viable at all:

      Gate 1 (Session Start): t_j(start) < SPH + SOP
          Not enough time exists between the recording's start and this
          seizure for a full preictal window to fit at all.

      Gate 2 (Inter-Seizure Gap): only checked if there's a previous
      seizure `i` in the same (edf_path, channel) group, and only if
      `postictal_time` is provided (see below).
          gap = t_j(start) - t_i(end)
          gap < postictal_time + sph + sop
          Not enough clearance exists after the previous seizure's
          recovery period for this seizure's preictal window to be
          extracted cleanly - it would overlap the previous seizure's
          ictal or recovery time.

    If EITHER gate trips, the seizure is not viable for preictal
    extraction: is_valid=False, start_time=stop_time=0.0. Otherwise: full
    window, is_valid=True.  Otherwise: full window, is_valid=True, plus a SOP buffer row (is_valid=False - not
    extracted as training data, but a real interval, not a zeroed/dropped
    one, so resolve_overlaps() still resolves it normally).

    Parameters:
        master_df (pd.DataFrame): Master annotations table with at least
            `edf_path`, `channel`, `start_time`, `stop_time`, and `label`
            columns.
        sph (numeric): Seizure Prediction Horizon - the preictal window
            length that gets extracted.
        sop (numeric): Seizure Occurrence Period - the buffer kept before
            seizure onset.
        postictal_time (numeric or None): Recovery window length used by
            Gate 2. If None (default), Gate 2 is skipped entirely - only
            Gate 1 (session start) applies. Pass this whenever you also
            plan to call add_exclusion_intervals() with the same value, so
            preictal viability and exclusion bounds stay consistent.

    Returns:
        pd.DataFrame: The original rows plus generated preictal rows
        (viable and dropped), sorted by split, edf_path, channel,
        start_time.

    Raises:
        ValueError: If master_df is None or empty.
    """
    if master_df is None:
        logger.error("master_df is None - cannot add preictal tags. Check that make_master_file() found valid records.")
        raise ValueError("master_df cannot be None. No valid records found in dataset.")

    if master_df.empty:
        logger.error("master_df is empty - cannot add preictal tags")
        raise ValueError("master_df is empty. No valid records found in dataset.")

    logger.info(
        f"Adding preictal tags (sph={sph}, sop={sop}, postictal_time={postictal_time}) "
        f"to {len(master_df)} rows"
    )
    preictal_rows = []
    valid_counts = {False: 0, True: 0}
    group_cols = ["edf_path", "channel"]

    for (edf_path, channel), group in master_df.groupby(group_cols):
        ictal = group.sort_values("start_time").reset_index(drop=True)

        for i in range(len(ictal)):
            row = ictal.iloc[i]
            j_start = row["start_time"]

            # Gate 1: session start
            gate1_fail = j_start < (sph + sop)

            # Gate 2: inter-seizure gap (only if postictal_time given and
            # there's a previous seizure to measure against)
            gate2_fail = False
            if postictal_time is not None and i > 0:
                i_end = float(ictal.iloc[i - 1]["stop_time"])
                gap = j_start - i_end
                gate2_fail = gap < (postictal_time + sph + sop)

            if gate1_fail or gate2_fail:
                preictal_start = 0.0 if gate1_fail else i_end
                preictal_end = j_start
                is_valid = False
                reason = "gate1 (session start)" if gate1_fail else "gate2 (inter-seizure gap)"
                logger.debug(
                    f"Preictal window dropped (is_valid=False, {reason}) for {edf_path} "
                    f"channel={channel} at ictal_start={j_start}"
                )
            else:
                preictal_start = j_start - sop - sph
                preictal_end = j_start - sop
                is_valid = True

            valid_counts[is_valid] += 1

            preictal_rows.append({
                **row.to_dict(),
                "label": f"p{row['label']}",
                "start_time": preictal_start,
                "stop_time": preictal_end,
                "is_valid": is_valid,
                "status": 1,
            })
            # Always block the SOP buffer
            if is_valid == 1:
                preictal_rows.append({
                    **row.to_dict(),
                    "label": f"p{row['label']}_sopbuffer",
                    "start_time": j_start - sop,
                    "stop_time": j_start,
                    "is_valid": False,
                    "status": 1,
                })

    if valid_counts[0]:
        logger.warning(
            f"{valid_counts[0]} preictal windows dropped entirely (is_valid=False, "
            f"gate failure), {valid_counts[1]} valid full-length windows kept"
        )

    preictal_df = pd.DataFrame(preictal_rows)
    result = pd.concat([master_df, preictal_df], ignore_index=True).sort_values(
        ["split", "edf_path", "channel", "start_time"]
    ).reset_index(drop=True)

    return result


def add_exclusion_intervals(master_df, postictal_time):
    """
    Add exclusion-interval rows covering post-seizure recovery time.

    Now unconditionally flat: every seizure gets a `postictal_time`-length
    exclusion window immediately after it ends, regardless of how close the
    next seizure is. This is deliberately simpler than the earlier
    merge-with-next-seizure version - viability of the NEXT seizure's
    preictal window (i.e., whether seizures are too close together) is now
    Gate 2's job in add_preictal_tags(), not this function's. This function
    only marks clean exclusion bounds so an interictal-window builder
    doesn't sample post-seizure recovery time as baseline activity.

    If a flat postictal window happens to run past the next seizure's
    start (seizures close together), that's expected - resolve_overlaps()
    resolves the conflict by priority (seiz/pseiz rows outrank x* rows), so
    the exclusion row never incorrectly displaces real ictal or preictal
    data; it only fills in genuinely unlabeled time.

    Must be called AFTER add_preictal_tags() if you want the same
    postictal_time used consistently for both Gate 2 and this function's
    exclusion bounds - pass the same value to both.

    Exclusion rows are labeled with an `x` prefix (e.g. `xfnsz`) and given
    is_valid=False (valid exclusion interval).

    Parameters:
        master_df (pd.DataFrame): Master annotations table. Only original
            ictal rows (not already prefixed `p` or `x`) generate exclusion
            rows.
        postictal_time (numeric): Flat recovery window length after every
            seizure.

    Returns:
        pd.DataFrame: The original rows plus generated exclusion rows,
        sorted by split, edf_path, channel, start_time.

    Raises:
        ValueError: If master_df is None or empty.
    """
    if master_df is None or master_df.empty:
        logger.error("master_df is None or empty - cannot add exclusion intervals")
        raise ValueError("master_df cannot be None or empty")

    logger.info(f"Adding exclusion intervals (postictal_time={postictal_time})")

    new_rows = []
    group_cols = ["edf_path", "channel"]

    for (edf_path, channel), group in master_df.groupby(group_cols):
        # Only original ictal rows generate exclusion rows - not preictal
        # ('p'-prefixed) and not exclusion rows from a prior call.
        ictal = group[~group["label"].astype(str).str.startswith(("p", "x"), na=False)]

        for _, row in ictal.iterrows():
            i_end = float(row["stop_time"])
            new_rows.append({
                **row.to_dict(),
                "label": f"x{row['label']}",
                "start_time": i_end,
                "stop_time": i_end + postictal_time,
                "is_valid": False,
                "status": 1,
            })

    if new_rows:
        full_df = pd.concat([master_df, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        full_df = master_df.copy()

    full_df = resolve_overlaps(full_df)

    return full_df.sort_values(["split", "edf_path", "channel", "start_time"]).reset_index(drop=True)


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
    # FIX: was `~label.startswith(("p","q","c","x"))`, which wrongly
    # excluded bare "cpsz"/"cnsz" ictal rows (they start with "c") from
    # "original_rows" -- see _is_original_ictal_label() docstring above.
    # That meant cpsz/cnsz seizures never got a postictal q* tag and never
    # participated in consecutive-seizure (c*) merging with neighboring
    # seizures, unlike every other seizure type.
    original_rows = master_df[master_df["label"].astype(str).apply(_is_original_ictal_label)]

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
    Priority: exclusion (x*) > preictal (p*) > original ictal

    Rows skip resolution only if they're zeroed gate-failure rows
    (is_valid=False, no `_sopbuffer` suffix) - these collapse to identical
    (edf_path, channel, 0, 0) rows across different seizures, which would
    falsely dedupe. SOP buffer rows are also is_valid=False but have real,
    distinct time ranges, so they're treated as active and still get
    carved/displaced by higher-priority rows like any other interval.
    """
    if df.empty:
        return df

    def _get_label_priority(label):
        lbl = str(label)
        # FIX: check bare seizure-type labels (e.g. "cpsz", "cnsz") FIRST,
        # via exact allow-list membership, before falling back to prefix
        # checks. Previously `lbl.startswith('c')` caught "cpsz"/"cnsz" and
        # mis-prioritized them as if they were "c"-prefixed consecutive
        # marker rows instead of original ictal rows -- see
        # _is_original_ictal_label() docstring above for the full story.
        if _is_original_ictal_label(lbl):
            return 1
        if lbl.startswith('x'):
            return 6
        if lbl.startswith('c'):
            return 5
        if lbl.startswith('p'):
            return 2
        if lbl == 'interictal': 
            return 0
        return 1

    df = df.copy()

    is_sopbuffer = df['label'].astype(str).str.endswith('_sopbuffer', na=False)
    dropped = df[~df['is_valid'] & ~is_sopbuffer]
    active = df[df['is_valid'] | is_sopbuffer].copy()

    active['priority'] = active['label'].apply(_get_label_priority)

    kept_rows = []
    for (edf_path, channel), group in active.groupby(['edf_path', 'channel']):
        group = group.sort_values(
            ['priority', 'start_time'], ascending=[False, True]
        )
        placed = []  # accepted (start, stop) intervals for this group so far

        for _, row in group.iterrows():
            segments = [(row['start_time'], row['stop_time'])]
            for (a_start, a_stop) in placed:
                next_segments = []
                for (s, e) in segments:
                    if a_stop <= s or a_start >= e:
                        next_segments.append((s, e))
                        continue
                    if a_start > s:
                        next_segments.append((s, a_start))
                    if a_stop < e:
                        next_segments.append((a_stop, e))
                segments = next_segments

            for (s, e) in segments:
                if e - s <= 0:
                    continue
                new_row = row.to_dict()
                new_row['start_time'] = s
                new_row['stop_time'] = e
                kept_rows.append(new_row)
                placed.append((s, e))

    active = pd.DataFrame(kept_rows)
    if not active.empty:
        active = active.drop(columns=['priority'])

    df = pd.concat([active, dropped], ignore_index=True)

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

    # FIX: was `~label.startswith(("p","q","c","x","bg"))`, which wrongly
    # excluded bare "cpsz"/"cnsz" ictal rows (they start with "c") from
    # "original_rows" -- see _is_original_ictal_label() docstring above.
    # Since this dataframe is used below to compute "reliable" coverage and
    # find the complement (uncovered gaps) to fill with synthetic "bg"
    # rows, excluding real cpsz/cnsz seizure spans made them look like
    # uncovered gaps -- i.e. this could stamp synthetic background labels
    # directly on top of real, un-derived cpsz/cnsz seizure activity.
    original_rows = master_df[
        master_df["label"].astype(str).apply(_is_original_ictal_label)
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


def chop_master_windows(
    master_df: pd.DataFrame,
    window_duration: float = 4.0,
    preictal_stride: float = None,
    preictal_adaptive: bool = False,
) -> pd.DataFrame:
    """
    Chop all event rows (preictal p*, original ictal, consecutive c*, postictal q*/x*)
    in master_df into fixed-length `window_duration`-second segments.

    Invalid rows (is_valid=False, e.g. SOP buffers _sopbuffer or gate failures)
    are preserved as-is so unusable intervals remain tracked.

    Parameters:
        master_df (pd.DataFrame): Annotations DataFrame after resolve_overlaps().
        window_duration (float): Length in seconds of each model window (default 4.0).
        preictal_stride (float or None): Step size in seconds between consecutive
            preictal windows. If None, defaults to `window_duration` (non-overlapping).
        preictal_adaptive (bool): If True, compute adaptive overlap step for
            preictal rows relative to background window duration.

    Returns:
        pd.DataFrame: New master DataFrame where every valid row is a fixed
        `window_duration`-length segment.
    """
    if master_df is None or master_df.empty:
        return master_df

    from pipeline.windows import segment_df_fixed, segment_df_adaptive

    df = master_df.copy()

    # Separate invalid rows (gate failures, _sopbuffer) from valid active rows
    is_sopbuffer = df["label"].astype(str).str.endswith("_sopbuffer", na=False)
    invalid_mask = ~df["is_valid"] | is_sopbuffer
    invalid_df = df[invalid_mask].copy()
    valid_df = df[~invalid_mask].copy()

    if valid_df.empty:
        return master_df.copy()

    # Background rows ('bg') were generated with window_duration. Any bg row
    # shorter than window_duration (e.g. truncated by overlap resolution) is dropped.
    bg_mask = valid_df["label"] == "bg"
    bg_df = valid_df[bg_mask].copy()
    if not bg_df.empty:
        bg_df = bg_df[(bg_df["stop_time"] - bg_df["start_time"]) >= window_duration - 1e-9]

    # Non-bg valid rows (preictal, original ictal, c*, q*, x*)
    non_bg_df = valid_df[~bg_mask].copy()

    if non_bg_df.empty:
        result = pd.concat([bg_df, invalid_df], ignore_index=True)
    else:
        # Preictal vs non-preictal
        is_pre = non_bg_df["label"].astype(str).str.startswith("p")
        pre_df = non_bg_df[is_pre]
        other_event_df = non_bg_df[~is_pre]

        # Segment other events (ictal, c*, q*, x*) with fixed non-overlapping windows
        segmented_other = segment_df_fixed(other_event_df, seg_time=window_duration, step_time=window_duration)

        # Segment preictal
        if preictal_adaptive and not bg_df.empty:
            total_len_bg = (bg_df["stop_time"] - bg_df["start_time"]).sum()
            segmented_pre = segment_df_adaptive(pre_df, seg_time=window_duration, total_len_bg=total_len_bg)
        else:
            p_step = preictal_stride if preictal_stride is not None else window_duration
            segmented_pre = segment_df_fixed(pre_df, seg_time=window_duration, step_time=p_step)

        result = pd.concat([bg_df, segmented_pre, segmented_other, invalid_df], ignore_index=True)

    if all(col in result.columns for col in ["split", "edf_path", "channel", "start_time"]):
        result = result.sort_values(
            by=["split", "edf_path", "channel", "start_time"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)

    logger.info(f"chop_master_windows: converted {len(master_df)} input rows into {len(result)} fixed-window rows (window_duration={window_duration}s)")
    return result