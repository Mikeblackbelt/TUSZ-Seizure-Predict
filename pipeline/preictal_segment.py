import os
import pandas as pd
from util import handle_logs

logger = handle_logs.get_logger("make_master_file", "logs/app.log")

SPLITS = ("train", "dev", "eval")


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

    for root, dirs, files in os.walk(dataset_path):
        for csv_file in [f for f in files if f.endswith(".csv")]:
            csv_path = os.path.join(root, csv_file)
            try:
                df = pd.read_csv(csv_path, comment="#")
                df.columns = df.columns.str.strip()
                new_tags = set(df["label"].unique())
                tags.update(new_tags)
                csv_count += 1
                logger.debug(f"Parsed {csv_path} - found tags: {new_tags}")
            except Exception as e:
                logger.error(f"Failed to parse {csv_path}: {e}")

    logger.info(f"Scanned {csv_count} CSV files - unique tags found: {tags}")
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
        logger.warning("No records found - master file not written")
        return None

    master = pd.concat(records, ignore_index=True)

    cols = ["edf_path", "csv_path", "split", "channel", "start_time", "stop_time", "label", "confidence", "status"]
    cols = [c for c in cols if c in master.columns]
    master = master[cols]

    master.to_csv(output_path, index=False)
    logger.info(f"Master file written to {output_path} with {len(master)} rows")
    return master


def add_preictal_tags(master_df, start_cutoff, max_duration):
    """
    Add preictal window rows to a master annotations DataFrame.
    
    For each input row, creates a matching row labeled with a `p` prefix and
    sets its time window to the interval ending `start_cutoff` seconds before
    the original `start_time` and extending back up to `max_duration` seconds.
    If the computed window would start before 0, it is clamped and marked with
    a status value.
    
    Parameters:
        master_df (pd.DataFrame): Master annotations table with at least
            `split`, `edf_path`, `start_time`, `stop_time`, and `label` columns.
        start_cutoff (numeric): Time gap to keep between the preictal window
            end and the original `start_time`.
        max_duration (numeric): Maximum length of each preictal window.
    
    Returns:
        pd.DataFrame: The original rows plus the generated preictal rows, sorted
        by split, EDF path, and start time.
    """
    logger.info(
        f"Adding preictal tags (start_cutoff={start_cutoff}, max_duration={max_duration}) "
        f"to {len(master_df)} rows"
    )
    preictal_rows = []
    status_counts = {0: 0, 1: 0, 2: 0}

    for _, row in master_df.iterrows():
        ictal_start = row["start_time"]
        raw_end = ictal_start - start_cutoff
        raw_start = raw_end - max_duration

        if raw_end <= 0:
            preictal_start = 0
            preictal_end = 0
            status = 2
        elif raw_start <= 0:
            preictal_start = 0
            preictal_end = raw_end
            status = 1
        else:
            preictal_start = raw_start
            preictal_end = raw_end
            status = 0

        status_counts[status] += 1
        if status != 0:
            logger.debug(
                f"Preictal window trimmed (status={status}) for {row['edf_path']} "
                f"at ictal_start={ictal_start}"
            )

        preictal_rows.append({
            **row.to_dict(),
            "label": f"p{row['label']}",
            "start_time": preictal_start,
            "stop_time": preictal_end,
            "status": status,
        })

    if status_counts[1] or status_counts[2]:
        logger.warning(
            f"{status_counts[1]} windows shortened (status=1), "
            f"{status_counts[2]} windows collapsed to [0,0] (status=2)"
        )

    preictal_df = pd.DataFrame(preictal_rows)
    result = pd.concat([master_df, preictal_df], ignore_index=True).sort_values(
        ["split", "edf_path", "start_time"]
    ).reset_index(drop=True)

    logger.info(f"Preictal tags added - master now has {len(result)} rows")
    return result