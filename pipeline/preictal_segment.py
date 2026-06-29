import os
import pandas as pd
from util import handle_logs

logger = handle_logs.get_logger("make_master_file", "logs/app.log")

def get_unique_tags(dataset_path):
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
    Walks the TUSZ dataset directory, reads all .csv annotation files,
    and writes a single master CSV with seizure info + source file paths.
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

    cols = ["edf_path", "csv_path", "channel", "start_time", "stop_time", "label", "confidence"]
    cols = [c for c in cols if c in master.columns]
    master = master[cols]

    master.to_csv(output_path, index=False)
    logger.info(f"Master file written to {output_path} with {len(master)} rows")
    return master


def add_preictal_tags(master_df, preictal_dur):
    """
    For each row in master_df, adds a new row with tag f'p{label}'
    spanning from max(0, start_time - preictal_dur) to start_time.
    """
    logger.info(f"Adding preictal tags with duration {preictal_dur}s to {len(master_df)} rows")
    preictal_rows = []
    clipped_count = 0

    for _, row in master_df.iterrows():
        preictal_start = max(0, row["start_time"] - preictal_dur)
        preictal_end = row["start_time"]

        if preictal_start == 0 and row["start_time"] < preictal_dur:
            logger.debug(f"Preictal window clipped to 0 for {row['edf_path']} at t={row['start_time']}")
            clipped_count += 1

        preictal_rows.append({
            **row.to_dict(),
            "label": f"p{row['label']}",
            "start_time": preictal_start,
            "stop_time": preictal_end,
        })

    if clipped_count:
        logger.warning(f"{clipped_count} preictal windows were clipped to t=0")

    preictal_df = pd.DataFrame(preictal_rows)
    result = pd.concat([master_df, preictal_df], ignore_index=True).sort_values(
        ["edf_path", "start_time"]
    ).reset_index(drop=True)

    logger.info(f"Preictal tags added - master now has {len(result)} rows")
    return result