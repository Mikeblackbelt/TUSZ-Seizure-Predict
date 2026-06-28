import os
import csv
import pandas as pd
from util import handle_logs

logger = handle_logs.get_logger("make_master_file", "logs/app.log")

def get_unique_tags(dataset_path):
    tags = set()

    for root, dirs, files in os.walk(dataset_path):
        for csv_file in [f for f in files if f.endswith(".csv")]:
            csv_path = os.path.join(root, csv_file)
            try:
                df = pd.read_csv(csv_path, comment="#")
                df.columns = df.columns.str.strip()
                tags.update(df["label"].unique())
            except Exception as e:
                logger.error(f"Failed to parse {csv_path}: {e}")

    return tags

def make_master_file(dataset_path, output_path="master.csv", allow_tag=None):
    """
    Walks the TUSZ dataset directory, reads all .csv annotation files,
    and writes a single master CSV with seizure info + source file paths.
    """
    records = []

    if allow_tag is None:
        allow_tag = get_unique_tags(dataset_path)

    for root, dirs, files in os.walk(dataset_path):
        csv_files = [f for f in files if f.endswith(".csv")]
        
        for csv_file in csv_files:
            csv_path = os.path.join(root, csv_file)
            edf_path = os.path.join(root, csv_file.replace(".csv", ".edf"))

            if not os.path.exists(edf_path):
                logger.warning(f"No matching .edf for {csv_path}, skipping")
                continue

            try:
                df = pd.read_csv(csv_path, comment="#")  # TUSZ csvs have header comments
                df.columns = df.columns.str.strip()

                # Only keep seizure rows (drop background)
                seizures = df[df["label"] in allow_tag].copy()

                if seizures.empty:
                    continue

                seizures["edf_path"] = edf_path
                seizures["csv_path"] = csv_path
                records.append(seizures)

            except Exception as e:
                logger.error(f"Failed to parse {csv_path}: {e}")

    if not records:
        logger.warning("No seizure records found.")
        return None

    master = pd.concat(records, ignore_index=True)

    # Reorder columns nicely
    cols = ["edf_path", "csv_path", "channel", "start_time", "stop_time", "label", "confidence"]
    cols = [c for c in cols if c in master.columns]  # only include cols that exist
    master = master[cols]

    master.to_csv(output_path, index=False)
    logger.info(f"Master file written to {output_path} with {len(master)} seizure segments")
    return master