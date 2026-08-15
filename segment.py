import argparse
import questionary
import subprocess
import sys
import os
import json
from pathlib import Path
from pipeline.recording_info import get_recording_info
from pipeline import preictal_segment
from util import handle_logs, verify_data
from util.handle_logs import load_config, save_config

CONFIG_FILE = "app_path.json"


def main():
    # Load existing config
    config = load_config()
   
    # Set up argument parser with config defaults
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the input dataset."
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default=config.get("applog", "logs\\app.log"),
        help="Path to the log file. Default is logs\\app.log"
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Save the provided arguments as defaults to app_path.json."
    )
   
    args = parser.parse_args()
   
    DATASET_PATH = args.input_path
    LOG_PATH = args.log_path
   
    # Save config if flag is used
    if args.save_config:
        new_config = {
            "input_path": DATASET_PATH,
            "applog": LOG_PATH,
        }
        save_config(new_config)
   
    LOGGER = handle_logs.get_logger("main", LOG_PATH)
   
    LOGGER.info("-" * 60)
    LOGGER.info("Starting pipeline")
    LOGGER.info(f"Dataset path: {DATASET_PATH}")
    LOGGER.info(f"Log path: {LOG_PATH}")
    LOGGER.info("-" * 60)

    # Input validation
    is_valid, message = verify_data.validate_input(DATASET_PATH)
    if not is_valid:
        LOGGER.error(f"Input validation failed: {message}")
        return

    # Run unit tests
    LOGGER.info("Running unit tests...")
    result = subprocess.run([
        sys.executable, "-m", "pytest", "testing/",
        "-v", "--tb=short", "--no-header",
    ])
    if result.returncode != 0:
        LOGGER.error("Unit tests failed. Aborted.")
        return
    LOGGER.info("Unit tests passed")

    # Tag selection
    LOGGER.info("Scanning for unique tags in dataset...")
    unique_tags = list(preictal_segment.get_unique_tags(DATASET_PATH))
    LOGGER.info(f"Available tags: {unique_tags}")

    selected_tags = questionary.checkbox(
        "Select all tags to make new preictal tags",
        choices=unique_tags,
    ).ask()

    LOGGER.info(f"User selected tags: {selected_tags}")

    # Timing parameters (SOP and SPH)
    sop = float(
        questionary.text(
            "SOP - buffer/safety clearance before seizure onset (seconds):",
            default="120"
        ).ask()
    )
    LOGGER.info(f"SOP buffer: {sop}s")

    sph = float(
        questionary.text(
            "SPH - preictal window length to extract (seconds):",
            default="420"
        ).ask()
    )
    LOGGER.info(f"SPH preictal duration: {sph}s")

    # Optional Exclusion Intervals (Postictal)
    use_exclusions = questionary.confirm(
        "Add exclusion (postictal) intervals?", 
        default=True
    ).ask()

    if use_exclusions:
        post_time = float(
            questionary.text(
                "Postictal exclusion time (seconds):", 
                default="1800"
            ).ask()
        )
        LOGGER.info(f"Postictal exclusion duration: {post_time}s")
    else:
        post_time = None

    # Output path
    new_master_path = questionary.text(
        "Output master file path:", 
        default="master_full.csv"
    ).ask()
    LOGGER.info(f"Output path: {new_master_path}")

    # Build master file
    LOGGER.info("Building master file...")
    master_df = preictal_segment.make_master_file(
        DATASET_PATH,
        output_path=new_master_path,
        allow_tag=selected_tags,
    )
    LOGGER.info("Master file built")

    # Normalize edf_path separators immediately, before anything else reads
    # or persists this column. make_master_file builds paths with the
    # current OS's os.path.join, so a master CSV built on Windows stores
    # backslash paths and one built on Linux stores forward-slash paths -
    # if the pipeline later runs on a different OS than the one that built
    # this CSV, downstream string-equality matching (extract_segments)
    # would silently fail for every row. Forward slashes are accepted by
    # both Windows and Linux file APIs, so normalizing here doesn't break
    # the get_recording_info() reads below on either OS, and every master
    # CSV this script writes from now on is OS-independent by construction.
    master_df["edf_path"] = master_df["edf_path"].astype(str).str.replace("\\", "/", regex=False)

    # Add preictal tags
    LOGGER.info("Adding preictal tags...")
    master_df = preictal_segment.add_preictal_tags(
        master_df, sph=sph, sop=sop, postictal_time=post_time
    )

    # Optional Exclusion Intervals
    if use_exclusions and post_time is not None:
        LOGGER.info("Adding exclusion intervals...")
        master_df = preictal_segment.add_exclusion_intervals(
            master_df=master_df,
            postictal_time=post_time,
        )

    LOGGER.info("Computing recording durations...")
    recording_durations = {}
    for edf_path in master_df["edf_path"].unique():
        info = get_recording_info(edf_path)
        recording_durations[edf_path] = info["n_times"] / info["sfreq"]

    LOGGER.info("Adding interictal tags...")
    master_df = preictal_segment.add_interictal_tags(
        master_df, recording_durations
    )
    # Save result
    master_df.to_csv(new_master_path, index=False)
    
    LOGGER.info("-" * 60)
    LOGGER.info(f"Pipeline complete - output at {new_master_path}")
    LOGGER.info(f"Output saved to: {new_master_path} ({len(master_df)} rows)")
    LOGGER.info("-" * 60)


if __name__ == "__main__":
    main()