import argparse
import questionary
import subprocess
import sys
import os
import json
from pathlib import Path

from pipeline import preictal_segment
from util import handle_logs, verify_data
from util.handle_logs import load_config, save_config
from testing.testing_segmentation import test_preictal

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
        help="Save the provided arguments as defaults to app_path.json. This will also allow for all logs to be saved to the specified log file, rather than just the main logs."
    )
   
    args = parser.parse_args()
   
    DATASET_PATH = args.input_path
    LOG_PATH = args.log_path
   
    # Save config if flag is used
    if args.save_config:
        new_config = {
            "input_path": DATASET_PATH,
            "applog": LOG_PATH,   # fixed key name
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

    # Timing parameters with nice defaults
    start_cutoff = float(
        questionary.text(
            "Preictal start cutoff (seconds):", 
            default="300"
        ).ask()
    )
    LOGGER.info(f"Start cutoff: {start_cutoff}s")

    max_duration = float(
        questionary.text(
            "Max preictal window duration (seconds):", 
            default="600"
        ).ask()
    )
    LOGGER.info(f"Max duration: {max_duration}s")

    # Optional postictal + consecutive
    use_post_consec = questionary.confirm(
        "Add postictal and consecutive tags?", 
        default=True
    ).ask()

    if use_post_consec:
        post_length = float(
            questionary.text("Postictal length (seconds):", default="300").ask()
        )
        consec_pre_length = float(
            questionary.text("Preictal length for consecutive detection:", default="600").ask()
        )
    else:
        post_length = consec_pre_length = None

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

    # Add preictal tags
    LOGGER.info("Adding preictal tags...")
    master_df = preictal_segment.add_preictal_tags(
        master_df, start_cutoff, max_duration
    )

    # Optional postictal + consecutive
    if use_post_consec:
        LOGGER.info("Adding postictal and consecutive tags...")
        master_df = preictal_segment.add_postictal_and_consecutive(
            master_df, 
            postictal_length=post_length, 
            preictal_length=consec_pre_length
        )

    # Save result
    master_df.to_csv(new_master_path, index=False)
   
    LOGGER.info("-" * 60)
    LOGGER.info(f"Pipeline complete - output at {new_master_path}")
    LOGGER.info(f"Output saved to: {new_master_path} ({len(master_df)} rows)")
    LOGGER.info("-" * 60)


if __name__ == "__main__":
    main()