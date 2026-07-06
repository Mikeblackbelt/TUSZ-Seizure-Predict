import argparse
import questionary
import subprocess
import sys
from pipeline import preictal_segment
from util import handle_logs, verify_data
from testing import test_preictal

parser = argparse.ArgumentParser()
parser.add_argument("input_path", type=str, help="Path to the input dataset.")
parser.add_argument("--log_path", type=str, default="logs\\app.log", help="Path to the log file. Default is logs\\app.log")
args = parser.parse_args()

DATASET_PATH = args.input_path
LOG_PATH = args.log_path

LOGGER = handle_logs.get_logger("main", LOG_PATH)

def main():
    """
    Run the preictal segmentation pipeline from the command line.
    
    Validates the input dataset, runs the test suite, prompts for tag and timing selections, builds the master file, applies preictal tags, and writes the updated CSV to the chosen output path.
    """
    LOGGER.info("-" * 60)
    LOGGER.info("Starting pipeline")
    LOGGER.info(f"Dataset path: {DATASET_PATH}")
    LOGGER.info(f"Log path:     {LOG_PATH}")
    LOGGER.info("-" * 60)

    verify_data.validate_input(DATASET_PATH)
    LOGGER.info("Running unit tests...")
    result = subprocess.run([
        sys.executable, "-m", "pytest", "testing/",
        "-v",
        "--tb=short",
        "--no-header",
    ])
    if result.returncode != 0:
        LOGGER.error("Unit tests failed. Aborted.")
        return
    LOGGER.info("Unit tests passed")

    LOGGER.info("Scanning for unique tags in dataset...")
    unique_tags = list(preictal_segment.get_unique_tags(DATASET_PATH))  # scan source
    LOGGER.info(f"Available tags: {unique_tags}")

    selected_tags = questionary.checkbox(
        "Select all tags to make new preictal tags",
        choices=unique_tags,
    ).ask()
    LOGGER.info(f"User selected tags: {selected_tags}")

    start_cutoff = float(input("Start cutoff - gap between ictal start and preictal window end (int or float): "))
    LOGGER.info(f"Start cutoff: {start_cutoff}s")

    max_duration = float(input("Max preictal window duration (int or float): "))
    LOGGER.info(f"Max duration: {max_duration}s")

    LOGGER.info("Prompting user for output file path")
    new_master_path = input("Output path for master/preictal file? (Must be an existing .csv) ")
    LOGGER.info(f"Output path: {new_master_path}")

    LOGGER.info("Building master file...")
    master_df = preictal_segment.make_master_file(
        DATASET_PATH,
        output_path=new_master_path,
        allow_tag=selected_tags,
    )
    LOGGER.info("Master file built")

    LOGGER.info("Adding preictal tags...")
    master_df = preictal_segment.add_preictal_tags(
        master_df,
        start_cutoff,
        max_duration,
    )

    master_df.to_csv(new_master_path, index=False)
    LOGGER.info("Preictal tags added and file updated")
    LOGGER.info("-" * 60)
    LOGGER.info(f"Pipeline complete - output at {new_master_path}")
    LOGGER.info("-" * 60)

if __name__ == "__main__":
    main()