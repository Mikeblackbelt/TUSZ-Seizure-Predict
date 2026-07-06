import argparse
import questionary
import subprocess
import sys
from pipeline import preictal_segment
from util import handle_logs, verify_data

parser = argparse.ArgumentParser()
parser.add_argument("input_path", type=str, help="Path to the input dataset.")
parser.add_argument("--log_path", type=str, default="logs\\app.log", help="Path to the log file.")
args = parser.parse_args()

DATASET_PATH = args.input_path
LOG_PATH = args.log_path

LOGGER = handle_logs.get_logger("main", LOG_PATH)


def main():
    LOGGER.info("-" * 60)
    LOGGER.info("Starting TUH EEG Master File Pipeline (Preictal + Postictal + Consecutive)")
    LOGGER.info(f"Dataset path: {DATASET_PATH}")
    LOGGER.info("-" * 60)

    verify_data.validate_input(DATASET_PATH)

    LOGGER.info("Running unit tests...")
    result = subprocess.run([
        sys.executable, "-m", "pytest", "testing/", "-v", "--tb=short", "--no-header"
    ])
    if result.returncode != 0:
        LOGGER.error("Unit tests failed. Aborted.")
        return
    LOGGER.info("Unit tests passed")

    unique_tags = list(preictal_segment.get_unique_tags(DATASET_PATH))
    selected_tags = questionary.checkbox(
        "Select tags to process", choices=unique_tags
    ).ask()

    start_cutoff = float(questionary.text("Preictal start cutoff (seconds):", default="300").ask())
    max_duration = float(questionary.text("Max preictal duration (seconds):", default="600").ask())

    use_post_consec = questionary.confirm("Add postictal and consecutive tags?", default=True).ask()
    
    if use_post_consec:
        post_length = float(questionary.text("Postictal length (seconds):", default="300").ask())
        consec_pre_length = float(questionary.text("Preictal length for consecutive detection:", default="600").ask())
    else:
        post_length = consec_pre_length = None

    # Output
    output_path = questionary.text("Output master file path:", default="master_full.csv").ask()

    LOGGER.info("Building master file...")
    master_df = preictal_segment.make_master_file(
        DATASET_PATH, output_path=output_path, allow_tag=selected_tags
    )

    LOGGER.info("Adding preictal tags...")
    master_df = preictal_segment.add_preictal_tags(master_df, start_cutoff, max_duration)

    if use_post_consec:
        LOGGER.info("Adding postictal and consecutive tags...")
        master_df = preictal_segment.add_postictal_and_consecutive(
            master_df, postictal_length=post_length, preictal_length=consec_pre_length
        )

    master_df.to_csv(output_path, index=False)
    
    LOGGER.info("-" * 60)
    LOGGER.info(f"Pipeline completed successfully!")
    LOGGER.info(f"Output saved to: {output_path} ({len(master_df)} rows)")
    LOGGER.info("-" * 60)


if __name__ == "__main__":
    main()