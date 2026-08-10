import argparse
import os
import subprocess
import sys

import questionary

from pipeline import preictal_segment
from pipeline.bipolar_montages import create_bipolar_montages
from pipeline.raw_eeg_extraction import concatenate_session_eeg
from pipeline.session_index import index_sessions
from util import handle_logs, verify_data
from util.handle_logs import load_config, save_config


def _checkpoint_exists(checkpoint_dir, session_key, stage):
    for suffix in (".npz", ".npy"):
        if os.path.exists(os.path.join(checkpoint_dir, f"{session_key}_{stage}{suffix}")):
            return True
    return False


def build_arg_parser(config):
    parser = argparse.ArgumentParser(
        description="Build a master annotation file and process EEG sessions into filtered checkpoints."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Root dataset path containing EDF and annotation files.",
    )
    parser.add_argument(
        "--master-output",
        type=str,
        default=config.get("master_output", "master_full.csv"),
        help="Output path for the generated master CSV.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=config.get("checkpoint_dir", "checkpoints"),
        help="Directory to save raw and proc checkpoint files.",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=config.get("applog", "logs\\app.log"),
        help="Path for application logging.",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default=config.get("tags", None),
        help="Comma-separated labels to include in the master file.",
    )
    parser.add_argument(
        "--sph",
        type=float,
        default=float(config.get("sph", 120.0)),
        help="Seizure occurrence period / buffer in seconds.",
    )
    parser.add_argument(
        "--sop",
        type=float,
        default=float(config.get("sop", 420.0)),
        help="Preictal duration in seconds.",
    )
    parser.add_argument(
        "--postictal-time",
        type=float,
        default=float(config.get("postictal_time", 1800.0)),
        help="Postictal exclusion duration in seconds.",
    )
    parser.add_argument(
        "--skip-exclusions",
        action="store_true",
        help="Do not add postictal exclusion intervals.",
    )
    parser.add_argument(
        "--process-sessions",
        action="store_true",
        help="Concatenate, filter, resample, and checkpoint all indexed EEG sessions.",
    )
    parser.add_argument(
        "--create-montage",
        action="store_true",
        help="Generate bipolar montage proc checkpoints from raw checkpoints.",
    )
    parser.add_argument(
        "--bandpass-low",
        type=float,
        default=float(config.get("bandpass_low", 0.5)),
        help="Low cutoff frequency for bandpass filtering.",
    )
    parser.add_argument(
        "--bandpass-high",
        type=float,
        default=float(config.get("bandpass_high", 40.0)),
        help="High cutoff frequency for bandpass filtering.",
    )
    parser.add_argument(
        "--notch-variance-threshold",
        type=float,
        default=float(config.get("notch_variance_threshold", 0.20)),
        help="Variance threshold for adaptive notch frequency detection.",
    )
    parser.add_argument(
        "--notch-power-percentile",
        type=float,
        default=float(config.get("notch_power_percentile", 80.0)),
        help="Power percentile threshold for adaptive notch frequency detection.",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run repository unit tests before execution.",
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Save provided settings as defaults to app_path.json.",
    )
    return parser


def prompt_for_tags(dataset_path, default_tags=None):
    tags = sorted(preictal_segment.get_unique_tags(dataset_path))
    if not tags:
        raise ValueError("No tags found in the dataset.")
    if default_tags:
        return default_tags
    selected = questionary.checkbox(
        "Select tags to include in the master file:",
        choices=tags,
    ).ask()
    if not selected:
        raise ValueError("At least one tag must be selected.")
    return selected


def run_unit_tests(logger):
    logger.info("Running unit tests...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "testing/", "-v", "--tb=short", "--no-header"],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("Unit tests failed.")
    logger.info("Unit tests passed")


def resolve_output_path(output_path, default_filename="master_full.csv"):
    if not output_path:
        return default_filename

    expanded_path = os.path.expanduser(output_path)
    if os.path.isdir(expanded_path):
        return os.path.join(expanded_path, default_filename)

    if expanded_path.endswith((os.sep, "/", "\\")):
        return os.path.join(expanded_path, default_filename)

    return expanded_path


def build_master_file(
    input_path,
    output_path,
    allow_tag,
    sph,
    sop,
    postictal_time,
    skip_exclusions,
    logger,
):
    logger.info("Building master file...")
    master_df = preictal_segment.make_master_file(
        input_path,
        output_path=output_path,
        allow_tag=allow_tag,
    )
    if master_df is None:
        raise RuntimeError("Master file generation failed.")

    logger.info("Adding preictal tags...")
    master_df = preictal_segment.add_preictal_tags(master_df, start_cutoff=sph, max_duration=sop)

    if not skip_exclusions:
        logger.info("Adding postictal and consecutive intervals...")
        master_df = preictal_segment.add_postictal_and_consecutive(
            master_df=master_df,
            postictal_time=postictal_time,
            preictal_duration=sop,
        )

    logger.info("Resolving label overlaps...")
    master_df = preictal_segment.resolve_overlaps(master_df)

    parent_dir = os.path.dirname(output_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    master_df.to_csv(output_path, index=False)
    logger.info(f"Master file saved to {output_path} ({len(master_df)} rows)")
    return master_df


def process_sessions(
    input_path,
    checkpoint_dir,
    create_montage_flag,
    bandpass_low,
    bandpass_high,
    notch_variance_threshold,
    notch_power_percentile,
    logger,
):
    logger.info("Indexing EEG sessions...")
    sessions = index_sessions(input_path)
    logger.info(f"Found {len(sessions)} indexed sessions")

    if not sessions:
        logger.warning("No sessions were found. Skipping EEG processing.")
        return

    for session_key, session in sessions.items():
        logger.info(f"Processing session: {session_key}")

        raw_checkpoint_exists = _checkpoint_exists(checkpoint_dir, session_key, "raw")
        proc_checkpoint_exists = _checkpoint_exists(checkpoint_dir, session_key, "proc")

        if raw_checkpoint_exists and (not create_montage_flag or proc_checkpoint_exists):
            logger.info(f"Skipping {session_key}: existing checkpoints already present")
            continue

        result = concatenate_session_eeg(
            session,
            session_key=session_key,
            output_dir=checkpoint_dir,
            apply_filtering=True,
            notch_variance_threshold=notch_variance_threshold,
            notch_power_percentile=notch_power_percentile,
            bandpass_low=bandpass_low,
            bandpass_high=bandpass_high,
        )
        if result is None:
            logger.warning(f"Session {session_key} had no EDF files. Skipping.")
            continue

        if create_montage_flag:
            montage = create_bipolar_montages(session_key, checkpoint_dir)
            if montage is None:
                logger.warning(f"Bipolar montage creation failed for {session_key}")


def main():
    config = load_config()
    parser = build_arg_parser(config)
    args = parser.parse_args()

    if args.save_config:
        save_config({
            "applog": args.log_path,
            "master_output": args.master_output,
            "checkpoint_dir": args.checkpoint_dir,
            "sph": args.sph,
            "sop": args.sop,
            "postictal_time": args.postictal_time,
            "bandpass_low": args.bandpass_low,
            "bandpass_high": args.bandpass_high,
            "notch_variance_threshold": args.notch_variance_threshold,
            "notch_power_percentile": args.notch_power_percentile,
        })

    resolved_master_output = resolve_output_path(args.master_output)

    logger = handle_logs.get_logger("pipeline_gpt2class", args.log_path)
    logger.info("-" * 60)
    logger.info("Starting pipeline_gpt2class")
    logger.info(f"Dataset path: {args.input_path}")
    logger.info(f"Master output: {resolved_master_output}")
    logger.info(f"Checkpoint dir: {args.checkpoint_dir}")
    logger.info("-" * 60)

    is_valid, message = verify_data.validate_input(args.input_path)
    if not is_valid:
        logger.error(f"Input validation failed: {message}")
        return

    if args.run_tests:
        try:
            run_unit_tests(logger)
        except RuntimeError as exc:
            logger.error(str(exc))
            return

    if args.tags:
        allow_tag = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
        if not allow_tag:
            logger.error("No valid tags were provided via --tags.")
            return
    else:
        try:
            allow_tag = prompt_for_tags(args.input_path)
        except ValueError as exc:
            logger.error(str(exc))
            return

    try:
        build_master_file(
            input_path=args.input_path,
            output_path=resolved_master_output,
            allow_tag=allow_tag,
            sph=args.sph,
            sop=args.sop,
            postictal_time=args.postictal_time,
            skip_exclusions=args.skip_exclusions,
            logger=logger,
        )
    except Exception as exc:
        logger.error(f"Master file pipeline failed: {exc}")
        return

    if args.process_sessions:
        process_sessions(
            input_path=args.input_path,
            checkpoint_dir=args.checkpoint_dir,
            create_montage_flag=args.create_montage,
            bandpass_low=args.bandpass_low,
            bandpass_high=args.bandpass_high,
            notch_variance_threshold=args.notch_variance_threshold,
            notch_power_percentile=args.notch_power_percentile,
            logger=logger,
        )

    logger.info("-" * 60)
    logger.info("Pipeline complete")
    logger.info("-" * 60)


if __name__ == "__main__":
    main()
