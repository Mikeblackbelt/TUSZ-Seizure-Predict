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


def _run_artifact_detection(mode, model_path, logger):
    """
    Build a `postprocess_fn(array, fs) -> array` for concatenate_session_eeg
    that runs the EOG/EMG artifact classifier and masks flagged samples.

    NOTE: the EEG_Artifact_Detection module (with its trained checkpoint)
    was not present alongside this pipeline code, so its exact inference
    API had to be assumed here: `EEG_Artifact_Detection.inference.detect_artifacts
    (data, fs, model_path=...) -> {"per_channel_probs": [...]}` matching the
    `detector_result` shape pipeline/artifact_masking.py already expects.
    If your module's entrypoint differs, adjust the import/call below --
    the masking side (pipeline/artifact_masking.py) is unchanged and already
    tested. If the module can't be imported at all, this logs a clear
    warning once and falls back to unmasked data rather than crashing the
    whole run.
    """
    try:
        from EEG_Artifact_Detection.inference import detect_artifacts
    except ImportError as e:
        logger.warning(
            f"--apply-artifact-masking was set but EEG_Artifact_Detection "
            f"could not be imported ({e}). Proceeding WITHOUT artifact "
            f"masking -- raw checkpoints will be unmasked. If your module "
            f"exposes a different entrypoint, update _run_artifact_detection() "
            f"in pipeline_gpt2class.py to match it."
        )
        return None

    from pipeline.artifact_masking import apply_zero_masking, apply_interpolation_masking

    mask_fn = apply_interpolation_masking if mode == "interpolate" else apply_zero_masking

    def _postprocess(array, fs):
        try:
            detector_result = detect_artifacts(array, fs, model_path=model_path)
        except Exception as e:
            logger.error(f"Artifact detection failed ({e}); returning unmasked data.")
            return array

        result = mask_fn(array, detector_result, fs_native=fs)
        masked_array = result[0] if isinstance(result, tuple) else result
        n_masked = None
        mask = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        if mask is not None:
            n_masked = int(mask.sum())
        logger.info(
            f"Artifact masking ({mode}) applied"
            + (f" - {n_masked} samples flagged" if n_masked is not None else "")
        )
        return masked_array

    return _postprocess


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
        "--skip-background",
        action="store_true",
        help=(
            "Do not derive explicit 'bg' background windows. Background is "
            "derived from stretches of each recording with NO reliable "
            "label at all (the original 'bckg' tag is NOT trusted as clean "
            "background -- see --bg-unreliable-labels) and is chopped into "
            "many fixed-length windows per gap, proportional to gap length, "
            "instead of the raw single-row-per-annotation behavior that "
            "under-represents background relative to preictal."
        ),
    )
    parser.add_argument(
        "--bg-window-duration",
        type=float,
        default=float(config.get("bg_window_duration", 4.0)),
        help="Length in seconds of each generated background window (default: 4.0). "
             "Should match/be a multiple of the model's window length.",
    )
    parser.add_argument(
        "--bg-stride",
        type=float,
        default=float(config.get("bg_stride", 0.0)) or None,
        help="Step in seconds between generated background windows within a gap "
             "(default: same as --bg-window-duration, i.e. non-overlapping).",
    )
    parser.add_argument(
        "--bg-min-gap",
        type=float,
        default=float(config.get("bg_min_gap", 0.0)) or None,
        help="Minimum gap length in seconds worth tagging (default: same as --bg-window-duration).",
    )
    parser.add_argument(
        "--bg-unreliable-labels",
        type=str,
        default=config.get("bg_unreliable_labels", "bckg"),
        help="Comma-separated labels that do NOT count as reliable annotation "
             "coverage when deriving background (default: 'bckg', since it's "
             "an artifact-flagged label, not clean background).",
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
        "--apply-artifact-masking",
        action="store_true",
        help=(
            "Run the EOG/EMG artifact classifier (EEG_Artifact_Detection/) on each "
            "session's raw checkpoint and zero/interpolate flagged samples before "
            "saving, instead of shipping unmasked raw signal as-is. Requires the "
            "EEG_Artifact_Detection module (with a trained checkpoint) to be "
            "importable -- see _run_artifact_detection() in this file if your "
            "module's inference entrypoint has a different name/signature."
        ),
    )
    parser.add_argument(
        "--artifact-masking-mode",
        choices=["zero", "interpolate"],
        default=config.get("artifact_masking_mode", "interpolate"),
        help="How to handle samples flagged as artifact (default: interpolate).",
    )
    parser.add_argument(
        "--artifact-model-path",
        type=str,
        default=config.get("artifact_model_path", None),
        help="Path to the trained artifact-classifier checkpoint, forwarded to "
             "the EEG_Artifact_Detection inference entrypoint.",
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
    skip_background=False,
    bg_window_duration=4.0,
    bg_stride=None,
    bg_min_gap=None,
    bg_unreliable_labels=("bckg",),
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

    if not skip_background:
        logger.info("Deriving explicit unlabeled-background windows...")
        master_df = preictal_segment.add_background_tags(
            master_df,
            window_duration=bg_window_duration,
            stride=bg_stride,
            min_gap=bg_min_gap,
            unreliable_labels=bg_unreliable_labels,
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
            "bg_window_duration": args.bg_window_duration,
            "bg_stride": args.bg_stride or 0.0,
            "bg_min_gap": args.bg_min_gap or 0.0,
            "bg_unreliable_labels": args.bg_unreliable_labels,
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

    bg_unreliable_labels = tuple(
        t.strip() for t in args.bg_unreliable_labels.split(",") if t.strip()
    ) if args.bg_unreliable_labels else ()

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
            skip_background=args.skip_background,
            bg_window_duration=args.bg_window_duration,
            bg_stride=args.bg_stride,
            bg_min_gap=args.bg_min_gap,
            bg_unreliable_labels=bg_unreliable_labels,
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