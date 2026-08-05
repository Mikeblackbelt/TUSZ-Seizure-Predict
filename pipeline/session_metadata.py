from pipeline.session_index import index_sessions
from util import handle_logs

logger = handle_logs.get_logger("session_metadata", "applog")

def _parse_duration(lines, csv_bi_path):
    """
    Extract the recording duration from a .csv_bi file's header line
    """
    if len(lines) < 3:
        logger.warning(f"{csv_bi_path}: file too short to contain a duration header")
        return None

    duration_line = lines[2]
    if "=" not in duration_line:
        logger.warning(f"{csv_bi_path}: unexpected duration header format: {duration_line!r}")
        return None

    value_str = duration_line.split("=")[-1].strip()
    number_str = value_str.split()[0] if value_str else ""

    try:
        return float(number_str)
    except ValueError:
        logger.warning(f"{csv_bi_path}: could not parse duration from {duration_line!r}")
        return None


def extract_session_metadata(session):
    """
    Parse a session's .csv_bi files (in order) and compute session-cumulative
    seizure start/end times.

    Each .csv_bi file's own seizure timestamps are relative to that single
    recording. For sessions with more than one recording, this shifts each
    recording's seizure times by the summed duration of every recording
    before it, so all seizures end up on one continuous session-relative
    timeline.

    Parameters:
        session (dict): One session record from index_sessions(), must
            contain "csv_bi_paths" (list[str], recording order matters).

    Returns:
        dict: {
            "recordings": list[{"path": str, "duration": float}],
            "seizures": list[{
                "recording_path": str,
                "start_time": float,          # raw, relative to its own recording
                "end_time": float,             # raw, relative to its own recording
                "cumulative_start_time": float,  # session-relative
                "cumulative_end_time": float,    # session-relative
            }],
            "total_duration": float,  # sum of all recording durations
        }
    """
    csv_bi_paths = session.get("csv_bi_paths", [])
    recordings = []
    seizures = []
    cumulative_time = 0.0
    skipped = 0

    for csv_bi_path in csv_bi_paths:
        try:
            with open(csv_bi_path) as f:
                lines = f.read().split("\n")
        except OSError as e:
            logger.error(f"Could not read {csv_bi_path}: {e}")
            skipped += 1
            continue

        duration = _parse_duration(lines, csv_bi_path)
        if duration is None:
            skipped += 1
            continue

        # trusts order session["csv_bi_paths"] arrives in
        recording_offset = cumulative_time
        cumulative_time += duration
        recordings.append({"path": csv_bi_path, "duration": duration})

        for row in lines[6:]:
            row = row.strip()
            if not row or row.startswith("#"):
                continue

            fields = row.split(",")
            if len(fields) < 4:
                continue

            label = fields[3].strip()
            if label != "seiz": 
                continue

            try:
                start = float(fields[1])
                end = float(fields[2])
            except ValueError:
                logger.warning(f"{csv_bi_path}: could not parse seizure row: {row!r}")
                continue

            seizures.append({
                "recording_path": csv_bi_path,
                "start_time": start,
                "end_time": end,
                "cumulative_start_time": start + recording_offset,
                "cumulative_end_time": end + recording_offset,
            })

    if skipped:
        logger.warning(f"Skipped {skipped}/{len(csv_bi_paths)} .csv_bi files for this session")

    logger.info(
        f"Extracted {len(seizures)} seizures across {len(recordings)} recordings "
        f"(total duration: {cumulative_time:.2f}s)"
    )

    return {
        "recordings": recordings,
        "seizures": seizures,
        "total_duration": cumulative_time,
    }

#test

from pipeline.session_metadata import extract_session_metadata

def main():
    sessions = index_sessions("dev")
    print(f"{len(sessions)} sessions found")

    for session_key, session in sessions.items():
        metadata = extract_session_metadata(session)
        print(metadata)


if __name__ == "__main__":
    main()
