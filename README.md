# Tuh-Preprocess

Preprocessing pipeline for the [TUSZ (Temple University Seizure) corpus](https://isip.piconepress.com/projects/tuh_eeg/), built to prepare EEG annotation data for a seizure prediction model. The pipeline scans a TUSZ dataset directory, builds a master CSV of seizure events, and generates **preictal**, **postictal**, and **consecutive** windows around each event so a downstream model can learn the full temporal structure surrounding a seizure, not just the moment before onset.

## What this does

1. Walks a TUSZ dataset directory and reads every `.csv` annotation file (each row is a labeled time segment (e.g. `fnsz`, `gnsz`, `bckg`) tied to a `.edf` recording and a specific `channel`).
2. Filters those rows down to a set of seizure tags you select.
3. Builds a single **master CSV** combining all selected events across the dataset, tagged with which TUSZ split (`train` / `dev` / `eval`) each row belongs to.
4. For every event row, generates a matching **preictal row** (`p{label}`) - the window of time *before* the seizure starts.
5. For every event row, generates either:
   - a **postictal row** (`q{label}`) - the window of time *after* the seizure ends, if the gap to the next seizure on the same channel is large enough, or
   - a **consecutive row** (`c{label1}{label2}`, or `c{label}2` if both seizures share a label) - spanning the gap to the next seizure, if that gap is too small to fit both a postictal and preictal window.
6. Resolves any remaining overlaps between generated windows sharing identical time boundaries on the same channel, keeping only the highest-priority row.
7. Writes the combined result (ictal + preictal + postictal/consecutive rows) back out as a single CSV, ready for downstream feature extraction / model training.

## Requirements

- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Full Segmentation Pipeline
```bash
python segment.py <path-to-tusz-dataset> [--log_path logs\app.log]
```

Running `segment.py` walks you through the full segmentation process interactively:

1. **Data validation** - confirms the input path exists, lists its contents, and asks you to visually confirm the data looks right before continuing.
2. **Unit tests** - runs the full test suite (`testing/`) before touching real data. If any test fails, the pipeline aborts.
3. **Tag selection** - scans the dataset for every unique seizure label present and lets you pick (via checkbox) which ones to include.
4. **Window configuration** - prompts for preictal (`start_cutoff`, `max_duration`), and postictal/consecutive (`postictal_length`, `preictal_length`) parameters (see below).
5. **Output path** - prompts for where to write the resulting CSV.
6. **Master file generation** - builds the master CSV of selected events.
7. **Preictal tagging** - appends a preictal row for every event row.
8. **Postictal and consecutive tagging** - appends either a postictal row or a consecutive row for every event row, depending on the gap to the next seizure on the same channel.
9. **Overlap resolution** - drops lower-priority rows that share identical time boundaries with a higher-priority row on the same channel, then writes the final combined CSV.

### Arguments

| Argument | Description | Default |
|---|---|---|
| `input_path` (positional) | Path to the root of the TUSZ dataset to process | - |
| `--log_path` | Path to write the run log to | `logs\app.log` |

## Pipeline internals

All core logic lives in `pipeline/preictal_segment.py`.

### `get_unique_tags(dataset_path)`

Recursively scans `dataset_path` for `.csv` annotation files and returns the set of every unique value in the `label` column across the whole dataset. Used to populate the tag-selection prompt and as the default filter (every tag) if none is explicitly provided.

### `get_split(path)`

Infers which TUSZ split (`train`, `dev`, or `eval`) a file belongs to by checking the path's directory components (case-insensitive), matching TUSZ's native layout (e.g. `.../edf/train/01_tcp_ar/...`). Returns `"unknown"` if none of the three match, and logs a warning when that happens.

### `make_master_file(dataset_path, output_path="master.csv", allow_tag=None)`

Walks `dataset_path`, and for every `.csv` annotation file with a matching `.edf` recording:

- Parses the CSV and filters rows down to `allow_tag` (defaults to every tag found via `get_unique_tags` if not provided).
- Tags each row with its `split` (via `get_split`) and a `status` of `-1` (not applicable - status only carries meaning for generated rows, see below).
- Skips files with no matching `.edf`, no rows in `allow_tag`, or that fail to parse, logging a count of each at the end of the scan.

Writes the combined result to `output_path` and returns it as a DataFrame. Returns `None` (and logs a warning) if no records were found.

**Output columns:** `edf_path`, `csv_path`, `split`, `channel`, `start_time`, `stop_time`, `label`, `confidence`, `status`

`status == -1` always identifies an original TUSZ row, never a generated one.

### `add_preictal_tags(master_df, start_cutoff, max_duration)`

For every row in `master_df`, generates a corresponding preictal row labeled `p{original_label}` (e.g. `fnsz` → `pfnsz`), representing the window of time *before* the seizure that a model should learn to recognize as a warning sign.

The window is computed as:

```
raw_end   = ictal_start - start_cutoff
raw_start = raw_end - max_duration
```

- `start_cutoff` - gap (in seconds) between the seizure's onset and the end of the preictal window. Use this to avoid labeling time immediately adjacent to seizure onset as "preictal," since EEG artifacts there can already resemble ictal activity.
- `max_duration` - the maximum length (in seconds) of the preictal window itself.

Both values are clamped to never go negative, and a `status` column on every preictal row records what (if anything) got trimmed:

| Status | Meaning |
|---|---|
| `-1` | Not a preictal row (original ictal/background row) |
| `0` | Nothing trimmed - full window of `max_duration` seconds applied |
| `1` | `raw_start` was trimmed to `0` - window shortened (still ends at `raw_end`) |
| `2` | `raw_end` was trimmed to `0` - window collapsed entirely to `[0, 0]` (e.g. seizure starts before `start_cutoff` has elapsed) |

The returned DataFrame contains both the original rows and the new preictal rows, sorted by `split`, `edf_path`, then `start_time`.

### `add_postictal_and_consecutive(master_df, postictal_length, preictal_length)`

For every original (non-preictal) row, grouped by `(edf_path, channel)` and walked in `start_time` order, decides between two outcomes based on the gap to the *next* seizure on that same channel:

```
gap = next_seizure.start_time - current_seizure.stop_time
```

- **If `gap < (postictal_length + preictal_length)`**: the two seizures are treated as consecutive - too close together for a full postictal window on the first and a full preictal window on the second to both fit without colliding. A single **consecutive row** is generated instead, labeled `c{label1}{label2}` (or `c{label}2` if both seizures share the same label), spanning from the first seizure's `stop_time` to `next.start_time - preictal_length`. If that computed end would fall at or before the start (i.e. the reserved preictal gap alone doesn't fit), the row is collapsed to zero-length and marked `status=2`; otherwise `status=0`. The loop then advances past both seizures (`i += 2`), since the second seizure's own postictal/consecutive decision is already accounted for by this row.
- **Otherwise**: a standard **postictal row** is generated, labeled `q{label}`, spanning from `stop_time` to `stop_time + postictal_length`, with `status=0`.
- **The last seizure in a channel's timeline** (no next seizure to compare against) always gets a standard postictal row.

After generating these rows, `resolve_overlaps` is called automatically on the combined DataFrame before it's returned, so callers don't need to invoke it separately.

The returned DataFrame contains the original rows, the new preictal rows (if already present), and the new postictal/consecutive rows, sorted by `split`, `edf_path`, `channel`, then `start_time`.

### `resolve_overlaps(df)`

Assigns each row a priority based on its label prefix:

```
consecutive (c)  >  preictal (p)  >  original ictal  >  postictal (q)
```

Rows are sorted by `(edf_path, channel, start_time, priority)` and then deduplicated on exact `(edf_path, channel, start_time, stop_time)` matches, keeping only the highest-priority row in each group. This means `resolve_overlaps` removes rows that share **identical time boundaries** with a higher-priority row on the same channel - it does not clip or trim rows whose time ranges only partially overlap with different start/stop values; such partial overlaps are left in the output as-is.

## Data validation (`util/verify_data.py`)

Before the pipeline touches any files, `validate_input(data_path)`:

1. Confirms `data_path` exists (`verify_data_path`).
2. Confirms it's a non-empty directory (`verify_data_integrity`).
3. Lists every file in the directory for a quick visual sanity check (`list_files_glob`).
4. Asks for interactive `y`/`n` confirmation before the run proceeds.

## Logging (`util/handle_logs.py`)

`get_logger(name, log_file=None, level=logging.DEBUG)` returns a standard `logging.Logger` that writes to both stdout and (if `log_file` is given) a file handler, auto-creating any missing parent directories. Loggers are keyed by `name` and re-fetching the same name returns the same configured logger rather than duplicating handlers.

## Testing

The full suite lives in `testing/` and runs automatically as the first step of `segment.py`. To run it manually:

```bash
python -m pytest testing/ -v
```

| File | Covers |
|---|---|
| `test_logging.py` | Logger creation and file output |
| `test_tags.py` | `get_unique_tags` - single/multiple files, nested directories, malformed CSVs, non-CSV files |
| `test_split.py` | `get_split` - train/dev/eval detection, case-insensitivity, unknown paths |
| `test_masterfile.py` | `make_master_file` - column output, tag filtering, missing `.edf` handling, empty directories |
| `test_preictal.py` | `add_preictal_tags` - window math, all three status codes, sort order, original rows left untouched |
| `test_postictal_consecutive.py` | `add_postictal_and_consecutive` - postictal vs. consecutive branching, same-label vs. different-label consecutive naming, last-seizure-in-channel handling |
| `test_resolve_overlaps.py` | `resolve_overlaps` - priority ordering, exact-boundary deduplication, partial overlaps left untouched |

Shared fixtures (`write_csv`, `write_edf`, `dataset_dir`) live in `testing/helpers.py`.

## Project structure

```
.
├── segment.py                  # CLI entry point
├── pipeline/
│   └── preictal_segment.py     # core scan / master-file / preictal / postictal / overlap logic
├── util/
│   ├── handle_logs.py          # shared logger factory
│   └── verify_data.py          # input path validation
├── testing/
│   ├── helpers.py              # shared pytest fixtures
│   ├── test_logging.py
│   ├── test_tags.py
│   ├── test_split.py
│   ├── test_masterfile.py
│   ├── test_preictal.py
│   ├── test_postictal_consecutive.py
│   └── test_resolve_overlaps.py
└── requirements.txt
```