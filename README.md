# Tuh-Preprocess

Preprocessing pipeline for the [TUSZ (Temple University Seizure) corpus](https://isip.piconepress.com/projects/tuh_eeg/), built to prepare EEG annotation data for a seizure prediction model. The pipeline scans a TUSZ dataset directory, builds a master CSV of seizure events, and generates corresponding **preictal** (pre-seizure) windows for each event so a downstream model can learn to predict an incoming seizure before it starts.

## What this does

1. Walks a TUSZ dataset directory and reads every `.csv` annotation file (each row is a labeled time segment (e.g. `fnsz`, `gnsz`, `bckg`) tied to a `.edf` recording).
2. Filters those rows down to a set of seizure tags you select.
3. Builds a single **master CSV** combining all selected events across the dataset, tagged with which TUSZ split (`train` / `dev` / `eval`) each row belongs to.
4. For every event row, generates a matching **preictal row** - a window of time *before* the seizure starts - using a configurable cutoff and duration.
5. Writes the combined result (ictal + preictal rows) back out as a single CSV, ready for downstream feature extraction / model training.

## Requirements

- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Preictal-Ictal Segmentation
```bash
python segment.py <path-to-tusz-dataset> [--log_path logs\app.log]
```

Running `segment.py` walks you through the preictal-ictal segmentation process interactively:

1. **Data validation** - confirms the input path exists, lists its contents, and asks you to visually confirm the data looks right before continuing. 
2. **Unit tests** - runs the full test suite (`testing/`) before touching real data. If any test fails, the pipeline aborts.
3. **Tag selection** - scans the dataset for every unique seizure label present and lets you pick (via checkbox) which ones to include.
4. **Preictal window configuration** - prompts for `start_cutoff` and `max_duration` (see below).
5. **Output path** - prompts for where to write the resulting CSV.
6. **Master file generation** - builds the master CSV of selected events.
7. **Preictal tagging** - appends a preictal row for every event row and writes the final combined CSV.

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
- Tags each row with its `split` (via `get_split`) and a `status` of `-1` (not applicable - status only carries meaning for preictal rows, see below).
- Skips files with no matching `.edf`, no rows in `allow_tag`, or that fail to parse, logging a count of each at the end of the scan.

Writes the combined result to `output_path` and returns it as a DataFrame. Returns `None` (and logs a warning) if no records were found.

**Output columns:** `edf_path`, `csv_path`, `split`, `channel`, `start_time`, `stop_time`, `label`, `confidence`, `status`

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

The returned DataFrame contains both the original rows and the new preictal rows, sorted by `split`, then `edf_path`, then `start_time`.

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

Shared fixtures (`write_csv`, `write_edf`, `dataset_dir`) live in `testing/helpers.py`.

## Project structure

```
.
├── segment.py                  # CLI entry point
├── pipeline/
│   └── preictal_segment.py     # core scan / master-file / preictal logic
├── util/
│   ├── handle_logs.py          # shared logger factory
│   └── verify_data.py          # input path validation
├── testing/
│   ├── helpers.py              # shared pytest fixtures
│   ├── test_logging.py
│   ├── test_tags.py
│   ├── test_split.py
│   ├── test_masterfile.py
│   └── test_preictal.py
└── requirements.txt
```