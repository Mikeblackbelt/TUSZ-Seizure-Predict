# Tuh-Preprocess

EEG annotation segmentation and signal processing pipeline for the Temple University Seizure (TUSZ) corpus.

`Tuh-Preprocess` converts raw TUSZ EDF files and CSV annotation files into standardized multi-channel session checkpoints (`.parquet` / `.npz`) and a structured master dataset (`master_full.csv`). Outputs interface directly with `TUSZ-Conformer-Prediction`.

---

## System Architecture

```
TUSZ Dataset Directory (EDFs + CSVs)
   │
   ├── 1. Annotation Engine (pipeline/preictal_segment.py)
   │     ├── Scans TUSZ annotation CSVs & filters allowed tags
   │     ├── Generates preictal windows (p*) with SPH / SOP buffers
   │     ├── Generates postictal (q*) and consecutive (c*) windows
   │     ├── Derives explicit background (bg) windows from unannotated gaps
   │     └── Resolves overlaps & tracks window status (-1, 0, 1, 2)
   │
   └── 2. Signal Engine (pipeline_gpt2class.py)
         ├── Parallel multi-worker session processing (--num-workers)
         ├── Concatenates multi-file EDF sessions (t000, t001, ...)
         ├── Resamples signal to 256 Hz & applies bandpass/notch filtering
         ├── Creates 19-channel 10-20 bipolar montages (proc stage)
         ├── Applies optional EOG/EMG artifact detection and masking
         └── Saves compressed .parquet checkpoints & _offsets.json files
```

---

## Pipeline Capabilities

### Annotation Engine (`pipeline/preictal_segment.py`)

* **Window Generation**: Derives **Preictal** (`p*`), **Postictal** (`q*`), **Consecutive** (`c*`), **Exclusion** (`x*`), and **Background** (`bg`) intervals.
* **Buffer Enforcement**: Applies Seizure Prediction Horizon (SPH) and Seizure Occurrence Period (SOP) safety bounds.
* **Status Column Tracking**:
  * `-1`: Original TUSZ annotation rows.
  * `1`: Valid generated window rows.
  * `2`: Collapsed gap window rows.
* **Explicit Background Derivation**: Background windows (`bg`) are generated from unannotated stretches of EEG rather than trusting artifact-flagged `bckg` tags.

### Signal Engine (`pipeline_gpt2class.py`, `raw_eeg_extraction.py`)

* **Multi-File Session Concatenation**: Combines multiple EDF files in a single recording session into one contiguous matrix.
* **Sample Offset Tracking**: Writes `{session_key}_offsets.json` recording the exact start sample for each EDF file within the concatenated session.
* **Bipolar Montages**: Converts raw channels into 19 standard TCP 10-20 bipolar channels (`proc` stage).
* **Artifact Masking**: Detects EOG/EMG noise using a trained SincNet/CNN classifier and applies interpolation or zero masking.
* **Parallel Execution**: Processes multiple EEG sessions in parallel using `--num-workers`.

---

## Installation & Dependencies

Requires Python 3.10+:

```bash
pip install -r requirements.txt
pip install -r EEG_Artifact_Detection/requirements.txt   # optional: for artifact classifier training
```

---

## Running the Pipeline (`pipeline_gpt2class.py`)

Run full master CSV generation and parallel session processing:

```bash
python pipeline_gpt2class.py /path/to/tusz/dataset \
  --master-output master_full.csv \
  --checkpoint-dir ./checkpoints \
  --sph 600 \
  --sop 300 \
  --postictal-time 1800 \
  --process-sessions \
  --create-montage \
  --num-workers 8
```

### CLI Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `input_path` | *(Required)* | Root directory of the TUSZ dataset containing EDF and CSV files. |
| `--master-output` | `master_full.csv` | Path for the generated master CSV. |
| `--checkpoint-dir` | `checkpoints` | Directory for `.parquet` checkpoints and offset files. |
| `--sph` | `120.0` | Seizure Prediction Horizon (preictal window length) in seconds. |
| `--sop` | `420.0` | Seizure Occurrence Period (pre-seizure buffer) in seconds. |
| `--postictal-time` | `1800.0` | Recovery window duration in seconds. |
| `--skip-background` | `False` | Disables background window derivation. |
| `--bg-window-duration` | `4.0` | Duration in seconds for derived background windows. |
| `--process-sessions` | `False` | Executes signal processing and checkpoint generation. |
| `--create-montage` | `False` | Generates 19-channel bipolar montage (`proc` stage) checkpoints. |
| `--num-workers` | `4` | Worker threads for parallel session processing. |
| `--apply-artifact-masking` | `False` | Enables EOG/EMG artifact detection and masking. |

---

## Master CSV Schema

`master_full.csv` contains all original and generated window records:

| Column | Type | Description |
| :--- | :--- | :--- |
| `edf_path` | `str` | Absolute path to the original `.edf` file. |
| `csv_path` | `str` | Absolute path to the original annotation `.csv` file. |
| `split` | `str` | Dataset split (`train`, `dev`, or `eval`). |
| `channel` | `str` | Specific channel (e.g. `FP1-F7`) or `TERM` (recording-wide). |
| `start_time` | `float` | Window start time in seconds relative to EDF start. |
| `stop_time` | `float` | Window stop time in seconds relative to EDF start. |
| `label` | `str` | Label tag (e.g. `fnsz`, `pfnsz`, `qfnsz`, `cfnsz`, `bg`). |
| `confidence` | `float` | Annotation confidence score. |
| `is_valid` | `bool` | `True` for valid full-length windows; `False` for Gate failures / SOP buffers. |
| `status` | `int` | Status code: `-1` = original, `1` = valid generated, `2` = collapsed gap. |

---

## Checkpoint File Layout

For each session, `process_sessions` outputs two files to `--checkpoint-dir`:

1. **`{session_key}_{stage}.parquet`**: Columnar Parquet table (columns = channels, rows = 256 Hz time samples).
2. **`{session_key}_offsets.json`**: JSON array tracking EDF file sample offsets within the session:
   ```json
   [
     {"edf_path": ".../t000.edf", "start_sample": 0},
     {"edf_path": ".../t001.edf", "start_sample": 153600}
   ]
   ```
