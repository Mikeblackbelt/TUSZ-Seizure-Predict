"""
Converts existing tuh-preprocess checkpoint arrays (`{session_key}_{stage}.npz`
or `.npy`) into `.parquet`, matching the schema `dataset.py`'s
`_load_parquet_array()` expects: one column per channel, one row per sample
(i.e. the transpose of the pipeline's (n_channels, n_samples) convention).

Why: parquet's columnar compression shrinks these checkpoints substantially
versus raw npz -- useful if you're tight on disk quota. Converting once
up front (rather than doing it lazily on first read) means training runs
don't pay any conversion cost, and you can delete the original npz/npy
after verifying the parquet round-trips correctly.

Usage:
    python convert_checkpoints_to_parquet.py /path/to/checkpoint_dir \
        --stage raw \
        --compression zstd \
        --delete-original

By default this DOES NOT delete the original npz/npy files -- pass
--delete-original once you've spot-checked a few converted files and are
confident the pipeline is reading them correctly. Every conversion is
verified (re-read the parquet, compare shape + values against the
original array) before the original is considered eligible for deletion,
so a failed/corrupt conversion never silently loses data.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def _load_source_array(path: str) -> np.ndarray:
    if path.endswith(".npz"):
        with np.load(path) as data:
            if "eeg" in data:
                arr = data["eeg"]
            elif "data" in data:
                arr = data["data"]
            else:
                arr = data[list(data.keys())[0]]
    else:
        arr = np.load(path)

    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    return arr


def _find_source_files(checkpoint_dir: str, stage: str) -> list[str]:
    suffix_npz = f"_{stage}.npz"
    suffix_npy = f"_{stage}.npy"
    files = []
    for fname in sorted(os.listdir(checkpoint_dir)):
        if fname.endswith(suffix_npz) or fname.endswith(suffix_npy):
            files.append(os.path.join(checkpoint_dir, fname))
    return files


def convert_one(
    src_path: str,
    out_dir: str,
    compression: str,
    channel_names: list[str] | None,
    row_group_size: int | None,
) -> tuple[str, bool, str]:
    """Converts one npz/npy checkpoint to parquet. Returns
    (dest_path, verified_ok, message)."""
    base = os.path.splitext(os.path.basename(src_path))[0]
    dest_path = os.path.join(out_dir, base + ".parquet")

    try:
        arr = _load_source_array(src_path)
    except Exception as e:
        return dest_path, False, f"failed to load source: {e}"

    if arr.ndim != 2:
        return dest_path, False, f"expected 2D (n_channels, n_samples) array, got shape {arr.shape}"

    n_channels, n_samples = arr.shape

    if channel_names is not None and len(channel_names) != n_channels:
        return dest_path, False, (
            f"--channel-names has {len(channel_names)} entries but array has "
            f"{n_channels} channels"
        )
    cols = channel_names if channel_names is not None else [f"ch_{i}" for i in range(n_channels)]

    # Transpose to (n_samples, n_channels) -- rows=samples, cols=channels,
    # matching what dataset.py's _load_parquet_array() expects.
    table = pa.table({col: arr[i] for i, col in enumerate(cols)})

    try:
        pq.write_table(
            table,
            dest_path,
            compression=compression,
            row_group_size=row_group_size,
        )
    except Exception as e:
        return dest_path, False, f"failed to write parquet: {e}"

    # Verify round-trip before caller considers deleting the original.
    try:
        reloaded = pq.read_table(dest_path).to_pandas().to_numpy(dtype=np.float32).T
        if reloaded.shape != arr.shape:
            return dest_path, False, f"shape mismatch after round-trip: {reloaded.shape} vs {arr.shape}"
        if not np.allclose(reloaded, arr, rtol=1e-5, atol=1e-6):
            return dest_path, False, "values mismatch after round-trip"
    except Exception as e:
        return dest_path, False, f"failed to verify round-trip: {e}"

    return dest_path, True, "ok"


def main():
    parser = argparse.ArgumentParser(
        description="Convert tuh-preprocess npz/npy checkpoints to parquet."
    )
    parser.add_argument("checkpoint_dir", help="Directory containing {session_key}_{stage}.npz/.npy files")
    parser.add_argument("--stage", default="raw", choices=["raw", "proc"], help="Which checkpoint stage to convert (default: raw)")
    parser.add_argument("--output-dir", default=None, help="Where to write .parquet files (default: same as checkpoint_dir)")
    parser.add_argument("--compression", default="zstd", choices=["snappy", "gzip", "zstd", "brotli", "none"], help="Parquet compression codec (default: zstd -- good ratio/speed tradeoff for float EEG data)")
    parser.add_argument("--row-group-size", type=int, default=None, help="Rows per parquet row group. Leave unset to let pyarrow choose; set explicitly if you plan to do row-group-level partial reads later.")
    parser.add_argument("--channel-names", default=None, help="Comma-separated channel names to use as column headers, in array-row order (default: ch_0, ch_1, ...)")
    parser.add_argument("--delete-original", action="store_true", help="Delete the source .npz/.npy ONLY after its parquet output is verified to round-trip correctly. Off by default.")
    parser.add_argument("--dry-run", action="store_true", help="List what would be converted without writing anything")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N files (useful for a quick spot-check before running on everything)")
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint_dir):
        print(f"Not a directory: {args.checkpoint_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.output_dir or args.checkpoint_dir
    os.makedirs(out_dir, exist_ok=True)

    channel_names = args.channel_names.split(",") if args.channel_names else None
    compression = None if args.compression == "none" else args.compression

    src_files = _find_source_files(args.checkpoint_dir, args.stage)
    if args.limit is not None:
        src_files = src_files[: args.limit]

    if not src_files:
        print(f"No _{args.stage}.npz/.npy files found in {args.checkpoint_dir}")
        return

    print(f"Found {len(src_files)} '{args.stage}' checkpoint file(s) to convert.")
    if args.dry_run:
        for f in src_files:
            print(f"  would convert: {f}")
        return

    ok_count = 0
    fail_count = 0
    deleted_count = 0
    total_src_bytes = 0
    total_dest_bytes = 0

    for src_path in tqdm(src_files, desc=f"Converting ({args.stage})"):
        dest_path, verified, msg = convert_one(
            src_path, out_dir, compression, channel_names, args.row_group_size
        )

        if not verified:
            fail_count += 1
            tqdm.write(f"[FAILED] {src_path} -> {msg}")
            continue

        ok_count += 1
        total_src_bytes += os.path.getsize(src_path)
        total_dest_bytes += os.path.getsize(dest_path)

        if args.delete_original:
            try:
                os.remove(src_path)
                deleted_count += 1
            except Exception as e:
                tqdm.write(f"[WARN] converted but failed to delete original {src_path}: {e}")

    print()
    print(f"Converted: {ok_count} | Failed: {fail_count}")
    if args.delete_original:
        print(f"Deleted originals: {deleted_count}")
    if total_src_bytes > 0:
        reduction_pct = 100 * (1 - total_dest_bytes / total_src_bytes)
        print(
            f"Size: {total_src_bytes / 1e9:.2f} GB -> {total_dest_bytes / 1e9:.2f} GB "
            f"({reduction_pct:.1f}% reduction)"
        )
    if fail_count > 0:
        print(f"\n{fail_count} file(s) failed conversion/verification -- originals for those were NOT deleted.")


if __name__ == "__main__":
    main()