import argparse
import io
import os
import shutil
from pathlib import Path

import numpy as np


def convert_checkpoint_file(input_path: str, output_path: str | None = None) -> str | None:
    source_path = Path(input_path)
    if output_path is None:
        output_path = str(source_path.with_suffix('.npz'))

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        if source_path.exists():
            source_path.unlink()
        return str(target_path)

    try:
        array = np.load(source_path, allow_pickle=False)
    except (EOFError, ValueError, OSError) as exc:
        print(f"Skipping unreadable checkpoint {source_path}: {exc}")
        return None

    try:
        if array.dtype.kind == 'f':
            array = array.astype(np.float32, copy=False)
        archive_bytes = io.BytesIO()
        np.savez_compressed(archive_bytes, data=array)
        target_path.write_bytes(archive_bytes.getvalue())
    except (OSError, ValueError) as exc:
        print(f"Failed to write checkpoint {target_path}: {exc}")
        return None

    if source_path.exists():
        source_path.unlink()
    return str(target_path)


def convert_directory(checkpoint_dir: str, output_dir: str | None = None) -> list[str]:
    converted = []
    source_dir = Path(checkpoint_dir)
    target_dir = Path(output_dir) if output_dir else source_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(source_dir.glob('*.npy')):
        if path.name.endswith('_offsets.npy'):
            continue
        if path.name.endswith('_raw.npy') or path.name.endswith('_proc.npy'):
            out_path = target_dir / f"{path.stem}.npz"
            result = convert_checkpoint_file(str(path), str(out_path))
            if result is not None:
                converted.append(result)
                if str(output_dir).lower() == 'none':
                    continue
                if target_dir != source_dir and out_path.exists():
                    try:
                        source_path = source_dir / path.name
                        if source_path.exists():
                            source_path.unlink()
                    except OSError:
                        pass
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert old .npy checkpoint files to compressed .npz checkpoint files.')
    parser.add_argument('checkpoint_dir', help='Directory containing checkpoint files to convert')
    parser.add_argument('--output-dir', default=None, help='Optional directory to write converted .npz files to')
    args = parser.parse_args()

    converted = convert_directory(args.checkpoint_dir, args.output_dir)
    print(f'Converted {len(converted)} checkpoint files')
    for item in converted:
        print(item)


if __name__ == '__main__':
    main()
