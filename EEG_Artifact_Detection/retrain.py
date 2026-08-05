"""Build a TUAR-derived training set for EEG_Artifact_Detection's MLPTrainer.

This script delegates TUAR dataset building and optional TUAR training to
module-level helpers.
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tuar_data_builder import build_tuar_dataset
from tuar_trainer import run_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_tuar_dataset_and_train")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tuar-path", required=True, help="Root folder of the TUAR corpus (walked recursively)")
    parser.add_argument("--out-datapath", default="data", help="Where to write train/val/test X.npy/Y.npy (MLPTrainer's --datapath)")
    parser.add_argument("--fallback-fs", type=float, default=256.0, help="Fallback sampling rate for .npy files with no <file>.fs.txt sidecar")
    parser.add_argument("--montage-dir", default=None, help="Directory to look for montage definition .txt files first")
    parser.add_argument("--label-threshold", type=float, default=0.5, help="Fraction of a window a class must cover to be labeled that class")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Fraction of PATIENTS held out for validation")
    parser.add_argument("--test-frac", type=float, default=0.15, help="Fraction of PATIENTS held out for test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-windows", type=int, default=2000, help="Cap on windows per split to keep dataset builds bounded (use -1 for unbounded)")
    parser.add_argument("--min-channel-frac", type=float, default=0.6,
                         help="A channel must appear in at least this fraction of scanned EDF files to be included in the auto-discovered canonical channel set")
    parser.add_argument("--max-channels", type=int, default=19,
                         help="Upper bound on how many canonical channels to use for multi-channel windows")
    parser.add_argument("--channel-scan-limit", type=int, default=None,
                         help="Only scan this many EDF file headers to discover canonical channels (default: scan all files -- header-only reads are cheap, but pass a limit to speed this up on very large corpora)")

    parser.add_argument("--train", action="store_true", help="Also train MLPTrainer on the built dataset")
    parser.add_argument("--model", default="MLP", choices=["MLP", "CNN", "SincNet", "CNN_Multi"])
    parser.add_argument("--pca", action="store_true")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--save-path", default="checkpoints")
    parser.add_argument("--outputpath", default="output")
    parser.add_argument("--log-file", default="train_log.txt")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--tmp-dir", default=None, help="Directory for temp build files (use a drive with space if building a large corpus)")
    args = parser.parse_args()

    build_tuar_dataset(
        args.tuar_path,
        args.out_datapath,
        args.fallback_fs,
        args.montage_dir,
        args.label_threshold,
        args.val_frac,
        args.test_frac,
        args.seed,
        args.max_windows,
        args.tmp_dir,
        min_channel_frac=args.min_channel_frac,
        max_channels=args.max_channels,
        channel_scan_limit=args.channel_scan_limit,
    ) 

    if args.train:
        run_training(args.out_datapath, args)


if __name__ == "__main__":
    main()