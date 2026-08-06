from pipeline.windows import segment_fixed, segment_adaptive
from pathlib import Path
import numpy as np

def load_saved_windows(folder):
    windows = []
    for p in sorted(Path(folder).glob("*.npy")):
        label = p.stem.rsplit("_", 1)[0]
        windows.append({"window": np.load(p), "label": label, "status": None,
                         "edf_path": None, "start_time": None, "stop_time": None})
    return windows

preictal_windows = load_saved_windows("preictal")
interictal_windows = load_saved_windows("interictal")

print(f"{len(preictal_windows)} preictal, {len(interictal_windows)} interictal")

SEG_TIME = 4.0
SFREQ = 256

# non-adaptive, fixed 4s segments, no overlap - for the majority class
interictal_segments = segment_fixed(interictal_windows, SEG_TIME, SFREQ)

# adaptive overlap, computed from the class-size ratio - for the minority class
total_len_inter = sum(w["window"].shape[1] for w in interictal_windows)
preictal_segments = segment_adaptive(
    preictal_windows, SEG_TIME, SFREQ, total_len_inter=total_len_inter
)

print(f"{len(interictal_segments)} interictal segments (fixed)")
print(f"{len(preictal_segments)} preictal segments (adaptive)")
import numpy as np

seg_len = int(SEG_TIME * SFREQ)

# 1. every segment is exactly seg_len samples
assert all(s["segment"].shape[1] == seg_len for s in interictal_segments)
assert all(s["segment"].shape[1] == seg_len for s in preictal_segments)

# 2. not garbage/empty data
assert not any(np.isnan(s["segment"]).any() for s in preictal_segments[:20])
assert not any(np.allclose(s["segment"], 0) for s in preictal_segments[:20])

# 3. the actual point of adaptive segmentation - did it close the class gap?
print(f"before: {len(preictal_windows)} pre windows vs {len(interictal_windows)} inter windows")
print(f"after:  {len(preictal_segments)} pre segs vs {len(interictal_segments)} inter segs")