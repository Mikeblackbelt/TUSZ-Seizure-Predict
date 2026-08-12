# 17 target EEG channels the pipeline extracts, in both conventions (AR,LE).
# Given edf file on has either present
CHANNELS_TO_INCLUDE = [
    'EEG T6-REF', 'EEG T5-REF', 'EEG T4-REF', 'EEG T3-REF',
    'EEG P4-REF', 'EEG P3-REF', 'EEG O2-REF', 'EEG O1-REF',
    'EEG FP2-REF', 'EEG FP1-REF', 'EEG F8-REF', 'EEG F7-REF',
    'EEG F4-REF', 'EEG F3-REF', 'EEG CZ-REF', 'EEG C4-REF',
    'EEG C3-REF', 'EEG T6-LE', 'EEG T5-LE', 'EEG T4-LE',
    'EEG T3-LE', 'EEG P4-LE', 'EEG P3-LE', 'EEG O2-LE',
    'EEG O1-LE', 'EEG FP2-LE', 'EEG FP1-LE', 'EEG F8-LE',
    'EEG F7-LE', 'EEG F4-LE', 'EEG F3-LE', 'EEG CZ-LE',
    'EEG C4-LE', 'EEG C3-LE',
]

N_TARGET_CHANNELS = 17

# Canonical row order that every "raw" checkpoint must be saved in. This is
# the single source of truth for channel order downstream (e.g.
# bipolar_montages.py indexes rows by this order) - it must NOT be
# redefined anywhere else, or the two definitions can silently drift apart.
CANONICAL_CHANNELS = [
    'FP1', 'F7', 'T3', 'T5', 'O1', 'FP2', 'F8', 'T4', 'T6', 'O2',
    'F3', 'C3', 'P3', 'F4', 'C4', 'P4', 'CZ'
]


def bare_channel_name(ch_name):
    """
    Strip a raw MNE channel label like 'EEG FP1-REF' or 'EEG FP1-LE' down
    to the bare electrode name 'FP1', so it can be matched against
    CANONICAL_CHANNELS regardless of which reference convention (AR/LE)
    the source .edf file used.
    """
    name = ch_name.strip()
    if name.upper().startswith("EEG "):
        name = name[4:]
    for suffix in ("-REF", "-LE"):
        if name.upper().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.strip().upper()


def reorder_to_canonical(data, ch_names):
    """
    Reorder a (n_channels, n_samples) array from whatever channel order
    `ch_names` reports (MNE's `include=` filter preserves each .edf
    file's own header order, not the order requested) into the fixed
    CANONICAL_CHANNELS row order.

    This must be called before concatenating/saving any checkpoint that
    downstream code (e.g. bipolar_montages.py) will index positionally
    by CANONICAL_CHANNELS - otherwise row i is not guaranteed to be the
    same physical electrode across files or sessions.

    Parameters:
        data (np.ndarray): shape (len(ch_names), n_samples).
        ch_names (list[str]): channel names as returned by the source
            (e.g. raw.ch_names), same order as data's rows.

    Returns:
        np.ndarray of shape (len(CANONICAL_CHANNELS), n_samples), row i
        corresponding to CANONICAL_CHANNELS[i].

    Raises:
        ValueError: if ch_names doesn't contain exactly one match for
            every entry in CANONICAL_CHANNELS (missing and/or duplicate
            channels are both treated as fatal, not silently ignored).
    """
    bare_names = [bare_channel_name(ch) for ch in ch_names]

    name_to_row = {}
    duplicates = set()
    for i, name in enumerate(bare_names):
        if name in name_to_row:
            duplicates.add(name)
        else:
            name_to_row[name] = i

    missing = [ch for ch in CANONICAL_CHANNELS if ch not in name_to_row]
    if missing or duplicates:
        raise ValueError(
            "Cannot map recording channels onto CANONICAL_CHANNELS: "
            f"missing={missing}, duplicated={sorted(duplicates)}, "
            f"got channel names={ch_names}"
        )

    row_order = [name_to_row[ch] for ch in CANONICAL_CHANNELS]
    return data[row_order, :]