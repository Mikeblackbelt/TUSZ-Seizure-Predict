import pandas as pd

def discard_invalid_preictal(csv_path, output_path, sph, sop):
    """Exclude non preictal and consecutive seizure code from the segmented csv files    
    """
    threshold = sph + sop
    df = pd.read_csv(csv_path).sort_values(
        ["edf_path", "channel", "start_time"]
    ).reset_index(drop=True)

    keep = []
    for _, g in df.groupby(["edf_path", "channel"], sort=False):
        g = g.reset_index(drop=True)
        label = g["label"].astype(str)
        is_p = label.str.startswith("p")
        is_c = label.str.startswith("c")
        is_ictal = ~(is_p | is_c | label.str.startswith("q"))

        prev_ictal_stop = None
        for i, row in g.iterrows():
            if is_c[i]:
                keep.append(row)
            elif is_p[i]:
                nxt = g.loc[i + 1:][is_ictal[i + 1:]]
                if nxt.empty:
                    continue
                j_start = nxt.iloc[0]["start_time"]
                gap = j_start - prev_ictal_stop if prev_ictal_stop is not None else float("inf")
                if gap > threshold:
                    keep.append(row)
            elif is_ictal[i]:
                prev_ictal_stop = row["stop_time"]

    result = pd.DataFrame(keep)
    result.to_csv(output_path, index=False)
    return result