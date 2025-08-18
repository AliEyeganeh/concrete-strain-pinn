# fix_pit_headers.py
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def is_numeric(series):
    return np.issubdtype(series.dtype, np.number)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run folder (e.g., out_ecc_scc_mcdo)")
    args = ap.parse_args()
    run = Path(args.run)

    files = sorted(run.glob("pit_*.csv"))
    if not files:
        raise SystemExit(f"No pit_*.csv files found in {run}")

    fixed = 0
    for f in files:
        df = pd.read_csv(f)
        # already good?
        if "u_temp" in df.columns and "u_iso" in df.columns:
            continue

        # find numeric columns
        num_cols = [c for c in df.columns if is_numeric(df[c])]
        if not num_cols:
            print(f"[skip] {f.name}: no numeric columns")
            continue

        # ensure there are two columns (fill iso with NaN if needed)
        if len(num_cols) == 1:
            df["__tmp_iso__"] = np.nan
            num_cols = [num_cols[0], "__tmp_iso__"]

        # rename the first two numeric columns
        col_temp, col_iso = num_cols[0], num_cols[1]
        rename_map = {col_temp: "u_temp", col_iso: "u_iso"}
        df = df.rename(columns=rename_map)
        # keep only needed columns (optional, but avoids surprises)
        keep = ["u_temp", "u_iso"]
        for c in df.columns:
            if c not in keep:
                # keep extra columns if you want; I’ll drop to be strict:
                pass
        df[keep].to_csv(f, index=False)
        fixed += 1
        print(f"[✓] {f.name}: set headers to u_temp, u_iso")

    print(f"[✓] Completed. Updated {fixed} file(s).")

if __name__ == "__main__":
    main()
