import pandas as pd
import numpy as np
from pathlib import Path
import math
import argparse

def phi_cdf(z):
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run folder containing test_details_cal.csv (e.g., out_ecc_scc_mcdo)")
    ap.add_argument("--split", default="test", choices=["test","val"], help="Which details file to use (default: test)")
    args = ap.parse_args()

    run = Path(args.run)
    det_path = run / f"{args.split}_details_cal.csv"
    if not det_path.exists():
        raise SystemExit(f"Missing {det_path}")

    det = pd.read_csv(det_path)

    # Expected columns (rename flexibly if needed)
    # y, mu_raw, sd_raw, mu_temp, sd_temp are typical in your pipeline
    col_y   = "y"
    col_mt  = "mu_temp" if "mu_temp" in det.columns else "mu_raw"
    col_st  = "sd_temp" if "sd_temp" in det.columns else "sd_raw"

    if col_y not in det.columns or col_mt not in det.columns or col_st not in det.columns:
        raise SystemExit(f"Required columns not found in {det_path}. Need y, mu_temp/sd_temp (or mu_raw/sd_raw). "
                         f"Found: {list(det.columns)}")

    # Compute PIT for temp-scale (or fallback to raw if temp not present)
    z_temp = (det[col_y].to_numpy(dtype=float) - det[col_mt].to_numpy(dtype=float)) / np.maximum(det[col_st].to_numpy(dtype=float), 1e-12)
    u_temp = phi_cdf(z_temp)

    # If an isotonic PIT already exists in your details, copy it; else fill NaN
    if "pit_iso" in det.columns:
        u_iso = det["pit_iso"].to_numpy(dtype=float)
    elif "u_iso" in det.columns:
        u_iso = det["u_iso"].to_numpy(dtype=float)
    else:
        u_iso = np.full_like(u_temp, np.nan, dtype=float)

    # Write one CSV per specimen named pit_<SPECIMEN>.csv with columns u_temp, u_iso
    if "specimen" not in det.columns:
        raise SystemExit(f"'specimen' column not found in {det_path}. Can't split per specimen.")

    for spec, g in det.groupby("specimen"):
        out = pd.DataFrame({
            "u_temp": u_temp[g.index],
            "u_iso":  u_iso[g.index]
        })
        out_path = run / f"pit_{spec}.csv"
        out.to_csv(out_path, index=False)

    print(f"[✓] Wrote per-specimen PIT files to {run} (pattern: pit_<SPECIMEN>.csv)")

if __name__ == "__main__":
    main()
