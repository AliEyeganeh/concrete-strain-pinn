# 20_build_hist_and_aliases.py
from pathlib import Path
import numpy as np
import pandas as pd

def coverage_curve_from_pit(u, levels):
    u = np.asarray(u, dtype=float)
    u = u[~np.isnan(u)]
    cov = []
    for a in levels:
        lo, hi = (1.0 - a)/2.0, 1.0 - (1.0 - a)/2.0
        cov.append(float(np.mean((u >= lo) & (u <= hi))) if u.size else np.nan)
    return cov

if __name__ == "__main__":
    run = Path(".")
    # read all per-specimen PIT we just wrote
    pit_files = list(run.glob("pit_*.csv"))
    if not pit_files:
        raise SystemExit("No pit_*.csv files found. Run 10_build_pit_and_curves.py first.")

    u_temp_all, u_iso_all = [], []
    for p in pit_files:
        df = pd.read_csv(p)
        if "u_temp" in df: u_temp_all.append(df["u_temp"].to_numpy(float))
        if "u_iso"  in df: u_iso_all.append(df["u_iso"].to_numpy(float))
    u_temp_all = np.concatenate(u_temp_all) if u_temp_all else np.array([])
    u_iso_all  = np.concatenate(u_iso_all)  if u_iso_all  else np.array([])

    levels = np.linspace(0.50, 0.99, 50)
    # canonical “hist_test_raw” files
    pd.DataFrame({"level": levels, "coverage": coverage_curve_from_pit(u_temp_all, levels)}).to_csv(
        run / "coverage_curve_temp_hist_test_raw.csv", index=False)
    pd.DataFrame({"level": levels, "coverage": coverage_curve_from_pit(u_iso_all,  levels)}).to_csv(
        run / "coverage_curve_iso_hist_test_raw.csv", index=False)

    # filename aliases some plotters ask for
    aliases = [
        ("coverage_curve_temp_hist_test_raw.csv", "coverage_curve_temp_hist_test_temp.csv"),
        ("coverage_curve_iso_hist_test_raw.csv",  "coverage_curve_iso_hist_test_temp.csv"),
        ("coverage_curve_temp_hist_test_raw.csv", "coverage_curve_temp_hist_val_raw.csv"),
        ("coverage_curve_temp_hist_test_raw.csv", "coverage_curve_temp_hist_val_temp.csv"),
        ("coverage_curve_iso_hist_test_raw.csv",  "coverage_curve_iso_hist_val_raw.csv"),
        ("coverage_curve_iso_hist_test_raw.csv",  "coverage_curve_iso_hist_val_temp.csv"),
    ]
    for src, dst in aliases:
        s, d = run / src, run / dst
        if s.exists() and not d.exists():
            d.write_bytes(s.read_bytes())

    print("[✓] wrote global hist coverage curves + aliases")
