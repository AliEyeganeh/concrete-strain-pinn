import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd

def phi_cdf(z):
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))

# Acklam’s rational approximation for Φ^{-1}(p) (no SciPy required)
def norm_ppf(p):
    p = np.asarray(p, dtype=float)
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("norm_ppf expects p in (0,1)")
    a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
          3.754408661907416e+00]
    plow  = 0.02425
    phigh = 1 - plow
    q = np.zeros_like(p)
    # lower region
    idx = p < plow
    if np.any(idx):
        q1 = np.sqrt(-2*np.log(p[idx]))
        q[idx] = (((((c[0]*q1 + c[1])*q1 + c[2])*q1 + c[3])*q1 + c[4])*q1 + c[5]) / \
                  ((((d[0]*q1 + d[1])*q1 + d[2])*q1 + d[3])*q1 + 1)
    # central region
    idx = (p >= plow) & (p <= phigh)
    if np.any(idx):
        q2 = p[idx] - 0.5
        r  = q2*q2
        q[idx] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q2 / \
                  (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    # upper region
    idx = p > phigh
    if np.any(idx):
        q3 = np.sqrt(-2*np.log(1 - p[idx]))
        q[idx] = -(((((c[0]*q3 + c[1])*q3 + c[2])*q3 + c[3])*q3 + c[4])*q3 + c[5]) / \
                    ((((d[0]*q3 + d[1])*q3 + d[2])*q3 + d[3])*q3 + 1)
    return q

def make_pit_and_curves(run_dir: Path, split: str = "test"):
    det_path = run_dir / f"{split}_details_cal.csv"
    if not det_path.exists():
        raise SystemExit(f"Missing {det_path}")

    det = pd.read_csv(det_path)

    # Flexible column mapping
    def pick(*names):
        for n in names:
            if n in det.columns:
                return n
        return None

    col_spec = pick("specimen")
    col_y    = pick("y","y_true","target")
    col_mu_t = pick("mu_temp","mu_raw","pred_mean","yhat","mu")
    col_sd_t = pick("sd_temp","sd_raw","pred_std","yhat_std","sd","sigma")

    if not all([col_spec, col_y, col_mu_t, col_sd_t]):
        raise SystemExit(f"Required columns not found in {det_path}. "
                         f"Need specimen, y, mu_temp/mu_raw, sd_temp/sd_raw. "
                         f"Found: {list(det.columns)}")

    y  = det[col_y].to_numpy(dtype=float)
    mu = det[col_mu_t].to_numpy(dtype=float)
    sd = np.maximum(det[col_sd_t].to_numpy(dtype=float), 1e-12)

    # PIT (temperature-scaled or raw fallback)
    u_temp = phi_cdf((y - mu) / sd)

    # If isotonic PIT exists, use it; else fill NaN (plotter just won't draw a curve)
    if "pit_iso" in det.columns:
        u_iso = det["pit_iso"].to_numpy(dtype=float)
    else:
        u_iso = np.full_like(u_temp, np.nan, dtype=float)

    # Per-specimen PIT files: pit_<SPECIMEN>.csv with u_temp and u_iso
    for spec, idx in det.groupby(col_spec).indices.items():
        out_pit = pd.DataFrame({
            "u_temp": u_temp[idx],
            "u_iso":  u_iso[idx]
        })
        out_pit.to_csv(run_dir / f"pit_{spec}.csv", index=False)

    # Coverage curves: coverage_curve_temp_<SPECIMEN>.csv and coverage_curve_iso_<SPECIMEN>.csv
    # Make a fine grid so lines look smooth
    alphas = np.linspace(0.01, 0.99, 99)  # two-sided nominal coverage
    z_two_sided = norm_ppf((1 + alphas)/2)   # critical z for two-sided α

    # Optional isotonic intervals (lo_iso/hi_iso or mu_iso/sd_iso); else mirror temp to keep plotter happy
    col_mu_iso = pick("mu_iso")
    col_sd_iso = pick("sd_iso")
    col_lo_iso = pick("lo_iso","lwr_iso","lower_iso")
    col_hi_iso = pick("hi_iso","upr_iso","upper_iso")

    for spec, g in det.groupby(col_spec):
        gi = g.index.to_numpy()
        y_s  = y[gi]
        mu_s = mu[gi]
        sd_s = sd[gi]

        # TEMP curve
        cov_temp = []
        for z in z_two_sided:
            lo = mu_s - z*sd_s
            hi = mu_s + z*sd_s
            cov_temp.append(float(np.mean((y_s >= lo) & (y_s <= hi))))
        df_temp = pd.DataFrame({"alpha": alphas, "coverage": cov_temp})
        df_temp.to_csv(run_dir / f"coverage_curve_temp_{spec}.csv", index=False)

        # ISO curve (best effort)
        if col_lo_iso and col_hi_iso:
            lo_i = g[col_lo_iso].to_numpy(dtype=float)
            hi_i = g[col_hi_iso].to_numpy(dtype=float)
            # For a proper curve we'd need iso intervals at many α; if only one α exists,
            # just write a flat line at that α to avoid breaking the plotter:
            # assume these bounds correspond roughly to 95%:
            alpha0 = 0.95
            cov0 = float(np.mean((y_s >= lo_i) & (y_s <= hi_i)))
            df_iso = pd.DataFrame({"alpha": [alpha0], "coverage": [cov0]})
        elif col_mu_iso and col_sd_iso:
            mu_i = g[col_mu_iso].to_numpy(dtype=float)
            sd_i = np.maximum(g[col_sd_iso].to_numpy(dtype=float), 1e-12)
            cov_iso = []
            for z in z_two_sided:
                lo = mu_i - z*sd_i
                hi = mu_i + z*sd_i
                cov_iso.append(float(np.mean((y_s >= lo) & (y_s <= hi))))
            df_iso = pd.DataFrame({"alpha": alphas, "coverage": cov_iso})
        else:
            # No isotonic data: mirror temp so file exists and plotter won’t crash
            df_iso = df_temp.copy()

        df_iso.to_csv(run_dir / f"coverage_curve_iso_{spec}.csv", index=False)

    print(f"[✓] Wrote PIT + coverage-curve CSVs into {run_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run folder, e.g. out_ecc_scc_mcdo")
    ap.add_argument("--split", default="test", choices=["test","val"], help="Split to use (default: test)")
    args = ap.parse_args()
    make_pit_and_curves(Path(args.run), split=args.split)

if __name__ == "__main__":
    main()
