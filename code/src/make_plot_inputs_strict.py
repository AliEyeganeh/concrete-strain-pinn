import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd

def norm_ppf(p):
    # Acklam’s approximation (no SciPy)
    p = np.asarray(p, dtype=float)
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("norm_ppf expects p in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1-0.02425
    q = np.zeros_like(p)
    lo = p < plow
    if np.any(lo):
        q1 = np.sqrt(-2*np.log(p[lo]))
        q[lo] = (((((c[0]*q1 + c[1])*q1 + c[2])*q1 + c[3])*q1 + c[4])*q1 + c[5]) / \
                 ((((d[0]*q1 + d[1])*q1 + d[2])*q1 + d[3])*q1 + 1)
    mid = (p >= plow) & (p <= phigh)
    if np.any(mid):
        q2 = p[mid] - 0.5
        r = q2*q2
        q[mid] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q2 / \
                  (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    hi = p > phigh
    if np.any(hi):
        q3 = np.sqrt(-2*np.log(1 - p[hi]))
        q[hi] = -(((((c[0]*q3 + c[1])*q3 + c[2])*q3 + c[3])*q3 + c[4])*q3 + c[5]) / \
                   ((((d[0]*q3 + d[1])*q3 + d[2])*q3 + d[3])*q3 + 1)
    return q

def build(run_dir: Path, split: str = "test"):
    det_path = run_dir / f"{split}_details_cal.csv"
    if not det_path.exists():
        raise SystemExit(f"Missing {det_path}")

    det = pd.read_csv(det_path)

    # Flexible column picking
    def pick(*names):
        for n in names:
            if n in det.columns:
                return n
        return None

    c_spec = pick("specimen")
    c_y    = pick("y","y_true","target")
    c_mu_t = pick("mu_temp","mu_raw","pred_mean","yhat","mu")
    c_sd_t = pick("sd_temp","sd_raw","pred_std","yhat_std","sd","sigma")
    c_u_iso= pick("pit_iso","u_iso")  # optional

    if not all([c_spec, c_y, c_mu_t, c_sd_t]):
        raise SystemExit(f"Need columns: specimen, y, mu_temp/mu_raw, sd_temp/sd_raw. Found: {list(det.columns)}")

    y  = det[c_y].to_numpy(float)
    mu = det[c_mu_t].to_numpy(float)
    sd = np.maximum(det[c_sd_t].to_numpy(float), 1e-12)

    # PIT (temp or raw fallback)
    # Φ via erf (vectorized)
    u_temp = 0.5 * (1.0 + np.vectorize(math.erf)((y - mu) / (sd * math.sqrt(2.0))))

    # Isotonic PIT if present; else NaN
    if c_u_iso:
        u_iso = det[c_u_iso].to_numpy(float)
    else:
        u_iso = np.full_like(u_temp, np.nan, dtype=float)

    # Write per-specimen PIT files (exact columns!)
    for spec, idx in det.groupby(c_spec).indices.items():
        pd.DataFrame({"u_temp": u_temp[idx], "u_iso": u_iso[idx]}).to_csv(run_dir / f"pit_{spec}.csv", index=False)

    # Coverage curves for TEMP and ISO: columns 'level','coverage'
    levels = np.linspace(0.50, 0.99, 50)  # two-sided nominal coverage
    z = norm_ppf((1 + levels)/2)

    # Optional isotonic parameters, else mirror TEMP
    c_mu_iso = pick("mu_iso"); c_sd_iso = pick("sd_iso")
    c_lo_iso = pick("lo_iso","lwr_iso","lower_iso")
    c_hi_iso = pick("hi_iso","upr_iso","upper_iso")

    for spec, g in det.groupby(c_spec):
        gi = g.index.to_numpy()
        y_s  = y[gi]
        mu_s = mu[gi]
        sd_s = sd[gi]

        cov_t = []
        for zz in z:
            lo = mu_s - zz*sd_s
            hi = mu_s + zz*sd_s
            cov_t.append(float(np.mean((y_s >= lo) & (y_s <= hi))))
        pd.DataFrame({"level": levels, "coverage": cov_t}).to_csv(run_dir / f"coverage_curve_temp_{spec}.csv", index=False)

        # ISO curve
        if c_lo_iso and c_hi_iso:
            lo_i = g[c_lo_iso].to_numpy(float)
            hi_i = g[c_hi_iso].to_numpy(float)
            # assume these represent ~95%; write single point to avoid crash
            pd.DataFrame({"level":[0.95], "coverage":[float(np.mean((y_s >= lo_i) & (y_s <= hi_i)))]}) \
              .to_csv(run_dir / f"coverage_curve_iso_{spec}.csv", index=False)
        elif c_mu_iso and c_sd_iso:
            mu_i = g[c_mu_iso].to_numpy(float)
            sd_i = np.maximum(g[c_sd_iso].to_numpy(float), 1e-12)
            cov_i = []
            for zz in z:
                lo = mu_i - zz*sd_i
                hi = mu_i + zz*sd_i
                cov_i.append(float(np.mean((y_s >= lo) & (y_s <= hi))))
            pd.DataFrame({"level": levels, "coverage": cov_i}).to_csv(run_dir / f"coverage_curve_iso_{spec}.csv", index=False)
        else:
            # No iso info: mirror TEMP (so the file exists and plotter is happy)
            pd.DataFrame({"level": levels, "coverage": cov_t}).to_csv(run_dir / f"coverage_curve_iso_{spec}.csv", index=False)

    print(f"[✓] Wrote PIT + coverage-curve files into {run_dir}")

if __name__ == "__main__":
    import math
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run folder, e.g. out_ecc_scc_mcdo")
    ap.add_argument("--split", default="test", choices=["test","val"])
    args = ap.parse_args()
    build(Path(args.run), split=args.split)
