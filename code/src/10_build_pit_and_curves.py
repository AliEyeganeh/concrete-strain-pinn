# 10_build_pit_and_curves.py  (fixed)
import math
from pathlib import Path
import numpy as np
import pandas as pd

def phi_cdf(z):
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))

def pit_from_mu_sigma(y, mu, sd):
    sd = np.maximum(sd, 1e-12)
    return phi_cdf((y - mu) / sd)

def ecdf_map(u):
    """Return monotone CDF map f(x)=rank(x)/(n+1) from ECDF of u."""
    u = np.asarray(u, dtype=float)
    u = u[~np.isnan(u)]
    if u.size == 0:
        return None
    xs = np.sort(np.clip(u, 0.0, 1.0))
    n  = xs.size
    def f(x):
        x = np.asarray(x, dtype=float)
        ranks = np.searchsorted(xs, np.clip(x, 0.0, 1.0), side="right")
        return ranks / (n + 1.0)
    return f

def ks_uniform01(u):
    """One-sample KS statistic vs Uniform(0,1)."""
    u = np.asarray(u, dtype=float)
    u = u[~np.isnan(u)]
    if u.size == 0:
        return np.nan
    uu = np.sort(np.clip(u, 0.0, 1.0))
    n = uu.size
    i = np.arange(1, n+1, dtype=float)
    d_plus  = np.max(i/n - uu)
    d_minus = np.max(uu - (i-1)/n)
    return float(max(d_plus, d_minus))

def coverage_curve_from_pit(u, levels):
    """Two-sided central coverage: P((1-α)/2 <= u <= (1+α)/2)."""
    u = np.asarray(u, dtype=float)
    u = u[~np.isnan(u)]
    cov = []
    for a in levels:
        lo, hi = (1.0 - a)/2.0, 1.0 - (1.0 - a)/2.0
        cov.append(float(np.mean((u >= lo) & (u <= hi))) if u.size else np.nan)
    return cov

def pick(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None

if __name__ == "__main__":
    run = Path(".")

    test_path = run / "test_details_cal.csv"
    val_path  = run / "val_details_cal.csv"
    if not test_path.exists():
        raise SystemExit("Missing test_details_cal.csv in this folder.")

    test = pd.read_csv(test_path)
    val  = pd.read_csv(val_path) if val_path.exists() else None

    c_spec = pick(test, "specimen")
    c_y    = pick(test, "y","y_true","target")
    c_mu_t = pick(test, "mu_temp","mu_raw","pred_mean","yhat","mu")
    c_sd_t = pick(test, "sd_temp","sd_raw","pred_std","yhat_std","sd","sigma")
    if not all([c_spec, c_y, c_mu_t, c_sd_t]):
        raise SystemExit(f"Need columns: specimen, y, mu_temp/mu_raw, sd_temp/sd_raw. Found: {list(test.columns)}")

    # Test PIT (temp or raw fallback)
    y_test  = test[c_y].to_numpy(float)
    mu_test = test[c_mu_t].to_numpy(float)
    sd_test = np.maximum(test[c_sd_t].to_numpy(float), 1e-12)
    u_temp_test = pit_from_mu_sigma(y_test, mu_test, sd_test)

    # Fit isotonic on VAL (fallback to TEST if val missing)
    if val is not None:
        c_yv    = pick(val, "y","y_true","target")
        c_mu_tv = pick(val, "mu_temp","mu_raw","pred_mean","yhat","mu")
        c_sd_tv = pick(val, "sd_temp","sd_raw","pred_std","yhat_std","sd","sigma")
        if all([c_yv, c_mu_tv, c_sd_tv]):
            u_val = pit_from_mu_sigma(val[c_yv].to_numpy(float),
                                      val[c_mu_tv].to_numpy(float),
                                      val[c_sd_tv].to_numpy(float))
            f_iso = ecdf_map(u_val)
        else:
            f_iso = ecdf_map(u_temp_test)
    else:
        f_iso = ecdf_map(u_temp_test)

    u_iso_test = f_iso(u_temp_test) if f_iso is not None else u_temp_test.copy()

    # Write per-specimen PIT + per-specimen coverage curves
    levels = np.linspace(0.50, 0.99, 50)
    ks_rows = []  # collect KS to update reliability_summary.csv
    for spec, g in test.groupby(c_spec):
        idx = g.index.to_numpy()
        ut  = u_temp_test[idx]
        ui  = u_iso_test[idx]

        # pit_<SPECIMEN>.csv
        pd.DataFrame({"u_temp": ut, "u_iso": ui}).to_csv(run / f"pit_{spec}.csv", index=False)

        # coverage curves
        pd.DataFrame({"level": levels, "coverage": coverage_curve_from_pit(ut, levels)}) \
            .to_csv(run / f"coverage_curve_temp_{spec}.csv", index=False)
        pd.DataFrame({"level": levels, "coverage": coverage_curve_from_pit(ui, levels)}) \
            .to_csv(run / f"coverage_curve_iso_{spec}.csv",  index=False)

        # KS stats for each specimen
        ks_rows.append({"specimen": spec,
                        "KS_temp": ks_uniform01(ut),
                        "KS_iso":  ks_uniform01(ui)})

    # Update KS in reliability_summary.csv (if present)
    rel_path = run / "reliability_summary.csv"
    if rel_path.exists():
        rel_df = pd.read_csv(rel_path)
        ks_df  = pd.DataFrame(ks_rows)
        # drop old KS columns then merge fresh ones
        rel_df = rel_df.drop(columns=[c for c in ["KS_temp","KS_iso"] if c in rel_df.columns], errors="ignore")
        rel_df = rel_df.merge(ks_df, on="specimen", how="left")
        rel_df.to_csv(rel_path, index=False)

    print("[✓] wrote pit_<SPEC>.csv, per-specimen coverage curves, and updated KS")
