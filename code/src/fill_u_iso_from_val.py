import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd

def phi_cdf(z):
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))

def pit_from_mu_sigma(y, mu, sd):
    sd = np.maximum(sd, 1e-12)
    return phi_cdf((y - mu) / sd)

def ecdf_fit(u):
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
    u = np.asarray(u, dtype=float)
    u = u[~np.isnan(u)]
    if u.size == 0:
        return np.nan
    u = np.sort(np.clip(u, 0.0, 1.0))
    n = u.size
    i = np.arange(1, n+1, dtype=float)
    d_plus  = np.max(i/n - u)
    d_minus = np.max(u - (i-1)/n)
    return float(max(d_plus, d_minus))

def pick(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run folder, e.g. out_ecc_scc_mcdo")
    ap.add_argument("--val_split", default="val", choices=["val","test"])
    ap.add_argument("--test_split", default="test", choices=["val","test"])
    args = ap.parse_args()

    run = Path(args.run)
    test_path = run / f"{args.test_split}_details_cal.csv"
    val_path  = run / f"{args.val_split}_details_cal.csv"
    if not test_path.exists():
        raise SystemExit(f"Missing {test_path}")

    test = pd.read_csv(test_path)
    val  = pd.read_csv(val_path) if val_path.exists() else None

    # Column mapping (flexible)
    c_spec = pick(test, "specimen")
    c_y    = pick(test, "y","y_true","target")
    c_mu_t = pick(test, "mu_temp","mu_raw","pred_mean","yhat","mu")
    c_sd_t = pick(test, "sd_temp","sd_raw","pred_std","yhat_std","sd","sigma")
    if not all([c_spec, c_y, c_mu_t, c_sd_t]):
        raise SystemExit(f"Need columns: specimen, y, mu_temp/mu_raw, sd_temp/sd_raw. Found: {list(test.columns)}")

    # PIT for test
    u_test = pit_from_mu_sigma(test[c_y].to_numpy(float),
                               test[c_mu_t].to_numpy(float),
                               test[c_sd_t].to_numpy(float))

    # Fit isotonic CDF maps (prefer per-specimen on VALIDATION; fallback global; fallback test)
    iso_maps = {}
    if val is not None:
        c_yv    = pick(val, "y","y_true","target")
        c_mu_tv = pick(val, "mu_temp","mu_raw","pred_mean","yhat","mu")
        c_sd_tv = pick(val, "sd_temp","sd_raw","pred_std","yhat_std","sd","sigma")
        c_specv = pick(val, "specimen")
        if all([c_yv, c_mu_tv, c_sd_tv, c_specv]):
            u_val = pit_from_mu_sigma(val[c_yv].to_numpy(float),
                                      val[c_mu_tv].to_numpy(float),
                                      val[c_sd_tv].to_numpy(float))
            val["_u_val"] = u_val
            # per-specimen ECDF if enough points; else remembered for global
            for spec, g in val.groupby(c_specv):
                if g.shape[0] >= 50:
                    f = ecdf_fit(g["_u_val"].to_numpy())
                    if f: iso_maps[spec] = f
            # global fallback ECDF
            f_global = ecdf_fit(u_val)
        else:
            f_global = ecdf_fit(u_test)  # worst-case fallback
    else:
        f_global = ecdf_fit(u_test)

    # Build and write outputs
    levels = np.linspace(0.50, 0.99, 50)
    ks_rows = []
    for spec, g in test.groupby(c_spec):
        idx = g.index.to_numpy()
        u_t = u_test[idx]
        # choose per-specimen map if available; else global; else identity
        f = iso_maps.get(spec, f_global if f_global is not None else (lambda x: np.asarray(x, float)))
        u_i = f(u_t)

        # write pit_<SPECIMEN>.csv
        pd.DataFrame({"u_temp": u_t, "u_iso": u_i}).to_csv(run / f"pit_{spec}.csv", index=False)

        # coverage curves (level, coverage)
        def cov_from_pit(u, levels):
            u = u[~np.isnan(u)]
            cov = []
            for a in levels:
                lo = (1-a)/2.0
                hi = 1.0 - lo
                cov.append(float(np.mean((u >= lo) & (u <= hi))) if u.size else np.nan)
            return cov

        pd.DataFrame({"level": levels, "coverage": cov_from_pit(u_t, levels)}).to_csv(run / f"coverage_curve_temp_{spec}.csv", index=False)
        pd.DataFrame({"level": levels, "coverage": cov_from_pit(u_i, levels)}).to_csv(run / f"coverage_curve_iso_{spec}.csv",  index=False)

        ks_rows.append({"specimen": spec, "KS_temp": ks_uniform01(u_t), "KS_iso": ks_uniform01(u_i)})

    # Refresh reliability_summary.csv KS columns
    rel_path = run / "reliability_summary.csv"
    ks_df = pd.DataFrame(ks_rows)
    if rel_path.exists():
        rel = pd.read_csv(rel_path)
        if "specimen" not in rel.columns and "dataset" in rel.columns:
            rel = rel.rename(columns={"dataset":"specimen"})
        rel = rel.drop(columns=[c for c in ["KS_temp","KS_iso"] if c in rel.columns], errors="ignore")
        rel = rel.merge(ks_df, on="specimen", how="left")
    else:
        # minimal file if none exists
        rel = ks_df.copy()
        rel["coverage_95"] = np.nan
        rel["crps"] = np.nan
        rel["rmse"] = np.nan
        rel["n"] = test.groupby(c_spec).size().reindex(rel["specimen"]).to_numpy()
        rel["model"] = "MC Dropout (posthoc)"
        rel["split"] = args.test_split
        rel["source_file"] = str(test_path)
    rel.to_csv(rel_path, index=False)
    print(f"[✓] Filled u_iso and wrote coverage curves + KS into {run}")

if __name__ == "__main__":
    main()
