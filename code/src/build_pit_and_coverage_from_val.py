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

def ecdf_map_fit(u):
    """
    Fit a monotone CDF-correction f using ECDF on validation PIT.
    Returns a function f(x) = rank(x)/(n+1), implemented with searchsorted.
    """
    u = np.asarray(u, dtype=float)
    u = u[~np.isnan(u)]
    x_sorted = np.sort(np.clip(u, 0.0, 1.0))
    n = len(x_sorted)
    if n == 0:
        # identity if we have nothing to fit
        return lambda x: np.asarray(x, dtype=float)
    def f(x):
        x = np.asarray(x, dtype=float)
        # step ECDF: right-rank/(n+1)
        ranks = np.searchsorted(x_sorted, np.clip(x, 0.0, 1.0), side="right")
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

def coverage_curve_from_pit(u, levels):
    """Two-sided central coverage from PIT: P((1-α)/2 <= u <= (1+α)/2)."""
    u = np.asarray(u, dtype=float)
    u = u[~np.isnan(u)]
    cov = []
    for a in levels:
        lo = (1.0 - a)/2.0
        hi = 1.0 - lo
        cov.append(float(np.mean((u >= lo) & (u <= hi))) if u.size else np.nan)
    return cov

def pick(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run folder, e.g. out_ecc_scc_mcdo")
    ap.add_argument("--val_split", default="val", choices=["val","test"], help="Validation split file prefix")
    ap.add_argument("--test_split", default="test", choices=["val","test"], help="Evaluation split file prefix")
    args = ap.parse_args()

    run = Path(args.run)
    val_path = run / f"{args.val_split}_details_cal.csv"
    test_path = run / f"{args.test_split}_details_cal.csv"
    if not test_path.exists():
        raise SystemExit(f"Missing {test_path}")
    if not val_path.exists():
        print(f"[i] {val_path} missing; will fit isotonic on test (less ideal).")

    # Load files
    test = pd.read_csv(test_path)
    val  = pd.read_csv(val_path) if val_path.exists() else test.copy()

    # Column mapping (flexible)
    c_spec = pick(test, "specimen")
    c_y    = pick(test, "y","y_true","target")
    c_mu_t = pick(test, "mu_temp","mu_raw","pred_mean","yhat","mu")
    c_sd_t = pick(test, "sd_temp","sd_raw","pred_std","yhat_std","sd","sigma")
    if not all([c_spec, c_y, c_mu_t, c_sd_t]):
        raise SystemExit(f"Need columns: specimen, y, mu_temp/mu_raw, sd_temp/sd_raw. Found: {list(test.columns)}")

    # Compute PIT (temp/raw) for val and test
    u_val  = pit_from_mu_sigma(val[c_y].to_numpy(float),  val[c_mu_t].to_numpy(float),  val[c_sd_t].to_numpy(float))
    u_test = pit_from_mu_sigma(test[c_y].to_numpy(float), test[c_mu_t].to_numpy(float), test[c_sd_t].to_numpy(float))

    # Fit isotonic mapping on validation PIT and apply to test
    f_iso = ecdf_map_fit(u_val)
    u_iso_test = f_iso(u_test)

    # Write per-specimen PIT files
    for spec, idx in test.groupby(c_spec).indices.items():
        out = pd.DataFrame({
            "u_temp": u_test[idx],
            "u_iso":  u_iso_test[idx]
        })
        out.to_csv(run / f"pit_{spec}.csv", index=False)

    # Coverage curves (exact headers your plotter wants: level, coverage)
    levels = np.linspace(0.50, 0.99, 50)
    for spec, g in test.groupby(c_spec):
        gi = g.index.to_numpy()
        u_t = u_test[gi]
        u_i = u_iso_test[gi]
        cov_t = coverage_curve_from_pit(u_t, levels)
        cov_i = coverage_curve_from_pit(u_i, levels)
        pd.DataFrame({"level": levels, "coverage": cov_t}).to_csv(run / f"coverage_curve_temp_{spec}.csv", index=False)
        pd.DataFrame({"level": levels, "coverage": cov_i}).to_csv(run / f"coverage_curve_iso_{spec}.csv",  index=False)

    # Update reliability_summary.csv with real KS_temp/KS_iso (merge on specimen)
    rel_path = run / "reliability_summary.csv"
    ks_rows = []
    for spec, g in test.groupby(c_spec):
        gi = g.index.to_numpy()
        ks_rows.append({
            "specimen": spec,
            "KS_temp": ks_uniform01(u_test[gi]),
            "KS_iso":  ks_uniform01(u_iso_test[gi]),
        })
    ks_df = pd.DataFrame(ks_rows)

    if rel_path.exists():
        rel = pd.read_csv(rel_path)
        if "specimen" not in rel.columns:
            # fall back if earlier version used 'dataset'
            if "dataset" in rel.columns:
                rel = rel.rename(columns={"dataset":"specimen"})
            else:
                rel["specimen"] = ks_df["specimen"]
        rel = rel.drop(columns=[c for c in ["KS_temp","KS_iso"] if c in rel.columns], errors="ignore")
        rel = rel.merge(ks_df, on="specimen", how="left")
    else:
        # Minimal rel when none exists
        rel = ks_df.copy()
        rel["coverage_95"] = np.nan
        rel["crps"] = np.nan
        rel["rmse"] = np.nan
        rel["n"] = test.groupby(c_spec).size().reindex(rel["specimen"]).to_numpy()
        rel["model"] = "MC Dropout (posthoc)"
        rel["split"] = args.test_split
        rel["source_file"] = str(test_path)

    rel.to_csv(rel_path, index=False)
    print(f"[✓] Wrote PIT + coverage curves and updated KS into {run}")

if __name__ == "__main__":
    main()
