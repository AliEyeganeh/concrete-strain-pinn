# make_reliability_summary.py
import os, glob, math, argparse
from pathlib import Path
import numpy as np
import pandas as pd

Z95 = 1.959963984540054  # 95% two-sided

def stdnorm_pdf(z):
    return np.exp(-0.5 * z**2) / math.sqrt(2*math.pi)

def stdnorm_cdf(z):
    return 0.5 * (1.0 + np.erf(z / math.sqrt(2)))

def crps_gaussian(y, mu, sigma):
    sigma = np.maximum(sigma, 1e-12)
    z = (y - mu) / sigma
    return sigma * (z * (2*stdnorm_cdf(z) - 1) + 2*stdnorm_pdf(z) - 1/math.sqrt(math.pi))

def coerce_columns(df):
    # Required
    y_names  = ["y","y_true","target","target_strain","strain_true","true","obs"]
    mu_names = ["yhat","y_pred","pred","pred_mean","mean","mu","y_pred_mean"]
    sd_names = ["yhat_std","y_pred_std","std","sigma","uncertainty","stddev"]
    lo_names = ["yhat_lo","y_pred_lo","lower","lo","lwr","lcl","ci_lo","ci_lower"]
    hi_names = ["yhat_hi","y_pred_hi","upper","hi","upr","ucl","ci_hi","ci_upper"]
    model_names   = ["model","method"]
    dataset_names = ["dataset","specimen","name"]

    def pick(cands):
        for c in df.columns:
            if c.lower() in cands:
                return c
        return None

    col_y   = pick(y_names)
    col_mu  = pick(mu_names)
    col_sd  = pick(sd_names)
    col_lo  = pick(lo_names)
    col_hi  = pick(hi_names)
    col_model   = pick(model_names)
    col_dataset = pick(dataset_names)
    return col_y, col_mu, col_sd, col_lo, col_hi, col_model, col_dataset

def infer_labels_from_path(path):
    stem = Path(path).stem.lower().replace("-", "_")
    tokens = stem.split("_")
    model = "unknown"
    if any(t in stem for t in ["mcdo","dropout","mc_dropout"]): model = "MC Dropout"
    if any(t in stem for t in ["ens","ensemble","deepens"]):    model = "Deep Ensemble"
    if "posthoc" in stem: model += " (Posthoc)" if model != "unknown" else "Posthoc"
    dataset = None
    # heuristic dataset pickup
    for tok in tokens:
        if tok in ["scc","ecc","uhpc","scc_ecc","scc_uhpc","ecc_uhpc"]:
            dataset = tok.upper().replace("_", "-")
    split = "test" if "test" in stem else ("val" if "val" in stem else ("cal" if "cal" in stem else None))
    return model, (dataset or stem), split

def compute_one(csv_path, z=Z95):
    df = pd.read_csv(csv_path)
    col_y, col_mu, col_sd, col_lo, col_hi, col_model, col_dataset = coerce_columns(df)
    if col_y is None or col_mu is None:
        raise ValueError("missing required columns y / yhat")
    y  = df[col_y].to_numpy(dtype=float)
    mu = df[col_mu].to_numpy(dtype=float)

    # Intervals/SD
    if col_lo is not None and col_hi is not None:
        lo = df[col_lo].to_numpy(dtype=float)
        hi = df[col_hi].to_numpy(dtype=float)
        sd_from_ci = (hi - lo) / (2*z)
        sd_from_ci = np.maximum(sd_from_ci, 1e-12)
        sd = df[col_sd].to_numpy(dtype=float) if col_sd is not None else sd_from_ci
    elif col_sd is not None:
        sd = np.maximum(df[col_sd].to_numpy(dtype=float), 1e-12)
        lo, hi = mu - z*sd, mu + z*sd
    else:
        lo = hi = sd = None

    coverage = np.mean((y >= lo) & (y <= hi)) if (lo is not None and hi is not None) else np.nan
    crps = np.mean(crps_gaussian(y, mu, sd)) if sd is not None else np.nan
    mae  = float(np.mean(np.abs(y - mu)))
    rmse = float(np.sqrt(np.mean((y - mu)**2)))
    n = len(df)

    model = df[col_model].iloc[0] if col_model else None
    dataset = df[col_dataset].iloc[0] if col_dataset else None
    inf_model, inf_dataset, split = infer_labels_from_path(csv_path)
    model = model or inf_model
    dataset = dataset or inf_dataset

    return {
        "model": model,
        "dataset": dataset,
        "split": split,
        "n": n,
        "coverage_95": (None if isinstance(coverage, float) and np.isnan(coverage) else float(coverage)),
        "crps":        (None if isinstance(crps, float)     and np.isnan(crps)     else float(crps)),
        "mae": float(mae),
        "rmse": float(rmse),
        "source_file": str(csv_path)
    }

def expand_patterns(run: Path, patterns: list[str], recursive: bool) -> list[str]:
    files = []
    if not patterns:
        patterns = ["**/*.csv"] if recursive else ["*.csv"]
    for pat in patterns:
        pat = pat.strip()
        if not pat:
            continue
        if recursive and "**" not in pat:
            pat = str(Path("**") / pat)
        files.extend(glob.glob(str(run / pat), recursive=True))
    # Deduplicate while keeping order
    seen, unique = set(), []
    for f in files:
        if f not in seen:
            seen.add(f); unique.append(f)
    return unique

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run directory containing CSVs (search is recursive if --recursive).")
    ap.add_argument("--glob", default="*details*.csv", help="Glob pattern(s) for CSVs. Use , ; or | to separate multiple patterns. Default: *details*.csv")
    ap.add_argument("--alpha", type=float, default=0.95, help="Nominal coverage (95%% default).")
    ap.add_argument("--recursive", action="store_true", help="Search recursively.")
    args = ap.parse_args()

    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)

    # Patterns list
    raw = args.glob.replace(";", ",").replace("|", ",")
    patterns = [p for p in (s.strip() for s in raw.split(",")) if p]

    files = expand_patterns(run, patterns, args.recursive)
    if not files:
        raise SystemExit(f"No CSV files found in {run} matching patterns: {patterns} (recursive={args.recursive})")

    print(f"[i] Found {len(files)} candidate CSVs. Filtering for usable columns...")
    rows, kept, skipped = [], 0, 0
    for f in files:
        try:
            rows.append(compute_one(f))
            kept += 1
        except Exception as e:
            print(f"[skip] {f}: {e}")
            skipped += 1

    if not rows:
        raise SystemExit("No valid files processed (none had recognizable columns).")

    out_csv = run / "reliability_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[✓] Wrote {out_csv.resolve()} with {len(rows)} rows. (kept {kept}, skipped {skipped})")

if __name__ == "__main__":
    main()
