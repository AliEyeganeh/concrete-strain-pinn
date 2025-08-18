# make_reliability_from_posthoc.py  (robust to specimen-prefixed filenames)
import argparse, math, glob
from pathlib import Path
import numpy as np
import pandas as pd

Z95 = 1.959963984540054

# ---------- helpers ----------
def phi_cdf(z):
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))

def ks_one_sample_uniform(u: np.ndarray) -> float:
    u = np.asarray(u, dtype=float)
    u = u[~np.isnan(u)]
    if u.size == 0:
        return np.nan
    u = np.sort(np.clip(u, 0.0, 1.0))
    n = u.size
    i = np.arange(1, n + 1, dtype=float)
    d_plus  = np.max(i / n - u)
    d_minus = np.max(u - (i - 1) / n)
    return float(max(d_plus, d_minus))

def find_any(run: Path, pattern: str):
    return [Path(p) for p in glob.glob(str(run / pattern), recursive=True)]

def pick(df: pd.DataFrame, *names):
    for n in names:
        if n in df.columns:
            return n
    return None

def coerce_metric_col(df: pd.DataFrame, base: str, split: str):
    # Try several common spellings like coverage95_test / coverage_95_test / cov95_test
    cands = [f"{base}95_{split}", f"{base}_95_{split}", f"{base}{split}_95", f"{base}_{split}_95"]
    for c in cands:
        if c in df.columns:
            return c
    # specific known names
    if base == "coverage":
        cands = [f"coverage_{split}", f"cov_{split}", f"cov95_{split}"]
    elif base == "crps":
        cands = [f"crps_{split}"]
    elif base == "rmse":
        cands = [f"rmse_{split}"]
    for c in cands:
        if c in df.columns:
            return c
    return None

def normalize_method(s: str) -> str:
    if not isinstance(s, str): return ""
    k = s.lower().replace("_","").replace("+","")
    if "tempscale" in k or (("temp" in k) and ("scale" in k)):
        return "temp-scale"
    if k == "raw":
        return "raw"
    if "tempiso" in k or "isotonic" in k or "iso" in k:
        return "temp+iso"
    # fallbacks
    if "temp" in k:  # sometimes written just "temp"
        return "temp-scale"
    return s.lower()

def compute_ks_from_details(details: pd.DataFrame, mu_col: str, sd_col: str) -> pd.Series:
    if mu_col not in details.columns or sd_col not in details.columns:
        return pd.Series(dtype=float)
    y  = details[pick(details, "y","y_true","target")].to_numpy(float)
    mu = details[mu_col].to_numpy(float)
    sd = np.maximum(details[sd_col].to_numpy(float), 1e-12)
    z  = (y - mu) / sd
    u  = phi_cdf(z)
    return details.groupby("specimen").apply(lambda g: ks_one_sample_uniform(u[g.index]))

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run folder (e.g., out_ecc_scc_mcdo)")
    ap.add_argument("--method", default="temp-scale", choices=["raw","temp-scale","temp+iso"],
                    help="Which calibration row to take (fallback to raw if missing).")
    ap.add_argument("--split",  default="test", choices=["val","test"],
                    help="Split to summarize from posthoc (default: test).")
    ap.add_argument("--spec",   default=None, help="Only emit this specimen (e.g., ECC_SCC).")
    args = ap.parse_args()

    run = Path(args.run)
    if not run.exists():
        raise SystemExit(f"Run folder not found: {run}")

    # 1) Load & merge ANY posthoc summary files
    post_files = find_any(run, "**/*posthoc*summary*.csv") or find_any(run, "posthoc_summary.csv")
    if not post_files:
        raise SystemExit(f"No posthoc summary CSV found in {run}. Looked for *posthoc*summary*.csv")
    post_dfs = []
    for p in post_files:
        try:
            df = pd.read_csv(p)
            # ensure specimen column; if missing, try infer from filename
            if "specimen" not in df.columns:
                stem = p.stem.lower()
                # crude inference: take first token before '_posthoc'
                toks = stem.split("_posthoc")[0].split("_")
                if toks:
                    df["specimen"] = toks[-1].upper()
            df["__source__"] = str(p)
            post_dfs.append(df)
        except Exception as e:
            print(f"[skip] {p.name}: {e}")
    if not post_dfs:
        raise SystemExit("No readable posthoc summary files.")
    post = pd.concat(post_dfs, ignore_index=True)

    # normalize method labels and filter
    if "method" in post.columns:
        post["method_norm"] = post["method"].apply(normalize_method)
        sel = post[post["method_norm"] == args.method]
        if sel.empty:
            # fallback order: temp-scale -> raw -> any available
            fallback = post[post["method_norm"] == "raw"]
            post_use = fallback if not fallback.empty else post
        else:
            post_use = sel
    else:
        post_use = post

    # if single specimen desired
    if args.spec:
        post_use = post_use[post_use["specimen"].astype(str) == args.spec]

    if post_use.empty:
        raise SystemExit("No rows in posthoc summary after filtering (method/spec).")

    # pick metric columns
    cov_col  = coerce_metric_col(post_use, "coverage", args.split) or f"coverage95_{args.split}"
    crps_col = coerce_metric_col(post_use, "crps", args.split)       or f"crps_{args.split}"
    rmse_col = coerce_metric_col(post_use, "rmse", args.split)       or f"rmse_{args.split}"
    for c in [cov_col, rmse_col]:
        if c not in post_use.columns:
            raise SystemExit(f"Missing required column '{c}' in merged posthoc summary.")
    # crps may be missing for some methods (e.g., temp+iso)
    has_crps = crps_col in post_use.columns

    # 2) Load details (any matching files; merged or per-specimen)
    det_files = find_any(run, "**/*test*_details_cal*.csv")
    if not det_files:
        print("[i] No test details found; will write reliability without KS/n.")
        det = pd.DataFrame()
    else:
        det_dfs = []
        for d in det_files:
            try:
                dd = pd.read_csv(d)
                if "specimen" not in dd.columns:
                    # try infer from filename for per-specimen files
                    spec = d.stem.split("_")[0]
                    dd["specimen"] = spec
                det_dfs.append(dd)
            except Exception as e:
                print(f"[skip] {d.name}: {e}")
        det = pd.concat(det_dfs, ignore_index=True) if det_dfs else pd.DataFrame()

    # 3) Compute KS (temp/raw/iso) from details if available
    ks_temp = ks_raw = ks_iso = pd.Series(dtype=float)
    if not det.empty:
        # flexible mu/sd columns
        mu_t = pick(det, "mu_temp","mu_raw","pred_mean","yhat","mu")
        sd_t = pick(det, "sd_temp","sd_raw","pred_std","yhat_std","sd","sigma")
        mu_r = pick(det, "mu_raw","mu_temp","pred_mean","yhat","mu")
        sd_r = pick(det, "sd_raw","sd_temp","pred_std","yhat_std","sd","sigma")
        if mu_t and sd_t:
            ks_temp = compute_ks_from_details(det, mu_t, sd_t)
        if mu_r and sd_r:
            ks_raw = compute_ks_from_details(det, mu_r, sd_r)
        if "pit_iso" in det.columns:
            ks_iso = det.groupby("specimen")["pit_iso"].apply(ks_one_sample_uniform)
        else:
            ks_iso = pd.Series(dtype=float)

    n_map = det.groupby("specimen").size().to_dict() if not det.empty else {}

    # 4) Build reliability_summary.csv rows
    rows = []
    for spec, g in post_use.groupby("specimen"):
        r = g.iloc[0]
        rows.append({
            "specimen": spec,
            "coverage_95": float(r[cov_col]),
            "crps": float(r[crps_col]) if has_crps and pd.notna(r[crps_col]) else np.nan,
            "rmse": float(r[rmse_col]),
            "KS_temp": float(ks_temp.get(spec, np.nan)) if not ks_temp.empty else np.nan,
            "KS_raw":  float(ks_raw.get(spec,  np.nan)) if not ks_raw.empty  else np.nan,
            "KS_iso":  float(ks_iso.get(spec,  np.nan)) if not ks_iso.empty  else np.nan,
            "n": int(n_map.get(spec, 0)) if n_map else pd.NA,
            "model": r.get("model", "MC Dropout (posthoc)"),
            "split": args.split,
            "source_file": r["__source__"]
        })

    out = pd.DataFrame(rows)
    (run / "reliability_summary.csv").write_text(out.to_csv(index=False))
    print(f"[✓] wrote {run/'reliability_summary.csv'} with {len(out)} rows")

if __name__ == "__main__":
    main()
