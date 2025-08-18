# 00_build_reliability.py
import math, glob
from pathlib import Path
import numpy as np
import pandas as pd

Z95 = 1.959963984540054

def phi_cdf(z):
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))

def ks_uniform01(u):
    u = np.asarray(u, dtype=float)
    u = u[~np.isnan(u)]
    if u.size == 0:
        return np.nan
    u = np.sort(np.clip(u, 0.0, 1.0))
    n = u.size
    i = np.arange(1, n+1, dtype=float)
    return float(max(np.max(i/n - u), np.max(u - (i-1)/n)))

def pick(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None

def normalize_method(s):
    if not isinstance(s, str): return ""
    k = s.lower().replace("_","").replace("+","")
    if "tempscale" in k or ("temp" in k and "scale" in k): return "temp-scale"
    if k == "raw": return "raw"
    if "tempiso" in k or "iso" in k: return "temp+iso"
    if "temp" in k: return "temp-scale"
    return s.lower()

def coerce_metric(df, base, split):
    cands = [f"{base}95_{split}", f"{base}_95_{split}", f"{base}{split}_95", f"{base}_{split}_95"]
    for c in cands:
        if c in df.columns: return c
    if base == "coverage":
        for c in [f"coverage_{split}", f"cov95_{split}", f"cov_{split}"]:
            if c in df.columns: return c
    if base == "crps":
        for c in [f"crps_{split}"]:
            if c in df.columns: return c
    if base == "rmse":
        for c in [f"rmse_{split}"]:
            if c in df.columns: return c
    return None

def infer_specimen_from_filename(path: Path):
    stem = path.stem.lower()
    # e.g., "scc_posthoc_summary" -> "SCC"
    toks = stem.replace("-", "_").split("_")
    if "posthoc" in toks:
        i = toks.index("posthoc")
        if i > 0:
            return toks[i-1].upper()
    # fallback: first token
    return toks[0].upper()

if __name__ == "__main__":
    run = Path(".")
    # --- collect any posthoc summary files (specimen-prefixed or merged) ---
    post_files = [Path(p) for p in glob.glob(str(run / "**/*posthoc*summary*.csv"), recursive=True)]
    if not post_files and (run / "posthoc_summary.csv").exists():
        post_files = [run / "posthoc_summary.csv"]
    if not post_files:
        raise SystemExit("No *posthoc*summary*.csv found in this folder.")

    post_list = []
    for p in post_files:
        try:
            df = pd.read_csv(p)
            if "specimen" not in df.columns:
                df["specimen"] = infer_specimen_from_filename(p)
            df["__source__"] = str(p)
            post_list.append(df)
        except Exception as e:
            print(f"[skip] {p.name}: {e}")
    post = pd.concat(post_list, ignore_index=True)

    # choose method: prefer temp-scale then fallback to raw
    if "method" in post.columns:
        post["method_norm"] = post["method"].apply(normalize_method)
        sel = post[post["method_norm"] == "temp-scale"]
        if sel.empty:
            sel = post[post["method_norm"] == "raw"]
        post = sel if not sel.empty else post

    # --- pick metric columns (test split) ---
    cov_col  = coerce_metric(post, "coverage", "test") or "coverage95_test"
    crps_col = coerce_metric(post, "crps", "test") or "crps_test"
    rmse_col = coerce_metric(post, "rmse", "test") or "rmse_test"
    for c in [cov_col, rmse_col]:
        if c not in post.columns:
            raise SystemExit(f"Missing required metric column '{c}' in merged posthoc summaries.")

    # --- collect test details (for KS and n) ---
    det_files = [Path(p) for p in glob.glob(str(run / "**/*test*_details_cal*.csv"), recursive=True)]
    det = pd.concat([pd.read_csv(d) for d in det_files], ignore_index=True) if det_files else pd.DataFrame()
    n_map = det.groupby("specimen").size().to_dict() if "specimen" in det.columns else {}

    # KS from temp/raw (if columns exist), iso if pit_iso present
    ks_temp = ks_raw = ks_iso = pd.Series(dtype=float)
    if not det.empty and "specimen" in det.columns:
        y  = det[pick(det, "y","y_true","target")].to_numpy(float)
        mu_t = det[pick(det, "mu_temp","mu_raw","pred_mean","yhat","mu")].to_numpy(float)
        sd_t = np.maximum(det[pick(det, "sd_temp","sd_raw","pred_std","yhat_std","sd","sigma")].to_numpy(float), 1e-12)
        u_t  = phi_cdf((y - mu_t) / sd_t)
        ks_temp = det.groupby("specimen").apply(lambda g: ks_uniform01(u_t[g.index]))
        # raw
        mu_r = det[pick(det, "mu_raw","mu_temp","pred_mean","yhat","mu")].to_numpy(float)
        sd_r = np.maximum(det[pick(det, "sd_raw","sd_temp","pred_std","yhat_std","sd","sigma")].to_numpy(float), 1e-12)
        u_r  = phi_cdf((y - mu_r) / sd_r)
        ks_raw = det.groupby("specimen").apply(lambda g: ks_uniform01(u_r[g.index]))
        # iso if available
        if "pit_iso" in det.columns:
            ks_iso = det.groupby("specimen")["pit_iso"].apply(ks_uniform01)

    # --- assemble reliability_summary.csv ---
    rows = []
    for spec, g in post.groupby("specimen"):
        r = g.iloc[0]
        rows.append({
            "specimen": spec,
            "coverage_95": float(r[cov_col]),
            "crps": float(r[crps_col]) if crps_col in post.columns and pd.notna(r.get(crps_col, np.nan)) else np.nan,
            "rmse": float(r[rmse_col]),
            "KS_temp": float(ks_temp.get(spec, np.nan)) if not ks_temp.empty else np.nan,
            "KS_raw":  float(ks_raw.get(spec,  np.nan)) if not ks_raw.empty  else np.nan,
            "KS_iso":  float(ks_iso.get(spec,  np.nan)) if not ks_iso.empty  else np.nan,
            "n": int(n_map.get(spec, 0)) if n_map else pd.NA,
            "model": r.get("model", "MC Dropout (posthoc)"),
            "split": "test",
            "source_file": r["__source__"]
        })
    out = pd.DataFrame(rows)
    out.to_csv("reliability_summary.csv", index=False)
    print("[✓] wrote reliability_summary.csv")
