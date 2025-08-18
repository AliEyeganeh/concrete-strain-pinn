# 30_plot_one_specimen.py — one-command, per-folder, per-specimen runner
import sys, os, re, shutil, subprocess, math, glob
from pathlib import Path
import numpy as np
import pandas as pd

# -------------------- small utils --------------------
def phi_cdf(z):
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))

def pit_from_mu_sigma(y, mu, sd):
    sd = np.maximum(sd, 1e-12)
    return phi_cdf((y - mu) / sd)

def ecdf_map(u):
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

def coverage_curve_from_pit(u, levels):
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

# -------------------- preflight builders --------------------
def ensure_reliability(run: Path):
    rel = run / "reliability_summary.csv"
    if rel.exists():
        return
    # Build a minimal reliability_summary from any *posthoc*summary* + details we can find
    post_files = [Path(p) for p in glob.glob(str(run / "**/*posthoc*summary*.csv"), recursive=True)]
    rows = []
    if post_files:
        for p in post_files:
            try:
                df = pd.read_csv(p)
                if "specimen" not in df.columns:
                    # infer from filename
                    spec = p.stem.split("_posthoc")[0].split("_")[-1].upper()
                    df["specimen"] = spec
                # prefer temp-scale, else raw, else first row per specimen
                if "method" in df.columns:
                    k = df["method"].astype(str).str.lower().str.replace("_","").str.replace("+","", regex=False)
                    df["__mn__"] = np.where(k.str.contains("tempscale") | (k.str.contains("temp") & k.str.contains("scale")), 0,
                                    np.where(k.eq("raw"), 1, 2))
                    df = df.sort_values(["specimen","__mn__"]).groupby("specimen").head(1)
                for _, r in df.iterrows():
                    rows.append({
                        "specimen": r["specimen"],
                        "coverage_95": float(r[r.index[r.index.str.fullmatch(r"coverage.*test.*", case=False)].tolist()[0]]) if any(df.columns.str.fullmatch(r"coverage.*test.*", case=False)) else np.nan,
                        "crps": float(r[r.index[r.index.str.fullmatch(r"crps.*test.*", case=False)].tolist()[0]]) if any(df.columns.str.fullmatch(r"crps.*test.*", case=False)) else np.nan,
                        "rmse": float(r[r.index[r.index.str.fullmatch(r"rmse.*test.*", case=False)].tolist()[0]]) if any(df.columns.str.fullmatch(r"rmse.*test.*", case=False)) else np.nan,
                        "model": r.get("model","MC Dropout (posthoc)"),
                        "split": "test",
                        "source_file": str(p)
                    })
            except Exception as e:
                print(f"[i] skip {p.name}: {e}")
    else:
        # last resort: pull specimen list from test_details_cal
        det = list(run.glob("*test*_details_cal*.csv"))
        if not det:
            raise SystemExit("No reliability_summary.csv and no posthoc/details files to infer from.")
        d = pd.read_csv(det[0], usecols=["specimen"])
        for s in sorted(d["specimen"].astype(str).unique()):
            rows.append({"specimen": s, "coverage_95": np.nan, "crps": np.nan, "rmse": np.nan,
                         "model": "MC Dropout (posthoc)", "split":"test", "source_file": str(det[0])})
    pd.DataFrame(rows).to_csv(rel, index=False)
    print("[✓] built reliability_summary.csv (minimal)")

def ensure_pit_and_curves_for_spec(run: Path, SPEC: str):
    pit = run / f"pit_{SPEC}.csv"
    need_build = True
    if pit.exists():
        df = pd.read_csv(pit)
        if "u_temp" in df.columns and "u_iso" in df.columns and not df["u_temp"].empty:
            need_build = False
    if not need_build:
        # still ensure coverage curves
        ensure_curves_from_pit(run, SPEC, df["u_temp"].to_numpy(float), df["u_iso"].to_numpy(float))
        return

    # Build using validation→test
    test_path = run / "test_details_cal.csv"
    if not test_path.exists():
        # try per-specimen files
        cand = list(run.glob(f"*{SPEC}*test*_details_cal*.csv"))
        if cand:
            test_path = cand[0]
        else:
            raise SystemExit(f"Missing test_details_cal.csv to build PIT for {SPEC}.")

    test = pd.read_csv(test_path)
    # Filter specimen
    if "specimen" not in test.columns:
        raise SystemExit(f"'specimen' column not in {test_path}")
    test = test[test["specimen"].astype(str) == SPEC]
    if test.empty:
        raise SystemExit(f"No rows for specimen {SPEC} in {test_path}")

    # Columns (flexible)
    c_y    = pick(test, "y","y_true","target")
    c_mu_t = pick(test, "mu_temp","mu_raw","pred_mean","yhat","mu")
    c_sd_t = pick(test, "sd_temp","sd_raw","pred_std","yhat_std","sd","sigma")
    if not all([c_y, c_mu_t, c_sd_t]):
        raise SystemExit(f"Need y, mu_temp/mu_raw, sd_temp/sd_raw in {test_path}. Got: {list(test.columns)}")

    u_temp = pit_from_mu_sigma(test[c_y].to_numpy(float),
                               test[c_mu_t].to_numpy(float),
                               test[c_sd_t].to_numpy(float))

    # Isotonic from validation (prefer per-specimen)
    val_path = run / "val_details_cal.csv"
    if val_path.exists():
        val = pd.read_csv(val_path)
        if "specimen" in val.columns:
            val = val[val["specimen"].astype(str) == SPEC]
        c_yv    = pick(val, "y","y_true","target")
        c_mu_tv = pick(val, "mu_temp","mu_raw","pred_mean","yhat","mu")
        c_sd_tv = pick(val, "sd_temp","sd_raw","pred_std","yhat_std","sd","sigma")
        if val is not None and all([c_yv,c_mu_tv,c_sd_tv]) and not val.empty:
            u_val = pit_from_mu_sigma(val[c_yv].to_numpy(float),
                                      val[c_mu_tv].to_numpy(float),
                                      val[c_sd_tv].to_numpy(float))
            f_iso = ecdf_map(u_val)
        else:
            f_iso = ecdf_map(u_temp)
    else:
        f_iso = ecdf_map(u_temp)

    u_iso = f_iso(u_temp) if f_iso is not None else np.copy(u_temp)
    pd.DataFrame({"u_temp": u_temp, "u_iso": u_iso}).to_csv(pit, index=False)
    print(f"[✓] wrote {pit.name}")

    ensure_curves_from_pit(run, SPEC, u_temp, u_iso)

def ensure_curves_from_pit(run: Path, SPEC: str, u_temp: np.ndarray, u_iso: np.ndarray):
    levels = np.linspace(0.50, 0.99, 50)
    cct = run / f"coverage_curve_temp_{SPEC}.csv"
    cci = run / f"coverage_curve_iso_{SPEC}.csv"
    if not cct.exists():
        pd.DataFrame({"level": levels, "coverage": coverage_curve_from_pit(u_temp, levels)}).to_csv(cct, index=False)
        print(f"[✓] wrote {cct.name}")
    if not cci.exists():
        pd.DataFrame({"level": levels, "coverage": coverage_curve_from_pit(u_iso, levels)}).to_csv(cci, index=False)
        print(f"[✓] wrote {cci.name}")

def ensure_global_hist_and_aliases(run: Path):
    # concat all PIT
    u_t_all, u_i_all = [], []
    for p in run.glob("pit_*.csv"):
        df = pd.read_csv(p)
        if "u_temp" in df: u_t_all.append(df["u_temp"].to_numpy(float))
        if "u_iso"  in df: u_i_all.append(df["u_iso"].to_numpy(float))
    if not u_t_all:
        return
    u_t_all = np.concatenate(u_t_all)
    u_i_all = np.concatenate(u_i_all) if u_i_all else np.array([])
    levels = np.linspace(0.50, 0.99, 50)
    t_raw = run / "coverage_curve_temp_hist_test_raw.csv"
    i_raw = run / "coverage_curve_iso_hist_test_raw.csv"
    if not t_raw.exists():
        pd.DataFrame({"level": levels, "coverage": coverage_curve_from_pit(u_t_all, levels)}).to_csv(t_raw, index=False)
    if not i_raw.exists():
        pd.DataFrame({"level": levels, "coverage": coverage_curve_from_pit(u_i_all, levels)}).to_csv(i_raw, index=False)
    # aliases some plotters ask for
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

# -------------------- plotter patching --------------------
PLOTTER_CANDIDATES = ["make_reliability_plots.py","../make_reliability_plots.py","../../make_reliability_plots.py"]

def find_plotter() -> Path:
    for cand in PLOTTER_CANDIDATES:
        p = Path(cand)
        if p.exists(): return p.resolve()
    raise SystemExit("Could not find make_reliability_plots.py. Put it here or one/two levels up.")

def patch_plotter(plotter_path: Path, run_dir: Path) -> Path:
    """
    Force the plotter to use RUN='.' and make load_pit tolerant:
      - define rRUN/RUN at the top
      - comment out ANY 'RUN = ...' lines (incl. 'RUN = rRUN')
      - replace stray rRUN tokens with RUN
      - rewrite out_* hard-coded joins/literals to use RUN
      - REPLACE the plotter's load_pit() with a tolerant version
    """
    import re
    src = plotter_path.read_text(encoding="utf-8", errors="ignore")

    # Comment out every RUN assignment (handles 'RUN = rRUN' and literals)
    src = re.sub(r'(?m)^\s*RUN\s*=\s*.*$', r'# (patched out) \g<0>', src)
    # Replace stray rRUN tokens with RUN
    src = re.sub(r'\brRUN\b', 'RUN', src)
    # Rewrite os.path.join("out_...", ...) to os.path.join(RUN, ...)
    src = re.sub(r"os\.path\.join\(\s*(['\"])out_[^'\"]+\1\s*,", r"os.path.join(RUN,", src)
    # Rewrite bare "out_..." literals to RUN
    src = re.sub(r"(['\"])out_[^'\"]+\1", "RUN", src)

    # Inject our RUN override + needed imports at the very top
    inject = (
        "import os, sys\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "rRUN = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()\n"
        "RUN  = rRUN\n"
        "print(f\"[plots] Using RUN folder: {RUN}\")\n"
    )
    patched = inject + src

    # Replace the plotter's load_pit() with a tolerant one.
    tolerant_load_pit = r'''
def load_pit(specimen):
    import os, numpy as np, pandas as pd
    p = os.path.join(RUN, f"pit_{specimen}.csv")
    df = pd.read_csv(p)
    # normalize headers
    norm = {c.strip().lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            key = n.lower()
            if key in norm:
                return norm[key]
        return None
    # choose u_temp-like column, or first numeric as fallback
    c_ut = pick("u_temp","pit_temp","temp","u","pit")
    if c_ut is None:
        num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if not num_cols:
            raise ValueError(f"{p} has no numeric columns; can't infer u_temp")
        c_ut = num_cols[0]
    # choose u_iso-like column, or synthesize
    c_ui = pick("u_iso","pit_iso","iso")
    if c_ui is None:
        df["u_iso"] = np.nan
        c_ui = "u_iso"
    return df[c_ut].dropna().values, df[c_ui].dropna().values
'''
    # Try to replace an existing definition; if not found, append ours.
    if re.search(r"(?s)def\s+load_pit\s*\(.*?\):", patched):
        patched = re.sub(r"(?s)def\s+load_pit\s*\(.*?\):.*?(?=\ndef\s|\n#|\Z)", tolerant_load_pit.strip(), patched, count=1)
    else:
        patched += "\n" + tolerant_load_pit + "\n"

    tmp = run_dir / "_patched_make_reliability_plots.py"
    tmp.write_text(patched, encoding="utf-8")
    return tmp

# -------------------- main --------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python 30_plot_one_specimen.py <SPECIMEN>")
    SPEC = sys.argv[1]

    run = Path(".").resolve()

    # 0) ensure reliability exists (or build minimal one)
    ensure_reliability(run)

    # 1) ensure PIT + per-specimen coverage exist for this SPEC (and are well-formed)
    ensure_pit_and_curves_for_spec(run, SPEC)

    # 2) ensure global hist + aliases exist
    ensure_global_hist_and_aliases(run)

    # 3) filter reliability_summary.csv to this SPEC (backup + restore)
    rel = run / "reliability_summary.csv"
    bak = run / "reliability_summary.ALL.csv"
    if not bak.exists():
        shutil.copyfile(rel, bak)
    df_all = pd.read_csv(bak)
    if "specimen" not in df_all.columns:
        raise SystemExit("reliability_summary.csv missing 'specimen' column.")
    df_one = df_all[df_all["specimen"].astype(str) == SPEC]
    if df_one.empty:
        raise SystemExit(f"Specimen '{SPEC}' not found. Available: {df_all['specimen'].unique()}")
    df_one.to_csv(rel, index=False)

    # 4) patch and run the plotter against this folder
    plotter = find_plotter()
    patched = patch_plotter(plotter, run)

    try:
        print(f"[→] python {patched.name} .")
        code = subprocess.call(["python", str(patched), "."])
        if code != 0:
            raise SystemExit(code)
        print(f"[✓] Plotted {SPEC} from {run.name}")
    finally:
        shutil.copyfile(bak, rel)
        try: patched.unlink()
        except Exception: pass
