
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All-Specimens Regression + Uncertainty Pipeline
===============================================

What this script does (end-to-end):
1) Scans a data/ folder for CSV files (each is a specimen)
   - Expected columns (case-insensitive): load, displacement, and one strain column.
   - Strain column can be named 'strain', 'target', or 'strain_channel_51'.
   - Optional feature columns, if present: 'load_rate', 'temp', 'rate', etc.
2) Builds a clean dataset per specimen with standardized features/targets.
3) Trains a compact MLP (PyTorch) with MC Dropout for predictive uncertainty.
4) Calibrates uncertainty on the validation set using:
   (a) Temperature scaling of predictive std (scalar tau > 0 via NLL minimization).
   (b) Isotonic CDF calibration (Kuleshov et al., 2018) with invertible mapping for quantiles.
5) Evaluates on validation and test:
   - RMSE (raw & calibrated)
   - Coverage at 95% (raw & calibrated)
   - Average interval width (raw & calibrated)
   - CRPS (Gaussian closed-form; raw & calibrated)
6) Writes tidy CSVs:
   - out_{run_name}/posthoc_summary.csv
   - out_{run_name}/val_details_cal.csv
   - out_{run_name}/test_details_cal.csv
7) Emits a LaTeX table for point accuracy:
   - out_{run_name}/tab_point_acc_posthoc.tex

Usage (from terminal):
    python all_specimens_pipeline.py \
        --data_dir ./data \
        --run_name all_specs_mcdo \
        --test_size 0.15 \
        --val_size 0.15 \
        --epochs 800 \
        --hidden 128 \
        --dropout_p 0.2 \
        --mc_samples 100 \
        --seed 42

Notes:
- The script does NOT assume any specific specimen list; it loads all *.csv in --data_dir.
- If your strain column is named differently, the script auto-detects it. You can also force it via --target_col.
- The isotonic calibration in regression follows the "Probabilistic Calibration" approach:
  We fit an isotonic regressor f(u) to map nominal CDF u = Phi((y - mu)/sigma) to calibrated CDF.
  To compute calibrated quantiles, we use the pseudo-inverse f^{-1} numerically.
"""
import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# Utils
# -----------------------------

def set_seeds(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def find_strain_col(df: pd.DataFrame, preferred: str = None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    lower_cols = {c.lower(): c for c in df.columns}
    candidates = ["strain_channel_51", "target", "strain", "epsilon", "eps", "y"]
    for k in candidates:
        if k in lower_cols:
            return lower_cols[k]
    # heuristic: last column if numeric and not load/disp
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for c in reversed(numeric_cols):
        if c.lower() not in ("load", "displacement", "disp", "x1", "x2", "time"):
            return c
    raise ValueError("Could not identify target strain column. Please use --target_col.")

def detect_feature_cols(df: pd.DataFrame, target_col: str) -> List[str]:
    lc = {c.lower(): c for c in df.columns}
    # Core features
    feats = []
    if "load" in lc: feats.append(lc["load"])
    if "displacement" in lc: feats.append(lc["displacement"])
    elif "disp" in lc: feats.append(lc["disp"])
    # Optional known features if present
    for opt in ["load_rate", "rate", "temp", "temperature"]:
        if opt in lc:
            feats.append(lc[opt])
    # Fallback: any other numeric columns except target
    if len(feats) == 0:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        feats = [c for c in numeric_cols if c != target_col]
    return feats

def crps_gaussian(y, mu, sigma):
    """
    Closed-form CRPS for a Gaussian forecast N(mu, sigma^2).
    Reference: Matheson & Winkler (1976); Gneiting et al. (2005)
    """
    y = np.asarray(y).reshape(-1)
    mu = np.asarray(mu).reshape(-1)
    sigma = np.asarray(sigma).reshape(-1)
    sigma = np.clip(sigma, 1e-8, None)
    z = (y - mu) / sigma
    from math import sqrt, pi
    # Phi and phi of standard normal
    Phi = 0.5 * (1.0 + erf(z / math.sqrt(2)))
    phi = np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    crps = sigma * (z * (2 * Phi - 1) + 2 * phi - 1.0 / math.sqrt(np.pi))
    return crps

def erf(x):
    # numpy doesn't have erf in top-level without scipy; use math.erf elementwise
    # We'll vectorize math.erf
    import math as _m
    x = np.asarray(x)
    vec = np.vectorize(_m.erf)
    return vec(x)

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def coverage_width(y_true, mu, sigma, alpha=0.95):
    z = float(scipy_norm_ppf(0.5 + alpha/2.0))
    lower = mu - z * sigma
    upper = mu + z * sigma
    inside = ((y_true >= lower) & (y_true <= upper)).astype(float)
    cov = float(np.mean(inside))
    avg_width = float(np.mean(upper - lower))
    return cov, avg_width, lower, upper

def scipy_norm_ppf(p):
    # Inverse CDF for standard normal without SciPy.
    # Use a rational approximation by Peter J. Acklam (2003).
    # Source: https://web.archive.org/web/20150910044753/http://home.online.no/~pjacklam/notes/invnorm/
    # Valid for 0<p<1.
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("p must be in (0,1)")
    p = np.asarray(p)

    # Coefficients in rational approximations
    a = [ -3.969683028665376e+01,  2.209460984245205e+02,
          -2.759285104469687e+02,  1.383577518672690e+02,
          -3.066479806614716e+01,  2.506628277459239e+00 ]

    b = [ -5.447609879822406e+01,  1.615858368580409e+02,
          -1.556989798598866e+02,  6.680131188771972e+01,
          -1.328068155288572e+01 ]

    c = [ -7.784894002430293e-03, -3.223964580411365e-01,
          -2.400758277161838e+00, -2.549732539343734e+00,
           4.374664141464968e+00,  2.938163982698783e+00 ]

    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00 ]

    plow  = 0.02425
    phigh = 1 - plow

    x = np.zeros_like(p, dtype=float)

    # Lower region
    mask = p < plow
    if np.any(mask):
        q = np.sqrt(-2*np.log(p[mask]))
        x[mask] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                   ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)

    # Central region
    mask = (p >= plow) & (p <= phigh)
    if np.any(mask):
        q = p[mask] - 0.5
        r = q*q
        x[mask] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
                   (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

    # Upper region
    mask = p > phigh
    if np.any(mask):
        q = np.sqrt(-2*np.log(1-p[mask]))
        x[mask] = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                     ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)

    return x

# -----------------------------
# Model
# -----------------------------

class MLPDropout(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout_p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)

def train_mlp(
    X_tr, y_tr, X_va, y_va,
    hidden=128, dropout_p=0.2, lr=1e-3, epochs=800, patience=50, seed=42, device="cpu"
):
    set_seeds(seed)
    in_dim = X_tr.shape[1]
    model = MLPDropout(in_dim, hidden=hidden, dropout_p=dropout_p).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).view(-1,1).to(device)
    X_va_t = torch.tensor(X_va, dtype=torch.float32).to(device)
    y_va_t = torch.tensor(y_va, dtype=torch.float32).view(-1,1).to(device)

    best_va = float("inf")
    best_state = None
    bad = 0
    for ep in range(1, epochs+1):
        model.train()
        opt.zero_grad()
        pred = model(X_tr_t)
        loss = loss_fn(pred, y_tr_t)
        loss.backward()
        opt.step()

        # validation
        model.eval()
        with torch.no_grad():
            va_pred = model(X_va_t)
            va_loss = loss_fn(va_pred, y_va_t).item()

        if va_loss < best_va - 1e-9:
            best_va = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

@torch.no_grad()
def predict_mc_dropout(model, X, scaler_y, mc_samples=100, device="cpu"):
    model.eval()
    # enable dropout at inference for MC sampling
    def enable_dropout(m):
        if isinstance(m, nn.Dropout):
            m.train()
    model.apply(enable_dropout)

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    preds = []
    for _ in range(mc_samples):
        p = model(X_t).cpu().numpy().reshape(-1)
        preds.append(p)
    preds = np.stack(preds, axis=1)
    mu = preds.mean(axis=1)
    std = preds.std(axis=1, ddof=1)
    return mu, std

# -----------------------------
# Calibration
# -----------------------------

def fit_temperature_sigma(y_val, mu_val, sigma_val, max_iter=2000, lr=0.05):
    """
    Fit a scalar tau > 0 s.t. calibrated sigma' = tau * sigma minimizes Gaussian NLL on validation.
    """
    y = torch.tensor(y_val, dtype=torch.float32).view(-1,1)
    mu = torch.tensor(mu_val, dtype=torch.float32).view(-1,1)
    sigma = torch.tensor(sigma_val, dtype=torch.float32).view(-1,1).clamp_min(1e-6)

    log_tau = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([log_tau], lr=lr)
    for _ in range(max_iter):
        opt.zero_grad()
        tau = torch.exp(log_tau)
        sig = tau * sigma
        # Gaussian NLL
        nll = 0.5 * torch.log(2*torch.pi*sig**2) + 0.5*((y - mu)/sig)**2
        loss = nll.mean()
        loss.backward()
        opt.step()
    tau = float(torch.exp(log_tau).detach().cpu().numpy().reshape(()))
    return tau

class IsotonicCalibratorCDF:
    """
    Maps nominal CDF u in (0,1) to calibrated CDF f(u) via isotonic regression.
    Provides a numerical inverse for quantile mapping.
    """
    def __init__(self):
        self.iso = None
        self.grid_x = None
        self.grid_y = None

    def fit(self, u_val: np.ndarray, y_val: np.ndarray):
        # Targets for isotonic: empirical CDF indicators 1{Y <= quantile implied by u}
        # Using Kuleshov's approach: use pairs (u_i, I_i), where I_i ~ Bernoulli(u_i) in calibrated case.
        # Here we use the "CDF calibration" trick with labels z_i = 1{y_i <= q_i}, where q_i is such that Phi(z_i)=u_i.
        # In practice, we use u as feature and the indicator as label.
        # However, we don't know q_i directly; instead we use u_i from model's (y,mu,sigma) relation:
        # u_i = Phi((y_i - mu_i)/sigma_i). Then label is 1{U_i' <= u_i}, with U_i' ~ Uniform(0,1).
        # A cleaner method (Gupta et al., 2021) uses the PIT values u_i themselves as targets against uniform.
        # We adopt the PIT approach: fit isotonic to map u -> calibrated u' close to Uniform(0,1).
        u = u_val.reshape(-1)
        # To fit isotonic, we need pairs (x, y). We'll sort by u and use the empirical CDF as target.
        # For uniform target, the expected CDF at sorted ranks is linspace(0,1).
        order = np.argsort(u)
        x_sorted = u[order]
        y_uniform_cdf = np.linspace(0, 1, len(u), endpoint=False) + 0.5/len(u)  # mid-ranks
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds='clip')
        self.iso.fit(x_sorted, y_uniform_cdf)
        # Save a dense grid for inverse mapping
        self.grid_x = np.linspace(0, 1, 2001)
        self.grid_y = self.iso.predict(self.grid_x)

    def forward_cdf(self, u: np.ndarray) -> np.ndarray:
        if self.iso is None:
            raise RuntimeError("IsotonicCalibratorCDF not fitted.")
        return self.iso.predict(u)

    def inverse_cdf(self, v: np.ndarray) -> np.ndarray:
        """
        Approximate inverse f^{-1}(v). We use the precomputed grid (x -> y) and invert by interpolation.
        """
        if self.grid_x is None or self.grid_y is None:
            raise RuntimeError("Calibrator grid not prepared.")
        v = np.clip(v, 0.0, 1.0)
        # Since grid_y is monotone increasing, we can interpolate x as function of y.
        return np.interp(v, self.grid_y, self.grid_x)

# -----------------------------
# Data Loading & Splits
# -----------------------------

def load_all_specimens(data_dir: Path, target_col: str = None) -> Dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    files = sorted(list(data_dir.glob("*.csv")))
    if len(files) == 0:
        raise FileNotFoundError(f"No CSV files found in {data_dir.resolve()}")

    specimens = {}
    for f in files:
        try:
            df = pd.read_csv(f)
        except UnicodeDecodeError:
            df = pd.read_csv(f, encoding="latin1")
        # normalize columns
        df.columns = [c.strip() for c in df.columns]
        tcol = find_strain_col(df, preferred=target_col)
        feats = detect_feature_cols(df, tcol)
        keep = feats + [tcol]
        dff = df[keep].dropna().reset_index(drop=True)
        # rename target to 'strain' for consistency
        dff = dff.rename(columns={tcol: "strain"})
        specimens[f.stem] = dff
    return specimens

def split_data(df: pd.DataFrame, test_size=0.15, val_size=0.15, seed=42):
    idx = np.arange(len(df))
    idx_tr, idx_te = train_test_split(idx, test_size=test_size, random_state=seed, shuffle=True)
    # further split train into train/val
    rel_val = val_size / (1 - test_size)
    idx_tr2, idx_va = train_test_split(idx_tr, test_size=rel_val, random_state=seed+1, shuffle=True)
    return idx_tr2, idx_va, idx_te

def fit_tau_for_coverage(y_val, mu_val, sigma_val, target=0.95):
    z = float(scipy_norm_ppf(0.5 + target/2.0))
    lo, hi = 0.1, 10.0  # search range for tau
    for _ in range(40):  # bisection
        mid = 0.5*(lo+hi)
        lo_b = mu_val - z * (sigma_val * mid)
        hi_b = mu_val + z * (sigma_val * mid)
        cov = float(np.mean((y_val >= lo_b) & (y_val <= hi_b)))
        if cov < target:  # need wider intervals
            lo = mid
        else:
            hi = mid
    return 0.5*(lo+hi)


# -----------------------------
# Main experiment per specimen
# -----------------------------

def run_one_specimen(df: pd.DataFrame, specimen: str, args, outdir: Path, device: str = "cpu") -> Dict:
    # Split
    idx_tr, idx_va, idx_te = split_data(df, args.test_size, args.val_size, args.seed)
    feat_cols = [c for c in df.columns if c != "strain"]
    X = df[feat_cols].values.astype(np.float32)
    y = df["strain"].values.astype(np.float32)

    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_va, y_va = X[idx_va], y[idx_va]
    X_te, y_te = X[idx_te], y[idx_te]

    # Scale
    x_scaler = StandardScaler().fit(X_tr)
    y_scaler = StandardScaler().fit(y_tr.reshape(-1,1))
    X_tr_s = x_scaler.transform(X_tr)
    X_va_s = x_scaler.transform(X_va)
    X_te_s = x_scaler.transform(X_te)
    y_tr_s = y_scaler.transform(y_tr.reshape(-1,1)).reshape(-1)
    y_va_s = y_scaler.transform(y_va.reshape(-1,1)).reshape(-1)
    y_te_s = y_scaler.transform(y_te.reshape(-1,1)).reshape(-1)

    # Train
    model = train_mlp(
        X_tr_s, y_tr_s, X_va_s, y_va_s,
        hidden=args.hidden, dropout_p=args.dropout_p, lr=args.lr,
        epochs=args.epochs, patience=args.patience, seed=args.seed, device=device
    )

    # Predict with MC Dropout (standardized space)
    mu_va_s, sd_va_s = predict_mc_dropout(model, X_va_s, y_scaler, mc_samples=args.mc_samples, device=device)
    mu_te_s, sd_te_s = predict_mc_dropout(model, X_te_s, y_scaler, mc_samples=args.mc_samples, device=device)

    # Inverse scale
    mu_va = y_scaler.inverse_transform(mu_va_s.reshape(-1,1)).reshape(-1)
    mu_te = y_scaler.inverse_transform(mu_te_s.reshape(-1,1)).reshape(-1)
    # For std: multiply by target scaler's std
    sigma_scale = float(y_scaler.scale_.reshape(()))
    sd_va = sd_va_s * sigma_scale
    sd_te = sd_te_s * sigma_scale

    # Raw metrics
    rmse_va_raw = rmse(y_va, mu_va)
    rmse_te_raw = rmse(y_te, mu_te)
    cov_va_raw, wid_va_raw, lo_va_raw, hi_va_raw = coverage_width(y_va, mu_va, sd_va, alpha=0.95)
    cov_te_raw, wid_te_raw, lo_te_raw, hi_te_raw = coverage_width(y_te, mu_te, sd_te, alpha=0.95)
    crps_va_raw = float(np.mean(crps_gaussian(y_va, mu_va, sd_va)))
    crps_te_raw = float(np.mean(crps_gaussian(y_te, mu_te, sd_te)))

        # === Calibration ===
    # Temperature calibration on VAL (coverage-targeted τ)
    tau = fit_tau_for_coverage(y_va, mu_va, sd_va, target=0.95)
    sd_va_cal = sd_va * tau
    sd_te_cal = sd_te * tau

    # Means don’t change under temp scaling
    rmse_va_cal_temp = rmse_va_raw
    rmse_te_cal_temp = rmse_te_raw
    cov_va_cal_temp, wid_va_cal_temp, lo_va_cal_temp, hi_va_cal_temp = coverage_width(y_va, mu_va, sd_va_cal, 0.95)
    cov_te_cal_temp, wid_te_cal_temp, lo_te_cal_temp, hi_te_cal_temp = coverage_width(y_te, mu_te, sd_te_cal, 0.95)
    crps_va_cal_temp = float(np.mean(crps_gaussian(y_va, mu_va, sd_va_cal)))
    crps_te_cal_temp = float(np.mean(crps_gaussian(y_te, mu_te, sd_te_cal)))

    # Isotonic CDF calibration (fit on VAL, AFTER temp scaling)
    z_val_temp = (y_va - mu_va) / np.clip(sd_va_cal, 1e-8, None)
    u_val_temp = 0.5 * (1 + erf(z_val_temp / np.sqrt(2.0)))
    iso = IsotonicCalibratorCDF()
    iso.fit(u_val_temp, y_va)

    # 95% bounds via isotonic (apply to both VAL and TEST)
    z95 = float(scipy_norm_ppf(0.975))
    u_low_nom  = 0.5 * (1 + erf((-z95) / np.sqrt(2.0)))
    u_high_nom = 0.5 * (1 + erf(( z95) / np.sqrt(2.0)))
    u_low_iso  = iso.inverse_cdf(np.array([u_low_nom]))[0]
    u_high_iso = iso.inverse_cdf(np.array([u_high_nom]))[0]
    z_low_iso  = scipy_norm_ppf(u_low_iso)
    z_high_iso = scipy_norm_ppf(u_high_iso)

    lo_va_iso = mu_va + z_low_iso  * sd_va_cal
    hi_va_iso = mu_va + z_high_iso * sd_va_cal
    lo_te_iso = mu_te + z_low_iso  * sd_te_cal
    hi_te_iso = mu_te + z_high_iso * sd_te_cal

    # Calibrated quantiles via inverse mapping for 95% PI
    z95 = float(scipy_norm_ppf(0.975))
    # For each sample, nominal quantiles => u_low/u_high
    def calibrated_bounds(mu, sd):
        u_low_nom  = 0.5*(1 + erf((-z95)/np.sqrt(2.0)))
        u_high_nom = 0.5*(1 + erf(( z95)/np.sqrt(2.0)))
        # Invert isotonic to get calibrated u*
        u_low_cal  = iso.inverse_cdf(np.array([u_low_nom]))[0]
        u_high_cal = iso.inverse_cdf(np.array([u_high_nom]))[0]
        # Map back to z* via Phi^{-1}
        z_low_cal  = scipy_norm_ppf(u_low_cal)
        z_high_cal = scipy_norm_ppf(u_high_cal)
        lower = mu + z_low_cal  * sd
        upper = mu + z_high_cal * sd
        return lower, upper

    lo_va_iso, hi_va_iso = calibrated_bounds(mu_va, sd_va_cal)  # chain temp->iso (recommended)
    lo_te_iso, hi_te_iso = calibrated_bounds(mu_te, sd_te_cal)

    cov_va_iso = float(np.mean((y_va >= lo_va_iso) & (y_va <= hi_va_iso)))
    cov_te_iso = float(np.mean((y_te >= lo_te_iso) & (y_te <= hi_te_iso)))
    wid_va_iso = float(np.mean(hi_va_iso - lo_va_iso))
    wid_te_iso = float(np.mean(hi_te_iso - lo_te_iso))

    # Assemble per-row details
    val_details = pd.DataFrame({
        "specimen": specimen,
        "mu_raw": mu_va, "sd_raw": sd_va,
        "mu_temp": mu_va, "sd_temp": sd_va_cal,
        "lo95_raw": lo_va_raw, "hi95_raw": hi_va_raw,
        "lo95_temp": lo_va_cal_temp, "hi95_temp": hi_va_cal_temp,
        "lo95_iso": lo_va_iso, "hi95_iso": hi_va_iso,
        "y": y_va
    })
    test_details = pd.DataFrame({
        "specimen": specimen,
        "mu_raw": mu_te, "sd_raw": sd_te,
        "mu_temp": mu_te, "sd_temp": sd_te_cal,
        "lo95_raw": lo_te_raw, "hi95_raw": hi_te_raw,
        "lo95_temp": lo_te_cal_temp, "hi95_temp": hi_te_cal_temp,
        "lo95_iso": lo_te_iso, "hi95_iso": hi_te_iso,
        "y": y_te
    })

    # Summary
    sumrows = []
    sumrows.append({
        "specimen": specimen, "method": "raw",
        "rmse_val": rmse_va_raw, "rmse_test": rmse_te_raw,
        "coverage95_val": cov_va_raw, "avg_width_val": wid_va_raw,
        "coverage95_test": cov_te_raw, "avg_width_test": wid_te_raw,
        "crps_val": crps_va_raw, "crps_test": crps_te_raw,
        "extra": json.dumps({"tau": 1.0})
    })
    sumrows.append({
        "specimen": specimen, "method": "temp-scale",
        "rmse_val": rmse_va_cal_temp, "rmse_test": rmse_te_cal_temp,
        "coverage95_val": cov_va_cal_temp, "avg_width_val": wid_va_cal_temp,
        "coverage95_test": cov_te_cal_temp, "avg_width_test": wid_te_cal_temp,
        "crps_val": crps_va_cal_temp, "crps_test": crps_te_cal_temp,
        "extra": json.dumps({"tau": tau})
    })
    sumrows.append({
        "specimen": specimen, "method": "temp+iso",
        "rmse_val": rmse_va_cal_temp, "rmse_test": rmse_te_cal_temp,
        "coverage95_val": cov_va_iso, "avg_width_val": wid_va_iso,
        "coverage95_test": cov_te_iso, "avg_width_test": wid_te_iso,
        "crps_val": np.nan, "crps_test": np.nan,
        "extra": json.dumps({"tau": tau, "iso_points": int(len(iso.grid_x))})
    })
    summary = pd.DataFrame(sumrows)

    # Write
    ensure_dir(outdir)
    # Append mode to allow multi-specimen accumulation outside caller if needed
    val_details.to_csv(outdir / f"{specimen}_val_details_cal.csv", index=False)
    test_details.to_csv(outdir / f"{specimen}_test_details_cal.csv", index=False)
    summary.to_csv(outdir / f"{specimen}_posthoc_summary.csv", index=False)
    return {
        "summary": summary,
        "val_details": val_details,
        "test_details": test_details,
        "feat_cols": feat_cols
    }

def make_latex_point_table(all_summary: pd.DataFrame, out_path: Path):
    # Keep RMSE (test) for methods; prefer calibrated variant for interval metrics; but for point accuracy RMSE is same for temp/iso
    # We'll display raw RMSE (test) per specimen.
    pivot = all_summary[all_summary["method"]=="raw"].pivot_table(
        index="specimen",
        values=["rmse_val", "rmse_test"],
        aggfunc="first"
    ).reset_index()
    # Build LaTeX
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Point prediction accuracy (RMSE, $\mu\varepsilon$). Lower is better.}")
    lines.append(r"\label{tab:point_acc}")
    lines.append(r"\begin{tabular}{lrr}")
    lines.append(r"\toprule")
    lines.append(r"Specimen & RMSE (Val) & RMSE (Test) \\")
    lines.append(r"\midrule")
    for _, row in pivot.iterrows():
        lines.append(f"{row['specimen']} & {row['rmse_val']:.2f} & {row['rmse_test']:.2f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines), encoding="utf-8")

# -----------------------------
# Orchestrator
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Folder with CSVs (one per specimen).")
    parser.add_argument("--run_name", type=str, default="out_all_specs")
    parser.add_argument("--target_col", type=str, default=None, help="Explicit target column if needed.")
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)

    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout_p", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mc_samples", type=int, default=100)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    set_seeds(args.seed)
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    data_dir = Path(args.data_dir)
    outdir = Path(f"./out_{args.run_name}")
    ensure_dir(outdir)

    specimens = load_all_specimens(data_dir, target_col=args.target_col)

    all_summaries = []
    for spec_name, df in specimens.items():
        print(f"[+] Training on specimen: {spec_name}  (n={len(df)})")
        res = run_one_specimen(df, spec_name, args, outdir, device=device)
        # Keep summary
        all_summaries.append(res["summary"])

    all_summary = pd.concat(all_summaries, ignore_index=True)
    all_summary.to_csv(outdir / "posthoc_summary.csv", index=False)

    # Merge all per-row details for convenience
    va_files = list(outdir.glob("*_val_details_cal.csv"))
    te_files = list(outdir.glob("*_test_details_cal.csv"))
    if len(va_files):
        pd.concat([pd.read_csv(f) for f in va_files], ignore_index=True).to_csv(outdir / "val_details_cal.csv", index=False)
    if len(te_files):
        pd.concat([pd.read_csv(f) for f in te_files], ignore_index=True).to_csv(outdir / "test_details_cal.csv", index=False)

    # LaTeX table
    make_latex_point_table(all_summary, outdir / "tab_point_acc_posthoc.tex")
    # === NEW: Reliability summary + PIT histograms (written by the pipeline) ===
    import math

    def _phi_cdf(z):
        z = np.asarray(z, dtype=float)
        return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))

    def _ks_uniform01(u):
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

    # Load the merged summary & details (they were just written above)
    posthoc_path = outdir / "posthoc_summary.csv"
    test_det_path = outdir / "test_details_cal.csv"

    if posthoc_path.exists() and test_det_path.exists():
        post = pd.read_csv(posthoc_path)
        det  = pd.read_csv(test_det_path)

        # Compute PIT (raw & temp-scale) per-row on the TEST split
        # Your details file has: y, mu_raw, sd_raw, mu_temp (== mu_raw), sd_temp
        z_raw  = (det["y"].to_numpy(dtype=float) - det["mu_raw"].to_numpy(dtype=float))  / np.maximum(det["sd_raw"].to_numpy(dtype=float),  1e-12)
        z_temp = (det["y"].to_numpy(dtype=float) - det["mu_temp"].to_numpy(dtype=float)) / np.maximum(det["sd_temp"].to_numpy(dtype=float), 1e-12)
        u_raw  = _phi_cdf(z_raw)
        u_temp = _phi_cdf(z_temp)

        det = det.copy()
        det["pit_raw"]  = u_raw
        det["pit_temp"] = u_temp

        # KS per specimen
        ks_raw_by_spec  = det.groupby("specimen")["pit_raw"].apply(_ks_uniform01)
        ks_temp_by_spec = det.groupby("specimen")["pit_temp"].apply(_ks_uniform01)

        # Select the temp-scale row from your posthoc summary as the reliability line
        # (CRPS is defined for raw & temp-scale; temp+iso has NaN CRPS by design.)
        rel_rows = []
        for spec, g in post.groupby("specimen"):
            sel = g[g["method"] == "temp-scale"]
            if sel.empty:
                # fall back to raw if temp-scale not present
                sel = g[g["method"] == "raw"]
            if sel.empty:
                continue
            row = sel.iloc[0]
            rel_rows.append({
                "specimen": spec,
                "coverage_95": float(row.get("coverage95_test", np.nan)),
                "crps":        float(row.get("crps_test",       np.nan)),
                "rmse":        float(row.get("rmse_test",       np.nan)),
                "KS_temp":     float(ks_temp_by_spec.get(spec, np.nan)),
                "KS_raw":      float(ks_raw_by_spec.get(spec,  np.nan)),
                "n":           int(det[det["specimen"] == spec].shape[0]) if "specimen" in det.columns else np.nan,
                "model":       "MC Dropout (posthoc)",
                "split":       "test",
                "source_file": str(posthoc_path)
            })

        rel_df = pd.DataFrame(rel_rows)
        rel_df.to_csv(outdir / "reliability_summary.csv", index=False)

        # PIT histograms (20 bins across [0,1]) for RAW and TEMP
        bins = np.linspace(0.0, 1.0, 21)
        hist_rows = []
        for spec, g in det.groupby("specimen"):
            # raw
            counts_raw, _ = np.histogram(g["pit_raw"].to_numpy(dtype=float), bins=bins)
            # temp
            counts_temp, _ = np.histogram(g["pit_temp"].to_numpy(dtype=float), bins=bins)
            for i in range(len(bins)-1):
                hist_rows.append({
                    "specimen": spec,
                    "bin_left": float(bins[i]),
                    "bin_right": float(bins[i+1]),
                    "count": int(counts_raw[i]),
                    "density": float(counts_raw[i] / max(1, len(g))),
                    "mode": "raw",
                    "split": "test"
                })
                hist_rows.append({
                    "specimen": spec,
                    "bin_left": float(bins[i]),
                    "bin_right": float(bins[i+1]),
                    "count": int(counts_temp[i]),
                    "density": float(counts_temp[i] / max(1, len(g))),
                    "mode": "temp",
                    "split": "test"
                })

        pit_df = pd.DataFrame(hist_rows)
        # write two convenience files for plotting
        pit_df[pit_df["mode"] == "temp"].to_csv(outdir / "pit_hist_test_temp.csv", index=False)
        pit_df[pit_df["mode"] == "raw"].to_csv(outdir / "pit_hist_test_raw.csv", index=False)
    else:
        print("[i] Skipping reliability/PIT export: missing posthoc_summary.csv or test_details_cal.csv")

    # (Optional) Expand the final print so you see the new files:
    have_rel = (outdir / "reliability_summary.csv").exists()
    have_ht  = (outdir / "pit_hist_test_temp.csv").exists()
    print(f"[✓] Wrote: {outdir/'posthoc_summary.csv'}  {outdir/'test_details_cal.csv'}  "
          f"{outdir/'tab_point_acc_posthoc.tex'}"
          f"{'  ' + str(outdir/'reliability_summary.csv') if have_rel else ''}"
          f"{'  ' + str(outdir/'pit_hist_test_temp.csv') if have_ht else ''}")


    print(f"[✓] Wrote: {outdir/'posthoc_summary.csv'}  {outdir/'val_details_cal.csv'}  {outdir/'test_details_cal.csv'}  {outdir/'tab_point_acc_posthoc.tex'}")

if __name__ == "__main__":
    main()
