import pandas as pd
from pathlib import Path

csv = Path("out_ecc_scc_mcdo") / "reliability_summary.csv"
df = pd.read_csv(csv)
if "KS_iso" not in df.columns:
    df["KS_iso"] = float("nan")
df.to_csv(csv, index=False)
print(f"[✓] Patched {csv} (added KS_iso)")
