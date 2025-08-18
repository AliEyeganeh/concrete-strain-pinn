# make_reliability_plots.py
# Creates PIT histograms and coverage curves for each specimen from the CSVs produced by
# finalize_results_and_reliability.py. Outputs PDFs under plots/reliability/.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RUN = r'out_scc_mcdo'  # <-- change if needed
OUTP = Path(RUN) / 'plots' / 'reliability'
OUTP.mkdir(parents=True, exist_ok=True)

def load_pit(specimen):
    df = pd.read_csv(os.path.join(RUN, f'pit_{specimen}.csv'))
    return df['u_temp'].dropna().values, df['u_iso'].dropna().values

def plot_pit(u, name, ks=None, fname='pit.pdf'):
    plt.figure()
    plt.hist(u, bins=20, range=(0,1), density=True, alpha=0.8, edgecolor='black')
    plt.axhline(1.0, linestyle='--')  # uniform density reference
    ttl = f'PIT Histogram — {name}'
    if ks is not None:
        ttl += f'  (KS={ks:.03f})'
    plt.title(ttl)
    plt.xlabel('u')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def load_coverage(specimen, mode):  # mode in {'temp','iso'}
    p = os.path.join(RUN, f'coverage_curve_{mode}_{specimen}.csv')
    return pd.read_csv(p)

def plot_coverage(cc_temp, cc_iso, name, fname='coverage.pdf'):
    plt.figure()
    plt.plot(cc_temp['level'], cc_temp['coverage'], marker='o', label='Temp')
    if cc_iso is not None:
        plt.plot(cc_iso['level'], cc_iso['coverage'], marker='s', label='Iso-after-Temp')
    xs = np.linspace(0.5, 0.99, 100)
    plt.plot(xs, xs, linestyle='--', label='Ideal')
    plt.xlabel('Nominal coverage level')
    plt.ylabel('Empirical coverage')
    plt.title(f'Coverage Curve — {name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

# Read reliability_summary for KS values (optional in titles)
rel = pd.read_csv(os.path.join(RUN, 'reliability_summary.csv'))
ks_map_temp = {r['specimen']: r['KS_temp'] for _, r in rel.iterrows()}
ks_map_iso  = {r['specimen']: r['KS_iso']  for _, r in rel.iterrows()}

# Discover specimens from the PIT files present
specimens = sorted([p.stem.replace('pit_','') for p in Path(RUN).glob('pit_*.csv')])

for s in specimens:
    # PIT plots
    u_temp, u_iso = load_pit(s)
    plot_pit(u_temp, f'{s} — Temp', ks_map_temp.get(s), OUTP / f'pit_temp_{s}.pdf')
    if len(u_iso) > 0 and not np.all(np.isnan(u_iso)):
        plot_pit(u_iso, f'{s} — Iso-after-Temp', ks_map_iso.get(s), OUTP / f'pit_iso_{s}.pdf')

    # Coverage curves
    cc_temp = load_coverage(s, 'temp')
    cc_iso  = None
    iso_path = Path(RUN) / f'coverage_curve_iso_{s}.csv'
    if iso_path.exists():
        cc_iso = pd.read_csv(iso_path)
    plot_coverage(cc_temp, cc_iso, s, OUTP / f'coverage_{s}.pdf')

print(f'[✓] Wrote PIT and coverage PDFs to: {OUTP.resolve()}')
