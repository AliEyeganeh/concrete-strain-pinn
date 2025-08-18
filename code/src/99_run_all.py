# 99_run_all.py
import sys, subprocess, shutil
from pathlib import Path

"""
Run this inside a run folder *after training*, for one specimen:
    python 99_run_all.py ECC_SCC
It will call:
  - 00_build_reliability.py
  - 10_build_pit_and_curves.py
  - 20_build_hist_and_aliases.py
  - 30_plot_one_specimen.py <SPECIMEN>
"""

SCRIPTS = [
    "00_build_reliability.py",
    "10_build_pit_and_curves.py",
    "20_build_hist_and_aliases.py",
]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python 99_run_all.py <SPECIMEN>")
    SPEC = sys.argv[1]
    here = Path(".")
    # sanity: ensure plotter exists somewhere
    if not (here/"make_reliability_plots.py").exists() and not (here/"../make_reliability_plots.py").exists():
        print("[i] make_reliability_plots.py not found in this folder; looking one level up...")
        if not (here/"../make_reliability_plots.py").exists():
            print("[!] Warning: plotting step will fail unless you add make_reliability_plots.py here or in the parent folder.")

    # run builders
    for s in SCRIPTS:
        if not (here/s).exists():
            raise SystemExit(f"Missing helper script: {s}")
        print(f"[→] python {s}")
        r = subprocess.call(["python", s])
        if r != 0:
            raise SystemExit(r)

    # plot one specimen
    if not (here/"30_plot_one_specimen.py").exists():
        raise SystemExit("Missing 30_plot_one_specimen.py")
    print(f"[→] python 30_plot_one_specimen.py {SPEC}")
    r = subprocess.call(["python", "30_plot_one_specimen.py", SPEC])
    if r != 0:
        raise SystemExit(r)
    print("[✓] Done.")
