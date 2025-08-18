# Concrete Strain Prediction with Physics-Informed Neural Networks (PINNs)

This repository contains the LaTeX manuscript and code for strain prediction in reinforced concrete joints using PINNs, deep ensembles, and uncertainty quantification.

## Repository Layout
- `paper/` — LaTeX source for the manuscript (`pa1-1.tex`), references, and figures.
- `code/` — Reproducible Python code and notebooks.
- `data/` — Placeholder only (no large files in Git). Consider Zenodo/OSF for datasets.
- `env/` — Environment files (e.g., `environment.yml` or `requirements.txt`).

## Reproduce
```bash
# LaTeX (example)
# latexmk -pdf -interaction=nonstopmode paper/pa1-1.tex

# Python (example)
# python -m venv .venv && source .venv/bin/activate
# pip install -r env/requirements.txt
# jupyter lab
