# PSBAF: Boiler Ash Fouling Prediction System

PSBAF is a graduation-design portfolio project for boiler ash fouling prediction. It integrates signal denoising, VMD decomposition, neural-network time-series forecasting, WOA-based hyperparameter optimization, ablation experiments, and a Flask visualization dashboard.

## Features

- Wavelet denoising for clean-factor time-series preprocessing.
- Variational Mode Decomposition (VMD) for multi-scale signal decomposition.
- TSMixer and TiDE forecasting models based on the Darts framework.
- Whale Optimization Algorithm (WOA) for global hyperparameter search.
- Ablation comparisons among TSMixer, TiDE, VMD, and WOA variants.
- Flask + AdminLTE + Pyecharts dashboard for interactive result visualization.

## Repository structure

```text
.
├── app.py                         # Flask dashboard entry point
├── data_cleaning.py               # Wavelet denoising utilities
├── data_decomposition.py          # VMD visualization script
├── data_decomposition_vmd.py      # VMD wrapper function
├── predict_*.py                   # Model training and prediction scripts
├── charts/                        # Pyecharts chart-rendering modules
├── templates/                     # Flask HTML templates
├── static/                        # Runtime static assets
├── *.csv                          # Saved prediction and optimization results
├── requirements.txt
└── .gitignore
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
python app.py
```

Default demo accounts:

| Role | Username | Password |
|---|---|---|
| User | `User` | `User` |
| Admin | `Admin` | `Admin` |

## Data note

The original `CF1.mat` file is not included in this GitHub-ready package by default. If you are allowed to publish the data, place it in the project root or adjust `DATA_FILE` in `data_cleaning.py`. For public repositories, a desensitized sample dataset is recommended.

## Main workflow

1. Run `data_cleaning.py` to inspect wavelet denoising results.
2. Run `data_decomposition.py` to inspect VMD decomposition and instantaneous frequency.
3. Run `predict_tsmixer.py`, `predict_tide.py`, and their WOA/VMD variants to generate prediction CSV files.
4. Run `app.py` to view the dashboard and ablation charts.

## Portfolio highlights

This project demonstrates practical skills in time-series data preprocessing, signal decomposition, neural-network forecasting, swarm-intelligence optimization, experimental comparison, and web-based result presentation.
