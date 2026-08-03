# Network Traffic Anomaly Detection with a Dense Autoencoder

Detects 24-hour base-station traffic profiles that deviate from "normal" behavior, using reconstruction error from a dense autoencoder as the anomaly score.

> **Note:** the original dataset is confidential and has been removed from this repo (including all real values/site IDs previously embedded in notebook outputs and result plots). This repo is being migrated to a synthetically generated dataset with injected, labeled anomalies so the methodology remains fully reproducible and results below can be regenerated end-to-end without any proprietary data. Results, plots, and file references below are being updated accordingly.

## Problem

A telecom network exposes hourly traffic volume for 743 base stations, each summarized as a 24-value daily profile (one value per hour, 0–23). The goal is to flag stations whose daily traffic *shape* is structurally unusual — e.g. an unexpected spike or drop pattern — with no labeled examples of "anomalous" behavior available to train against.

## Approach

1. **Normalize** each profile by its own daily mean, so the model learns hourly *shape* rather than absolute traffic volume. This puts small and large stations on equal footing.
2. **Split** into train/validation (80/20) and run a **grid search** over 6 encoder architectures × 2 learning rates × 3 random seeds, selecting the combination with the lowest mean validation MSE.
3. **Retrain** the selected architecture with early stopping; cache the trained weights and training history to disk so the notebook doesn't retrain on every run.
4. **Score** every station by reconstruction error (RMSE between actual and reconstructed profile).
5. **Threshold** at the 95th percentile of the *training-set* error distribution (assumed to represent typical behavior) and flag stations above it as anomalies.
6. **Validate visually**: plot actual vs. reconstructed curves for the most anomalous and most typical stations to confirm flagged profiles genuinely fail to reconstruct, since no ground-truth labels exist.

## Results

*Pending regeneration on synthetic data (see note above). The original analysis — run on the confidential dataset — selected a `24 → 8 → 24` architecture and flagged ~5% of stations as anomalous, validated visually via actual-vs-reconstructed plots; those specific figures and counts are not reproduced here since they derive from proprietary data.*

## Repo structure

```
NetworkAnomaliesProject/
├── README.md
├── requirements.txt
└── SDA.ipynb                    # full analysis, notebook entry point
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook SDA.ipynb
```

Run all cells top to bottom. The notebook expects a `dataset.xlsx` file (743 rows × 24 hourly columns + a `site` identifier column) in the project directory — currently pending replacement with a synthetic data generator. Cached-artifact filenames (`tuning_results.csv`, `sda_autoencoder.keras`, `sda_training_history.pkl`) are produced on first run and reused on subsequent runs; delete them, or set the `FORCE_RETRAIN_*` flags near the top of their respective cells to `True`, to force a fresh run (~10–15 min on CPU).

## Limitations & next steps

- No ground-truth anomaly labels exist for this dataset, so precision/recall can't be computed — validation is visual/qualitative, not statistical.
- The threshold is a single global percentile; it doesn't account for stations whose normal behavior is inherently more variable than others.
- Each day is scored independently; there's no use of history, so a station that is *consistently* unusual every day looks identical to a one-off event.
- Next: benchmark against a simpler baseline (PCA reconstruction error or per-hour z-scores) to quantify what the autoencoder adds; extend to sequence models (e.g. LSTM autoencoder) if multi-day data becomes available; sanity-check precision against any known incident/outage records.

## Tech stack

Python, pandas, NumPy, scikit-learn, TensorFlow/Keras, Matplotlib
