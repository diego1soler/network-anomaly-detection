# Network Traffic Anomaly Detection with a Dense Autoencoder

Detects 24-hour base-station traffic profiles that deviate from "normal" behavior, using reconstruction error from a dense autoencoder as the anomaly score.

> **Note on data:** the original analysis was run on a confidential telecom dataset. The dataset used here merges a **synthetic dataset** — generated to match the aggregate per-hour statistics of the original (mean/std of log-traffic and an hour-to-hour correlation structure), with labeled anomalies (outages, spikes, phase shifts, inversions) injected on top — with the original real profiles, which are included with their site identifiers replaced by the same anonymized `S####` scheme (approved for publication by the data owner). Only the identifiers were changed; only the synthetic portion carries ground-truth anomaly labels, since that's the only part where the "true" answer is known by construction.

## Problem

A telecom network exposes hourly traffic volume for base stations, each summarized as a 24-value daily profile (one value per hour, 0–23). The goal is to flag stations whose daily traffic *shape* is structurally unusual — e.g. an unexpected spike, outage, or phase shift — without assuming labeled examples of "anomalous" behavior are available at training time.

## Approach

1. **Generate** a synthetic dataset (743 stations) calibrated to the real data's aggregate per-hour statistics, with ~5% labeled anomalies injected (outage / spike / phase-shift / inversion), then **merge** in the real (re-identified) profiles — 1,486 stations total.
2. **Normalize** each profile by its own daily mean, so the model learns hourly *shape* rather than absolute traffic volume.
3. **Split** into train/validation (80/20) and run a **grid search** over 6 encoder architectures × 2 learning rates × 3 random seeds, selecting the combination with the lowest mean validation MSE.
4. **Retrain** the selected architecture with early stopping; cache the trained weights and training history to disk so the notebook doesn't retrain on every run.
5. **Score** every station by reconstruction error (RMSE between actual and reconstructed profile) and **threshold** at the 95th percentile of the training-set error distribution.
6. **Validate** both visually (actual vs. reconstructed curves) and quantitatively — precision, recall, F1, ROC-AUC, and per-anomaly-type recall against the injected ground truth (synthetic subset only, since that's the only subset with known labels).

## Results

- Selected architecture: `24 → 8 → 24` (single 8-unit bottleneck), learning rate 5e-4, mean validation MSE ≈ 0.0137 (averaged over 3 seeds).
- **75 of 1,486 stations (5.0%)** flagged as anomalous at the 95th-percentile threshold (0.204) — 51/743 among the synthetic subset, 24/743 among the real subset.
- Against the injected ground truth (synthetic subset only): **precision 0.71, recall 0.95, F1 0.81, ROC-AUC 0.99**.
- Recall by injected anomaly type: spike 1.00, phase-shift 1.00, inversion 1.00, outage 0.80 — nearly every injected anomaly type is caught at this threshold.

![Actual vs. reconstructed traffic profiles for the 5 most anomalous stations](images/reconstruction_anomalous.png)

*The 5 highest-error stations (all from the synthetic/injected-spike subset) — the model's reconstruction (orange) is visibly damped relative to the actual spike (blue).*

![Actual vs. reconstructed traffic profiles for the 5 most typical stations](images/reconstruction_typical.png)

*For contrast, the 5 lowest-error stations (all from the real subset): reconstruction tracks the actual profile almost exactly.*

![ROC curve for reconstruction error as an anomaly score](images/roc_curve.png)

*ROC-AUC of 0.99 on the labeled (synthetic) subset — reconstruction error is a strong ranking signal for the injected anomalies, well above chance.*

## Repo structure

```
NetworkAnomaliesProject/
├── README.md
├── requirements.txt
├── SDA.ipynb                    # full analysis, notebook entry point
├── tuning_results.csv           # cached architecture/lr search results
├── sda_autoencoder.keras        # cached trained model
├── sda_training_history.pkl     # cached training loss history
└── images/                      # figures embedded in this README
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook SDA.ipynb
```

Run all cells top to bottom. The synthetic portion of the dataset is generated in-notebook; the real (re-identified) portion is loaded from a local `dataset.xlsx` if present, and the notebook falls back to synthetic-only data automatically if that file is absent — so the headline numbers above (1,486 rows) reflect the full merged dataset, while a from-scratch run without `dataset.xlsx` will reproduce the same methodology on 743 synthetic rows only, with different specific numbers. The architecture search and final model training are cached to disk (`tuning_results.csv`, `sda_autoencoder.keras`, `sda_training_history.pkl`) — delete those files, or set the `FORCE_RETRAIN_*` flags near the top of their respective cells to `True`, to reproduce the search/training from scratch (~10–15 min on CPU).

## Limitations & next steps

- Ground-truth labels only exist for the synthetic subset, so precision/recall/F1/ROC-AUC above are computed on that subset only — the real subset's flagged stations (24/743) have no way to be checked against a known answer here.
- The injected anomaly types are a designer's guess at what real anomalies look like; real-world anomalies may take different or subtler forms than outage/spike/phase-shift/inversion.
- The threshold is a single global percentile; it doesn't account for stations whose normal behavior is inherently more variable than others.
- Each day is scored independently; there's no use of history, so a station that is *consistently* unusual every day looks identical to a one-off event.
- Next: benchmark against a simpler baseline (PCA reconstruction error or per-hour z-scores); extend to sequence models (e.g. LSTM autoencoder) if multi-day data becomes available; cross-check the 24 flagged real stations against actual incident/outage records if any exist.

## Tech stack

Python, pandas, NumPy, scikit-learn, TensorFlow/Keras, Matplotlib
