# Explainable Multimodal Physics-Aware Deep Learning for EV Battery
### ML Review 1 · Group AD1 · SRM Institute of Science and Technology

**Assigned technique:** Robust linear regression (Huber · RANSAC · Theil-Sen · Quantile)

| | |
|---|---|
| Swastik Mukherjee | RA2411026010181 |
| Prahladh Alarpati | RA2411026010171 |
| Aayan Noori | RA2411026010165 |

---

## What this project does

Estimates an EV battery cell's **State of Health** from only the **first 10 minutes** of
a discharge, using 15 physics-meaningful features drawn from three sensor streams
(voltage, current, temperature), fitted with robust linear regression.

**Result:** 1.14 SOH percentage points mean absolute error on a cell never seen in
training (R² = 0.953), against 7.03 pp for a predict-the-mean baseline. Under 20%
corrupted training labels, ordinary least squares degrades 29× while the robust fit
degrades 1.1×.

## Deliverables

| File | What it is |
|---|---|
| `src/01_model_robust_regression.py` | The model: feature engineering, 5 robust fits, contamination study, explainability |
| `src/02_statistical_analysis.py` | Statistical analysis of a second, different battery dataset (matplotlib + seaborn) |
| `src/make_ppt.py` | Builds the deck; every slide number is read from `outputs/*.csv` |
| `src/viz_style.py` | Shared chart styling (colour-blind-safe palette) |
| `EXPLANATION.md` | Plain-English walkthrough of both scripts + viva prep |
| `ML_Review1_EV_Battery_SOH_AD1.pptx` | 12-slide deck |

## Setup and run

```bash
pip3 install pandas numpy scipy scikit-learn matplotlib seaborn python-pptx pillow

python3 src/02_statistical_analysis.py     # statistics  -> outputs/figures/s*.png
python3 src/01_model_robust_regression.py  # model       -> outputs/figures/f*.png
python3 src/make_ppt.py                    # deck
```

## Data sources

- **NASA PCoE Li-ion Battery Ageing** (cells B0005/6/7/18) — 169,766 within-cycle sensor
  samples across 636 discharge cycles. Used for the model.
- **HNEI 18650 NMC-LCO Ageing** — 15,064 cycle summaries across 14 cells. Used for the
  statistical analysis.

Both are included in `data/`.
