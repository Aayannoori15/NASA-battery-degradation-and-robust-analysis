"""
===============================================================================
 Explainable Multimodal Physics-Aware Robust Regression for EV Battery SOH
===============================================================================
 Course     : Machine Learning  -  ML Review 1  (Group AD1)
 Technique  : Robust Linear Regression  (Huber / RANSAC / Theil-Sen / Quantile)
 Dataset    : NASA PCoE Li-ion battery ageing set - 4 cells, 636 discharge cycles
 Team       : Swastik Mukherjee   RA2411026010181
              Prahladh Alarpati   RA2411026010171
              Aayan Noori         RA2411026010165
===============================================================================

WHAT THIS SCRIPT DOES, IN ONE PARAGRAPH
---------------------------------------
An EV battery slowly loses capacity as it ages.  The number that describes this
is State of Health (SOH): today's usable capacity divided by the capacity the
cell had when it was new.  Measuring SOH properly means fully discharging the
pack, which no car can afford to do.  So we estimate it instead.  We take only
the FIRST 10 MINUTES of a discharge, pull out physics-meaningful numbers from
three sensor streams (voltage, current, temperature), and fit a straight-line
model from those numbers to SOH.  Because real battery sensors glitch, we fit
that line with ROBUST regression, which refuses to be dragged around by a few
bad readings - and we prove that it refuses, by deliberately corrupting the
training labels and watching ordinary least squares fall apart while the robust
fits stay put.

HOW TO RUN
----------
    python3 src/01_model_robust_regression.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import (
    LinearRegression, HuberRegressor, RANSACRegressor,
    TheilSenRegressor, QuantileRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from viz_style import use_project_style, SERIES, SEQ, DIVERGING, INK, INK_SOFT, save

warnings.filterwarnings("ignore")
use_project_style()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "nasa_discharge_multimodal.csv")
FIGS = os.path.join(ROOT, "outputs", "figures")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(FIGS, exist_ok=True)

RNG = np.random.default_rng(42)

# --- physical constants and design choices, all in one place ------------------
NOMINAL_CAPACITY_AH = 2.00    # rated capacity of these 18650 cells
V_FULL_OCV = 4.20             # open-circuit voltage of a fully charged cell
WINDOW_S = 600.0              # the 10-minute observation window (our hard rule)
E_ACT_J_PER_MOL = 20_000.0    # activation energy for SEI growth, ~20 kJ/mol
R_GAS = 8.314                 # J / (mol K)

TRAIN_CELLS = ["B0005", "B0006", "B0007"]
TEST_CELLS = ["B0018"]


def banner(txt):
    print("\n" + "=" * 78)
    print(txt)
    print("=" * 78)


# =============================================================================
# STEP 1  -  LOAD THE RAW MULTIMODAL SIGNALS
# =============================================================================
# The CSV is NOT one row per cycle. It is one row per *sample* inside a cycle:
# roughly 270 readings of voltage, current and temperature taken while the cell
# is being discharged. So 169,766 rows really means 636 discharge cycles.
banner("STEP 1  |  Loading raw multimodal sensor data")
raw = pd.read_csv(DATA)
raw = raw.sort_values(["Battery", "id_cycle", "Time"]).reset_index(drop=True)

print(f"rows (individual sensor samples) : {len(raw):,}")
print(f"cells                            : {sorted(raw.Battery.unique())}")
print(f"discharge cycles                 : {raw.groupby(['Battery','id_cycle']).ngroups}")
print(f"ambient temperature              : {raw.ambient_temperature.unique()} degC")
print("\nthe three modalities we actually use:")
print(raw[["Voltage_measured", "Current_measured", "Temperature_measured"]].describe().T
      [["mean", "std", "min", "max"]].round(3))


# =============================================================================
# STEP 2  -  WHY WE CANNOT USE THE WHOLE CYCLE  (the leakage trap)
# =============================================================================
# The tempting feature is "how long did the discharge last". It is also cheating:
# the dataset's Capacity label was computed by integrating current over exactly
# that duration, so duration and label are the same number wearing a hat.
banner("STEP 2  |  Leakage check - why the full cycle is off limits")
full = raw.groupby(["Battery", "id_cycle"]).agg(
    duration_s=("Time", "max"), capacity_ah=("Capacity", "first")).reset_index()
leak = np.corrcoef(full.duration_s, full.capacity_ah)[0, 1]
print(f"corr(full discharge duration, capacity label) = {leak:.4f}")
print("-> 0.997 is not a model, it is the label rewritten. Full-cycle duration,")
print("   total Ah and total Wh are therefore BANNED from the feature set.")
print(f"-> Every feature below is computed from the first {WINDOW_S:.0f} s only.")
print(f"   Shortest cycle in the data is {full.duration_s.min():.0f} s, so the")
print("   window always exists and we never pad or extrapolate.")


# =============================================================================
# STEP 3  -  PHYSICS-AWARE FEATURE ENGINEERING FROM THE 10-MINUTE WINDOW
# =============================================================================
# This is the heart of the project. Each feature is a quantity a battery
# engineer would recognise, not an anonymous number from an autoencoder.
# That is what makes the final linear model *explainable*: every coefficient
# multiplies something with a name and a unit.
def extract_features(g: pd.DataFrame) -> pd.Series:
    """Turn one discharge cycle (a few hundred sensor rows) into one feature row."""
    t = g["Time"].to_numpy(float)
    v = g["Voltage_measured"].to_numpy(float)
    i = g["Current_measured"].to_numpy(float)
    temp = g["Temperature_measured"].to_numpy(float)
    amb = float(g["ambient_temperature"].iloc[0])

    t = t - t[0]                       # clock starts at zero for every cycle
    w = t <= WINDOW_S                  # the only data we are allowed to look at
    tw, vw, iw, tempw = t[w], v[w], i[w], temp[w]
    i_abs = np.abs(iw)

    def v_at(sec):
        """Voltage at a fixed wall-clock second (linear interpolation)."""
        return float(np.interp(sec, tw, vw))

    def t_at_voltage(level):
        """First second at which the falling voltage crosses `level`."""
        below = np.where(vw <= level)[0]
        return float(tw[below[0]]) if len(below) else np.nan

    # ---- modality 1: ELECTRICAL -------------------------------------------
    # Ohm's law on the very first loaded sample. A fresh cell sags a little
    # under a 2 A load; an aged cell sags a lot, because its internal
    # resistance has grown. This single feature is the classic ageing marker.
    r_internal_mohm = (V_FULL_OCV - vw[0]) / i_abs[0] * 1000.0

    # Slope of the voltage plateau. Steeper plateau = less charge left to give.
    mid = (tw >= 100) & (tw <= WINDOW_S)
    dvdt_mv_per_s = np.polyfit(tw[mid], vw[mid], 1)[0] * 1000.0

    # Partial incremental-capacity style timing: how many seconds the cell
    # spends crossing the 3.9 V -> 3.8 V band. Aged cells cross it faster.
    t_39 = t_at_voltage(3.90)
    t_38 = t_at_voltage(3.80)
    band_39_38_s = t_38 - t_39 if np.isfinite(t_38) and np.isfinite(t_39) else np.nan

    wh_window = float(np.trapezoid(vw * i_abs, tw)) / 3600.0   # energy, Wh
    i_rms = float(np.sqrt(np.mean(iw ** 2)))                   # load actually applied

    # ---- modality 2: THERMAL ----------------------------------------------
    # Heat is the fingerprint of resistance: P = I^2 R. A cell whose
    # resistance has grown runs hotter for the same current.
    t_rise_k = float(tempw.max() - tempw[0])
    temp_at_600 = float(np.interp(WINDOW_S, tw, tempw))
    dtdt_mk_per_s = np.polyfit(tw, tempw, 1)[0] * 1000.0       # milli-K per second
    thermal_dose_kh = float(np.trapezoid(tempw - amb, tw)) / 3600.0  # K*h above ambient

    # ---- physics-aware couplings: where the two modalities multiply -------
    # Joule heating energy actually dissipated in the window (W*s -> Wh).
    ohmic_heat_wh = (i_rms ** 2) * (r_internal_mohm / 1000.0) * WINDOW_S / 3600.0
    # Arrhenius rate factor: side reactions speed up exponentially with
    # temperature. Scaled x1e4 only so the number is readable.
    t_mean_k = float(np.mean(tempw)) + 273.15
    arrhenius = np.exp(-E_ACT_J_PER_MOL / (R_GAS * t_mean_k)) * 1e4
    # Degrees of heating bought per watt-hour delivered - a pure efficiency loss.
    heat_per_wh = t_rise_k / wh_window if wh_window > 0 else np.nan

    return pd.Series({
        # electrical
        "r_internal_mohm": r_internal_mohm,
        "v_at_60s": v_at(60),
        "v_at_300s": v_at(300),
        "v_at_600s": v_at(600),
        "dvdt_mv_per_s": dvdt_mv_per_s,
        "band_39_38_s": band_39_38_s,
        "wh_window": wh_window,
        "i_rms": i_rms,
        # thermal
        "t_rise_k": t_rise_k,
        "temp_at_600s": temp_at_600,
        "dtdt_mk_per_s": dtdt_mk_per_s,
        "thermal_dose_kh": thermal_dose_kh,
        # couplings
        "ohmic_heat_wh": ohmic_heat_wh,
        "arrhenius_x1e4": arrhenius,
        "heat_per_wh": heat_per_wh,
        # bookkeeping (never used as inputs)
        "capacity_ah": float(g["Capacity"].iloc[0]),
    })


banner("STEP 3  |  Physics-aware feature engineering")
feat = (raw.groupby(["Battery", "id_cycle"], sort=True)
           .apply(extract_features, include_groups=False)
           .reset_index()
           .rename(columns={"id_cycle": "cycle"}))

# The label. SOH is just a percentage of the rated capacity.
feat["soh"] = feat["capacity_ah"] / NOMINAL_CAPACITY_AH

FEATURES = [
    "r_internal_mohm", "v_at_60s", "v_at_300s", "v_at_600s", "dvdt_mv_per_s",
    "band_39_38_s", "wh_window", "i_rms",
    "t_rise_k", "temp_at_600s", "dtdt_mk_per_s", "thermal_dose_kh",
    "ohmic_heat_wh", "arrhenius_x1e4", "heat_per_wh",
]
MODALITY = {f: ("Electrical" if f in FEATURES[:8] else
                "Thermal" if f in FEATURES[8:12] else "Coupled")
            for f in FEATURES}

before = len(feat)
feat = feat.dropna(subset=FEATURES + ["soh"]).reset_index(drop=True)
print(f"cycles -> feature rows : {before}  (dropped {before - len(feat)} with gaps)")
print(f"features per cycle     : {len(FEATURES)}  "
      f"({sum(v=='Electrical' for v in MODALITY.values())} electrical, "
      f"{sum(v=='Thermal' for v in MODALITY.values())} thermal, "
      f"{sum(v=='Coupled' for v in MODALITY.values())} physics couplings)")
print(f"SOH range              : {feat.soh.min():.3f} to {feat.soh.max():.3f}")
feat.to_csv(os.path.join(OUT, "cycle_features.csv"), index=False)
print(feat[FEATURES].describe().T[["mean", "std", "min", "max"]].round(3))


# =============================================================================
# STEP 4  -  TRAIN / TEST SPLIT BY CELL, NOT BY ROW
# =============================================================================
# Two cycles of the same cell are near-duplicates. Splitting rows at random
# would put twins on both sides and flatter the model. We hold out an entire
# unseen cell instead - the question a BMS actually faces.
banner("STEP 4  |  Grouped split - hold out a whole unseen cell")
tr = feat[feat.Battery.isin(TRAIN_CELLS)].reset_index(drop=True)
te = feat[feat.Battery.isin(TEST_CELLS)].reset_index(drop=True)
X_tr, y_tr = tr[FEATURES].to_numpy(), tr["soh"].to_numpy()
X_te, y_te = te[FEATURES].to_numpy(), te["soh"].to_numpy()
groups = tr["Battery"].to_numpy()
print(f"train : {TRAIN_CELLS} -> {len(tr)} cycles")
print(f"test  : {TEST_CELLS} -> {len(te)} cycles  (model has never seen this cell)")


# =============================================================================
# STEP 5  -  THE MODELS
# =============================================================================
# All five are straight lines: SOH = b0 + b1*x1 + ... + b15*x15. They differ
# only in WHAT THEY PUNISH when a point does not fit.
#   OLS       squares the error   -> one bad point can bend the whole line
#   Huber     squares small errors, switches to absolute past 1.35 sigma
#   RANSAC    fits on random subsets, keeps the fit that most points agree with
#   Theil-Sen takes the median of slopes over many subsamples
#   Quantile  fits the conditional median (pure absolute error)
def models():
    return {
        "OLS (baseline)": Pipeline([("sc", StandardScaler()),
                                    ("m", LinearRegression())]),
        "Huber": Pipeline([("sc", StandardScaler()),
                           ("m", HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=2000))]),
        "RANSAC": Pipeline([("sc", StandardScaler()),
                            ("m", RANSACRegressor(estimator=LinearRegression(),
                                                  min_samples=0.6, residual_threshold=None,
                                                  max_trials=300, random_state=0))]),
        "Theil-Sen": Pipeline([("sc", StandardScaler()),
                               ("m", TheilSenRegressor(max_subpopulation=5000, random_state=0))]),
        "Quantile (median)": Pipeline([("sc", StandardScaler()),
                                       ("m", QuantileRegressor(quantile=0.5, alpha=1e-4,
                                                               solver="highs"))]),
    }


def score(name, yt, yp):
    return {"Model": name,
            "MAE (SOH)": mean_absolute_error(yt, yp),
            "RMSE (SOH)": float(np.sqrt(mean_squared_error(yt, yp))),
            "R2": r2_score(yt, yp),
            "MAE (% pts)": mean_absolute_error(yt, yp) * 100}


banner("STEP 5  |  Clean data - how do the five fits compare?")
clean_rows, fitted = [], {}
for name, mdl in models().items():
    mdl.fit(X_tr, y_tr)
    fitted[name] = mdl
    clean_rows.append(score(name, y_te, mdl.predict(X_te)))

# Two honesty baselines: predict the training mean, and use the single best feature.
clean_rows.append(score("Baseline: train mean", y_te, np.full_like(y_te, y_tr.mean())))
best_single = max(FEATURES, key=lambda f: abs(np.corrcoef(tr[f], y_tr)[0, 1]))
single = Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]).fit(
    tr[[best_single]], y_tr)
clean_rows.append(score(f"Baseline: {best_single} only", y_te, single.predict(te[[best_single]])))

clean = pd.DataFrame(clean_rows).sort_values("MAE (SOH)").reset_index(drop=True)
print(clean.round(4).to_string(index=False))
clean.to_csv(os.path.join(OUT, "results_clean.csv"), index=False)

# Leave-one-cell-out cross-validation on the training cells, for an honest
# spread rather than a single lucky split.
banner("STEP 5b |  Leave-one-cell-out CV on the training cells")
cv = GroupKFold(n_splits=len(TRAIN_CELLS))
cv_rows = []
for name, mdl in models().items():
    pred = cross_val_predict(mdl, X_tr, y_tr, groups=groups, cv=cv)
    cv_rows.append(score(name, y_tr, pred))
cvdf = pd.DataFrame(cv_rows).sort_values("MAE (SOH)")
print(cvdf.round(4).to_string(index=False))
cvdf.to_csv(os.path.join(OUT, "results_cv.csv"), index=False)


# =============================================================================
# STEP 6  -  THE POINT OF ROBUSTNESS: CORRUPT THE LABELS ON PURPOSE
# =============================================================================
# Real capacity labels come from coulomb counters that drift, reset and
# occasionally report nonsense. We simulate exactly that: pick p% of training
# cycles and replace their SOH with a badly wrong value. The TEST set stays
# clean, so any loss of accuracy is the model being fooled, nothing else.
banner("STEP 6  |  Contamination study - inject faulty sensor labels")
LEVELS = [0, 5, 10, 15, 20, 30]
rob_rows = []
for p in LEVELS:
    y_bad = y_tr.copy()
    n_bad = int(round(len(y_bad) * p / 100))
    if n_bad:
        idx = RNG.choice(len(y_bad), n_bad, replace=False)
        # two realistic fault modes, half each: a stuck-low reading and a
        # wild spike well outside the physical range.
        half = n_bad // 2
        y_bad[idx[:half]] = 0.35 + RNG.normal(0, 0.03, half)
        y_bad[idx[half:]] = 1.60 + RNG.normal(0, 0.05, n_bad - half)
    for name, mdl in models().items():
        mdl.fit(X_tr, y_bad)
        r = score(name, y_te, mdl.predict(X_te))
        r["Contamination %"] = p
        rob_rows.append(r)

rob = pd.DataFrame(rob_rows)
pivot = rob.pivot(index="Contamination %", columns="Model", values="MAE (% pts)")
print("\nTest MAE in SOH percentage points, as training labels get dirtier:")
print(pivot.round(2).to_string())
rob.to_csv(os.path.join(OUT, "results_contamination.csv"), index=False)

# The fair way to read that table is the WORST row for each model, not the last
# one: a model you would trust in a car must survive its unluckiest batch.
worst = pivot.max()
blow = (worst / pivot.loc[0]).sort_values()
print("\nWorst-case MAE over all contamination levels (the number that matters):")
for m, w in worst.sort_values().items():
    print(f"   {m:<18} {w:6.2f} pp   ({blow[m]:5.1f}x its own clean error)")
print(f"\nOLS degrades up to {blow['OLS (baseline)']:.0f}x; Huber at most "
      f"{blow['Huber']:.1f}x. That gap is the whole argument for robust regression.")
print("Theil-Sen is the instructive failure: its theoretical breakdown point is")
print("about 29%, and sure enough it holds to 15% and then collapses.")
ols_blowup, hub_blowup = blow["OLS (baseline)"], blow["Huber"]


# =============================================================================
# STEP 7  -  EXPLAINABILITY
# =============================================================================
banner("STEP 7  |  Explaining the chosen model (Huber)")
CHOSEN = "Huber"
huber = fitted[CHOSEN]
coef = huber.named_steps["m"].coef_
scaler = huber.named_steps["sc"]

expl = pd.DataFrame({
    "feature": FEATURES,
    "modality": [MODALITY[f] for f in FEATURES],
    "std_coef": coef,                       # effect of a 1-sigma change in SOH units
    "effect_pp_per_sd": coef * 100,         # same thing in SOH percentage points
    "corr_with_soh": [np.corrcoef(tr[f], y_tr)[0, 1] for f in FEATURES],
}).sort_values("std_coef", key=np.abs, ascending=False).reset_index(drop=True)
print(expl.round(4).to_string(index=False))
expl.to_csv(os.path.join(OUT, "explainability_coefficients.csv"), index=False)

# Bootstrap the coefficients so we can say which signs we actually trust.
boot = np.zeros((400, len(FEATURES)))
for b in range(400):
    s = RNG.choice(len(X_tr), len(X_tr), replace=True)
    boot[b] = (Pipeline([("sc", StandardScaler()),
                         ("m", HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=1000))])
               .fit(X_tr[s], y_tr[s]).named_steps["m"].coef_)
lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
stable = [(f, c) for f, c, l, h in zip(FEATURES, coef, lo, hi) if l * h > 0]
print(f"\n{len(stable)} of {len(FEATURES)} coefficients keep the same sign in 95% of"
      " bootstrap resamples:")
for f, c in sorted(stable, key=lambda x: -abs(x[1])):
    print(f"   {f:<18} {c:+.4f} SOH per 1 SD  ({MODALITY[f]})")

# One cycle, fully decomposed: prediction = intercept + sum of contributions.
z = scaler.transform(X_te)
contrib = z * coef
pick = int(np.argmin(te["soh"].to_numpy()))     # the most degraded test cycle
case = pd.DataFrame({"feature": FEATURES, "contribution": contrib[pick]}) \
         .sort_values("contribution", key=np.abs, ascending=False)
print(f"\nWorked example - cell {te.Battery[pick]}, cycle {te.cycle[pick]}:")
print(f"   true SOH      {te.soh[pick]*100:.1f}%")
print(f"   predicted SOH {huber.predict(X_te[[pick]])[0]*100:.1f}%")
print(f"   intercept     {huber.named_steps['m'].intercept_*100:.1f}%  then:")
for _, r in case.head(5).iterrows():
    print(f"   {r.feature:<18} {r.contribution*100:+6.2f} pp")
case.to_csv(os.path.join(OUT, "explainability_single_cycle.csv"), index=False)

# Multicollinearity - a real caveat for linear models with 15 related features.
corr = tr[FEATURES].corr().abs()
pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(FEATURES)
         for b in FEATURES[i+1:] if corr.loc[a, b] > 0.9]
print(f"\n{len(pairs)} feature pairs are correlated above 0.9 - so individual")
print("coefficients share credit and must be read as a group, not in isolation:")
for a, b, c in sorted(pairs, key=lambda x: -x[2])[:5]:
    print(f"   {a} <-> {b}: {c:.3f}")


# =============================================================================
# STEP 8  -  FIGURES
# =============================================================================
banner("STEP 8  |  Writing figures")

# --- F1: the ageing we are trying to predict -------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.0))
for k, (cell, g) in enumerate(feat.groupby("Battery")):
    ax.plot(g.cycle, g.soh * 100, color=SERIES[k], label=cell, lw=2)
    ax.annotate(cell, (g.cycle.iloc[-1], g.soh.iloc[-1] * 100), xytext=(6, 0),
                textcoords="offset points", color=SERIES[k], fontweight="bold",
                fontsize=9.5, va="center")
ax.axhline(80, color=INK_SOFT, ls="--", lw=1.2)
ax.annotate("80% = end of automotive life", (6, 80.8), color=INK_SOFT, fontsize=9)
ax.set_xlim(0, feat.cycle.max() * 1.08)
ax.set_title("Capacity fade across four NASA cells", loc="left")
ax.set_xlabel("Discharge cycle"); ax.set_ylabel("State of Health (%)")
ax.legend(loc="lower left", ncol=4, title=None)
save(fig, f"{FIGS}/f1_soh_trajectories.png")

# --- F2: the three modalities, fresh cycle vs aged cycle -------------------
cell = "B0005"
sub = raw[raw.Battery == cell]
c_new, c_old = sub.id_cycle.min(), sub.id_cycle.max()
fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.3))
panels = [("Voltage_measured", "Terminal voltage (V)", "Electrical"),
          ("Current_measured", "Load current (A, held at 2 A)", "Electrical"),
          ("Temperature_measured", "Cell temperature (degC)", "Thermal")]
for ax, (col, lab, mod) in zip(axes, panels):
    for k, (cyc, nm) in enumerate([(c_new, f"cycle {c_new} (fresh)"),
                                   (c_old, f"cycle {c_old} (aged)")]):
        g = sub[sub.id_cycle == cyc]
        ax.plot(g.Time - g.Time.iloc[0], g[col], color=SERIES[k], label=nm, lw=1.8)
    ax.axvspan(0, WINDOW_S, color=SERIES[3], alpha=0.14, lw=0)
    ax.set_title(f"{mod}: {lab.split(' (')[0]}", loc="left", fontsize=11)
    ax.set_xlabel("Seconds into discharge"); ax.set_ylabel(lab)
axes[0].legend(loc="lower left")
axes[2].annotate("shaded = the only\nwindow we may use", (WINDOW_S * 1.15, 26),
                 fontsize=8.5, color=INK_SOFT)
fig.suptitle(f"Cell {cell}: the same discharge seen by three sensors, new vs end of life",
             x=0.005, ha="left", fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, f"{FIGS}/f2_multimodal_signals.png")

# --- F3: how each feature relates to SOH (signed -> diverging ramp) --------
e = expl.sort_values("corr_with_soh")
fig, ax = plt.subplots(figsize=(6.4, 4.6))
cols = [SERIES[0] if c > 0 else SERIES[1] for c in e.corr_with_soh]
ax.barh(e.feature, e.corr_with_soh, color=cols, height=0.68)
for y, (f, c) in enumerate(zip(e.feature, e.corr_with_soh)):
    ax.annotate(f"{c:+.2f}", (c, y), xytext=(6 if c > 0 else -6, 0),
                textcoords="offset points", va="center",
                ha="left" if c > 0 else "right", fontsize=8.5, color=INK_SOFT)
ax.axvline(0, color=INK_SOFT, lw=1)
ax.set_xlim(-1.15, 1.15)
ax.set_title("Each physics feature vs SOH (Pearson r, training cells)", loc="left")
ax.set_xlabel("correlation with State of Health"); ax.set_ylabel("")
ax.grid(True, axis="x", alpha=0.9); ax.grid(False, axis="y")
save(fig, f"{FIGS}/f3_feature_correlation.png")

# --- F4: feature-feature redundancy (single-hue magnitude ramp) -----------
fig, ax = plt.subplots(figsize=(6.6, 5.4))
sns.heatmap(tr[FEATURES].corr(), cmap=DIVERGING, center=0, vmin=-1, vmax=1,
            square=True, linewidths=1.2, linecolor="white",
            cbar_kws={"shrink": 0.75, "label": "Pearson r"}, ax=ax)
ax.set_title("Features overlap heavily - read coefficients as a group", loc="left")
plt.setp(ax.get_xticklabels(), rotation=55, ha="right", fontsize=8)
plt.setp(ax.get_yticklabels(), fontsize=8)
save(fig, f"{FIGS}/f4_feature_corr_matrix.png")

# --- F5: clean-data model comparison --------------------------------------
cmp = clean[~clean.Model.str.startswith("Baseline")].sort_values("MAE (% pts)")
fig, ax = plt.subplots(figsize=(7.0, 3.6))
bars = ax.bar(cmp.Model, cmp["MAE (% pts)"], color=SERIES[0], width=0.6)
bars[int(np.argmax(cmp.Model.values == CHOSEN))].set_color(SERIES[1])
base = clean.loc[clean.Model == "Baseline: train mean", "MAE (% pts)"].iloc[0]
ax.axhline(base, color=INK_SOFT, ls="--", lw=1.2)
ax.annotate(f"predict-the-mean baseline: {base:.2f} pp", (0.02, base * 0.93),
            xycoords=("axes fraction", "data"), fontsize=9, color=INK_SOFT)
for b, v in zip(bars, cmp["MAE (% pts)"]):
    ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width()/2, v), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=9.5, color=INK)
ax.set_title("Error on the unseen cell B0018 (lower is better)", loc="left")
ax.set_ylabel("Mean absolute error (SOH percentage points)"); ax.set_xlabel("")
plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
save(fig, f"{FIGS}/f5_model_comparison.png")

# --- F6: THE headline chart - robustness under label corruption -----------
# Log y-axis on purpose: the three robust fits sit on top of each other near
# 1 pp while OLS climbs to 31 pp. On a linear axis they would be one smudge.
fig, ax = plt.subplots(figsize=(7.4, 4.2))
order = ["OLS (baseline)", "Theil-Sen", "Huber", "Quantile (median)", "RANSAC"]
styles = ["-", "-", "-", "--", ":"]
for k, (m, ls) in enumerate(zip(order, styles)):
    ax.plot(pivot.index, pivot[m], color=SERIES[k], marker="o", label=m,
            lw=2.2, ls=ls, markeredgecolor="white", markeredgewidth=0.8)
ax.set_yscale("log")
ax.set_yticks([1, 2, 5, 10, 20, 30], ["1", "2", "5", "10", "20", "30"])
ax.set_xlim(-1, LEVELS[-1] + 1)
peak = pivot["OLS (baseline)"].max()
px = int(pivot["OLS (baseline)"].idxmax())
ax.annotate(f"OLS peaks at {peak:.0f} pp\n= {peak/pivot.loc[0,'OLS (baseline)']:.0f}x its clean error",
            (px, peak), xytext=(-18, -62), textcoords="offset points",
            fontsize=9, color=SERIES[0], ha="right",
            arrowprops=dict(arrowstyle="-", color=SERIES[0], lw=1))
ax.annotate("Huber / Quantile / RANSAC stay here, within 1.1x",
            (11.5, 1.72), fontsize=9, color=INK_SOFT)
ax.set_title("Robust fits ignore faulty capacity labels; OLS does not", loc="left")
ax.set_xlabel("Share of training labels corrupted (%)")
ax.set_ylabel("Test MAE on clean data (SOH pp, log scale)")
ax.legend(loc="upper left", ncol=2)
ax.grid(True, axis="y", which="both", alpha=0.6)
save(fig, f"{FIGS}/f6_contamination.png")

# --- F7: standardized coefficients with bootstrap intervals ---------------
o = np.argsort(np.abs(coef))
fig, ax = plt.subplots(figsize=(6.8, 4.8))
cols = [SERIES[0] if coef[i] > 0 else SERIES[1] for i in o]
ax.barh(np.arange(len(o)), coef[o] * 100, color=cols, height=0.66)
ax.errorbar(coef[o] * 100, np.arange(len(o)),
            xerr=[(coef[o]-lo[o]) * 100, (hi[o]-coef[o]) * 100],
            fmt="none", ecolor=INK_SOFT, elinewidth=1.1, capsize=3)
ax.set_yticks(np.arange(len(o)), [FEATURES[i] for i in o], fontsize=9)
ax.axvline(0, color=INK_SOFT, lw=1)
ax.set_title("What the model learned: SOH change per 1 SD of each feature", loc="left")
ax.set_xlabel("effect on SOH (percentage points per standard deviation)")
ax.grid(True, axis="x", alpha=0.9); ax.grid(False, axis="y")
ax.annotate("bars = Huber coefficient   |   whiskers = 95% bootstrap interval",
            (0, -0.15), xycoords="axes fraction", fontsize=8.5, color=INK_SOFT)
save(fig, f"{FIGS}/f7_coefficients.png")

# --- F8: predicted vs actual on the held-out cell -------------------------
yp = huber.predict(X_te)
fig, ax = plt.subplots(figsize=(4.9, 4.4))
lims = [min(y_te.min(), yp.min()) * 100 - 1.5, max(y_te.max(), yp.max()) * 100 + 1.5]
ax.fill_between(lims, [l - 2 for l in lims], [l + 2 for l in lims],
                color=SERIES[0], alpha=0.10, lw=0, label="+/- 2 pp band")
ax.plot(lims, lims, color=INK_SOFT, ls="--", lw=1.2, label="perfect prediction")
ax.scatter(y_te * 100, yp * 100, s=26, color=SERIES[0], alpha=0.75,
           edgecolor="white", linewidth=0.7)
inside = np.mean(np.abs(yp - y_te) * 100 <= 2) * 100
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_title(f"Held-out cell B0018: {inside:.0f}% of cycles within 2 pp", loc="left")
ax.set_xlabel("Measured SOH (%)"); ax.set_ylabel("Predicted SOH (%)")
ax.legend(loc="upper left"); ax.grid(True, alpha=0.9)
save(fig, f"{FIGS}/f8_pred_vs_actual.png")

# --- F9: residual diagnostics ---------------------------------------------
res = (yp - y_te) * 100
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))
axes[0].axhline(0, color=INK_SOFT, lw=1.2)
axes[0].scatter(yp * 100, res, s=24, color=SERIES[0], alpha=0.75,
                edgecolor="white", linewidth=0.7)
axes[0].set_title("Residuals vs prediction", loc="left")
axes[0].set_xlabel("Predicted SOH (%)"); axes[0].set_ylabel("Error (pp)")
sns.histplot(res, bins=24, color=SERIES[0], ax=axes[1], edgecolor="white")
axes[1].axvline(0, color=INK_SOFT, lw=1.2)
axes[1].set_title(f"Error distribution (bias {res.mean():+.2f} pp)", loc="left")
axes[1].set_xlabel("Error (pp)"); axes[1].set_ylabel("Cycles")
fig.tight_layout()
save(fig, f"{FIGS}/f9_residuals.png")

# --- F10: what RANSAC threw away ------------------------------------------
y_bad = y_tr.copy()
idx = RNG.choice(len(y_bad), int(0.15 * len(y_bad)), replace=False)
y_bad[idx] = 0.35 + RNG.normal(0, 0.03, len(idx))
rs = models()["RANSAC"].fit(X_tr, y_bad)
mask = rs.named_steps["m"].inlier_mask_
key = "r_internal_mohm"
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.scatter(tr[key][mask], y_bad[mask] * 100, s=26, color=SERIES[0], alpha=0.8,
           edgecolor="white", linewidth=0.7, label=f"kept as inlier (n={mask.sum()})")
ax.scatter(tr[key][~mask], y_bad[~mask] * 100, s=34, color=SERIES[1], alpha=0.9,
           marker="X", edgecolor="white", linewidth=0.7,
           label=f"rejected as outlier (n={(~mask).sum()})")
ax.set_title("RANSAC finds the injected sensor faults on its own", loc="left")
ax.set_xlabel("Internal resistance proxy (milliohm)")
ax.set_ylabel("Training label: SOH (%)")
ax.legend(loc="upper right")
save(fig, f"{FIGS}/f10_ransac_inliers.png")

# --- F11: single-cycle explanation ----------------------------------------
top = case.head(7).iloc[::-1]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
cols = [SERIES[0] if c > 0 else SERIES[1] for c in top.contribution]
ax.barh(top.feature, top.contribution * 100, color=cols, height=0.64)
for y, c in enumerate(top.contribution * 100):
    ax.annotate(f"{c:+.2f} pp", (c, y), xytext=(6 if c > 0 else -6, 0),
                textcoords="offset points", va="center",
                ha="left" if c > 0 else "right", fontsize=9, color=INK_SOFT)
ax.axvline(0, color=INK_SOFT, lw=1)
lo_x, hi_x = (top.contribution * 100).min(), (top.contribution * 100).max()
ax.set_xlim(lo_x * 1.55, hi_x * 1.45)   # room for the end labels, both sides
ax.set_title(f"Why the model called cycle {te.cycle[pick]} of B0018 "
             f"{huber.predict(X_te[[pick]])[0]*100:.0f}% healthy", loc="left")
ax.set_xlabel("contribution to the prediction (pp, added to the intercept)")
ax.grid(True, axis="x", alpha=0.9); ax.grid(False, axis="y")
save(fig, f"{FIGS}/f11_single_cycle_explanation.png")

# --- F12: predicted SOH trajectory on the held-out cell -------------------
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.plot(te.cycle, y_te * 100, color=SERIES[0], lw=2.2, label="Measured SOH")
ax.plot(te.cycle, yp * 100, color=SERIES[1], lw=1.8, ls="--", label="Predicted from 10 min")
ax.fill_between(te.cycle, yp * 100 - 2, yp * 100 + 2, color=SERIES[1], alpha=0.14, lw=0)
ax.set_title("Cell B0018: predicted ageing curve never seen in training", loc="left")
ax.set_xlabel("Discharge cycle"); ax.set_ylabel("State of Health (%)")
ax.legend(loc="lower left")
save(fig, f"{FIGS}/f12_trajectory_heldout.png")


# =============================================================================
# STEP 9  -  SUMMARY FOR THE SLIDES
# =============================================================================
banner("STEP 9  |  Headline numbers")
best = clean.iloc[0]
h = clean[clean.Model == CHOSEN].iloc[0]
print(f"cycles / features            : {len(feat)} / {len(FEATURES)}")
print(f"observation window           : first {WINDOW_S:.0f} s of a discharge")
print(f"best model on unseen cell    : {best.Model}  MAE {best['MAE (% pts)']:.2f} pp, "
      f"R2 {best['R2']:.3f}")
print(f"chosen model ({CHOSEN})        : MAE {h['MAE (% pts)']:.2f} pp, R2 {h['R2']:.3f}")
print(f"predict-the-mean baseline    : MAE {base:.2f} pp")
print(f"cycles within 2 pp           : {inside:.0f}%")
print(f"leave-one-cell-out CV, best  : {cvdf.iloc[0].Model} "
      f"{cvdf.iloc[0]['MAE (% pts)']:.2f} pp  (OLS: "
      f"{cvdf[cvdf.Model=='OLS (baseline)']['MAE (% pts)'].iloc[0]:.2f} pp)")
print(f"OLS worst case, dirty labels : {worst['OLS (baseline)']:.2f} pp "
      f"({ols_blowup:.0f}x its clean error)")
print(f"Huber worst case             : {worst['Huber']:.2f} pp "
      f"({hub_blowup:.1f}x its clean error)")
print(f"stable coefficients          : {len(stable)} of {len(FEATURES)}")
print("\nAll figures in outputs/figures/, all tables in outputs/.")
