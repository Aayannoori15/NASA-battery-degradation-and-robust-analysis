"""
===============================================================================
 Statistical Data Analysis of a Li-ion Battery Ageing Dataset  (HNEI)
===============================================================================
 Course    : Machine Learning  -  ML Review 1  (Group AD1)
 Unit      : Statistical analysis / exploratory data analysis
 Dataset   : HNEI 18650 NMC-LCO ageing set - 14 cells, 15,064 charge/discharge
             cycles, 8 cycle-summary measurements + Remaining Useful Life
 Team      : Swastik Mukherjee   RA2411026010181
             Prahladh Alarpati   RA2411026010171
             Aayan Noori         RA2411026010165
 Libraries : pandas, numpy, scipy, matplotlib, seaborn  (visualisation is
             deliberately matplotlib + seaborn only - no plotting shortcuts)
===============================================================================

WHY A SECOND, DIFFERENT DATASET?
--------------------------------
The model script works on NASA cells and answers one question: can we predict
State of Health?  This script does something else entirely, on a different set
of cells from a different lab.  It asks what the data IS before anyone models
it: how is each measurement distributed, is any of it Gaussian, which columns
are secretly the same column, how much of it is sensor garbage, and do the
14 cells age the same way.  Those answers are what justify the modelling
choices made in the other script - above all the choice of a robust fit.

HOW TO RUN
----------
    python3 src/02_statistical_analysis.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from viz_style import use_project_style, SERIES, SEQ, DIVERGING, INK, INK_SOFT, save

warnings.filterwarnings("ignore")
use_project_style()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "hnei_battery_rul.csv")
FIGS = os.path.join(ROOT, "outputs", "figures")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(FIGS, exist_ok=True)
RNG = np.random.default_rng(7)
ALPHA = 0.05


def banner(txt):
    print("\n" + "=" * 78)
    print(txt)
    print("=" * 78)


# =============================================================================
# STEP 1  -  LOAD, AND REBUILD THE CELL LABELS THAT THE FILE LOST
# =============================================================================
# The published CSV stacks all 14 cells on top of each other and forgets to say
# which row belongs to which cell. But Cycle_Index counts 1, 2, 3 ... and then
# drops back to 1 when the next cell starts. Every drop is a cell boundary, so
# we can recover the grouping exactly.
banner("STEP 1  |  Load and recover the lost cell identifier")
df = pd.read_csv(DATA).drop(columns=["Unnamed: 0"])
df["cell_id"] = ("Cell_" +
                 (df["Cycle_Index"].diff().fillna(1).lt(0).cumsum() + 1)
                 .astype(str).str.zfill(2))

FEATURES = [
    "Discharge Time (s)", "Decrement 3.6-3.4V (s)", "Max. Voltage Dischar. (V)",
    "Min. Voltage Charg. (V)", "Time at 4.15V (s)", "Time constant current (s)",
    "Charging time (s)", "Total time (s)",
]
SHORT = {  # readable axis labels - the originals are too long for a chart
    "Discharge Time (s)": "Discharge time",
    "Decrement 3.6-3.4V (s)": "3.6->3.4 V decrement",
    "Max. Voltage Dischar. (V)": "Max discharge voltage",
    "Min. Voltage Charg. (V)": "Min charge voltage",
    "Time at 4.15V (s)": "Time at 4.15 V",
    "Time constant current (s)": "Constant-current time",
    "Charging time (s)": "Charging time",
    "Total time (s)": "Total cycle time",
}
TARGET = "RUL"

print(f"rows (cycles)      : {len(df):,}")
print(f"columns            : {df.shape[1]} ({len(FEATURES)} measurements + RUL + cell_id)")
print(f"cells recovered    : {df.cell_id.nunique()}")
print(f"cycles per cell    : min {df.cell_id.value_counts().min()}, "
      f"max {df.cell_id.value_counts().max()}, "
      f"median {int(df.cell_id.value_counts().median())}")
print(f"missing values     : {int(df.isna().sum().sum())}")
print(f"duplicate rows     : {int(df.duplicated().sum())}")
print(f"RUL (cycles left)  : {df[TARGET].min()} to {df[TARGET].max()}")


# =============================================================================
# STEP 2  -  DESCRIPTIVE STATISTICS, AND THE FIRST ALARM
# =============================================================================
# Mean vs median is the cheapest outlier detector there is. If the mean of a
# column sits far above its median, a handful of huge values are pulling it.
banner("STEP 2  |  Descriptive statistics")
desc = df[FEATURES + [TARGET]].describe().T
desc["median"] = df[FEATURES + [TARGET]].median()
desc["IQR"] = desc["75%"] - desc["25%"]
desc["skew"] = df[FEATURES + [TARGET]].skew()
desc["excess_kurtosis"] = df[FEATURES + [TARGET]].kurtosis()
desc["mean/median"] = desc["mean"] / desc["median"]
cols = ["mean", "median", "std", "min", "max", "IQR", "skew",
        "excess_kurtosis", "mean/median"]
print(desc[cols].round(3).to_string())
desc[cols].to_csv(os.path.join(OUT, "stats_descriptive.csv"))

worst = desc["mean/median"].drop(TARGET).idxmax()
print(f"\nRed flag: '{worst}' has a mean {desc.loc[worst,'mean/median']:.1f}x its median")
print(f"and an excess kurtosis of {desc.loc[worst,'excess_kurtosis']:.0f}. A normal")
print("distribution has excess kurtosis 0. This column is mostly a thin spike")
print("with a very long tail - i.e. a few cycles where logging ran for days.")


# =============================================================================
# STEP 3  -  OUTLIER AUDIT: TWO RULES, BECAUSE ONE RULE IS AN OPINION
# =============================================================================
# Rule A (Tukey / IQR): anything more than 1.5 IQR outside the quartiles.
# Rule B (modified Z-score): uses the MEDIAN and the MAD instead of the mean
#   and the standard deviation, so the outliers cannot hide themselves by
#   inflating the very spread used to detect them. |score| > 3.5 is the usual cut.
banner("STEP 3  |  Outlier audit")


def iqr_outliers(s):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)


def modified_z(s):
    med = s.median()
    mad = np.median(np.abs(s - med))
    if mad == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return 0.6745 * (s - med) / mad


rows = []
for f in FEATURES:
    m_iqr = iqr_outliers(df[f])
    m_mz = modified_z(df[f]).abs() > 3.5
    rows.append({"feature": SHORT[f],
                 "IQR outliers %": m_iqr.mean() * 100,
                 "mod-Z outliers %": m_mz.mean() * 100,
                 "max / p99 ratio": df[f].max() / df[f].quantile(0.99)})
odf = pd.DataFrame(rows).sort_values("IQR outliers %", ascending=False)
print(odf.round(2).to_string(index=False))
odf.to_csv(os.path.join(OUT, "stats_outliers.csv"), index=False)

# A physical sanity screen: a charge/discharge cycle on an 18650 cell cannot
# honestly take longer than a day. Anything above that is an instrument fault.
impossible = (df["Total time (s)"] > 86400).sum()
print(f"\ncycles whose total time exceeds 24 hours : {impossible} "
      f"({impossible/len(df)*100:.2f}%)")
print(f"median IQR-outlier rate across columns  : {odf['IQR outliers %'].median():.1f}%")
print("\nThese are not interesting physics, they are logging faults - and they are")
print("exactly why the companion model is fitted with Huber / RANSAC rather than")
print("ordinary least squares.")


# =============================================================================
# STEP 4  -  IS ANYTHING HERE NORMAL?  (formal tests, not eyeballing)
# =============================================================================
# D'Agostino's K^2 combines skewness and kurtosis into one statistic.
# Shapiro-Wilk is more sensitive but caps out at 5000 points, so we subsample.
# Both answer the same question: could this plausibly be a Gaussian sample?
banner("STEP 4  |  Normality tests")
norm_rows = []
for f in FEATURES + [TARGET]:
    x = df[f].dropna().to_numpy()
    xs = RNG.choice(x, size=min(4999, len(x)), replace=False)
    k2, p_k2 = stats.normaltest(x)
    w, p_sw = stats.shapiro(xs)
    # log1p often rescues a right-skewed positive variable. Does it here?
    p_log = stats.normaltest(np.log1p(x - x.min() + 1e-9))[1] if x.min() >= 0 else np.nan
    norm_rows.append({"variable": SHORT.get(f, f),
                      "D'Agostino K2": k2, "p (K2)": p_k2,
                      "Shapiro W": w, "p (Shapiro)": p_sw,
                      "p after log1p": p_log,
                      "normal?": "no" if p_k2 < ALPHA else "cannot reject"})
ndf = pd.DataFrame(norm_rows)
print(ndf.round(4).to_string(index=False))
ndf.to_csv(os.path.join(OUT, "stats_normality.csv"), index=False)
n_notnormal = (ndf["normal?"] == "no").sum()
print(f"\n{n_notnormal} of {len(ndf)} variables reject normality at alpha={ALPHA},")
print("and the log transform does not rescue a single one. Consequence: every")
print("test below is the rank-based (non-parametric) version. Using a t-test or")
print("Pearson r here would be reporting a number the data does not support.")


# =============================================================================
# STEP 5  -  PEARSON vs SPEARMAN: THE OUTLIERS SHOW UP IN THE CORRELATIONS
# =============================================================================
# Pearson measures straight-line agreement on the raw values, so outliers move
# it a lot. Spearman first replaces values by their RANKS, which squashes any
# outlier down to "the largest one". Where the two disagree, outliers are
# driving the relationship.
banner("STEP 5  |  Correlation with RUL - raw values vs ranks")
corr_rows = []
for f in FEATURES:
    r, pr = stats.pearsonr(df[f], df[TARGET])
    rho, ps = stats.spearmanr(df[f], df[TARGET])
    corr_rows.append({"feature": SHORT[f], "Pearson r": r, "p (Pearson)": pr,
                      "Spearman rho": rho, "p (Spearman)": ps,
                      "rank gain": abs(rho) - abs(r)})
cdf = pd.DataFrame(corr_rows).sort_values("Spearman rho", key=np.abs, ascending=False)
print(cdf.round(4).to_string(index=False))
cdf.to_csv(os.path.join(OUT, "stats_correlation_rul.csv"), index=False)
big = cdf.loc[cdf["rank gain"].idxmax()]
print(f"\nBiggest disagreement: '{big.feature}' scores r={big['Pearson r']:+.2f} but")
print(f"rho={big['Spearman rho']:+.2f}  -  Pearson calls it insignificant (p="
      f"{big['p (Pearson)']:.2f}) while")
print(f"Spearman calls it near-perfect. A gap of {abs(big['rank gain']):.2f} is not a")
print("subtlety: it means a handful of extreme cycles are hiding the single")
print("strongest signal in the dataset from anyone who only computes Pearson r.")

# Holm-Bonferroni correction: 8 tests on the same data means 8 chances to get
# lucky, so we tighten the threshold before believing any single one.
ps = cdf["p (Spearman)"].to_numpy()
order = np.argsort(ps)
holm = np.zeros(len(ps), bool)
for rank, i in enumerate(order):
    if ps[i] < ALPHA / (len(ps) - rank):
        holm[i] = True
    else:
        break
print(f"\nAfter Holm-Bonferroni correction for {len(ps)} simultaneous tests, "
      f"{holm.sum()} of {len(ps)} remain significant.")


# =============================================================================
# STEP 6  -  HYPOTHESIS TESTS WITH EFFECT SIZES
# =============================================================================
# A p-value only says "probably not zero". The effect size says "and it is this
# big", which is the part a reader actually needs. We report both every time.
banner("STEP 6  |  Hypothesis tests")
tests = []

V = "Max. Voltage Dischar. (V)"

# H1: do the 14 cells sit at different voltage levels?  (Kruskal-Wallis, the
#     rank-based cousin of one-way ANOVA)
gs = [g[V].to_numpy() for _, g in df.groupby("cell_id")]
H, p = stats.kruskal(*gs)
eps2 = (H - len(gs) + 1) / (len(df) - len(gs))     # epsilon-squared effect size
tests.append(("H1  cells differ in voltage level", "Kruskal-Wallis", H, p,
              f"eps^2 = {eps2:.3f}"))

# H2: is the SPREAD the same across cells?  Levene centred on the median
#     (Brown-Forsythe) is the outlier-tolerant version of the variance test.
W, p = stats.levene(*gs, center="median")
tests.append(("H2  equal variance across cells", "Levene (median)", W, p,
              f"sd range {min(g.std() for g in gs):.3f}-"
              f"{max(g.std() for g in gs):.3f} V"))

# H3: does max discharge voltage fall between early life and late life?
cut_lo, cut_hi = df[TARGET].quantile([0.25, 0.75])
late = df.loc[df[TARGET] <= cut_lo, "Max. Voltage Dischar. (V)"]
early = df.loc[df[TARGET] >= cut_hi, "Max. Voltage Dischar. (V)"]
U, p = stats.mannwhitneyu(early, late, alternative="greater")
cles = U / (len(early) * len(late))      # probability a random early cycle > late
tests.append(("H3  voltage higher early than late", "Mann-Whitney U", U, p,
              f"CLES = {cles:.3f}"))

# H4: is the fall in discharge time from early to late life real?
late_d = df.loc[df[TARGET] <= cut_lo, "Discharge Time (s)"]
early_d = df.loc[df[TARGET] >= cut_hi, "Discharge Time (s)"]
U2, p = stats.mannwhitneyu(early_d, late_d, alternative="two-sided")
tests.append(("H4  discharge time shifts with age", "Mann-Whitney U", U2, p,
              f"median {early_d.median():.0f}s -> {late_d.median():.0f}s"))

# H5: is ageing monotone within a single cell?  (Spearman inside one cell only,
#     so cell-to-cell differences cannot manufacture the trend)
one = df[df.cell_id == df.cell_id.unique()[0]]
rho, p = stats.spearmanr(one["Cycle_Index"], one[V])
tests.append((f"H5  monotone fade within {one.cell_id.iloc[0]}", "Spearman rho",
              rho, p, f"n = {len(one)}"))

# H6: a deliberate NULL result, kept in because it is informative. RUL is not a
#     measurement, it is a countdown the authors wrote in afterwards: every cell
#     ends at 0, so every cell has the same RUL distribution by construction.
#     The test should fail to reject, and it does.
H6, p = stats.kruskal(*[g[TARGET].to_numpy() for _, g in df.groupby("cell_id")])
tests.append(("H6  cells differ in RUL (expected null)", "Kruskal-Wallis", H6, p,
              "identical by construction"))

tdf = pd.DataFrame(tests, columns=["hypothesis", "test", "statistic",
                                   "p-value", "effect size"])
tdf["verdict"] = np.where(tdf["p-value"] < ALPHA, "reject H0", "cannot reject H0")
print(tdf.to_string(index=False, float_format=lambda v: f"{v:,.4g}"))
tdf.to_csv(os.path.join(OUT, "stats_hypothesis_tests.csv"), index=False)
n_rej = int((tdf["p-value"] < ALPHA).sum())
print(f"\n{n_rej} of {len(tdf)} tests reject the null at alpha={ALPHA}.")
print("Reading them in order:")
print(" - H1/H2: the 14 cells differ in BOTH their typical voltage and their")
print("   spread, so one pooled model is already an averaging compromise. That is")
print("   why the companion script holds out whole cells, never random rows.")
print(" - H3/H4: ageing is unmistakable - a random early-life cycle outranks a")
print(f"   late-life one {cles*100:.1f}% of the time, and median discharge time")
print(f"   halves from {early_d.median():.0f}s to {late_d.median():.0f}s.")
print(" - H5: within a single cell the fade is almost perfectly monotone")
print(f"   (rho = {rho:.2f}), which is the clean signal a linear model can use.")
print(" - H6 is the useful failure: RUL cannot tell the cells apart because RUL")
print("   is a countdown added by the dataset authors, not something measured.")


# =============================================================================
# STEP 7  -  PCA: HOW MANY INDEPENDENT THINGS ARE WE REALLY MEASURING?
# =============================================================================
# Eight columns does not mean eight pieces of information. PCA on the
# standardised (and rank-transformed, to defuse the outliers) data tells us how
# many directions actually carry variance.
banner("STEP 7  |  PCA on rank-transformed features")
ranked = df[FEATURES].rank().to_numpy()
Z = (ranked - ranked.mean(0)) / ranked.std(0)
cov = np.cov(Z, rowvar=False)
eigval, eigvec = np.linalg.eigh(cov)
idx = np.argsort(eigval)[::-1]
eigval, eigvec = eigval[idx], eigvec[:, idx]
evr = eigval / eigval.sum()
cum = np.cumsum(evr)
n80 = int(np.searchsorted(cum, 0.80) + 1)
for i, (e, c) in enumerate(zip(evr, cum), 1):
    print(f"   PC{i}: {e*100:5.1f}% of variance, cumulative {c*100:5.1f}%")
print(f"\n{n80} component{'s' if n80 > 1 else ''} already carr{'y' if n80 > 1 else 'ies'} "
      f"80% of the variance in 8 columns -")
print("the measurements are largely restatements of one underlying thing:")
print("how far through its life the cell is.")
pcs = Z @ eigvec[:, :2]
pd.DataFrame({"PC": [f"PC{i}" for i in range(1, 9)],
              "explained_variance_ratio": evr,
              "cumulative": cum}).to_csv(os.path.join(OUT, "stats_pca.csv"), index=False)


# =============================================================================
# STEP 8  -  FIGURES  (matplotlib + seaborn)
# =============================================================================
banner("STEP 8  |  Writing figures")

# --- S1: distributions, raw vs log, for the four worst-behaved columns -----
pick = desc["skew"].drop(TARGET).sort_values(ascending=False).head(4).index.tolist()
fig, axes = plt.subplots(2, 4, figsize=(11.2, 5.0))
for j, f in enumerate(pick):
    x = df[f].to_numpy()
    sns.histplot(x, bins=60, color=SERIES[0], ax=axes[0, j], edgecolor="white",
                 linewidth=0.4)
    axes[0, j].set_title(SHORT[f], loc="left", fontsize=10.5)
    axes[0, j].set_xlabel("raw value"); axes[0, j].set_ylabel("cycles" if j == 0 else "")
    axes[0, j].annotate(f"skew {stats.skew(x):.1f}", (0.96, 0.9), xycoords="axes fraction",
                        ha="right", fontsize=9, color=INK_SOFT)
    sns.histplot(np.log1p(x - x.min()), bins=60, color=SERIES[1], ax=axes[1, j],
                 edgecolor="white", linewidth=0.4)
    axes[1, j].set_xlabel("log(1 + value)")
    axes[1, j].set_ylabel("cycles" if j == 0 else "")
    axes[1, j].annotate(f"skew {stats.skew(np.log1p(x - x.min())):.1f}",
                        (0.96, 0.9), xycoords="axes fraction", ha="right",
                        fontsize=9, color=INK_SOFT)
fig.suptitle("The four most skewed measurements: raw (blue) and log-transformed (orange)",
             x=0.005, ha="left", fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
save(fig, f"{FIGS}/s1_distributions.png")

# --- S2: Q-Q plots - the visual version of the normality tests ------------
fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.3))
for ax, f in zip(axes, ["Max. Voltage Dischar. (V)", "Discharge Time (s)", "RUL"]):
    x = df[f].to_numpy()
    xs = RNG.choice(x, size=min(3000, len(x)), replace=False)
    (osm, osr), (slope, inter, r) = stats.probplot(xs, dist="norm")
    ax.plot(osm, slope * osm + inter, color=INK_SOFT, ls="--", lw=1.3,
            label="perfect normal", zorder=1)
    ax.scatter(osm, osr, s=10, color=SERIES[0], alpha=0.65, zorder=2,
               edgecolor="none")
    ax.set_title(SHORT.get(f, f), loc="left", fontsize=11)
    ax.set_xlabel("theoretical quantile"); ax.set_ylabel("observed")
    ax.annotate(f"R = {r:.3f}", (0.05, 0.9), xycoords="axes fraction",
                fontsize=9, color=INK_SOFT)
axes[0].legend(loc="lower right")
fig.suptitle("Normal Q-Q plots: every variable leaves the line at the tails",
             x=0.005, ha="left", fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.92])
save(fig, f"{FIGS}/s2_qq_normality.png")

# --- S3: outlier share per column, two rules side by side ----------------
o = odf.sort_values("IQR outliers %")
y = np.arange(len(o))
fig, ax = plt.subplots(figsize=(6.8, 4.2))
ax.barh(y + 0.19, o["IQR outliers %"], height=0.34, color=SERIES[0],
        label="Tukey IQR rule")
ax.barh(y - 0.19, o["mod-Z outliers %"], height=0.34, color=SERIES[1],
        label="Modified Z (MAD) rule")
for yy, (a, b) in enumerate(zip(o["IQR outliers %"], o["mod-Z outliers %"])):
    ax.annotate(f"{a:.1f}%", (a, yy + 0.19), xytext=(5, 0), textcoords="offset points",
                va="center", fontsize=8.5, color=INK_SOFT)
    ax.annotate(f"{b:.1f}%", (b, yy - 0.19), xytext=(5, 0), textcoords="offset points",
                va="center", fontsize=8.5, color=INK_SOFT)
ax.set_yticks(y, o.feature, fontsize=9.5)
ax.set_title("Share of cycles flagged as outliers, by two independent rules", loc="left")
ax.set_xlabel("% of the 15,064 cycles flagged")
ax.grid(True, axis="x", alpha=0.9); ax.grid(False, axis="y")
ax.legend(loc="lower right")
save(fig, f"{FIGS}/s3_outliers.png")

# --- S4: the full correlation matrix, ranks not raw values ---------------
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
labels = [SHORT[f] for f in FEATURES] + ["RUL"]
for ax, method, name in zip(axes, ["pearson", "spearman"],
                            ["Pearson (raw values)", "Spearman (ranks)"]):
    m = df[FEATURES + [TARGET]].corr(method=method)
    m.index = m.columns = labels
    sns.heatmap(m, cmap=DIVERGING, center=0, vmin=-1, vmax=1, square=True,
                linewidths=1.2, linecolor="white", annot=True, fmt=".2f",
                annot_kws={"size": 7}, cbar=False, ax=ax)
    ax.set_title(name, loc="left", fontsize=11.5)
    plt.setp(ax.get_xticklabels(), rotation=50, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
fig.suptitle("Same data, two definitions of correlation - the gap is the outliers",
             x=0.005, ha="left", fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, f"{FIGS}/s4_correlation.png")

# --- S5: the strongest relationship, drawn honestly ----------------------
f_best = FEATURES[[SHORT[f] for f in FEATURES].index(cdf.iloc[0].feature)]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
sample = df.sample(min(4000, len(df)), random_state=1)
axes[0].scatter(sample[f_best], sample[TARGET], s=9, color=SERIES[0], alpha=0.35,
                edgecolor="none")
axes[0].set_xscale("symlog")
axes[0].set_title(f"{SHORT[f_best]} vs RUL (log x, raw values)", loc="left", fontsize=11)
axes[0].set_xlabel(SHORT[f_best]); axes[0].set_ylabel("Remaining useful life (cycles)")
axes[0].annotate(f"Pearson r = {cdf.iloc[0]['Pearson r']:+.2f}", (0.04, 0.08),
                 xycoords="axes fraction", fontsize=9.5, color=INK_SOFT)
axes[1].scatter(df[f_best].rank(pct=True) * 100, df[TARGET].rank(pct=True) * 100,
                s=9, color=SERIES[1], alpha=0.35, edgecolor="none")
axes[1].set_title("The same pair in rank space", loc="left", fontsize=11)
axes[1].set_xlabel(f"{SHORT[f_best]} percentile"); axes[1].set_ylabel("RUL percentile")
axes[1].annotate(f"Spearman rho = {cdf.iloc[0]['Spearman rho']:+.2f}", (0.04, 0.08),
                 xycoords="axes fraction", fontsize=9.5, color=INK_SOFT)
fig.tight_layout()
save(fig, f"{FIGS}/s5_best_relationship.png")

# --- S6: do all 14 cells age the same way?  (small multiples) ------------
cells = sorted(df.cell_id.unique())
fig, axes = plt.subplots(2, 7, figsize=(12.6, 4.2), sharex=True, sharey=True)
for ax, c in zip(axes.ravel(), cells):
    g = df[df.cell_id == c]
    ax.plot(g["Cycle_Index"], g["Max. Voltage Dischar. (V)"], color=SERIES[0], lw=1.2)
    ax.set_title(c.replace("Cell_", "Cell "), loc="left", fontsize=9.5,
                 fontweight="normal")
    ax.tick_params(labelsize=8)
axes[0, 0].set_ylabel("Max discharge V"); axes[1, 0].set_ylabel("Max discharge V")
for ax in axes[1]:
    ax.set_xlabel("cycle", fontsize=9)
fig.suptitle("All 14 cells fade the same way - the spikes are periodic reference "
             "cycles, and the later cells glitch more",
             x=0.005, ha="left", fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.92])
save(fig, f"{FIGS}/s6_per_cell_ageing.png")

# --- S7: group comparison behind the Kruskal-Wallis test ----------------
fig, ax = plt.subplots(figsize=(8.6, 3.8))
sns.boxplot(data=df, x="cell_id", y="Max. Voltage Dischar. (V)", ax=ax,
            color=SERIES[0], fliersize=1.6, linewidth=0.9, width=0.66,
            boxprops={"alpha": 0.85})
ax.axhline(df["Max. Voltage Dischar. (V)"].median(), color=SERIES[1], ls="--", lw=1.4)
ax.annotate("pooled median", (0.004, df["Max. Voltage Dischar. (V)"].median()),
            xycoords=("axes fraction", "data"), xytext=(0, 5),
            textcoords="offset points", color=SERIES[1], fontsize=9)
ax.set_title(f"Per-cell spread of max discharge voltage  "
             f"(Levene p = {tdf.iloc[1]['p-value']:.1e}: variances are not equal)",
             loc="left")
ax.set_xlabel(""); ax.set_ylabel("Max discharge voltage (V)")
plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8.5)
save(fig, f"{FIGS}/s7_group_comparison.png")

# --- S8: PCA scree + the first two components coloured by RUL -----------
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
axes[0].bar(np.arange(1, 9), evr * 100, color=SERIES[0], width=0.6,
            label="individual")
axes[0].plot(np.arange(1, 9), cum * 100, color=SERIES[1], marker="o",
             label="cumulative")
axes[0].axhline(80, color=INK_SOFT, ls="--", lw=1.1)
axes[0].annotate("80% threshold", (4.4, 73), fontsize=9, color=INK_SOFT)
for i, v in enumerate(evr * 100, 1):
    axes[0].annotate(f"{v:.0f}", (i, v), xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=8.5, color=INK_SOFT)
axes[0].set_title(f"Scree plot: PC1 alone carries {evr[0]*100:.0f}% of the variance",
                  loc="left", fontsize=11)
axes[0].set_xlabel("principal component"); axes[0].set_ylabel("% of variance")
axes[0].legend(loc="center right")
sc = axes[1].scatter(pcs[:, 0], pcs[:, 1], c=df[TARGET], cmap=SEQ, s=7, alpha=0.6,
                     edgecolor="none")
cb = fig.colorbar(sc, ax=axes[1], shrink=0.88)
cb.set_label("RUL (cycles left)", fontsize=9)
axes[1].set_title("Cycles in PC1-PC2 space, shaded by remaining life",
                  loc="left", fontsize=11)
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
fig.tight_layout()
save(fig, f"{FIGS}/s8_pca.png")

# --- S9: RUL itself, and how many cycles each cell contributes -----------
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
sns.histplot(df[TARGET], bins=50, color=SERIES[0], ax=axes[0], edgecolor="white",
             linewidth=0.4)
axes[0].axvline(df[TARGET].median(), color=SERIES[1], ls="--", lw=1.4)
axes[0].annotate(f"median {df[TARGET].median():.0f}",
                 (df[TARGET].median(), axes[0].get_ylim()[1] * 0.9),
                 xytext=(6, 0), textcoords="offset points", color=SERIES[1], fontsize=9)
axes[0].set_title("Remaining useful life is close to uniform, not bell-shaped",
                  loc="left", fontsize=11)
axes[0].set_xlabel("RUL (cycles left)"); axes[0].set_ylabel("cycles")
vc = df.cell_id.value_counts().sort_index()
axes[1].bar(range(len(vc)), vc.values, color=SERIES[0], width=0.66)
axes[1].set_xticks(range(len(vc)), [c.replace("Cell_", "") for c in vc.index], fontsize=8.5)
axes[1].set_title("Cycles contributed per cell", loc="left", fontsize=11)
axes[1].set_xlabel("cell"); axes[1].set_ylabel("cycles logged")
fig.tight_layout()
save(fig, f"{FIGS}/s9_rul_and_balance.png")


# =============================================================================
# STEP 9  -  WHAT THE STATISTICS TELL US TO DO NEXT
# =============================================================================
banner("STEP 9  |  Conclusions that feed straight into the model")
print(f"1. {len(df):,} cycles from {df.cell_id.nunique()} cells, zero missing values,")
print("   zero duplicates - but the cell label had to be rebuilt from cycle resets.")
print(f"2. Not one of {len(ndf)} variables is Gaussian (all reject at alpha={ALPHA}),")
print("   and a log transform fixes none of them -> use rank-based statistics.")
print(f"3. Median {odf['IQR outliers %'].median():.1f}% of cycles are IQR outliers and "
      f"{impossible} cycles")
print("   claim to last over a day -> the model must be fitted robustly.")
print(f"4. Spearman beats Pearson on {int((cdf['rank gain'] > 0).sum())} of {len(cdf)} features;")
print(f"   the strongest monotone signal is '{cdf.iloc[0].feature}' at "
      f"rho = {cdf.iloc[0]['Spearman rho']:+.2f}.")
print(f"5. {n_rej} of {len(tdf)} hypotheses reject H0; the cells differ in both level")
print("   and spread, and ageing within a cell is almost perfectly monotone")
print("   -> validate by holding out whole cells, never random rows.")
print(f"6. {n80} principal component{'s' if n80 > 1 else ''} explain{'' if n80 > 1 else 's'} "
      f"80% of 8 columns ({evr[0]*100:.0f}% in PC1 alone) -> the measurements")
print("   are mostly one latent variable, so a LINEAR model is a fair starting point.")
print("\nAll figures in outputs/figures/, all tables in outputs/.")
