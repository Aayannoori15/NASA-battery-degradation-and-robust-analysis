# Explaining Our Code in Plain English
### Group AD1 — Explainable Multimodal Physics-Aware Deep Learning for EV Battery
**Swastik Mukherjee** RA2411026010181 · **Prahladh Alarpati** RA2411026010171 · **Aayan Noori** RA2411026010165

This file is written so that any one of us can open it the night before the review
and be able to explain every line of our code without memorising anything. No jargon
is used without being unpacked first.

---

## 0. The one-paragraph version

An electric car's battery slowly wears out. The number that describes how worn out it
is called **State of Health (SOH)** — today's capacity divided by the capacity it had
when new. The honest way to measure SOH is to drain the battery completely and count
how much came out, which obviously no car can do while someone is driving it. So we
estimate it instead. We take **only the first ten minutes** of a discharge, pull out
fifteen numbers that a battery engineer would recognise (internal resistance, how fast
the voltage sags, how hot the cell gets, and so on), and fit a **straight line** from
those fifteen numbers to SOH. We fit that line using **robust regression**, which is
a version of linear regression that refuses to be dragged around by a few wrong
readings. Then we prove it refuses, by deliberately feeding the model faulty labels
and watching ordinary regression fall apart while ours does not.

**Headline result:** mean error of **1.14 percentage points of SOH** on a battery cell
the model had never seen, against **7.03 pp** for guessing the average. When we corrupt
20% of the training labels, ordinary least squares gets **29× worse** and our robust
fit gets **1.1× worse**.

---

## 1. What is in the folder

```
ML-project/
├── data/
│   ├── nasa_discharge_multimodal.csv   NASA PCoE: 4 cells, 169,766 sensor samples
│   └── hnei_battery_rul.csv            HNEI: 14 cells, 15,064 cycle summaries
├── src/
│   ├── viz_style.py                    shared chart styling for both scripts
│   ├── 01_model_robust_regression.py   THE MODEL (robust linear regression)
│   ├── 02_statistical_analysis.py      THE STATISTICS (matplotlib + seaborn)
│   └── make_ppt.py                     builds the slide deck from the results
├── outputs/
│   ├── figures/                        21 PNG charts, all used in the deck
│   └── *.csv                           every table the slides quote
├── EXPLANATION.md                      this file
└── ML_Review1_EV_Battery_SOH_AD1.pptx  the deck
```

Run order:

```bash
python3 src/02_statistical_analysis.py     # ~15 s
python3 src/01_model_robust_regression.py  # ~60 s
python3 src/make_ppt.py                    # ~5 s
```

---

## 2. Why two different datasets?

Because the two scripts do two different jobs, and using the same data for both would
make the second script look like a repeat of the first.

| | Dataset A — **NASA PCoE** | Dataset B — **HNEI** |
|---|---|---|
| Used by | `01_model_robust_regression.py` | `02_statistical_analysis.py` |
| What it is | 4 cells cycled to death at 24 °C | 14 cells, a different lab and chemistry |
| Shape | one row per **sensor sample inside a cycle** | one row per **cycle** |
| Columns | voltage, current, temperature over time | 8 summary measurements per cycle |
| Label | measured capacity (Ah) | remaining useful life (cycles left) |
| Why this one | it has the **three sensor streams** we need for "multimodal" | it is big and messy enough for a real statistics study |

---

# PART A — `01_model_robust_regression.py`

## Step 1 — Loading the data

```python
raw = pd.read_csv(DATA)
raw = raw.sort_values(["Battery", "id_cycle", "Time"])
```

The CSV has 169,766 rows, but that is **not** 169,766 batteries or even 169,766 cycles.
It is one row per *reading*. About 270 readings are taken while a single discharge
happens, so the file is really **636 discharge cycles** across 4 cells. Getting this
right matters: if you treat each row as an independent example you will think you have
a hundred times more data than you do.

> **If asked:** "How many data points do you actually have?" → 636. The 169k is raw
> sensor telemetry that we compress down to 636 feature rows.

## Step 2 — The leakage trap (the most important part of the whole project)

The obvious feature is *how long the discharge lasted* — an old battery runs out
sooner. We checked it:

```python
corr(full discharge duration, capacity label) = 0.9975
```

A correlation of 0.997 is suspiciously perfect, and here is why: the dataset's capacity
label was **computed by integrating the current over exactly that duration**. Current is
held at a constant 2 A, so capacity ≈ 2 A × duration. Duration and the label are the
same number wearing different units.

This is called **data leakage**: the answer has quietly leaked into the inputs. A model
using it would report R² ≈ 1.0 and would be completely useless, because in a real car
you do not have the full duration — that is the thing you are trying to avoid measuring.

So we **banned** total duration, total Ah and total Wh, and restricted every feature to
the **first 600 seconds** of the discharge. The shortest cycle in the whole dataset is
2,108 s, so the ten-minute window always exists and we never have to pad anything.

> **If asked:** "Why only 600 seconds?" → Two reasons. It removes the leakage, and it
> matches the real problem: a BMS wants SOH from a short window, not a full discharge.

## Step 3 — Feature engineering: where "physics-aware" and "multimodal" come from

This is the heart of the project. Each feature is a quantity with a name, a unit and a
physical meaning — that is what makes the final model **explainable**. A neural network
would invent 128 anonymous numbers; we build 15 that we can defend one by one.

**Multimodal = three different sensor streams**, not three copies of one.

### Modality 1 — Electrical (8 features)

```python
r_internal_mohm = (V_FULL_OCV - vw[0]) / i_abs[0] * 1000
```
Plain English: a fully charged cell sits at about **4.2 V** when nothing is connected.
The moment you draw 2 A, the voltage drops. **Ohm's law** says that drop ÷ current =
resistance. A fresh cell sags a little; an old cell sags a lot, because a layer of gunk
(the SEI layer) has built up on the electrode. This single number is the classic ageing
marker, and in our model it is the second-strongest feature.

- `v_at_60s`, `v_at_300s`, `v_at_600s` — voltage at three fixed stopwatch readings.
- `dvdt_mv_per_s` — the **slope** of the voltage plateau in millivolts per second. A
  healthy cell holds a flat plateau; a worn one slides down it.
- `band_39_38_s` — how many seconds it takes to fall from 3.90 V to 3.80 V. This is a
  cheap stand-in for *incremental capacity analysis*, a standard battery diagnostic.
- `wh_window`, `i_rms` — energy delivered in the window, and the load actually applied.

### Modality 2 — Thermal (4 features)

Heat is the fingerprint of resistance, because **power lost as heat = I²R**. A cell with
higher internal resistance runs hotter for the same current. So temperature is not a
duplicate of voltage, it is a second, independent witness.

- `t_rise_k` — how many degrees it warmed up.
- `temp_at_600s` — temperature at the ten-minute mark.
- `dtdt_mk_per_s` — heating rate.
- `thermal_dose_kh` — the area under the temperature curve above ambient, in K·h.
  Think of it as "how much heat exposure the cell accumulated".

### The couplings — where the two modalities multiply (3 features)

This is the "physics-aware" part in the strict sense: terms taken from battery
degradation theory rather than from the data.

- `ohmic_heat_wh = I²·R·t` — the Joule heating energy actually dissipated.
- `arrhenius_x1e4 = exp(−Eₐ / (R·T))` — the **Arrhenius equation**. Chemical side
  reactions that eat the battery speed up *exponentially* with temperature, not
  linearly. Feeding the model the exponential term means a straight line can capture a
  curved physical law. Eₐ = 20 kJ/mol is the standard literature value for SEI growth.
- `heat_per_wh` — degrees of heating bought per watt-hour delivered: a pure efficiency
  loss, and a direct sign of a degraded cell.

> **If asked:** "Where is the deep learning?" → Be honest. The project title is the
> research area; our assigned technique for this review is robust linear regression.
> What we have built is the **physics-aware, explainable, multimodal feature pipeline**
> that such a deep model would sit on top of, with a transparent linear model in the
> slot where the network would go. We state exactly that on the conclusion slide, and
> the next step in Future Work is a physics-informed neural network on the same inputs.

## Step 4 — Splitting train from test, the strict way

```python
TRAIN_CELLS = ["B0005", "B0006", "B0007"]   # 504 cycles
TEST_CELLS  = ["B0018"]                     # 132 cycles
```

We did **not** shuffle rows randomly. Cycle 50 and cycle 51 of the same cell are
practically twins; a random split would put one twin in training and the other in
testing, and the model would score brilliantly for the wrong reason.

Instead we hold out an **entire cell**. The model has never seen B0018 in any form. That
is exactly the question a real BMS faces: here is a battery I have never met, how
healthy is it?

## Step 5 — The five models, and the one idea that separates them

Every one of them fits the same straight line:

> SOH = b₀ + b₁·x₁ + b₂·x₂ + … + b₁₅·x₁₅

They differ **only in what they punish when a point does not fit**.

| Model | How it punishes an error | What that means |
|---|---|---|
| **OLS** (ordinary least squares) | squares it | a point 10× off counts 100× as much → one bad reading bends the whole line |
| **Huber** | squares small errors, switches to plain absolute past 1.35σ | normal points behave normally, outliers stop getting extra votes |
| **RANSAC** | ignores it entirely if it disagrees with the crowd | fits on random subsets, keeps the fit most points agree with, labels the rest outliers |
| **Theil-Sen** | takes the **median** of slopes over many pairs | medians are immune to extremes — but only up to ~29% bad data |
| **Quantile (median)** | plain absolute error | fits the conditional median instead of the conditional mean |

All five are wrapped in a `Pipeline` with `StandardScaler`, so each feature is converted
to "number of standard deviations from its average". Two reasons: the optimisers
converge better, and — more importantly for us — the coefficients become **directly
comparable**, which is what makes the explainability chart meaningful.

### Results on clean data (held-out cell B0018)

| Model | MAE (SOH pp) | R² |
|---|---|---|
| RANSAC | **1.08** | 0.958 |
| OLS | 1.08 | 0.958 |
| Huber | 1.14 | 0.953 |
| Quantile | 1.30 | 0.949 |
| Theil-Sen | 1.39 | 0.946 |
| *Baseline: predict the mean* | *7.03* | *−0.04* |

Note the honest part: **on clean data, plain OLS is just as good.** That is the expected
answer, and saying so is better than pretending otherwise. The reason to prefer robust
regression shows up in the next two tests.

### Leave-one-cell-out cross-validation

Train on two cells, test on the third, three times round:

| Model | MAE (pp) |
|---|---|
| Theil-Sen | **3.73** |
| Huber | 5.22 |
| Quantile | 7.60 |
| OLS / RANSAC | 9.96 |

With only two cells to learn from, OLS is **worse than useless** (R² = −0.36, i.e. worse
than guessing the average). The robust fits transfer to an unseen cell far better.

## Step 6 — The experiment that justifies the word "robust"

Real capacity labels come from coulomb counters, which drift, reset and occasionally
report nonsense. So we simulated exactly that: take p% of the **training** labels and
replace them with sensor-fault values — half stuck low at 35% SOH, half spiking to 160%
(which is physically impossible). The **test set stays clean**, so any loss of accuracy
is the model being fooled, nothing else.

| Corrupted training labels | OLS | Theil-Sen | Huber | Quantile | RANSAC |
|---|---|---|---|---|---|
| 0% | 1.08 | 1.39 | 1.14 | 1.30 | 1.08 |
| 10% | 6.34 | 1.40 | 1.17 | 1.21 | 1.08 |
| 20% | **30.95** | 22.94 | 1.29 | 1.25 | 1.17 |
| 30% | 7.98 | 16.00 | 1.15 | 1.17 | 1.08 |

Read the **worst row** for each model, not the last one — a model you would put in a car
has to survive its unluckiest batch.

- **OLS peaks at 30.95 pp — 29× its own clean error.** At that point it is not predicting
  battery health, it is predicting noise.
- **Huber's worst is 1.29 pp — 1.1× its clean error.** It barely notices.
- **Theil-Sen is the instructive failure.** Theory says its breakdown point is about 29%,
  meaning it can tolerate just under 29% bad data and then collapses. It holds to 15%
  and collapses at 20%, exactly on cue. We left this in the results on purpose: it shows
  we understand *why* each method is robust, not just that it is.

> **If asked:** "Why does OLS get *better* again at 30%?" → Our two fault modes are
> symmetric (half too low, half too high). At 30% they happen to roughly cancel out in
> the squared-error sum. That is luck, not robustness — which is exactly why we report
> the worst case rather than the endpoint.

## Step 7 — Explainability

Three separate things, because "explainable" is used loosely and we wanted to be precise.

**(a) Global — what did the model learn overall?**
Because the inputs were standardised, each coefficient reads as *"SOH changes by this
much when this feature moves one standard deviation"*. The biggest are `v_at_600s`
(+27.9 pp), `r_internal_mohm` (−26.7 pp) and `ohmic_heat_wh` (+24.2 pp). The signs make
physical sense: higher voltage still available after ten minutes means a healthier cell;
higher internal resistance means a worn one.

**(b) How much do we trust each coefficient?**
We bootstrapped: resample the training set 400 times with replacement, refit, and look
at the spread. **8 of the 15** coefficients keep the same sign in 95% of resamples. The
other 7 are not reliable and we say so on the slide rather than quietly dropping them.

**(c) Local — why this one prediction?**
Because the model is linear, a prediction is literally *intercept + a contribution per
feature*. For the most degraded test cycle (B0018, cycle 132): true SOH 67.1%, predicted
67.4%, starting from a 79.4% intercept and then −21.5 pp from temperature, −20.1 pp from
voltage at 600 s, −17.4 pp from temperature rise, and so on. Every prediction comes with
a receipt.

**The caveat we do not hide:** 20 pairs of our features correlate above 0.9 (for example
`r_internal_mohm` and `ohmic_heat_wh` at 0.998 — one is computed from the other). When
features overlap this much, they **share credit**, so an individual coefficient should be
read as part of a group, not as a standalone causal effect. This is on the slide.

---

# PART B — `02_statistical_analysis.py`

A completely separate study on a completely different dataset, using **pandas, numpy,
scipy, matplotlib and seaborn**. Its job is to describe what the data *is*, before
anyone models it.

## Step 1 — Recovering a column the dataset lost

The published HNEI CSV stacks all 14 cells on top of each other and **forgets to say
which row belongs to which cell**. But `Cycle_Index` counts 1, 2, 3… and then drops back
to 1 when the next cell starts. Every drop is a boundary:

```python
df["cell_id"] = "Cell_" + (df["Cycle_Index"].diff().lt(0).cumsum() + 1).astype(str).str.zfill(2)
```

That recovers all 14 cells exactly. Without it, every per-cell analysis below would be
impossible. (Nice detail to mention in a viva — it shows we actually looked at the data.)

## Step 2 — Descriptive statistics, and the cheapest outlier detector there is

Compare each column's **mean** to its **median**. If the mean sits far above the median,
a handful of huge values are dragging it. `Discharge Time (s)` has a mean **2.9× its
median** and an **excess kurtosis of 340** (a normal distribution has 0, and anything
above about 10 is already extreme). That column is a thin spike with a tail stretching
for days — literally: the longest "cycle" in the file lasted **11.1 days**.

A second red flag from the same table: `Decrement 3.6-3.4V (s)` has a **minimum of
−397,646 seconds**. A duration cannot be negative, so that is an instrument fault we can
point at without any statistics at all.

## Step 3 — Outliers, checked two ways on purpose

One rule is an opinion; two agreeing rules is evidence.

- **Tukey / IQR rule:** flag anything more than 1.5×IQR outside the quartiles.
- **Modified Z-score:** uses the **median** and the **MAD** (median absolute deviation)
  instead of the mean and standard deviation. This matters — outliers inflate the
  standard deviation, so a normal Z-score lets them hide inside the spread they
  themselves created. The MAD version cannot be fooled that way.

Result: a median of **1.8%** of cycles flagged, and **182 cycles** that claim to last
over 24 hours, which is physically impossible for an 18650 cell. These are logging
faults, not interesting physics — and they are precisely why the companion model is
fitted with Huber and RANSAC instead of OLS. **The two scripts connect here.**

## Step 4 — Is anything normal? (No.)

Two formal tests on all 9 variables:
- **D'Agostino's K²** combines skewness and kurtosis into one statistic.
- **Shapiro–Wilk** is more sensitive but caps out at 5,000 points, so we subsample.

**9 of 9 variables reject normality**, every one at p < 0.001, and a log transform
rescues **none** of them. The Q-Q plots say the same thing visually: every variable
tracks the straight line in the middle and then peels away at the tails.

Consequence, and this is the point of the step: every test afterwards is the
**rank-based (non-parametric)** version. Using a t-test or Pearson r here would be
reporting a number the data does not support.

## Step 5 — The best finding in the whole project

For each feature we computed correlation with RUL two ways:
- **Pearson r** — straight-line agreement on the raw values. Outliers move it a lot.
- **Spearman ρ** — the same calculation on the **ranks**. An outlier is squashed down to
  "the largest one", so it cannot dominate.

| Feature | Pearson r | Spearman ρ |
|---|---|---|
| **Discharge time** | **+0.01** (p = 0.14, *not significant*) | **+0.96** |
| Time at 4.15 V | +0.18 | +0.96 |
| 3.6→3.4 V decrement | +0.01 | +0.96 |
| Max discharge voltage | +0.78 | +0.93 |

Read the first row again. **Pearson says discharge time has no relationship with
remaining life at all.** Spearman says it is the single strongest signal in the dataset.
Both are computed on the same column. The difference is entirely the handful of
corrupted cycles — and anyone who computed only Pearson r would have thrown away their
best feature.

We also applied a **Holm–Bonferroni correction**, because testing 8 features on one
dataset gives 8 chances to get lucky. All 8 survive.

## Step 6 — Six hypothesis tests, with effect sizes

A p-value only says "probably not zero". The **effect size** says "and it is this big",
which is the part a reader actually needs. We report both every time.

| | Hypothesis | Test | Result |
|---|---|---|---|
| H1 | cells differ in voltage level | Kruskal–Wallis | reject (p ≈ 10⁻¹²⁴, ε² = 0.04) |
| H2 | variances equal across cells | Levene (median) | reject (p ≈ 10⁻⁶⁰) |
| H3 | voltage higher early than late | Mann–Whitney U | reject, **CLES = 0.982** |
| H4 | discharge time shifts with age | Mann–Whitney U | reject, median 2069 s → 1032 s |
| H5 | monotone fade within one cell | Spearman ρ | reject, **ρ = −0.96** |
| H6 | cells differ in RUL | Kruskal–Wallis | **cannot reject** |

- **CLES 0.982** means: pick a random early-life cycle and a random late-life cycle, and
  98.2% of the time the early one has the higher voltage. Ageing is unmistakable.
- **H6 is a deliberate null result that we kept in.** RUL is not a measurement — it is a
  countdown the dataset authors wrote in afterwards. Every cell ends at 0, so every cell
  has the same RUL distribution *by construction*. The test should fail to reject, and it
  does. Keeping a negative result in the report is good practice, and it also warns
  anyone who tries to use RUL as if it were sensor data.

## Step 7 — PCA: how many independent things are we really measuring?

Eight columns does not mean eight pieces of information. We ran PCA on the
**rank-transformed, standardised** data (ranks first, so the outliers cannot hijack the
components).

**PC1 alone explains 91.1% of the variance.** All eight measurements are mostly
restatements of one underlying thing: how far through its life the cell is.

That is a direct argument for the modelling choice in the other script. One dominant
latent axis means a **linear** model is a fair starting point — you do not need a deep
network to follow a single direction.

---

## 3. How the two scripts fit together (say this in the viva)

> The statistics script told us four things about battery data: it is **not Gaussian**,
> it is **full of sensor outliers**, the measurements collapse onto **one latent ageing
> axis**, and **cells differ from each other**. Each of those decided something in the
> model script. Not Gaussian + outliers → fit robustly, not by least squares. One latent
> axis → a linear model is a reasonable hypothesis. Cells differ → validate by holding
> out whole cells, never random rows. The analysis is not decoration; it is the reason
> the model is built the way it is.

---

## 4. Honest limitations (have these ready — examiners ask)

1. **Only 4 cells** in the modelling dataset, all at one ambient temperature (24 °C). We
   cannot claim the coefficients transfer to a hot climate or a cold one.
2. **Constant-current discharge**, not a real drive cycle. Real EV loads vary
   second to second.
3. **20 feature pairs correlate above 0.9**, so individual coefficients share credit.
4. **One held-out cell.** Leave-one-cell-out CV gives a wider, more honest error (3.7 pp
   at best) than the single-split number (1.1 pp). Both are on the slides.
5. **No deep network yet**, by design — the review's assigned technique is robust linear
   regression, and we built the explainable feature layer such a network would need.

## 5. Likely questions, short answers

**"Why not just use a neural network?"** With 636 examples and 15 features, a network
would overfit and give no reason for its answers. We would rather have a defensible 1.14
pp than an unexplainable 1.0 pp. The physics features we built are the hard part; the
network can be dropped in later.

**"Isn't 1.14 pp suspiciously good?"** It is good because the physics is strong, not
because we cheated — we deliberately banned the leaking feature (duration, r = 0.997)
and held out a whole cell. The leave-one-cell-out number of 3.7 pp is the more
conservative figure, and we show it too.

**"What does 'robust' actually mean here?"** It means the fit has a high *breakdown
point*: the share of corrupted data it can absorb before its answer becomes arbitrary.
OLS has a breakdown point of 0% — a single extreme point can move the line anywhere.
Theil–Sen's is about 29%, which our experiment confirms.

**"Why percentage points and not percent?"** Because SOH is already a percentage. An
error of "1.14 percentage points" means predicting 81.1% when the truth was 80.0% — it
avoids the confusion of saying "1% error" on a quantity measured in %.

**"What is multimodal about it?"** Three physically distinct sensor streams — voltage,
current, temperature — plus terms that multiply them together. Temperature is not a copy
of voltage: it carries resistance information through I²R, which is a second, independent
witness to the same ageing.
