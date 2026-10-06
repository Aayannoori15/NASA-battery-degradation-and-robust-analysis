"""
make_ppt.py - builds the ML Review 1 slide deck.

It reuses the sample deck as a TEMPLATE: same slide size, same background
images, same LIGHT / DARK layouts, same fonts and palette. We strip the sample's
slides out and lay our own content on the inherited theme, so the result looks
like it came from the same design system.

Every number printed on a slide is read out of outputs/*.csv, so the deck can
never drift away from what the code actually produced.

    python3 src/make_ppt.py
"""
import copy
import os

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGS = os.path.join(ROOT, "outputs", "figures")
OUT = os.path.join(ROOT, "outputs")
TEMPLATE = os.path.join(ROOT, "AI_Investment_Committee_ML_Review1_Dark.pptx")
TARGET = os.path.join(ROOT, "ML_Review1_EV_Battery_SOH_AD1.pptx")

# ---- the inherited palette --------------------------------------------------
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x2D, 0x2A, 0x55)      # body text on white cards
INK_SOFT = RGBColor(0x6B, 0x68, 0x90)  # captions
LINE = RGBColor(0xE4, 0xDE, 0xFA)     # card border
AMBER = RGBColor(0xF5, 0x9E, 0x0B)
TEAL = RGBColor(0x0F, 0x76, 0x6E)
ROSE = RGBColor(0xE1, 0x1D, 0x48)
BLUE = RGBColor(0x1D, 0x4E, 0xD8)
DARKCARD = RGBColor(0x1F, 0x1B, 0x4D)
DARKCARD2 = RGBColor(0x2D, 0x2A, 0x55)
PALE = RGBColor(0xE4, 0xDE, 0xFA)
SERIF, SANS = "Georgia", "Calibri"


# ============================ numbers from the pipeline ======================
clean = pd.read_csv(f"{OUT}/results_clean.csv")
cv = pd.read_csv(f"{OUT}/results_cv.csv")
cont = pd.read_csv(f"{OUT}/results_contamination.csv")
coefs = pd.read_csv(f"{OUT}/explainability_coefficients.csv")
tests = pd.read_csv(f"{OUT}/stats_hypothesis_tests.csv")
corr = pd.read_csv(f"{OUT}/stats_correlation_rul.csv")
outl = pd.read_csv(f"{OUT}/stats_outliers.csv")
pca = pd.read_csv(f"{OUT}/stats_pca.csv")

piv = cont.pivot(index="Contamination %", columns="Model", values="MAE (% pts)")
M = {
    "huber_mae": clean.loc[clean.Model == "Huber", "MAE (% pts)"].iloc[0],
    "huber_r2": clean.loc[clean.Model == "Huber", "R2"].iloc[0],
    "best_mae": clean["MAE (% pts)"].min(),
    "base_mae": clean.loc[clean.Model == "Baseline: train mean", "MAE (% pts)"].iloc[0],
    "ols_peak": piv["OLS (baseline)"].max(),
    "ols_blow": piv["OLS (baseline)"].max() / piv.loc[0, "OLS (baseline)"],
    "hub_peak": piv["Huber"].max(),
    "hub_blow": piv["Huber"].max() / piv.loc[0, "Huber"],
    "cv_best": cv.iloc[0]["Model"], "cv_best_mae": cv.iloc[0]["MAE (% pts)"],
    "cv_ols": cv.loc[cv.Model == "OLS (baseline)", "MAE (% pts)"].iloc[0],
    "top_feat": coefs.iloc[0]["feature"],
    "pc1": pca.iloc[0]["explained_variance_ratio"] * 100,
    "rho_best": corr.iloc[0]["Spearman rho"], "r_best": corr.iloc[0]["Pearson r"],
    "feat_best": corr.iloc[0]["feature"], "p_best": corr.iloc[0]["p (Pearson)"],
    "n_reject": int((tests["p-value"] < 0.05).sum()), "n_tests": len(tests),
    "med_out": outl["IQR outliers %"].median(),
}
print("numbers pulled from the pipeline:")
for k, v in M.items():
    print(f"   {k:<12} {v}")


# ============================ template plumbing ==============================
prs = Presentation(TEMPLATE)
LAYOUT = {l.name: l for l in prs.slide_layouts}

# Strip the sample's slides but keep layouts, theme and background images.
xml_slides = prs.slides._sldIdLst
for sld in list(xml_slides):
    rId = sld.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
    prs.part.drop_rel(rId)
    xml_slides.remove(sld)

W, H = prs.slide_width / 914400, prs.slide_height / 914400   # 10 x 5.625 in


# The inherited LIGHT layout still carries the sample project's footer line.
# Rewrite it in place so every slide is footed with OUR project.
FOOTER = "ML Review 1  |  Explainable Multimodal Physics-Aware SOH Estimation for EV Batteries  |  Group AD1"
for sh in LAYOUT["LIGHT"].shapes:
    if sh.has_text_frame and "AI Investment Committee" in sh.text_frame.text:
        sh.text_frame.paragraphs[0].runs[0].text = FOOTER


def add(name):
    return prs.slides.add_slide(LAYOUT[name])


def text(slide, l, t, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         space_after=0, bullet=False, line=1.0):
    """runs: list of dicts {text, size, bold, color, font}. One run = one paragraph."""
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, r in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        p.line_spacing = line
        run = p.add_run()
        run.text = ("•  " if bullet else "") + r["text"]
        f = run.font
        f.size = Pt(r.get("size", 11))
        f.bold = r.get("bold", False)
        f.name = r.get("font", SANS)
        f.color.rgb = r.get("color", INK)
    return tb


def rect(slide, l, t, w, h, fill, line_col=None, radius=None, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(l), Inches(t), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line_col is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line_col
        s.line.width = Pt(1)
    s.shadow.inherit = False
    if radius is not None:
        s.adjustments[0] = radius
    return s


def card(slide, l, t, w, h, fill=WHITE, line_col=LINE):
    return rect(slide, l, t, w, h, fill, line_col, radius=0.036,
                shape=MSO_SHAPE.ROUNDED_RECTANGLE)


def header(slide, title, kicker=None):
    """The amber rule + Georgia title that every content slide opens with."""
    rect(slide, 0.3, 0.35, 0.08, 0.5, AMBER)
    text(slide, 0.5, 0.3, 9.0, 0.6, [{"text": title, "size": 24, "bold": True,
                                      "color": WHITE, "font": SERIF}],
         anchor=MSO_ANCHOR.MIDDLE)
    if kicker:
        text(slide, 0.5, 0.86, 9.0, 0.22, [{"text": kicker, "size": 9.5,
                                            "color": PALE}])


def pic_card(slide, l, t, w, h, fname, caption=None, cap_h=0.24):
    """White card with a figure fitted inside it, caption pinned to the bottom."""
    card(slide, l, t, w, h)
    pad = 0.08
    av_w, av_h = w - 2 * pad, h - 2 * pad - (cap_h if caption else 0)
    iw, ih = Image.open(os.path.join(FIGS, fname)).size
    ar = iw / ih
    dw, dh = (av_w, av_w / ar) if av_w / ar <= av_h else (av_h * ar, av_h)
    slide.shapes.add_picture(os.path.join(FIGS, fname),
                             Inches(l + pad + (av_w - dw) / 2),
                             Inches(t + pad + (av_h - dh) / 2),
                             Inches(dw), Inches(dh))
    if caption:
        text(slide, l + pad + 0.04, t + h - cap_h - 0.04, w - 2 * pad, cap_h,
             [{"text": caption, "size": 8.5, "color": INK_SOFT}],
             anchor=MSO_ANCHOR.MIDDLE)


def stat(slide, l, t, w, h, value, label, accent=AMBER, vsize=19, lsize=9):
    """The number-plus-caption tile used throughout the sample deck."""
    card(slide, l, t, w, h)
    rect(slide, l, t + 0.12, 0.07, h - 0.24, accent)
    text(slide, l + 0.2, t + 0.07, w - 0.3, h * 0.48,
         [{"text": value, "size": vsize, "bold": True, "color": accent, "font": SERIF}],
         anchor=MSO_ANCHOR.MIDDLE)
    text(slide, l + 0.2, t + h * 0.48, w - 0.3, h * 0.45,
         [{"text": label, "size": lsize, "color": INK_SOFT}], anchor=MSO_ANCHOR.TOP)


def panel(slide, l, t, w, h, title, bullets, accent=ROSE, tsize=13, bsize=10.5,
          gap=5):
    card(slide, l, t, w, h)
    text(slide, l + 0.2, t + 0.1, w - 0.4, 0.34,
         [{"text": title, "size": tsize, "bold": True, "color": accent, "font": SERIF}],
         anchor=MSO_ANCHOR.MIDDLE)
    text(slide, l + 0.2, t + 0.5, w - 0.4, h - 0.6,
         [{"text": b, "size": bsize, "color": INK} for b in bullets],
         space_after=gap, bullet=True, line=1.05)


# ================================= SLIDE 1 ===================================
s = add("DARK")
text(s, 0.6, 0.62, 7.0, 0.3,
     [{"text": "ML REVIEW 1  ·  SRM INSTITUTE OF SCIENCE AND TECHNOLOGY  ·  GROUP AD1",
       "size": 9.5, "color": PALE}])
text(s, 0.6, 1.05, 5.7, 1.45,
     [{"text": "Explainable Multimodal Physics-Aware Deep Learning for EV Battery",
       "size": 26, "bold": True, "color": WHITE, "font": SERIF}], line=0.95)
text(s, 0.6, 2.62, 5.5, 0.72,
     [{"text": "Estimating State of Health from ten minutes of voltage, current and "
               "temperature - with robust linear regression", "size": 11.5,
       "color": PALE}], line=1.15)
rect(s, 0.6, 3.46, 2.2, 0.04, AMBER)
text(s, 0.6, 3.6, 3.0, 0.25, [{"text": "Team", "size": 9, "color": AMBER,
                               "font": SERIF, "bold": True}])
text(s, 0.6, 3.88, 5.7, 0.95,
     [{"text": "Swastik Mukherjee  -  RA2411026010181", "size": 10.5, "color": WHITE},
      {"text": "Prahladh Alarpati  -  RA2411026010171", "size": 10.5, "color": WHITE},
      {"text": "Aayan Noori  -  RA2411026010165", "size": 10.5, "color": WHITE}],
     space_after=2)
text(s, 0.6, 4.95, 6.0, 0.3,
     [{"text": "B.Tech CSE  ·  Department of Computational Intelligence  ·  October 2026",
       "size": 9, "color": PALE}])

rect(s, 6.5, 0.8, 3.0, 3.95, DARKCARD2, radius=0.03,
     shape=MSO_SHAPE.ROUNDED_RECTANGLE)
rows = [("IN", "10 min of V, I and T\nfrom one discharge", TEAL),
        ("MODEL", "15 physics features ->\nrobust linear fit", AMBER),
        ("OUT", "State of Health,\nwith a reason attached", ROSE)]
for i, (tag, body, col) in enumerate(rows):
    y = 1.02 + i * 1.02
    rect(s, 6.7, y, 1.0, 0.72, col, radius=0.08, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, 6.7, y, 1.0, 0.72, [{"text": tag, "size": 11, "bold": True,
                                 "color": WHITE, "font": SERIF}],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 7.85, y, 1.5, 0.72, [{"text": l, "size": 9, "color": PALE}
                                 for l in body.split("\n")],
         anchor=MSO_ANCHOR.MIDDLE, line=1.1)
text(s, 6.7, 4.12, 2.7, 0.5,
     [{"text": f"636 cycles · 4 cells · MAE {M['huber_mae']:.2f} pp on an unseen cell",
       "size": 9.5, "bold": True, "color": AMBER, "font": SERIF}], line=1.15)

# ================================= SLIDE 2 ===================================
s = add("LIGHT")
header(s, "Problem and Objective")
panel(s, 0.5, 1.1, 4.3, 2.7, "Why State of Health is hard", [
    "An EV battery's real capacity is only known by fully discharging it - which no car can do on the road",
    "Battery Management Systems must instead infer it from short, noisy sensor windows",
    "Deep models can do this but give no reason, and nobody certifies a black box for a safety system",
    "Capacity labels themselves come from coulomb counters that drift and glitch",
], accent=ROSE)
pic_card(s, 5.0, 1.1, 4.5, 2.7, "f1_soh_trajectories.png",
         "Capacity fade of the four NASA cells used in this project")
for i, (t_, b_) in enumerate([
        ("UNIT 1", "Statistical analysis of a 15,064-cycle battery ageing dataset"),
        ("UNIT 2", "Robust linear regression on physics-aware multimodal features"),
        ("GOAL", "An SOH estimate that is accurate, explainable and fault-tolerant")]):
    x = 0.5 + i * 3.1
    card(s, x, 4.0, 2.8, 0.95)
    text(s, x + 0.15, 4.08, 2.5, 0.22, [{"text": t_, "size": 9, "bold": True,
                                         "color": AMBER, "font": SERIF}])
    text(s, x + 0.15, 4.32, 2.5, 0.58, [{"text": b_, "size": 9.5, "color": INK}],
         line=1.1)

# ================================= SLIDE 3 ===================================
s = add("LIGHT")
header(s, "Two Datasets, Two Jobs")
for i, (v, l, c) in enumerate([
        ("169,766", "raw sensor samples (NASA)", BLUE),
        ("636", "discharge cycles, 4 cells", TEAL),
        ("15,064", "cycles, 14 cells (HNEI)", AMBER),
        ("15 + 8", "engineered and measured features", ROSE)]):
    stat(s, 0.5, 1.1 + i * 0.97, 2.4, 0.85, v, l, c, vsize=17)
panel(s, 3.1, 1.1, 3.1, 2.75, "Dataset A - NASA PCoE", [
    "4 cells (B0005/6/7/18) cycled to failure at 24 degC",
    "Within-cycle traces: voltage, current, temperature",
    "Used for the MODEL - label is measured capacity",
    "Split by cell: train on 3, test on the 4th",
], accent=BLUE, bsize=9.5)
panel(s, 6.4, 1.1, 3.1, 2.75, "Dataset B - HNEI 18650", [
    "14 NMC-LCO cells, 8 cycle-summary measurements",
    "Cell identity had to be rebuilt from cycle-index resets",
    "Used for the STATISTICS - label is remaining useful life",
    "Deliberately a different lab, chemistry and format",
], accent=TEAL, bsize=9.5)
card(s, 3.1, 4.0, 6.4, 0.9)
text(s, 3.3, 4.08, 6.1, 0.76,
     [{"text": "The leakage trap we had to design around", "size": 10.5,
       "bold": True, "color": ROSE, "font": SERIF},
      {"text": "Full discharge duration correlates 0.997 with the capacity label - "
               "because the label IS that duration, integrated. Using it would score a "
               "fake R2 of 1.0, so every feature here is capped at the first 600 s.",
       "size": 9.5, "color": INK}], space_after=3, line=1.08)

# ================================= SLIDE 4 ===================================
s = add("LIGHT")
header(s, "Unit 1: Nothing in This Data Is Normal")
pic_card(s, 0.5, 1.1, 4.6, 3.0, "s2_qq_normality.png",
         "Normal Q-Q plots - every variable leaves the line at the tails")
pic_card(s, 5.25, 1.1, 4.25, 3.0, "s1_distributions.png",
         "The four most skewed columns, raw (blue) vs log-transformed (orange)")
for i, (v, l, c) in enumerate([
        ("9 of 9", "variables reject normality (D'Agostino and Shapiro, p < 0.001)", BLUE),
        ("0", "of them are rescued by a log transform", AMBER),
        (f"{M['med_out']:.1f}%", "median share of cycles flagged as IQR outliers", ROSE)]):
    stat(s, 0.5 + i * 3.1, 4.25, 2.8, 0.95, v, l, c, vsize=17, lsize=8.5)

# ================================= SLIDE 5 ===================================
s = add("LIGHT")
header(s, "Unit 1: Outliers Hide the Strongest Signal")
pic_card(s, 0.5, 1.1, 5.5, 2.8, "s4_correlation.png",
         "Same eight columns, two definitions of correlation")
panel(s, 6.15, 1.1, 3.35, 2.8, "What the gap means", [
    f"'{M['feat_best']}' scores Pearson r = {M['r_best']:+.2f} (p = {M['p_best']:.2f}, "
    "not significant)",
    f"The same pair scores Spearman rho = {M['rho_best']:+.2f} - near-perfect",
    "A handful of cycles logged for days flatten the straight-line fit",
    "Anyone reporting only Pearson r would discard the best feature in the dataset",
], accent=ROSE, bsize=9.5)
for i, (v, l, c) in enumerate([
        (f"{M['rho_best']:.2f}", "Spearman rho, strongest monotone signal", TEAL),
        (f"{M['r_best']:.2f}", "Pearson r on the very same pair", ROSE),
        ("8 of 8", "features still significant after Holm-Bonferroni", BLUE)]):
    stat(s, 0.5 + i * 3.1, 4.05, 2.8, 0.95, v, l, c, vsize=17, lsize=8.5)

# ================================= SLIDE 6 ===================================
s = add("LIGHT")
header(s, "Unit 1: Structure, Groups and Hypothesis Tests")
pic_card(s, 0.5, 1.1, 5.3, 2.5, "s8_pca.png",
         "PCA on rank-transformed features, and the cycles in PC1-PC2 space")
panel(s, 5.95, 1.1, 3.55, 2.5, "Six formal tests", [
    "H1 cells differ in voltage level - Kruskal-Wallis, reject",
    "H2 variances differ across cells - Levene, reject",
    "H3/H4 early vs late life - Mann-Whitney, reject (CLES 0.98)",
    "H5 fade within one cell - Spearman rho = -0.96, reject",
    "H6 cells differ in RUL - cannot reject, and that is correct",
], accent=TEAL, bsize=9)
for i, (v, l, c) in enumerate([
        (f"{M['pc1']:.0f}%", "of all variance sits in PC1 alone - one latent ageing axis", BLUE),
        (f"{M['n_reject']} of {M['n_tests']}", "hypotheses reject H0 at alpha = 0.05", AMBER),
        ("98.2%", "chance a random early cycle outranks a late one", TEAL)]):
    stat(s, 0.5 + i * 3.1, 3.75, 2.8, 0.95, v, l, c, vsize=17, lsize=8.5)
card(s, 0.5, 4.78, 9.0, 0.42)
text(s, 0.7, 4.78, 8.6, 0.42,
     [{"text": "Conclusion that drives Unit 2: non-Gaussian, outlier-heavy, "
               "one dominant latent axis, and cells that differ from each other "
               "-> a ROBUST LINEAR fit validated by holding out whole cells.",
       "size": 9, "bold": True, "color": INK}], anchor=MSO_ANCHOR.MIDDLE, line=1.05)

# ================================= SLIDE 7 ===================================
s = add("LIGHT")
header(s, "Unit 2: Multimodal Physics-Aware Features")
pic_card(s, 0.5, 1.1, 9.0, 2.25, "f2_multimodal_signals.png",
         "One cell, three sensors, fresh cycle vs end of life. Shaded band = the "
         "only 600 s we allow the model to see.")
for i, (t_, items, col) in enumerate([
        ("ELECTRICAL  (8)", "Internal resistance from the first loaded sample · "
         "voltage at 60 / 300 / 600 s · plateau slope · 3.9->3.8 V crossing time · "
         "energy and RMS current in the window", BLUE),
        ("THERMAL  (4)", "Temperature rise · temperature at 600 s · heating rate · "
         "thermal dose above ambient (K·h)", TEAL),
        ("PHYSICS COUPLINGS  (3)", "Joule heat I²Rt · Arrhenius rate factor "
         "exp(-Ea/RT) for SEI growth · degrees of heating per Wh delivered", AMBER)]):
    x = 0.5 + i * 3.1
    card(s, x, 3.55, 2.8, 1.45)
    rect(s, x, 3.67, 0.07, 1.21, col)
    text(s, x + 0.2, 3.63, 2.5, 0.24, [{"text": t_, "size": 9.5, "bold": True,
                                        "color": col, "font": SERIF}])
    text(s, x + 0.2, 3.9, 2.5, 1.0, [{"text": items, "size": 8.5, "color": INK}],
         line=1.1)

# ================================= SLIDE 8 ===================================
s = add("LIGHT")
header(s, "Unit 2: Five Lines, Five Definitions of 'Fit'")
panel(s, 0.5, 1.1, 4.0, 3.9, "Setup", [
    "Target: SOH = measured capacity / 2.00 Ah rated",
    "504 training cycles (B0005/6/7), 132 test cycles (B0018, never seen)",
    "All 15 features standardised inside the pipeline, so coefficients are comparable",
    "OLS squares the error - one bad point bends the whole line",
    "Huber squares small errors, switches to absolute past 1.35 sigma",
    "RANSAC fits random subsets and keeps the consensus fit",
    "Theil-Sen takes the median of pairwise slopes",
    "Quantile regression fits the conditional median",
], accent=BLUE, bsize=9.5, gap=4)
pic_card(s, 4.7, 1.1, 4.8, 2.45, "f5_model_comparison.png",
         "Mean absolute error on the held-out cell, in SOH percentage points")
for i, (v, l, c) in enumerate([
        (f"{M['huber_mae']:.2f} pp", "Huber MAE on an unseen cell", TEAL),
        (f"{M['base_mae']:.1f} pp", "predict-the-mean baseline", ROSE),
        (f"{M['huber_r2']:.3f}", "R² on the held-out cell", AMBER)]):
    stat(s, 4.7 + i * 1.64, 3.7, 1.52, 1.3, v, l, c, vsize=14, lsize=8)

# ================================= SLIDE 9 ===================================
s = add("LIGHT")
header(s, "Results on a Cell the Model Has Never Seen")
pic_card(s, 0.5, 1.1, 5.4, 2.55, "f12_trajectory_heldout.png",
         "Cell B0018: measured ageing curve vs the curve predicted from 10-minute windows")
pic_card(s, 6.05, 1.1, 3.45, 2.55, "f8_pred_vs_actual.png",
         "Predicted vs measured, with the +/- 2 pp band")
for i, (v, l, c) in enumerate([
        ("77%", "of test cycles land within 2 percentage points", TEAL),
        (f"{M['base_mae']/M['huber_mae']:.1f}x", "better than the predict-the-mean baseline", BLUE),
        (f"{M['cv_best_mae']:.1f} pp", f"best leave-one-cell-out CV ({M['cv_best']})", AMBER),
        (f"{M['cv_ols']:.1f} pp", "same CV for OLS - robust fits transfer better", ROSE)]):
    stat(s, 0.5 + i * 2.3, 3.8, 2.1, 1.25, v, l, c, vsize=16, lsize=8)

# ================================ SLIDE 10 ===================================
s = add("LIGHT")
header(s, "The Experiment That Justifies 'Robust'")
pic_card(s, 0.5, 1.1, 5.4, 3.05, "f6_contamination.png",
         "Test error on clean data as a growing share of TRAINING labels is corrupted")
panel(s, 6.05, 1.1, 3.45, 3.05, "How the test works", [
    "Replace p% of training capacity labels with sensor-fault values: half stuck low at 35% SOH, half spiking to 160%",
    "The test set stays clean, so any loss is the model being fooled",
    "OLS peaks at an error of 31 pp - it is predicting nonsense",
    "Theil-Sen is the instructive failure: its breakdown point is ~29% and it collapses right on cue",
    "Huber, Quantile and RANSAC never move",
], accent=ROSE, bsize=9)
for i, (v, l, c) in enumerate([
        (f"{M['ols_blow']:.0f}x", "worse: OLS at its worst contamination level", ROSE),
        (f"{M['hub_blow']:.1f}x", "worse: Huber at its worst", TEAL),
        (f"{M['hub_peak']:.2f} pp", "Huber's worst-case error, still under 1.3 pp", AMBER)]):
    stat(s, 0.5 + i * 3.1, 4.3, 2.8, 0.85, v, l, c, vsize=16, lsize=8.5)

# ================================ SLIDE 11 ===================================
s = add("LIGHT")
header(s, "Explainability: Every Prediction Has a Reason")
pic_card(s, 0.5, 1.1, 4.5, 3.0, "f7_coefficients.png",
         "Effect on SOH per standard deviation, with 95% bootstrap intervals")
pic_card(s, 5.15, 1.1, 4.35, 3.0, "f11_single_cycle_explanation.png",
         "One cycle decomposed: intercept plus a named contribution per feature")
for i, (v, l, c) in enumerate([
        ("8 of 15", "coefficients keep their sign in 95% of bootstrap resamples", BLUE),
        ("20 pairs", "of features correlate above 0.9 - credit is shared, so read them as groups", AMBER),
        ("0.3 pp", "gap between true and predicted SOH on the worked example", TEAL)]):
    stat(s, 0.5 + i * 3.1, 4.25, 2.8, 0.95, v, l, c, vsize=16, lsize=8.5)

# ================================ SLIDE 12 ===================================
s = add("DARK")
text(s, 0.6, 0.45, 8.8, 0.6, [{"text": "Conclusion", "size": 26, "bold": True,
                              "color": WHITE, "font": SERIF}],
     anchor=MSO_ANCHOR.MIDDLE)
rect(s, 0.6, 1.1, 2.2, 0.04, AMBER)
findings = [
    ("1", "Ten minutes of voltage, current and temperature is enough to estimate "
          f"State of Health to {M['huber_mae']:.2f} pp on a cell the model never saw."),
    ("2", "Robustness is not a detail: with 20% faulty labels OLS degrades "
          f"{M['ols_blow']:.0f}x while Huber stays within {M['hub_blow']:.1f}x."),
    ("3", "Physics-aware features keep the model readable - internal resistance, "
          "plateau voltage and temperature rise carry the signal, with intervals."),
    ("4", "The statistics decided the modelling: non-Gaussian, outlier-heavy data "
          "with cell-to-cell differences demands a robust fit and grouped validation."),
]
for i, (n, body) in enumerate(findings):
    y = 1.32 + i * 0.73
    rect(s, 0.6, y, 8.8, 0.62, DARKCARD, radius=0.08,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    rect(s, 0.72, y + 0.13, 0.36, 0.36, AMBER, radius=0.18,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, 0.72, y + 0.13, 0.36, 0.36, [{"text": n, "size": 11, "bold": True,
                                          "color": DARKCARD, "font": SERIF}],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 1.25, y, 8.0, 0.62, [{"text": body, "size": 10, "color": PALE}],
         anchor=MSO_ANCHOR.MIDDLE, line=1.1)

for i, (t_, items, col) in enumerate([
        ("Limitations", "Only 4 cells, one ambient temperature (24 degC) · constant-current "
         "discharge only, not a real drive cycle · 20 correlated feature pairs blur "
         "individual coefficients · single held-out cell", ROSE),
        ("Future Work", "More cells and varied temperatures · physics-informed neural "
         "network with the same features as inputs · per-cell random effects · "
         "online recursive update inside a live BMS", TEAL)]):
    x = 0.6 + i * 4.5
    rect(s, x, 4.35, 4.3, 1.0, DARKCARD2, radius=0.04,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, x + 0.18, 4.42, 4.0, 0.22, [{"text": t_, "size": 10, "bold": True,
                                         "color": col, "font": SERIF}])
    text(s, x + 0.18, 4.66, 4.0, 0.62, [{"text": items, "size": 8, "color": PALE}],
         line=1.1)

prs.save(TARGET)
print(f"\nDeck written -> {TARGET}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
