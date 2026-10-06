"""
viz_style.py
------------
One place where every chart in this project gets its look.

Why a separate file?  Because a deck where each figure has its own fonts and its
own colours looks like four people made it.  Both analysis scripts import this
module, so every PNG that lands in outputs/figures/ comes out of the same system.

The palette is a colour-blind-safe categorical set (validated for adjacent-pair
separation under protanopia / deuteranopia / tritanopia).  Slots are assigned in
a FIXED order and never cycled - series 1 is always blue, series 2 always orange.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------------
# Categorical palette - assign in this order, never shuffle, never cycle.
# ----------------------------------------------------------------------------
SERIES = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# Scatter / bubble charts compare every pair at once, so they are capped at the
# first three slots (those three stay separable for all pairs, not just neighbours).
SCATTER_SAFE = SERIES[:3]

# Single-hue ramp for magnitude (heatmaps, density).  Light -> dark, one hue.
SEQ = "mako"          # seaborn's perceptually uniform blue-green ramp
DIVERGING = "vlag"    # two poles + neutral grey middle, for signed quantities

INK = "#17171a"        # primary text
INK_SOFT = "#5b5b66"   # secondary text / axis labels
GRID = "#e3e3e8"       # recessive gridlines
SURFACE = "#ffffff"    # the figures sit inside white cards on the dark slides


def use_project_style():
    """Apply the project-wide matplotlib/seaborn defaults. Call once per script."""
    sns.set_theme(style="whitegrid")
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlepad": 10,
        "axes.labelsize": 10.5,
        "axes.labelcolor": INK_SOFT,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def finish(ax, title=None, xlabel=None, ylabel=None, note=None):
    """Small helper so every axes ends up labelled the same way."""
    if title:
        ax.set_title(title, loc="left")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.9)
    ax.grid(False, axis="x")
    if note:
        ax.annotate(note, xy=(0, -0.22), xycoords="axes fraction",
                    fontsize=8.5, color=INK_SOFT)
    return ax


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"   figure saved -> {path}")
