"""House plot style for all notebooks.

Why a shared style: the notebooks are published as a website, and default
matplotlib (boxed axes, harsh spines, saturated primaries) reads as noise next
to styled prose. One `apply_plot_style()` call per notebook gives every figure
the same voice: recessive chrome (grid and axes fade behind the data), a
categorical palette whose *order* is chosen for color-vision-deficiency safety,
and ink colors that match the site's text hierarchy.

The palette details matter more than they look like they should:

- The eight categorical colors are used in this fixed order, never cycled or
  re-sorted. The order maximizes the minimum perceptual distance between
  *adjacent* pairs under CVD simulation (worst adjacent pair: deltaE 9.1 in
  OKLab x100, above the 8.0 target) — so stacked or neighboring series stay
  distinguishable for colorblind readers.
- Slots 3-5 (magenta, yellow, aqua) sit below 3:1 contrast on a light surface,
  so plots that use them should carry direct labels or a legend (all of ours
  do).
- Figures keep a light surface (#fcfcfb) even on dark site pages: a static PNG
  cannot switch with the theme, and a consistently light, softly-bordered
  figure beats a mismatched dark one.
"""

import matplotlib as mpl
from cycler import cycler

# Categorical palette (fixed order — see module docstring before reordering)
CATEGORICAL = [
    "#2a78d6",  # blue
    "#008300",  # green
    "#e87ba4",  # magenta
    "#eda100",  # yellow
    "#1baf7a",  # aqua
    "#eb6834",  # orange
    "#4a3aa7",  # violet
    "#e34948",  # red
]

_SURFACE = "#fcfcfb"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"


def apply_plot_style() -> None:
    """Apply the house matplotlib style. Call once, right after imports."""
    mpl.rcParams.update(
        {
            # Surface and chrome: recessive, so data carries the figure
            "figure.facecolor": _SURFACE,
            "axes.facecolor": _SURFACE,
            "savefig.facecolor": _SURFACE,
            "axes.edgecolor": _BASELINE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": _GRIDLINE,
            "grid.linewidth": 0.8,
            # Ink hierarchy mirrors the site's text colors
            "text.color": _INK_PRIMARY,
            "axes.titlecolor": _INK_PRIMARY,
            "axes.labelcolor": _INK_SECONDARY,
            "xtick.color": _INK_MUTED,
            "ytick.color": _INK_MUTED,
            "xtick.labelcolor": _INK_SECONDARY,
            "ytick.labelcolor": _INK_SECONDARY,
            # Series colors: fixed CVD-safe order
            "axes.prop_cycle": cycler(color=CATEGORICAL),
            # Marks: 2px lines, >=8px markers (readable at notebook width)
            "lines.linewidth": 2.0,
            "lines.markersize": 6.0,
            # Type
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "figure.titleweight": "semibold",
        }
    )
