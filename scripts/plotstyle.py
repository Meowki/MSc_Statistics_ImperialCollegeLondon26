from pathlib import Path

import matplotlib as mpl


# colour palette
NAVY = "#002147"
PROCESS = "#006EAF"
CYAN = "#0091D4"
TANGERINE = "#EC7300"
TANGERINE_DK = "#B75500"
GREY = "#8A96A0"
TEXT = "#000000"
GRID = "#E6EEF4"

# figure sizes
FULL_W = 5.9
HALF_W = 2.9
SINGLE_H = 3.4

# matplotlib settings
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "mathtext.fontset": "dejavusans",
    "text.color": TEXT,
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.edgecolor": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    "axes.prop_cycle": mpl.cycler(color=[NAVY, TANGERINE, PROCESS, CYAN, TANGERINE_DK, GREY]),
    "figure.constrained_layout.use": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

# baseline styles
MODEL_STYLE = {
    "04": (TANGERINE, "o", "--"),
    "25": (NAVY, "s", "-"),
}

# calibration styles
CAL_STYLE = {
    "raw": (TANGERINE, "o", "--"),
    "recal_global": (CYAN, "s", "-."),
    "recal_stratified": (NAVY, "D", "-"),
}

# policy styles
POLICY_STYLE = {
    "fixed_global": (NAVY, "o", "-"),
    "fixed_stratified": (PROCESS, "s", "-"),
    "topk_global": (TANGERINE, "^", "--"),
    "topk_per_type": (TANGERINE_DK, "D", "--"),
    "controlled_rate_168h": (CYAN, "v", "-."),
    "qchart_ewma": (GREY, "P", ":"),
}

# reference lines
REFERENCE_STYLE = {
    "color": GREY,
    "linestyle": "--",
    "linewidth": 1.0,
}

# policy colours
POLICY_COLOR = {key: value[0] for key, value in POLICY_STYLE.items()}

# figure output
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"


def light_grid(ax, axis="both"):
    """light print-safe grid"""
    ax.grid(
        True,
        axis=axis,
        color=GRID,
        linewidth=0.7,
        zorder=0,
    )


def save_fig(fig, name):
    """pdf and png output"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300)
