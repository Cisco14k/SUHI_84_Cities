"""Nature-portfolio figure style: Liberation Sans (Arial metric), 7 pt text, 180 mm double-column width, NPG palette."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
NPG = dict(red="#E64B35", cyan="#4DBBD5", teal="#00A087", navy="#3C5488", salmon="#F39B7F", lav="#8491B4", mint="#91D1C2",
           dred="#DC0000", brown="#7E6148", beige="#B09C85", grey="#7F7F7F")
SRC = {"Tree canopy": NPG["teal"], "Open green": "#8FB339", "Water": NPG["navy"], "Bare": NPG["brown"]}
ZONE = {"Tropical": NPG["red"], "Arid": "#E6A119", "Temperate": NPG["cyan"], "Continental": NPG["navy"]}
ZMARK = {"Tropical": "o", "Arid": "^", "Temperate": "s", "Continental": "D"}
MM = 1 / 25.4; W2 = 180 * MM; W1 = 88 * MM
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.3, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.1, "lines.markersize": 3.5, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 450, "pdf.fonttype": 42, "svg.fonttype": "none",
                     "legend.handlelength": 1.6, "legend.borderaxespad": 0.3, "axes.titlepad": 3})
FIG = "/home/claude/n84/figs"


def label(ax, s, x=-0.16, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", dpi=450); fig.savefig(f"{FIG}/{name}.pdf"); plt.close(fig)
