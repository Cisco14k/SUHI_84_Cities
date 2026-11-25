"""Fig. 1: study design."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/n84ns/src")
from style import *
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
C = pd.read_csv("/home/claude/n84ns/data/cities.csv")
zmap = lambda k: "Tropical" if k[0] == "A" else "Arid" if k[0] == "B" else "Temperate" if k[0] == "C" else "Continental"
C["zone"] = C.koppen.map(zmap)
land = gpd.read_file("/home/claude/new/data/ne/ne_110m_land.shp")
cmap = LinearSegmentedColormap.from_list("tc", ["#F2F2F2", NPG["mint"], NPG["teal"], "#0B5D4E"])
fig = plt.figure(figsize=(W2, 128 * MM))
ax = fig.add_axes([0.05, 0.43, 0.80, 0.57])
land.plot(ax=ax, color="#E6E6E6", lw=0)
for z in ["Tropical", "Arid", "Temperate", "Continental"]:
    d = C[C.zone == z]
    sc = ax.scatter(d.lon, d.lat, c=d.cool_0_120.clip(0, 0.3), cmap=cmap, vmin=0, vmax=0.3, marker=ZMARK[z], s=16, edgecolor="#333333", lw=0.35,
                    label=f"{z} ({len(d)})", zorder=3)
ax.set_xlim(-170, 180); ax.set_ylim(-50, 72); ax.set_aspect(1.25); ax.axis("off")
leg = ax.legend(loc="lower left", title="Köppen zone (n)", title_fontsize=6.3, handletextpad=0.3, borderaxespad=0.2, bbox_to_anchor=(0.0, 0.02))
for h in leg.legend_handles: h.set_facecolor("white"); h.set_edgecolor("#333333")
cax = fig.add_axes([0.87, 0.52, 0.012, 0.33])
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 0.3)), cax=cax); cb.outline.set_linewidth(0.4)
cb.set_label("Canopy cooling within 120 m\n(°C per 10 pp)", fontsize=6.3); cb.ax.tick_params(labelsize=6)
ax.text(-0.01, 1.0, "a", transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
# b scenes by decade and region
ax = fig.add_axes([0.09, 0.09, 0.235, 0.26])
regions = ["Africa"] * 12 + ["Asia"] * 12 + ["C. America"] * 12 + ["Europe"] * 12 + ["N. America"] * 12 + ["Oceania"] * 12 + ["S. America"] * 12
C["region"] = regions
S = C.scenes.str.split("/", expand=True).astype(int); S.columns = ["1985–94", "1995–04", "2005–14", "2015–25"]; S["region"] = C.region
G = S.groupby("region").sum()
cols = [NPG["red"], NPG["cyan"], NPG["teal"], NPG["navy"], NPG["salmon"], NPG["lav"], NPG["brown"]]
bottom = np.zeros(4)
for i, (r, row) in enumerate(G.iterrows()):
    ax.bar(range(4), row.values, bottom=bottom, color=cols[i], width=0.7, label=r, lw=0); bottom += row.values
ax.set_xticks(range(4)); ax.set_xticklabels(G.columns, fontsize=6); ax.set_ylabel("Clear scenes analysed")
ax.legend(fontsize=5.6, ncol=2, loc="upper left", columnspacing=0.6, handlelength=0.9, bbox_to_anchor=(0, 1.13))
ax.set_ylim(0, 4600)
ax.text(-0.33, 1.13, "b", transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
# c rings schematic
ax = fig.add_axes([0.36, 0.05, 0.25, 0.33]); ax.set_aspect(1); ax.axis("off")
rs = np.random.default_rng(5)
for r, a in zip([600, 480, 360, 240, 120], [0.06, 0.08, 0.10, 0.12, 0.16]):
    ax.add_patch(Circle((0, 0), r, fc=NPG["mint"], alpha=a, ec="#555555", lw=0.5))
for _ in range(32):
    rr = rs.uniform(90, 600); t = rs.uniform(0, 2 * np.pi)
    ax.add_patch(Rectangle((rr * np.cos(t) - 30, rr * np.sin(t) - 30), 60, 60, fc=NPG["teal"], ec="none", alpha=0.9))
ax.add_patch(Rectangle((-30, -30), 60, 60, fc=NPG["red"], ec="k", lw=0.5))
for r in (120, 240, 360, 480, 600):
    ax.text(r * 0.707 + 6, r * 0.707 + 6, f"{r}", fontsize=5.6, color="#333333")
ax.set_xlim(-680, 680); ax.set_ylim(-1000, 680)
ax.text(0, -820, "Receptor (red), ring shares of canopy, open green,\nwater and bare ground; built-up cover is the reference", ha="center", va="top", fontsize=5.8)
ax.text(-0.05, 1.03, "c", transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
# d analysis chain
ax = fig.add_axes([0.64, 0.04, 0.35, 0.37]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
steps = [("Landsat LST and decadal land cover", "12,014 scenes, 84 city-regions, 1985–2025", NPG["navy"]),
         ("Scene-level double machine learning", "36 ring treatments; neighbourhood confounders", NPG["teal"]),
         ("Correlated-error pooling", "block bootstrap; random effects; frequency, reach", NPG["lav"]),
         ("Sensor-footprint model", "PSF, resampling, ring geometry; physical length L", NPG["red"]),
         ("60 m kernel DML", "WorldCover exposures; decay length L; ring widths", NPG["salmon"]),
         ("Modelled residential surface cooling", "residents, heat indicators, planting designs", NPG["cyan"]),
         ("Planning kernels and scenarios", "held-out forecasts; NEX-GDDP-CMIP6, 10 models", NPG["brown"])]
for i, (h, s, c) in enumerate(steps):
    y = 0.92 - i * 0.14
    ax.add_patch(FancyBboxPatch((0.02, y - 0.055), 0.96, 0.11, boxstyle="round,pad=0.0,rounding_size=0.02", fc="#F7F7F7", ec="#CCCCCC", lw=0.5))
    ax.add_patch(Rectangle((0.02, y - 0.055), 0.018, 0.11, fc=c, ec="none"))
    ax.text(0.07, y + 0.022, h, fontsize=6.3, fontweight="bold", va="center"); ax.text(0.07, y - 0.027, s, fontsize=5.7, va="center", color="#444444")
    if i < len(steps) - 1:
        ax.annotate("", xy=(0.5, y - 0.085), xytext=(0.5, y - 0.055), arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5))
ax.text(0.0, 1.08, "d", transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
save(fig, "Fig1")
