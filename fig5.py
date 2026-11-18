"""Extended Data Fig. 9: Lagrangian moisture and heat sources of the overpass air."""
import sys, json, os, numpy as np, pandas as pd, geopandas as gpd
sys.path.insert(0, "/home/claude/n84/src")
from style import *
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
ROOT = "/home/claude/n84"
C = pd.read_csv(f"{ROOT}/data/cities_inc.csv")
zmap = lambda k: "Tropical" if k[0] == "A" else "Arid" if k[0] == "B" else "Temperate" if k[0] == "C" else "Continental"
C["zone"] = C.koppen.map(zmap)
L = pd.read_csv(f"{ROOT}/data/out/lagr_city.csv")
L20 = L[L.year == 2020].merge(C[["city", "zone", "cool_0_120", "reach_interp", "aridity", "canopy_pct", "lat", "lon"]], on="city")
REL = np.load(f"{ROOT}/data/out/lagr_rel.npz"); MAPS = np.load(f"{ROOT}/data/out/lagr_maps.npz")
QC = LinearSegmentedColormap.from_list("q", ["#FFFFFF", "#91D1C2", "#4DBBD5", "#3C5488"])
HC = LinearSegmentedColormap.from_list("h", ["#FFFFFF", "#F39B7F", "#E64B35", "#7E1F12"])
DG = [10, 25, 50, 100, 175, 250, 500, 750, 1000, 1500, 2000]
fig = plt.figure(figsize=(W2, 150 * MM))
gs = fig.add_gridspec(3, 3, left=0.065, right=0.985, top=0.965, bottom=0.06, wspace=0.42, hspace=0.5)
ext = [-805, 805, -805, 805]
for k, (key, cmap, lab, ttl) in enumerate((("q", QC, "a", "moisture uptake"), ("h", HC, "b", "sensible-heat gain"))):
    ax = fig.add_subplot(gs[0, k]); A = REL[key] / (10 * 10)          # per km2
    A = np.where(A > 0, A, np.nan); vmax = np.nanpercentile(A, 99.7)
    im = ax.imshow(A, origin="lower", extent=ext, cmap=cmap, norm=LogNorm(vmin=vmax / 300, vmax=vmax), interpolation="nearest")
    for r in (50, 250, 750):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, lw=0.6, ls="--", color="#333333"))
        ax.text(r * 0.72, -r * 0.72, f"{r}", fontsize=5, color="#333333", ha="left", va="top")
    ax.annotate("", xy=(700, 700), xytext=(450, 700), arrowprops=dict(arrowstyle="->", lw=0.8, color="k"))
    ax.text(575, 740, "wind", fontsize=5.4, ha="center", va="bottom")
    ax.set_xlim(-800, 800); ax.set_ylim(-800, 800); ax.set_aspect("equal")
    ax.set_xlabel("Along-wind distance (km)"); ax.set_ylabel("Crosswind distance (km)" if k == 0 else "")
    ax.set_title(f"{ttl}, {int(REL['n'])} city-years", fontsize=6, color="#444444", pad=3)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02); cb.ax.tick_params(labelsize=5.4)
    if k == 0: cb.set_label("share of total per km$^2$", fontsize=5.6)
    label(ax, lab)
# c cumulative share vs distance
ax = fig.add_subplot(gs[0, 2])
for key, col, nm in (("cq", NPG["navy"], "moisture"), ("ch", NPG["red"], "sensible heat")):
    M = np.array([[r[f"{key}_{g}"] for g in DG] for _, r in L20.iterrows()])
    q = np.nanpercentile(M, [25, 50, 75], 0)
    ax.fill_between(DG, 100 * q[0], 100 * q[2], color=col, alpha=0.18, lw=0); ax.plot(DG, 100 * q[1], color=col, lw=1.2, label=nm)
ax.set_xscale("log"); ax.set_xticks([10, 50, 250, 1000]); ax.set_xticklabels(["10", "50", "250", "1,000"]); ax.set_xlim(10, 2000)
ax.set_ylim(0, 100); ax.set_xlabel("Distance from city (km)"); ax.set_ylabel("Cumulative share (%)")
ax.legend(loc="upper left", fontsize=5.8, frameon=False)
ax.text(0.98, 0.04, f"{len(L20)} city-regions, 2020\nmedian and IQR", transform=ax.transAxes, fontsize=5.4, ha="right", va="bottom", color="#444444")
label(ax, "c")
# d world map: share of moisture from land
ax = fig.add_subplot(gs[1, :2])
land = gpd.read_file("/home/claude/new/data/ne/ne_110m_land.shp"); land.plot(ax=ax, color="#E8E8E8", lw=0)
LCM = LinearSegmentedColormap.from_list("lnd", ["#3C5488", "#8491B4", "#F2F2F2", "#B09C85", "#7E6148"])
sc = ax.scatter(L20.lon, L20.lat, c=100 * L20.land_q, cmap=LCM, vmin=0, vmax=100, s=8 + 40 * L20.q_250, edgecolor="#333333", lw=0.3, zorder=3)
ax.set_xlim(-170, 180); ax.set_ylim(-50, 70); ax.set_aspect("equal"); ax.axis("off")
cax = ax.inset_axes([0.62, 0.06, 0.3, 0.035]); cb = fig.colorbar(sc, cax=cax, orientation="horizontal"); cb.ax.tick_params(labelsize=5.4)
cb.set_label("Moisture taken up over land (%)", fontsize=5.8)
for s_ in (0.1, 0.3, 0.5):
    ax.scatter([], [], s=8 + 40 * s_, color="white", edgecolor="#333333", lw=0.3, label=f"{int(100 * s_)}%")
ax.legend(title="within 250 km", title_fontsize=5.4, fontsize=5.4, loc="lower left", frameon=False, ncol=3, columnspacing=0.6, handletextpad=0.1)
label(ax, "d", x=0.0)
# f meta: cooling within 120 m vs overpass VPD, coloured by zone
ax = fig.add_subplot(gs[1, 2])
for z in ZONE:
    d = L20[L20.zone == z]; ax.scatter(d.vpd_ov, d.cool_0_120, s=7, color=ZONE[z], lw=0, label=z)
import statsmodels.formula.api as smf
m = smf.ols("cool_0_120 ~ np.log(vpd_ov)", L20).fit(cov_type="HC3")
xx = np.linspace(L20.vpd_ov.min(), L20.vpd_ov.max(), 50); ax.plot(xx, m.params.iloc[0] + m.params.iloc[1] * np.log(xx), color="k", lw=0.8, ls="--")
ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks([0.5, 1, 2, 4]); ax.set_xticklabels(["0.5", "1", "2", "4"]); ax.set_ylim(-0.02, 0.5)
ax.set_xlabel("Overpass vapour-pressure deficit (kPa)"); ax.set_ylabel("Cooling 0–120 m (°C per 10 pp)")
ax.set_title(f"slope {m.params.iloc[1]:+.3f} per ln kPa, P = {m.pvalues.iloc[1]:.2f}".replace("-", "−"), fontsize=5.8, color="#444444", pad=3)
ax.legend(loc="upper left", fontsize=5.2, frameon=False, handletextpad=0.1, borderaxespad=0.1, ncol=2, columnspacing=0.5)
label(ax, "f")
# e example city maps
ex = [("Bamako", 3), ("Guangzhou", 19), ("Berlin", 36)]
gsi = gs[2, :].subgridspec(1, 3, wspace=0.08)
for k, (nm, i) in enumerate(ex):
    ax = fig.add_subplot(gsi[0, k]); key = f"{i:02d}"
    if f"{key}_q" in MAPS.files:
        q = MAPS[f"{key}_q"]; la = MAPS[f"{key}_lat"]; lo = MAPS[f"{key}_lon"]; oc = MAPS[f"{key}_oc"]
        q = np.where(q > 0, q, np.nan); vmax = np.nanpercentile(q, 99.5)
        ax.pcolormesh(lo, la, q, cmap=QC, norm=LogNorm(vmin=vmax / 100, vmax=vmax), shading="nearest", rasterized=True)
        ax.contour(lo, la, oc.astype(float), levels=[0.5], colors="#555555", linewidths=0.4)
        c = C.iloc[i]; ax.plot(c.lon, c.lat, marker="*", color=NPG["red"], ms=6, mec="k", mew=0.3)
        ax.set_xlim(lo.min(), lo.max()); ax.set_ylim(la.min(), la.max())
        dlon = 500 / (111.2 * np.cos(np.radians(c.lat))); x0 = lo.min() + 0.06 * (lo.max() - lo.min()); y0 = la.min() + 0.06 * (la.max() - la.min())
        ax.plot([x0, x0 + dlon], [y0, y0], color="k", lw=1.2, solid_capstyle="butt"); ax.text(x0 + dlon / 2, y0 + 0.3, "500 km", fontsize=5.2, ha="center", va="bottom")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(nm, fontsize=6, pad=2)
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(0.5)
    ax.set_aspect("auto")
    if k == 0: label(ax, "e", x=-0.12)
save(fig, "EDFig9")
