"""Fig. 6: kernels in held-out cities, residents within reach by income group, and projected warming against what canopy within reach can offset."""
import sys, numpy as np, pandas as pd, geopandas as gpd
sys.path.insert(0, "/home/claude/n84/src")
from style import *
from sitab import t, ci
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
SEQ = LinearSegmentedColormap.from_list("npgseq", ["#F7FBF9", "#91D1C2", "#00A087", "#3C5488"])
from matplotlib.lines import Line2D
ROOT = "/home/claude/n84"
C = pd.read_csv(f"{ROOT}/data/cities_inc.csv")
M = pd.read_csv(f"{ROOT}/data/out/proj_citymed.csv")
INC = ["Low & lower-middle", "Upper-middle", "High"]; ICOL = {INC[0]: NPG["red"], INC[1]: NPG["salmon"] if False else "#E6A119", INC[2]: NPG["navy"]}
ILAB = {INC[0]: "Low/lower-\nmiddle", INC[1]: "Upper-\nmiddle", INC[2]: "High"}
SCEN = ["ssp245", "ssp585"]; SLAB = {"ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}
SCOL = {"ssp126": NPG["teal"], "ssp245": NPG["cyan"], "ssp585": NPG["red"]}
fig = plt.figure(figsize=(W2, 112 * MM))
gs = fig.add_gridspec(2, 3, left=0.075, right=0.985, top=0.95, bottom=0.08, wspace=0.45, hspace=0.55, height_ratios=[1, 1.05])
rs = np.random.default_rng(3)
# a forecast skill (canopy and all sources, water counted)
ax = fig.add_subplot(gs[0, 0]); T = t(5)
cols = ["Step rule, 450 m", "InVEST kernel, 450 m", "InVEST kernel, fitted dcool", "Exponential, fitted", "Empirical curve"]
labs = ["Step\n≤450 m", "InVEST\n450 m", "InVEST\nfitted", "Exp.\nfitted", "Empi-\nrical"]
for k, (tg, c, mk) in enumerate((("tree canopy", NPG["teal"], "o"), ("all sources", NPG["navy"], "s"))):
    r = T[(T.iloc[:, 0] == "Counted") & (T.Targets == tg)].iloc[0]
    v = np.array([ci(r[c_]) for c_ in cols]); x = np.arange(len(cols)) + (k - 0.5) * 0.22
    ax.errorbar(x, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt=mk, color=c, ms=3.2, lw=0, elinewidth=0.8, label=tg.capitalize())
ax.axhline(0, color="k", lw=0.5); ax.set_xticks(range(len(cols))); ax.set_xticklabels(labs, fontsize=5.6)
ax.set_ylabel("Skill (1 − MSE/MSE$_{const}$)"); ax.set_ylim(-0.05, 0.75); ax.legend(loc="upper left", fontsize=5.8, handletextpad=0.2)
label(ax, "a")
# b residents within 240 m by income group
ax = fig.add_subplot(gs[0, 1])
for k, g in enumerate(INC):
    d = C[C.inc2 == g]; x = k + rs.uniform(-0.18, 0.18, len(d))
    ax.scatter(x, d.res240, s=6, color=ICOL[g], alpha=0.75, lw=0)
    pw = np.sum(d.res240 * d.popm) / d.popm.sum()
    ax.plot([k - 0.3, k + 0.3], [pw, pw], color="k", lw=1.4); ax.plot([k - 0.3, k + 0.3], [d.res240.median()] * 2, color="k", lw=0.8, ls="--")
ax.set_xticks(range(3)); ax.set_xticklabels([ILAB[g] + f"\n({(C.inc2 == g).sum()})" for g in INC], fontsize=5.8)
ax.set_ylabel("Residents ≤240 m of canopy (%)"); ax.set_ylim(0, 100); ax.set_xlim(-0.55, 2.55)
ax.legend(handles=[Line2D([], [], color="k", lw=1.4, label="population-weighted"), Line2D([], [], color="k", lw=0.8, ls="--", label="median city-region")],
          loc="upper left", fontsize=5.6, frameon=False)
label(ax, "b")
# c projected warm-season Tmax change by income group and scenario, 2081-2100 (filled) and 2041-2060 (open)
ax = fig.add_subplot(gs[0, 2])
for j, s in enumerate(SCEN):
    for k, g in enumerate(INC):
        for p, mk, off in (("2041-2060", "o", -0.09), ("2081-2100", "o", 0.09)):
            d = M[(M.scenario == s) & (M.period == p) & (M.inc2 == g)]
            if not len(d): continue
            x = k + (j - 0.5) * 0.36 + off; q = np.percentile(d.dtx, [25, 50, 75])
            ax.plot([x, x], [q[0], q[2]], color=SCOL[s], lw=1)
            ax.scatter([x], [q[1]], s=12, color=SCOL[s] if p == "2081-2100" else "white", edgecolor=SCOL[s], lw=0.8, zorder=3)
ax.set_xticks(range(3)); ax.set_xticklabels([ILAB[g] for g in INC], fontsize=5.8); ax.set_xlim(-0.55, 2.55)
ax.set_ylabel("Δ warm-season $T_{max}$ (°C)"); ax.set_ylim(0, 8.2)
h = [Line2D([], [], color=SCOL[s], marker="o", ms=3, lw=1, label=SLAB[s]) for s in SCEN] + \
    [Line2D([], [], color="grey", marker="o", mfc="white", ms=3, lw=0, label="2041–2060"), Line2D([], [], color="grey", marker="o", ms=3, lw=0, label="2081–2100")]
ax.legend(handles=h, loc="upper left", fontsize=5.4, ncol=2, columnspacing=0.6, handletextpad=0.3, frameon=False)
label(ax, "c")
# d canopy within 240 m needed to offset 2081-2100 warming (SSP2-4.5) against present canopy
ax = fig.add_subplot(gs[1, 0])
d = M[(M.scenario == "ssp245") & (M.period == "2081-2100")]
for g in INC:
    dd = d[d.inc2 == g]; cap = dd.need >= 99.9
    ax.scatter(dd.canopy_pct[~cap], dd.need[~cap], s=7, color=ICOL[g], lw=0, alpha=0.85, label=ILAB[g].replace("\n", ""))
    ax.scatter(dd.canopy_pct[cap], np.full(cap.sum(), 103), s=9, marker="^", color=ICOL[g], lw=0, alpha=0.85, clip_on=False)
ax.plot([0, 60], [0, 60], color=NPG["grey"], ls=":", lw=0.8); ax.axhline(30, color=NPG["green"] if "green" in NPG else NPG["teal"], ls="--", lw=0.7)
ax.text(59, 97, "▲ ≥100 pp (capped)", fontsize=5.2, ha="right", va="top", color="#444444")
ax.text(59, 31, "30% canopy (3-30-300)", fontsize=5.2, ha="right", va="bottom", color="#444444")
ax.set_xlabel("Present urban canopy (%)"); ax.set_ylabel("Added canopy ≤240 m to offset\n2081–2100 warming (pp)"); ax.set_xlim(0, 60); ax.set_ylim(0, 106); ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.set_yticklabels(["0", "20", "40", "60", "80", "≥100"])
ax.legend(loc="lower right", fontsize=5.4, handletextpad=0.1, frameon=False, borderaxespad=0.1, bbox_to_anchor=(1.0, 0.0))
label(ax, "d")
# e map: fraction of 2081-2100 SSP2-4.5 warming offset by +10 pp canopy within 240 m
ax = fig.add_subplot(gs[1, 1:])
land = gpd.read_file("/home/claude/new/data/ne/ne_110m_land.shp"); land.plot(ax=ax, color="#E8E8E8", lw=0)
d = d.merge(C[["city", "lat", "lon"]], on="city")
v = np.clip(100 * d.offset, 0, 30)
sc = ax.scatter(d.lon, d.lat, c=v, cmap=SEQ, vmin=0, vmax=30, s=14, edgecolor="#333333", lw=0.3, zorder=3)
ax.set_xlim(-170, 180); ax.set_ylim(-50, 70); ax.set_aspect("equal"); ax.axis("off")
cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02, aspect=40, shrink=0.6)
cb.set_label("Share of 2081–2100 warming (SSP2-4.5) offset by +10 pp canopy within 240 m (%)", fontsize=6); cb.ax.tick_params(labelsize=5.8)
label(ax, "e", x=0.0)
save(fig, "Fig5")
