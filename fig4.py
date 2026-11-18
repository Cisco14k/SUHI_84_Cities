"""Fig. 4: four decades, shade and weather (from the published estimates)."""
import sys, re, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/n84/src")
from style import *
from sitab import t, ci, num
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(W2, 105 * MM))
gs = fig.add_gridspec(2, 3, left=0.07, right=0.985, top=0.94, bottom=0.12, wspace=0.42, hspace=0.62)
gb = fig.add_gridspec(1, 2, left=0.07, right=0.985, top=gs.top - 0.0, bottom=0.10, width_ratios=[1, 1.25], wspace=0.95)
gb = fig.add_gridspec(1, 2, left=0.07, right=0.975, top=0.42, bottom=0.12, width_ratios=[1, 1.2], wspace=0.95)
mids = np.array([60, 180, 300, 420])
# a decades
ax = fig.add_subplot(gs[0, 0])
D = t(13).iloc[8:12]; cols = [NPG["brown"], NPG["salmon"], NPG["cyan"], NPG["navy"]]
for i, (_, r) in enumerate(D.iterrows()):
    v = np.array([ci(r[c]) for c in ["0–120 m", "120–240 m", "240–360 m", "360–480 m"]])
    ax.errorbar(mids + (i - 1.5) * 7, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color=cols[i], marker="o", ms=2.8, lw=1, elinewidth=0.6, label=f"{r['Group']} ({int(r['k'])})")
ax.axhline(0, color="k", lw=0.5); ax.set_xlim(0, 480); ax.set_ylim(-0.04, 0.26); ax.set_xticks([0, 120, 240, 360, 480])
ax.set_xlabel("Distance from receptor (m)"); ax.set_ylabel("Canopy cooling (°C per 10 pp)"); ax.legend(loc="upper right", fontsize=5.8); label(ax, "a")
# b reach by decade, balanced panel, and long differences
ax = fig.add_subplot(gs[0, 1])
B = t(14); b10 = B[B.Panel.str.contains("≥ 10 scenes per")]; b5 = B[B.Panel.str.contains("≥ 5 scenes per")]
x = np.arange(4)
ax.plot(x, b5["Practical reach (m)"].astype(float), marker="s", color=NPG["lav"], label="balanced, ≥5 scenes (73)")
ax.plot(x, b10["Practical reach (m)"].astype(float), marker="o", color=NPG["teal"], label="balanced, ≥10 scenes (65)")
R = t(10); ld = R[R.Analysis.str.startswith("Long differences")]
ax.scatter([1.5, 2.5], ld["Practical reach, pooled (m)"].astype(float), marker="D", color=NPG["red"], s=14, zorder=3, label="long differences")
ax.axhline(300, color=NPG["grey"], ls=":", lw=0.8); ax.axhline(450, color=NPG["red"], ls="--", lw=0.7)
ax.set_xticks(x); ax.set_xticklabels(["1985–94", "1995–04", "2005–14", "2015–25"], fontsize=6)
ax.set_ylim(140, 470); ax.set_ylabel("Practical reach (m)"); ax.legend(loc="lower center", fontsize=5.6, ncol=1)
ax.text(3.3, 455, "δ = 0.025", fontsize=5.4, color=NPG["red"], ha="right", va="bottom"); ax.text(3.3, 305, "δ = 0.05", fontsize=5.4, color=NPG["grey"], ha="right", va="bottom"); label(ax, "b")
# c sun quadrants
T = t(15)
for j, (dirn, lab, axpos) in enumerate((("Sun", "c", gs[0, 2]), ("Wind", "d", gb[0, 0]))):
    ax = fig.add_subplot(axpos); d = T[T.Direction == dirn]
    for k, (col, c) in enumerate((("Facing quadrant", NPG["red"] if dirn == "Sun" else NPG["navy"]), ("Opposite quadrant", NPG["grey"]), ("Lateral quadrants", NPG["mint"]))):
        ax.bar(np.arange(5) + (k - 1) * 0.26, d[col].map(num), width=0.26, color=c, lw=0,
               label=(("sun-facing" if dirn == "Sun" else "upwind") if k == 0 else ("opposite" if k == 1 else "lateral")))
    ax.axhline(0, color="k", lw=0.5); ax.set_xticks(range(5)); ax.set_xticklabels(["0–\n120", "120–\n240", "240–\n360", "360–\n480", "480–\n600"], fontsize=5.8)
    ax.set_xlabel("Ring (m)"); ax.set_ylabel("Canopy cooling (°C per 10 pp)"); ax.set_ylim(-0.03, 0.235); ax.set_xlim(-0.6, 4.6)
    ax.legend(loc="upper right", fontsize=5.8)
    ax.set_title("overpass wind, 2024–2025 (78 cities)" if dirn == "Wind" else "solar azimuth, 1985–2025 (84 cities)", fontsize=6, color="#444444", pad=3)
    label(ax, lab)
# e modulation
ax = fig.add_subplot(gb[0, 1])
M = t(4); M = M[M.Model.str.startswith("Canopy, monthly") | M.Model.str.startswith("Canopy, overpass")]
labs = []; vals = []
for _, r in M.iterrows():
    v = ci(r.iloc[2]); vals.append(v); labs.append(r.Driver.replace(" (monthly)", "").replace("anomaly", "anom.") + (" †" if "overpass" in r.Model else ""))
vals = np.array(vals) / 1000; y = np.arange(len(vals))[::-1]
sig = np.sign(vals[:, 1]) == np.sign(vals[:, 2])
ax.errorbar(vals[:, 0], y, xerr=[vals[:, 0] - vals[:, 1], vals[:, 2] - vals[:, 0]], fmt="none", ecolor="#555555", elinewidth=0.7)
ax.scatter(vals[:, 0], y, c=[NPG["teal"] if s else "white" for s in sig], edgecolor=NPG["teal"], s=16, zorder=3, lw=0.8)
ax.axvline(0, color="k", lw=0.5); ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=6)
ax.set_xlabel("Δ cooling within 240 m (°C per 10 pp per s.d.)")
ax.set_title("† overpass hour, 2024–2025; others monthly, 1985–2021; filled P < 0.05", fontsize=5.8, color="#444444", pad=3)
ax.set_xlim(-0.014, 0.014); label(ax, "e", x=-0.33)
save(fig, "Fig4")
