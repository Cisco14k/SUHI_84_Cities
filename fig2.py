"""Fig. 2: magnitude, frequency and reach of surface cooling (from the published scene-pooled estimates)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/n84ns/src")
from style import *
from sitab import table3, t, num, rng
import matplotlib.pyplot as plt
T = table3(); C = pd.read_csv("/home/claude/n84ns/data/cities.csv")
fig = plt.figure(figsize=(W2, 112 * MM))
gs = fig.add_gridspec(2, 3, left=0.075, right=0.985, top=0.95, bottom=0.085, wspace=0.42, hspace=0.55)
# a: profiles
ax = fig.add_subplot(gs[0, 0])
a = T[(T.zone == "All")]
c = a[a.source == "Tree canopy"]
ax.fill_between(c.mid, c.pilo, c.pihi, color=SRC["Tree canopy"], alpha=0.13, lw=0, label="95% PI, canopy")
for s, mk in (("Water", "D"), ("Tree canopy", "o"), ("Open green", "s")):
    d = a[a.source == s]
    ax.errorbar(d.mid, d.cool, yerr=[d.cool - d.lo, d.hi - d.cool], color=SRC[s], marker=mk, ms=3, lw=1, capsize=0, elinewidth=0.7, label=s)
ax.axhline(0, color="k", lw=0.5); ax.set_xlim(0, 1800); ax.set_ylim(-0.12, 0.62)
ax.set_xlabel("Distance from receptor (m)"); ax.set_ylabel("Surface cooling (°C per 10 pp)")
ax.legend(loc="upper right", ncol=1); label(ax, "a")
# b: frequency
ax = fig.add_subplot(gs[0, 1])
F = t(8); F.columns = [x.strip() for x in F.columns]
F["Source"] = F["Source"].replace("​", np.nan).ffill()
F["mid"] = F["Ring (m)"].map(lambda s: np.mean([float(x) for x in s.split("–")]))
for s, mk in (("Tree canopy", "o"), ("Open green", "s"), ("Water", "D")):
    d = F[F.Source == s]
    ax.plot(d.mid, d["F0.05"], marker=mk, color=SRC[s], label=s)
ax.axhline(0.5, color=NPG["grey"], ls="--", lw=0.6)
ax.set_ylim(0, 1.05); ax.set_xlim(0, 600); ax.set_xticks([0, 200, 400, 600])
ax.set_xlabel("Distance from receptor (m)"); ax.set_ylabel("P(cooling ≥ 0.05 °C on a clear day)")
ax.legend(loc="upper right"); label(ax, "b")
# c: zones
ax = fig.add_subplot(gs[0, 2])
for i, z in enumerate(["Tropical", "Arid", "Temperate", "Continental"]):
    d = T[(T.source == "Tree canopy") & (T.zone == z)]; off = (i - 1.5) * 9
    ax.errorbar(d.mid + off, d.cool, yerr=[d.cool - d.lo, d.hi - d.cool], color=ZONE[z], marker=ZMARK[z], ms=3, lw=1, elinewidth=0.7,
                label=f"{z} ({int(d.k.iloc[0])})")
ax.axhline(0, color="k", lw=0.5); ax.set_xlim(0, 860); ax.set_ylim(-0.06, 0.31)
ax.set_xlabel("Distance from receptor (m)"); ax.set_ylabel("Canopy cooling (°C per 10 pp)")
ax.legend(loc="upper right"); label(ax, "c")
# d: reach vs threshold
ax = fig.add_subplot(gs[1, 0])
R = t(9); R.columns = [x.strip() for x in R.columns]; R["Source"] = R["Source"].replace("​", np.nan).ffill()
for s, mk in (("Tree canopy", "o"), ("Water", "D")):
    d = R[R.Source == s]; th = d.iloc[:, 1].astype(float).values
    v = [num(str(x).split(" ")[0]) for x in d.iloc[:, 2]]; lohi = [rng(x) for x in d.iloc[:, 2]]
    ax.errorbar(th + (0.002 if s == "Water" else 0), v, yerr=[[vv - l for vv, (l, h) in zip(v, lohi)], [h - vv for vv, (l, h) in zip(v, lohi)]],
                color=SRC[s], marker=mk, ms=3.5, lw=1, label=s)
ax.axhline(450, color=NPG["red"], ls="--", lw=0.7); ax.axhline(300, color=NPG["grey"], ls=":", lw=0.8)
ax.text(0.104, 456, "InVEST default (450 m)", ha="right", va="bottom", fontsize=6, color=NPG["red"])
ax.text(0.104, 306, "3-30-300 access (300 m)", ha="right", va="bottom", fontsize=6, color=NPG["grey"])
ax.set_xlim(0.015, 0.108); ax.set_ylim(150, 500); ax.set_xticks([0.025, 0.05, 0.1])
ax.set_xlabel("Threshold δ (°C per 10 pp)"); ax.set_ylabel("Threshold reach of pooled profile (m)")
ax.legend(loc="lower left"); label(ax, "d")
# e: city reaches by zone
ax = fig.add_subplot(gs[1, 1])
zmap = lambda k: "Tropical" if k[0] == "A" else "Arid" if k[0] == "B" else "Temperate" if k[0] == "C" else "Continental"
C["zone"] = C.koppen.map(zmap)
rs = np.random.default_rng(3)
for i, z in enumerate(["Tropical", "Arid", "Temperate", "Continental"]):
    v = C[C.zone == z].reach_interp.values
    q1, md, q3 = np.percentile(v, [25, 50, 75])
    ax.add_patch(plt.Rectangle((i - 0.28, q1), 0.56, q3 - q1, color=ZONE[z], alpha=0.25, lw=0))
    ax.plot([i - 0.28, i + 0.28], [md, md], color=ZONE[z], lw=1.4)
    ax.scatter(i + rs.uniform(-0.2, 0.2, len(v)), v, s=4, color=ZONE[z], alpha=0.85, lw=0)
ax.axhline(450, color=NPG["red"], ls="--", lw=0.7); ax.axhline(300, color=NPG["grey"], ls=":", lw=0.8)
ax.set_xticks(range(4)); ax.set_xticklabels([f"{z if len(z) < 5 else z[:4] + '.'}\n({(C.zone == z).sum()})" for z in ["Tropical", "Arid", "Temperate", "Continental"]], fontsize=6)
ax.set_xlim(-0.6, 3.6); ax.set_ylim(-20, 760); ax.set_ylabel("Threshold reach of city-regions (m)"); label(ax, "e")
# f: per hectare
ax = fig.add_subplot(gs[1, 2])
for s, mk in (("Water", "D"), ("Tree canopy", "o"), ("Open green", "s")):
    d = a[a.source == s].copy(); area = np.pi * (d.r1 ** 2 - d.r0 ** 2) / 1e4 * 0.1     # ha in 10 pp of the ring
    v = d.cool / area * 1000
    ax.plot(d.mid, np.where(v > 0.1, v, np.nan), marker=mk, color=SRC[s], label=s)
ax.set_yscale("log"); ax.set_ylim(0.3, 3000); ax.set_xlim(0, 1800)
ax.set_xlabel("Distance from receptor (m)"); ax.set_ylabel("Cooling per hectare (m°C ha$^{-1}$)")
ax.legend(loc="upper right"); label(ax, "f")
save(fig, "Fig2")
