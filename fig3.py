"""Fig. 3: the reach is physical, not instrumental. a-c, sensor-footprint forward model in 12 WorldCover cities;
d, 60 m canopy rings with WorldCover exposures (2014-2023 Landsat 8/9); e, kernel profile likelihood; f, practical reach vs threshold and its sources of spread."""
import sys, json, os, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/n84/src")
from style import *
from sitab import table3
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = "/home/claude/n84"
P = pd.read_csv(f"{ROOT}/data/out/psf_profiles.csv"); FIT = json.load(open(f"{ROOT}/data/out/psf_fit.json"))
T3 = table3(); obs = T3[T3.source.str.contains("anopy") & T3.zone.str.startswith("All")].sort_values("r0")
SEN = ["ETM+ (60 m)", "TIRS (100 m)", "TM (120 m)", "TIRS, with PSF halo"]
SCOL = {SEN[0]: NPG["cyan"], SEN[1]: NPG["navy"], SEN[2]: NPG["brown"], SEN[3]: NPG["lav"]}
SLAB = {SEN[0]: "ETM+ (60 m)", SEN[1]: "TIRS (100 m)", SEN[2]: "TM (120 m)", SEN[3]: "TIRS + halo"}
mids = obs.mid.values[:6]
fig = plt.figure(figsize=(W2, 112 * MM))
gs = fig.add_gridspec(2, 3, left=0.07, right=0.985, top=0.95, bottom=0.09, wspace=0.55, hspace=0.55)
# a: no physical lateral cooling (L = 0): what the instrument alone produces
ax = fig.add_subplot(gs[0, 0])
for s in SEN:
    d = P[(P.L == 0) & (P.sensor == s)].groupby("ring").cool.agg(["mean", "std", "count"]).iloc[:6]
    ax.errorbar(mids + (SEN.index(s) - 1.5) * 6, d["mean"], yerr=1.96 * d["std"] / np.sqrt(d["count"]), color=SCOL[s], marker="o", ms=2.6, lw=0.9, elinewidth=0.6, label=SLAB[s])
ax.errorbar(mids, obs.cool.values[:6], yerr=[obs.cool.values[:6] - obs.lo.values[:6], obs.hi.values[:6] - obs.cool.values[:6]], color="k", marker="s", ms=2.8, lw=1.1, elinewidth=0.6, label="observed")
ax.axhline(0, color="k", lw=0.5); ax.set_xlim(0, 720); ax.set_xticks([0, 240, 480, 720]); ax.set_ylim(-0.06, 0.32)
ax.set_xlabel("Distance from receptor (m)"); ax.set_ylabel("Apparent cooling (°C per 10 pp)")
ax.set_title("no lateral cooling ($L$ = 0)", fontsize=6, color="#444444", pad=3)
ax.legend(loc="upper right", fontsize=5.4, handlelength=1.4, frameon=False)
label(ax, "a")
# b: TIRS apparent profiles for a range of physical e-folding lengths, scaled to the observed 120-480 m rings
ax = fig.add_subplot(gs[0, 1])
Ls = [30, 45, 60, 90, 120]; cm = plt.get_cmap("viridis")
LC = {30: NPG["mint"], 45: NPG["teal"], 60: NPG["navy"], 90: NPG["lav"], 120: NPG["salmon"]}
o = obs.cool.values[:6]; se = (obs.hi.values[:6] - obs.lo.values[:6]) / 3.92
for L in Ls:
    d = P[(P.L == L) & (P.sensor == SEN[1])].groupby("ring").cool.mean().values[:6]
    a = np.sum(d[1:4] * o[1:4] / se[1:4] ** 2) / np.sum(d[1:4] ** 2 / se[1:4] ** 2)
    ax.plot(mids, a * d, color=LC[L], lw=1, marker="o", ms=2.2, label=f"$L$ = {L} m")
ax.errorbar(mids, o, yerr=1.96 * se, color="k", marker="s", ms=2.8, lw=0, elinewidth=0.7, label="observed", zorder=4)
ax.axhline(0, color="k", lw=0.5); ax.set_xlim(0, 720); ax.set_xticks([0, 240, 480, 720]); ax.set_ylim(-0.06, 0.5)
ax.set_xlabel("Distance from receptor (m)"); ax.set_ylabel("Scaled cooling (°C per 10 pp)")
ax.set_title("TIRS forward model", fontsize=6, color="#444444", pad=3)
ax.legend(loc="upper right", fontsize=5.4, ncol=2, columnspacing=0.6, handlelength=1.2, frameon=False)
label(ax, "b")
# c: fitted L and the reach with and without the sensor
ax = fig.add_subplot(gs[0, 2])
for k, s in enumerate(SEN):
    f = FIT[s]; y = 3 - k
    ax.plot(f["reach_app_ci"], [y + 0.12] * 2, color=SCOL[s], lw=1.2); ax.scatter([f["reach_apparent"]], [y + 0.12], color=SCOL[s], s=14, marker="o", zorder=3)
    ax.plot(f["reach_true_ci"], [y - 0.12] * 2, color=SCOL[s], lw=1.2, ls=(0, (2, 1))); ax.scatter([f["reach_true"]], [y - 0.12], color="white", edgecolor=SCOL[s], s=14, lw=0.9, zorder=3)
ax.axvline(285, color=NPG["grey"], ls=":", lw=0.8)
ax.set_yticks([3, 2, 1, 0]); ax.set_yticklabels([SLAB[s] + f"\n$L$ = {FIT[s]['L']:.0f} ({FIT[s]['L_ci'][0]:.0f}–{FIT[s]['L_ci'][1]:.0f}) m" for s in SEN], fontsize=5.6); ax.set_xlim(240, 340); ax.set_ylim(-0.6, 4.4)
ax.set_xlabel("Practical reach (m)")
ax.legend(handles=[Line2D([], [], color="k", marker="o", ms=3, lw=1, label="apparent (with sensor)"),
                   Line2D([], [], color="k", marker="o", mfc="white", ms=3, lw=1, ls=(0, (2, 1)), label="physical (sensor removed)")],
          loc="upper center", fontsize=5.4, frameon=False, ncol=1)
label(ax, "c", x=-0.62)
H = json.load(open(f"{ROOT}/data/out/hr_pool.json")) if os.path.exists(f"{ROOT}/data/out/hr_pool.json") else None
# d: 60 m rings
ax = fig.add_subplot(gs[1, 0])
if H:
    r = H["r60"]; m = np.array(r["mid"]); mu = np.array(r["mu"]); s_ = np.array(r["se"]); keep = m <= 900
    for c in np.array(r["city"]):
        ax.plot(m[keep], c[keep], color=NPG["mint"], lw=0.4, alpha=0.6)
    ax.errorbar(m[keep], mu[keep], yerr=1.96 * s_[keep], color=NPG["teal"], marker="o", ms=2.8, lw=1.1, elinewidth=0.6)
    Lh = H["kernel"]["Lhat"]
    ax.text(0.97, 0.95, f"{len(r['city'])} city-regions, WorldCover\nexposures, Landsat 8–9", transform=ax.transAxes, fontsize=5.4, ha="right", va="top", color="#444444")
ax.axhline(0, color="k", lw=0.5); ax.set_xlim(0, 900); ax.set_xticks([0, 300, 600, 900])
ax.set_xlabel("Distance from receptor (m)"); ax.set_ylabel("Canopy cooling (°C per 10 pp)")
label(ax, "d")
# e: kernel profile likelihood
ax = fig.add_subplot(gs[1, 1])
if H:
    k = H["kernel"]; ax.plot(k["L"], 100 * np.array(k["dsse_rel"]), color=NPG["navy"], marker="o", ms=2.8, lw=1)
    ax.axvspan(*k["Lci"], color=NPG["navy"], alpha=0.12, lw=0); ax.axvline(k["Lhat"], color=NPG["navy"], lw=0.8, ls="--")
    for s in SEN[:2]:
        f = FIT[s]; ax.errorbar([f["L"]], [0.45 * max(100 * np.array(k["dsse_rel"]))], xerr=[[f["L"] - f["L_ci"][0]], [f["L_ci"][1] - f["L"]]],
                                fmt="D", color=SCOL[s], ms=3, elinewidth=0.8)
    ax.text(0.97, 0.95, f"$\\hat L$ = {k['Lhat']:.0f} m ({k['Lci'][0]:.0f}–{k['Lci'][1]:.0f})\n95% of kernel within {k['reach95']:.0f} m", transform=ax.transAxes,
            fontsize=5.6, ha="right", va="top")
from matplotlib.ticker import NullLocator
ax.set_xscale("log"); ax.xaxis.set_minor_locator(NullLocator()); ax.set_xticks([20, 40, 80, 160, 320]); ax.set_xticklabels(["20", "40", "80", "160", "320"])
ax.set_xlabel("Kernel e-folding length $L$ (m)"); ax.set_ylabel("Increase in residual SS (%)")
label(ax, "e")
# f: reach vs threshold for canopy, PSF-corrected, by zone
ax = fig.add_subplot(gs[1, 2])
from sitab import t
T9 = t(9); T9 = T9.iloc[:3]
th = np.array([0.025, 0.05, 0.1])
pr = [list(map(float, __import__("re").findall(r"[\d.]+", x)))[:3] for x in T9.iloc[:, 2]]
pr = np.array(pr)
ax.errorbar(th, pr[:, 0], yerr=[pr[:, 0] - pr[:, 1], pr[:, 2] - pr[:, 0]], color="k", marker="s", ms=3, lw=1, elinewidth=0.6, label="observed, pooled")
# exponential kernel prediction of the reach for the fitted L and amplitude (TIRS fit) at each threshold
f = FIT[SEN[1]]; prof = np.array(f["prof_true"][:6]); mm = mids
def reach(p, dlt):
    k = np.argmax(p < dlt)
    return mm[k - 1] + (mm[k] - mm[k - 1]) * (p[k - 1] - dlt) / (p[k - 1] - p[k]) if k > 0 else 0
ax.plot(th, [reach(np.array(f["prof_app"][:6]), x) for x in th], color=NPG["navy"], marker="o", ms=3, lw=1, label="forward model, apparent")
ax.plot(th, [reach(prof, x) for x in th], color=NPG["navy"], marker="o", mfc="white", ms=3, lw=1, ls=(0, (2, 1)), label="forward model, physical")
ax.axhline(300, color=NPG["grey"], ls=":", lw=0.8); ax.axhline(450, color=NPG["red"], ls="--", lw=0.7)
ax.text(0.1, 455, "InVEST default", fontsize=5.4, color=NPG["red"], ha="right", va="bottom")
ax.text(0.1, 305, "3-30-300", fontsize=5.4, color=NPG["grey"], ha="right", va="bottom")
from matplotlib.ticker import NullFormatter, NullLocator
ax.set_xscale("log"); ax.xaxis.set_minor_locator(NullLocator()); ax.set_xticks(th); ax.set_xticklabels(["0.025", "0.05", "0.10"]); ax.set_xlim(0.02, 0.125)
ax.set_ylim(180, 480); ax.set_xlabel("Threshold δ (°C per 10 pp)"); ax.set_ylabel("Practical reach (m)")
ax.legend(loc="lower left", fontsize=5.4, frameon=False)
label(ax, "f")
save(fig, "Fig3")
