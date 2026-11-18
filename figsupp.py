"""Supplementary Figs 1-3 and Extended Data Figs 9-10."""
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
which = sys.argv[1:] or ["s1", "s2", "s3", "ed9", "ed10"]
H = json.load(open(f"{ROOT}/data/out/hr_pool.json")) if os.path.exists(f"{ROOT}/data/out/hr_pool.json") else None

if "s1" in which and H and "fetch" in H:
    F = pd.read_csv(f"{ROOT}/data/out/hr_fetch.csv"); La = sorted(F.La.unique())
    fig, axs = plt.subplots(1, 4, figsize=(W2, 55 * MM)); fig.subplots_adjust(left=0.08, right=0.99, bottom=0.22, top=0.9, wspace=0.6)
    ax = axs[0]; FF = H["fetch"]; LL = sorted(int(x) for x in FF)
    dd = np.array([FF[str(l)]["diff_mean"] for l in LL]); se_ = np.array([FF[str(l)]["diff_se"] for l in LL])
    ax.plot(LL, [FF[str(l)]["up"] for l in LL], color=NPG["teal"], marker="^", ms=2.6, lw=0.8, label="upwind")
    ax.plot(LL, [FF[str(l)]["down"] for l in LL], color=NPG["lav"], marker="v", ms=2.6, lw=0.8, label="downwind")
    ax.errorbar(LL, dd, yerr=1.96 * se_, color=NPG["navy"], marker="o", ms=2.8, lw=1, elinewidth=0.7, label="difference")
    ax.axhline(0, color="k", lw=0.5); ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks(LL); ax.set_xticklabels([str(l) for l in LL])
    ax.set_xlabel("Fetch length $L_a$ (m)"); ax.set_ylabel("Canopy cooling (°C per 10 pp)"); ax.legend(loc="center right", fontsize=5.2, frameon=False)
    label(ax, "a"); axs = axs[1:]
    ax = axs[0]; d = F[F.La == 120]
    ax.scatter(d.ws, d["diff"], s=5, color=NPG["navy"], alpha=0.6, lw=0)
    r = H["fetch"]["120"]; xx = np.linspace(0, d.ws.max(), 10)
    ax.plot(xx, r["diff_w"] - r["slope_ws"] * np.mean(d.ws) + r["slope_ws"] * xx, color=NPG["red"], lw=1)
    ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("Overpass wind speed, ERA5 100 m (m s$^{-1}$)"); ax.set_ylabel("Upwind − downwind, $L_a$ = 120 m\n(°C per 10 pp)")
    ax.set_ylim(np.percentile(d["diff"], 1) - 0.05, np.percentile(d["diff"], 99) + 0.05)
    ax.text(0.98, 0.97, (f"slope {r['slope_ws']:+.3f} ± {1.96 * r['slope_se']:.3f} per m s$^{{-1}}$\nP = {r['slope_p']:.2f}").replace("-0", "−0"), transform=ax.transAxes, fontsize=5.6, ha="right", va="top")
    label(ax, "b")
    ax = axs[1]
    ax.bar(np.arange(len(La)), [100 * H["fetch"][str(l)]["frac_fetch_better"] for l in La], color=NPG["teal"], width=0.6)
    ax.axhline(50, color=NPG["grey"], ls=":", lw=0.8); ax.set_xticks(range(len(La))); ax.set_xticklabels([str(l) for l in La])
    ax.set_xlabel("Fetch length $L_a$ (m)"); ax.set_ylabel("Scenes where fetch kernel\nfits better than isotropic (%)"); ax.set_ylim(0, 100)
    label(ax, "c")
    ax = axs[2]
    ax.bar(np.arange(len(La)), [100 * H["fetch"][str(l)]["dsse_rel"] for l in La], color=NPG["lav"], width=0.6)
    ax.axhline(0, color="k", lw=0.5); ax.set_xticks(range(len(La))); ax.set_xticklabels([str(l) for l in La])
    ax.set_xlabel("Fetch length $L_a$ (m)"); ax.set_ylabel("Residual SS, fetch − isotropic (%)")
    label(ax, "d")
    save(fig, "SuppFig1")

if "s2" in which:
    P1 = pd.read_csv(f"{ROOT}/data/out/psf_profiles.csv"); P1 = P1[P1.sensor == "TIRS (100 m)"]
    P2 = pd.read_csv(f"{ROOT}/data/out/psf_profiles_shifted.csv")
    fig, axs = plt.subplots(1, 3, figsize=(W2, 55 * MM), sharey=True); fig.subplots_adjust(left=0.07, right=0.985, bottom=0.2, top=0.88, wspace=0.12)
    for ax, L, lab in zip(axs, (0, 60, 120), "abc"):
        for P, col, nm, mk in ((P1, NPG["navy"], "edges 0, 120, 240, … m", "o"), (P2, NPG["red"], "edges 0, 180, 300, … m", "s")):
            d = P[P.L == L].groupby("mid").cool.mean(); d = d[d.index <= 900]
            ax.plot(d.index, d.values, color=col, marker=mk, ms=2.8, lw=1, label=nm)
            if "cool_true" in P:
                t = P[P.L == L].groupby("mid").cool_true.mean(); t = t[t.index <= 900]
                ax.plot(t.index, t.values, color=col, ls=":", lw=0.8)
        ax.axhline(0, color="k", lw=0.5); ax.set_title(f"TIRS, $L$ = {L} m", fontsize=6.5, pad=3); ax.set_xlabel("Ring mid-distance (m)")
        ax.set_xlim(0, 900); label(ax, lab)
    axs[0].set_ylabel("Cooling (°C per 10 pp)"); axs[0].legend(loc="upper right", fontsize=5.6, frameon=False)
    axs[1].legend(handles=[Line2D([], [], color="k", lw=1, label="apparent (sensor)"), Line2D([], [], color="k", lw=0.8, ls=":", label="physical (no sensor)")],
                  loc="upper right", fontsize=5.6, frameon=False)
    save(fig, "SuppFig2")

if "s3" in which and H:
    k = H["kernel"]; Lc = np.array(k["Lcity"]); cities = k["cities"]
    o = np.argsort(Lc)
    fig, ax = plt.subplots(figsize=(120 * MM, 70 * MM)); fig.subplots_adjust(left=0.3, right=0.97, bottom=0.14, top=0.97)
    zc = [ZONE[C.zone[C.city == c].iloc[0]] for c in np.array(cities)[o]]
    ax.barh(np.arange(len(o)), Lc[o], color=zc, height=0.7)
    ax.axvline(k["Lhat"], color="k", ls="--", lw=0.8); ax.axvspan(*k["Lci"], color="k", alpha=0.08, lw=0)
    ax.set_yticks(np.arange(len(o))); ax.set_yticklabels(np.array(cities)[o], fontsize=5.6); ax.set_xlabel("City-specific kernel length $\\hat L$ (m)")
    ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks([20, 40, 80, 160, 320]); ax.set_xticklabels(["20", "40", "80", "160", "320"])
    ax.legend(handles=[Line2D([], [], color=v, lw=4, label=z) for z, v in ZONE.items() if z in set(C.zone[C.city.isin(cities)])], loc="lower right", fontsize=5.6, frameon=False)
    save(fig, "SuppFig3")

if "ed9" in which:
    MAPS = np.load(f"{ROOT}/data/out/lagr_maps.npz")
    QC = LinearSegmentedColormap.from_list("q", ["#FFFFFF", "#91D1C2", "#4DBBD5", "#3C5488"])
    HC = LinearSegmentedColormap.from_list("h", ["#FFFFFF", "#F39B7F", "#E64B35", "#7E1F12"])
    ex = [3, 19, 12, 36, 48, 73, 74, 64]
    ex = [i for i in ex if f"{i:02d}_q" in MAPS.files][:8]
    fig = plt.figure(figsize=(W2, 95 * MM)); gs = fig.add_gridspec(2, len(ex), left=0.04, right=0.99, top=0.93, bottom=0.04, wspace=0.06, hspace=0.12)
    for k, i in enumerate(ex):
        for r, (key, cm) in enumerate((("q", QC), ("h", HC))):
            ax = fig.add_subplot(gs[r, k]); A = MAPS[f"{i:02d}_{key}"]; la = MAPS[f"{i:02d}_lat"]; lo = MAPS[f"{i:02d}_lon"]; oc = MAPS[f"{i:02d}_oc"]
            A = np.where(A > 0, A, np.nan); vmax = np.nanpercentile(A, 99.5)
            ax.pcolormesh(lo, la, A, cmap=cm, norm=LogNorm(vmin=vmax / 100, vmax=vmax), shading="nearest", rasterized=True)
            ax.contour(lo, la, oc.astype(float), levels=[0.5], colors="#555555", linewidths=0.35)
            c = C.iloc[i]; ax.plot(c.lon, c.lat, marker="*", color="k" if key == "h" else NPG["red"], ms=5, mec="k", mew=0.3)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(0.5)
            if r == 0: ax.set_title(c.city.replace(" (Australia)", ""), fontsize=6, pad=2)
            if k == 0: ax.set_ylabel("moisture" if key == "q" else "sensible heat", fontsize=6.5)
    fig.text(0.005, 0.95, "a", fontsize=8, fontweight="bold"); fig.text(0.005, 0.48, "b", fontsize=8, fontweight="bold")
    save(fig, "SuppFig4")

if "ed10" in which:
    M = pd.read_csv(f"{ROOT}/data/out/proj_citymed.csv"); M = M[M.period == "2081-2100"]
    SC = {"ssp245": NPG["cyan"], "ssp585": NPG["red"]}
    order = M[M.scenario == "ssp245"].sort_values("dtx").city.tolist()
    fig, axs = plt.subplots(1, 2, figsize=(W2, 170 * MM), gridspec_kw=dict(width_ratios=[1.3, 1])); fig.subplots_adjust(left=0.13, right=0.985, bottom=0.065, top=0.98, wspace=0.08)
    y = {c: i for i, c in enumerate(order)}
    ax = axs[0]
    for s, col in SC.items():
        d = M[M.scenario == s]
        ax.errorbar(d.dtx, [y[c] for c in d.city], xerr=[d.dtx - d.dtx_lo, d.dtx_hi - d.dtx], fmt="o", ms=2, color=col, elinewidth=0.4, label={"ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}[s])
    ax.set_yticks(range(len(order))); ax.set_yticklabels([c.replace(" (Australia)", "") for c in order], fontsize=4.6); ax.set_ylim(-1, len(order))
    ax.set_xlabel("Δ warm-season $T_{max}$, 2081–2100 vs 1995–2014 (°C)"); ax.legend(loc="lower right", fontsize=5.6, frameon=False)
    label(ax, "a", x=-0.2)
    ax = axs[1]; d = M[M.scenario == "ssp245"]
    ax.barh([y[c] for c in d.city], 100 * d.offset, color=[ {"Low & lower-middle": NPG["red"], "Upper-middle": "#E6A119", "High": NPG["navy"]}[g] for g in d.inc2], height=0.7)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([]); ax.set_ylim(-1, len(order))
    ax.set_xlabel("Share of SSP2-4.5 warming offset by\n+10 pp canopy within 240 m (%)")
    ax.legend(handles=[Line2D([], [], color=v, lw=4, label=k) for k, v in {"Low & lower-middle": NPG["red"], "Upper-middle": "#E6A119", "High": NPG["navy"]}.items()], loc="lower right", fontsize=5.6, frameon=False)
    label(ax, "b", x=-0.05)
    save(fig, "EDFig10")
