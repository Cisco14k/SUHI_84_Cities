"""Fig. 4 (Nature Sustainability): who receives canopy cooling.
a, modelled canopy cooling delivered to residents' cells by income group (84 city-regions); b, delivered cooling against urban canopy cover;
c, share of residents beyond 240 m of a canopy patch, all residents against the hottest third (21 city-regions, 60 m);
d, residential cooling of designs relative to one park under common rules; e, residents reached by >= 0.05 degC;
f, sensitivity of the income gap in delivered cooling (low/lower-middle minus high) to covariates and to omitting world regions."""
# Methods: Fig4.png
# 1. ESA WorldCover 2021 cover fractions at 120 m and GlobPOP 2020 residents allocated to urban cells by built-up fraction (access84.py).
# 2. Delivered cooling of a cell = 10 x sum over rings of the pooled canopy ring effect (Table 1) x the ring's canopy fraction; population-weighted.
# 3. Income-group contrasts by least squares with HC3 errors (log aridity and absolute latitude, or canopy cover, as covariates); leave-one-region-out.
# 4. Heat co-location: median-scene surface-temperature excess at 60 m in the 21 re-analysis city-regions; hottest third of residents by excess.
# 5. Heat co-location: thirds of residents by observed excess, by excess plus modelled canopy cooling (no-canopy counterfactual) and by built-up fraction within 240 m.
# 6. Planting designs (planting3.py, feasible footprints, common sets of city-regions): 5, 20 or 100 ha of added canopy as +30 pp in single cells, ~2 ha units or one park, best locations, same
#    eligible land and resident rules for all designs; medians over 84 city-regions, 95% intervals from city bootstraps and ring-effect draws.
# 7. Income gap: least squares with HC3 errors, equal or population weights; leave-one-region-out point estimates.
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/n84ns/src")
from style import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = "/home/claude/n84ns"
C = pd.read_csv(f"{ROOT}/data/out/cities_merged.csv"); CF = json.load(open(f"{ROOT}/data/out/cityfx.json"))
H = pd.read_csv(f"{ROOT}/data/out/heat21.csv")
INC = ["Low & lower-middle", "Upper-middle", "High"]; ICOL = {INC[0]: NPG["red"], INC[1]: "#E6A119", INC[2]: NPG["navy"]}
ILAB = {INC[0]: "Low/lower-\nmiddle", INC[1]: "Upper-\nmiddle", INC[2]: "High"}
rs = np.random.default_rng(4)
fig = plt.figure(figsize=(W2, 118 * MM))
gs = fig.add_gridspec(2, 3, left=0.075, right=0.985, top=0.95, bottom=0.10, wspace=0.62, hspace=0.62)
# a delivered cooling by income group
ax = fig.add_subplot(gs[0, 0])
for k, g in enumerate(INC):
    d = C[C.inc2 == g]; x = k + rs.uniform(-0.18, 0.18, len(d))
    ax.scatter(x, d.dT_pw, s=6, color=ICOL[g], alpha=0.75, lw=0)
    pw = np.sum(d.dT_pw * d.residents) / d.residents.sum()
    ax.plot([k - 0.3, k + 0.3], [pw, pw], color="k", lw=1.4); ax.plot([k - 0.3, k + 0.3], [d.dT_pw.median()] * 2, color="k", lw=0.8, ls="--")
ax.set_xticks(range(3)); ax.set_xticklabels([ILAB[g] + f"\n({(C.inc2 == g).sum()})" for g in INC], fontsize=5.8)
ax.set_ylabel("Modelled residential\nsurface cooling (°C)"); ax.set_ylim(0, 2.1); ax.set_xlim(-0.55, 2.55)
ax.legend(handles=[Line2D([], [], color="k", lw=1.4, label="population-weighted"), Line2D([], [], color="k", lw=0.8, ls="--", label="median city-region")],
          loc="upper left", fontsize=5.6, frameon=False)
label(ax, "a")
# b delivered cooling vs canopy cover
ax = fig.add_subplot(gs[0, 1])
for g in INC:
    d = C[C.inc2 == g]; ax.scatter(100 * d.canopy_urban, d.dT_pw, s=7, color=ICOL[g], lw=0, alpha=0.85, label=ILAB[g].replace("-\n", "-").replace("\n", " "))
x = 100 * C.canopy_urban.values; y = C.dT_pw.values; b = np.polyfit(x, y, 1); xx = np.linspace(0, x.max(), 50)
ax.plot(xx, np.polyval(b, xx), color=NPG["grey"], lw=0.8, ls="--")
r = np.corrcoef(x, y)[0, 1]
ax.text(0.97, 0.05, f"r = {r:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8)
ax.set_xlabel("Urban canopy cover (%)"); ax.set_ylabel("Modelled residential\nsurface cooling (°C)"); ax.set_ylim(0, 2.1); ax.set_xlim(0, 39)
ax.legend(loc="upper left", fontsize=5.5, frameon=False, handletextpad=0.1)
label(ax, "b")
# c heat co-location under three heat indicators
ax = fig.add_subplot(gs[0, 2])
H2 = pd.read_csv(f"{ROOT}/data/out/heat21b.csv")
cols = [("all_beyond", "All", NPG["navy"]), ("hot_beyond", "Hottest,\nobserved", NPG["red"]),
        ("hot_beyond_cf", "Hottest,\nno canopy", NPG["salmon"]), ("hot_beyond_b", "Most\nbuilt-up", NPG["brown"])]
for _, rw in H2.iterrows():
    ax.plot(range(4), [100 * rw[c] for c, _, _ in cols], color=NPG["grey"], lw=0.4, alpha=0.5)
for k, (c, nm, col) in enumerate(cols):
    ax.scatter(np.full(len(H2), k), 100 * H2[c], s=6, color=col, lw=0, zorder=3)
    ax.plot([k - 0.25, k + 0.25], [100 * H2[c].median()] * 2, color=col, lw=1.6, zorder=4)
ax.set_xticks(range(4)); ax.set_xticklabels([nm for _, nm, _ in cols], fontsize=5.1); ax.set_xlim(-0.5, 3.5)
ax.set_ylabel("Residents >240 m from canopy (%)"); ax.set_ylim(0, 105)
ax.set_title(f"{len(H2)} city-regions, 60 m; thirds of residents", fontsize=5.8, color="#444444")
label(ax, "c")
# d planting designs relative to one park, common feasible sets
P3 = json.load(open(f"{ROOT}/data/out/planting3.json")); B = [5, 20, 100]
ser = [("E1_R0", "D30", "Dispersed, all eligible land", NPG["teal"], "o", "full"),
       ("E1_R0", "D30", "Dispersed, all eligible land, saturating", NPG["teal"], "o", "none"),
       ("E2_R0", "D30", "Dispersed, less-occupied land", NPG["cyan"], "s", "full"),
       ("E2_R0", "PN", "Units of ~2 ha, less-occupied land", NPG["navy"], "D", "full"),
       ("E1_R1", "D30", "Dispersed, displacement in all", NPG["red"], "^", "full")]
ax = fig.add_subplot(gs[1, 0])
for j, (er, dn, nm, col, mk, fill) in enumerate(ser):
    xs = np.arange(3) + (j - 2) * 0.14
    v = [P3[f"{er}_{b}_{dn}_vs_P1"] for b in B]
    if "saturating" in nm:
        med = [x["sat"] for x in v]; lo = [x["sat_ci"][0] for x in v]; hi = [x["sat_ci"][1] for x in v]
    else:
        med = [x["median"] for x in v]; lo = [x["ci"][0] for x in v]; hi = [x["ci"][1] for x in v]
    ax.errorbar(xs, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], fmt=mk, color=col, ms=3.0, lw=0.8, capsize=1.2,
                mfc=col if fill == "full" else "white", label=nm)
ax.axhline(1, color="k", lw=0.5, ls=":"); ax.set_yscale("log"); ax.set_ylim(0.6, 16)
ax.set_yticks([0.5, 1, 2, 4, 8]); ax.set_yticklabels(["0.5", "1", "2", "4", "8"])
ax.set_xticks(range(3)); ax.set_xticklabels([f"{b} ha" for b in B]); ax.set_xlabel("Canopy budget"); ax.set_xlim(-0.5, 2.5)
ax.set_ylabel("Residential cooling relative to one park")
ax.legend(loc="upper left", fontsize=4.9, frameon=False, handletextpad=0.2, borderaxespad=0.05, labelspacing=0.2)
label(ax, "d")
# e coverage against threshold, 100 ha, no displacement, all eligible land
ax = fig.add_subplot(gs[1, 1]); TH = ["02", "05", "10", "20"]; tx = [0.02, 0.05, 0.10, 0.20]
for dn, nm, c, mk in (("D30", "dispersed", NPG["teal"], "o"), ("PN", "units of ~2 ha", NPG["navy"], "D"), ("MIX", "half park, half dispersed", NPG["lav"], "s"), ("P1", "one park", NPG["brown"], "v")):
    ax.plot(tx, [P3[f"reach{t}_E1_R0_100_{dn}"] for t in TH], marker=mk, color=c, ms=3, lw=1, label=nm)
ax.set_xscale("log"); ax.set_xticks(tx); ax.set_xticklabels(["0.02", "0.05", "0.1", "0.2"])
from matplotlib.ticker import NullFormatter, NullLocator
ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("Cooling threshold (°C)"); ax.set_ylabel("Residents reached (%)"); ax.set_ylim(0, 27)
ax.legend(loc="upper right", fontsize=5.2, frameon=False, handletextpad=0.3, labelspacing=0.25, ncol=2, columnspacing=0.8)
ax.set_title("100 ha, all eligible land, no displacement", fontsize=5.6, color="#444444")
label(ax, "e")
# f sensitivity of the income gap
ax = fig.add_subplot(gs[1, 2])
I = CF["income"]["dT_pw"]; IW = CF["income_weighted"]["dT_pw"]
rows = [("Unadjusted", CF["unadjusted_LLM"], NPG["grey"]), ("Aridity, latitude", I["aridity_lat"]["LLM"], NPG["navy"]),
        ("…population-weighted", IW["aridity_lat"], NPG["lav"]), ("Canopy cover", I["canopy"]["LLM"], NPG["teal"])]
nfull = len(rows)
for reg, v in CF["loro"].items(): rows.append((f"no {reg}", [v["benefit"], np.nan, np.nan], NPG["grey"]))
for i, (nm, v, c) in enumerate(rows):
    y = len(rows) - 1 - i
    if np.isfinite(v[1]): ax.plot([max(v[1], -0.95), min(v[2], 0.75)], [y, y], color=c, lw=1)
    ax.scatter([v[0]], [y], s=12, color=c, zorder=3, marker="o" if i < nfull else "D")
ax.axvline(0, color="k", lw=0.5)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=5.3)
ax.set_xlabel("Low/lower-middle − high (°C)"); ax.set_xlim(-0.95, 0.8)
label(ax, "f", x=-0.66)
save(fig, "Fig4")
