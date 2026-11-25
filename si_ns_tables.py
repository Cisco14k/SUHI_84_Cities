"""Supplementary Tables 18-25 for the Nature Sustainability revision -> txt/si_ns_tables.md"""
import json, numpy as np, pandas as pd
ROOT = "/home/claude/n84ns"; O = f"{ROOT}/data/out"
HW = json.load(open(f"{O}/hrw_pool.json")); CF = json.load(open(f"{O}/cityfx.json")); F21 = json.load(open(f"{O}/forecast21.json"))
H = pd.read_csv(f"{O}/heat21.csv"); C = pd.read_csv(f"{O}/cities_merged.csv")
m = lambda x: f"{x:.3f}".replace("-", "−"); out = []
# Table 18
out.append("**Supplementary Table 18 \\| Threshold reach by ring width and kernel length by confounder set (60 m re-analysis).**\n")
out.append("| Ring width (m) | Ring 1 | Ring 2 | Ring 3 | Ring 4 | *R*~0.025~ (m) | *R*~0.05~ (m) | *R*~0.10~ (m) | Kernel-implied *R*~0.05~ (m) |")
out.append("|:--|--:|--:|--:|--:|:--|:--|:--|--:|")
for w in ("60", "90", "120", "180"):
    d = HW["widths"][w]; r = d["reach"]
    f = lambda k: f"{r[k][0]:.0f} ({r[k][1]:.0f}–{r[k][2]:.0f})"
    out.append(f"| {w} | " + " | ".join(m(x) for x in d["mu"][:4]) + f" | {f('0.025')} | {f('0.05')} | {f('0.1')} | {HW['kernel_pred']['reach'][w]['0.05']:.0f} |")
out.append("\n| Confounder set | *L̂* (m) | 95% CI (m) | 4.74 *L̂* (m) | *R*~0.05~, 120 m rings (m) |")
out.append("|:--|--:|:--|--:|--:|")
lab = {"V0": "Main", "V1": "Without receptor NDVI", "V2": "Without shares at 1.8–6 km"}
for v in ("V0", "V1", "V2"):
    d = HW["variants"][v]; rr = d.get("reach120", HW["widths"]["120"]["reach"]["0.05"][0])
    out.append(f"| {lab[v]} | {d['Lhat']:.0f} | {d['Lci'][0]:.0f}–{d['Lci'][1]:.0f} | {4.744 * d['Lhat']:.0f} | {rr:.0f} |")
out.append(f"\nPooled canopy cooling (°C per 10 pp) in the first four rings of each width, {HW['n_cities']} city-regions and {HW['n_scenes']} Landsat 8–9 scenes, WorldCover exposures, 60 m grid; reaches interpolated between ring mid-distances (equation (5)) with 400 city bootstraps. Kernel-implied reach: amplitude fitted to the 120 m profile beyond the first ring times the mass of the exponential kernel with *L̂* in each ring. Kernel lengths by profile likelihood with 1,000 city bootstraps.\n")
# Table 19
S = CF["income_summary"]; I = CF["income"]
out.append("**Supplementary Table 19 \\| Access to and modelled residential surface cooling from canopy by World Bank income group, 2015–2025 canopy and 2020 residents.**\n")
out.append("| Income group | City-regions | Residents (million) | Median canopy (%) | Residents ≤240 m of a patch (%) | Residents with ≥10% canopy within 240 m (%) | Modelled residential surface cooling (°C) | Residents receiving ≥0.5 °C (%) | Gain if canopy within 240 m reached 30% (°C) |")
out.append("|:--|--:|--:|--:|--:|--:|--:|--:|--:|")
for g in ("Low & lower-middle", "Upper-middle", "High"):
    d = S[g]; out.append(f"| {g} | {d['n']} | {d['residents_m']:.1f} | {d['canopy_med']:.1f} | {d['acc_pw']:.0f} | {d['can10_pw']:.0f} | {d['dT_pw']:.2f} | {d['dT05_pw']:.0f} | {d['gap30_pw']:.2f} |")
ci = lambda v, f="{:.2f}": (f.format(v[0]) + " (" + f.format(v[1]) + " to " + f.format(v[2]) + ")").replace("-", "−")
out.append("\n| Outcome; covariates | Low and lower-middle vs high | Upper-middle vs high |")
out.append("|:--|:--|:--|")
nm = {"res240": "Residents ≤240 m (pp, published)", "acc_pct": "Residents ≤240 m (pp, WorldCover)", "can10_pct": "Residents with ≥10% canopy within 240 m (pp)",
      "dT_pw": "Modelled residential surface cooling (°C)", "dT05_pct": "Residents receiving ≥0.5 °C (pp)"}
IW = CF["income_weighted"]["dT_pw"]
out.append(f"| Modelled residential surface cooling (°C); none | {ci(CF['unadjusted_LLM'])} | |")
out.append(f"| Modelled residential surface cooling (°C); log aridity, latitude, population-weighted | {ci(IW['aridity_lat'])} | |")
out.append(f"| Modelled residential surface cooling (°C); canopy cover, population-weighted | {ci(IW['canopy'])} | |")
for y in nm:
    for cv, cl in (("aridity_lat", "log aridity, latitude"), ("canopy", "canopy cover")):
        f = "{:.2f}" if y == "dT_pw" else "{:.1f}"
        out.append(f"| {nm[y]}; {cl} | {ci(I[y][cv]['LLM'], f)} | {ci(I[y][cv]['UM'], f)} |")
out.append("\n| Left-out world region | Median reach (m) | Median cooling 0–120 m (°C) | Access gap (pp) | Modelled-cooling gap (°C) |")
out.append("|:--|--:|--:|--:|--:|")
for r, v in CF["loro"].items():
    out.append(f"| {r} | {v['median_reach']:.0f} | {v['median_cool']:.3f} | {v['access_pub']:.1f} | {v['benefit']:.2f} |".replace("-", "−"))
out.append(f"\nPopulation-weighted shares and means over residents of urban cells (ESA WorldCover 2021 at 120 m; GlobPOP 2020). Modelled residential surface cooling uses the pooled canopy ring effects of Table 1 and excludes the receptor's own canopy. Least squares with HC3 errors, 95% CIs; city-regions weighted equally unless stated. Gaps in the last table are low and lower-middle minus high income, controlling for log aridity and absolute latitude. Leaving out one city-region at a time, the modelled-cooling difference ranged over {CF['loco_range']['benefit'][0]:.2f} to {CF['loco_range']['benefit'][1]:.2f} °C with equal weights and {CF['loco_weighted'][0]:.2f} to {CF['loco_weighted'][1]:.2f} °C with population weights. Replication of the published access measure: Spearman ρ = {CF['replication']['spearman']:.2f}.\n".replace("-0", "−0"))
# Table 20
out.append("**Supplementary Table 20 \\| Heat co-location in the 21 re-analysis city-regions under three heat indicators.**\n")
out.append("| City-region | Residents (million) | Residents >240 m (%) | Hottest third by observed excess, >240 m (%) | Hottest third without modelled canopy cooling, >240 m (%) | Most built-up third, >240 m (%) | Cooling ratio, hottest ÷ others, observed | Cooling ratio, hottest ÷ others, no canopy | Cooling ratio, built-up ÷ others |")
out.append("|:--|--:|--:|--:|--:|--:|--:|--:|--:|")
for _, r in H.iterrows():
    out.append(f"| {r.city} | {r.residents / 1e6:.2f} | {100 * r.all_beyond:.0f} | {100 * r.hot_beyond:.0f} | {100 * r.hot_beyond_cf:.0f} | {100 * r.hot_beyond_b:.0f} | {r.dT_hot / r.dT_cool:.2f} | {r.dT_hot_cf / r.dT_cool_cf:.2f} | {r.dT_hot_b / r.dT_cool_b:.2f} |".replace("-", "−"))
mm = H.median(numeric_only=True)
out.append(f"| Median | | {100 * mm.all_beyond:.0f} | {100 * mm.hot_beyond:.0f} | {100 * mm.hot_beyond_cf:.0f} | {100 * mm.hot_beyond_b:.0f} | {(H.dT_hot / H.dT_cool).median():.2f} | {(H.dT_hot_cf / H.dT_cool_cf).median():.2f} | {(H.dT_hot_b / H.dT_cool_b).median():.2f} |")
out.append("\nSurface-temperature excess of 60 m urban cells over the rural background (median over clear Landsat 8–9 scenes); residents from GlobPOP 2020; patches of canopy ≥ 0.5 covering ≥ 2 ha (WorldCover 2021); modelled residential surface cooling with the pooled 60 m ring effects. Thirds of residents ranked by observed excess, by excess plus the modelled cooling of surrounding canopy (no-canopy counterfactual) and by built-up cover within 240 m; cooling ratios are population-weighted modelled cooling of the top third divided by that of the other residents.\n")
# Table 21
P3 = json.load(open(f"{O}/planting3.json"))
out.append("**Supplementary Table 21 \\| Residential cooling of planting designs relative to one park, on common sets of city-regions in which every design is feasible.**\n")
out.append("| Eligible land; displacement | Budget (ha) | City-regions | +10 pp cells | +20 pp cells | +30 pp cells | Units of ~2 ha | Half park, half +30 pp | +30 pp cells, saturating |")
out.append("|:--|--:|--:|:--|:--|:--|:--|:--|:--|")
LN = {"E1": "All", "E2": "Less-occupied"}; RN = {"R0": "none", "R1": "proportional"}
for en in ("E1", "E2"):
    for rn in ("R0", "R1"):
        for B in (5, 20, 100):
            cells = []
            for a_ in ("D10", "D20", "D30", "PN", "MIX"):
                v = P3[f"{en}_{rn}_{B}_{a_}_vs_P1"]; cells.append(f"{v['median']:.2f} ({v['ci'][0]:.2f}–{v['ci'][1]:.2f})")
            v = P3[f"{en}_{rn}_{B}_D30_vs_P1"]; cells.append(f"{v['sat']:.2f} ({v['sat_ci'][0]:.2f}–{v['sat_ci'][1]:.2f})")
            out.append(f"| {LN[en]}; {RN[rn]} | {B} | {P3['common_n'][f'{en}_{B}']} | " + " | ".join(cells) + " |")
out.append(f"\nMedians over the common set of city-regions of the ratio of residential cooling (person-°C) to that of one round park with the same added canopy area; 95% intervals from 2,000 city bootstraps combined with draws of the ring effects (Table 1; saturating column, city bootstraps only). Conversion never exceeds a cell's built-up cover and is scaled down, never up, to the budget. Eligible land: urban cells with built-up ≥ 0.3 and canopy < 0.2; less-occupied: those with residents at or below the city-region median (median {100 * P3['e2_share']:.0f}% of eligible area; open green space and bare ground cover {P3['open_mean_E2']:.0f}% of these cells against {P3['open_mean_E1']:.0f}% of all eligible cells). Proportional displacement removes residents of a converted cell in proportion to the converted share of its built-up cover, in every design. Saturating: ring cooling rises with canopy fraction as (1 − e^−f/0.3^)/(1 − e^−1/0.3^). Greedy placement; placement uncertainty is not included.\n")
# Table 22
out.append("**Supplementary Table 22 \\| Held-out forecast skill and paired differences, 21 re-analysis city-regions (120 m canopy rings).**\n")
out.append("| Forecast or contrast | Skill or difference | 95% CI |")
out.append("|:--|--:|:--|")
for k, nmk in (("invest450", "InVEST kernel, 450 m"), ("invest_fit", f"InVEST kernel, fitted (d~cool~ = {F21['dcool_fitted_median']:.0f} m)"), ("empirical", "Empirical curve"),
               ("invest_fit-invest450", "Fitted minus 450 m"), ("empirical-invest450", "Empirical minus 450 m"), ("empirical-invest_fit", "Empirical minus fitted")):
    v = F21[k]; out.append(f"| {nmk} | {v[0]:.2f} | {v[1]:.2f} to {v[2]:.2f} |".replace("-", "−").replace("d~cool~", "*d*~cool~"))
out.append("\nSkill = 1 − Σ*w*(*μ* − *f*)^2^/Σ*w*(*μ* − *f*~c~)^2^ pooled over held-out city-regions; 2,000 paired city bootstraps.\n")
Cz = pd.read_csv(f"{ROOT}/data/cities_inc.csv"); Cz["zone"] = Cz.koppen.str[0].map({"A": "Tropical", "B": "Arid", "C": "Temperate", "D": "Continental"})
out.append("**Supplementary Table 23 \\| City-regions by income group, climate zone and world region.**\n")
regs = sorted(Cz.region.unique()); zs = ["Tropical", "Arid", "Temperate", "Continental"]; G3 = ("Low & lower-middle", "Upper-middle", "High")
out.append("| Category | Low and lower-middle income | Upper-middle income | High income | Total |")
out.append("|:--|--:|--:|--:|--:|")
for col, vals, pre in (("zone", zs, "Zone: "), ("region", regs, "Region: ")):
    for v_ in vals:
        out.append(f"| {pre}{v_} | " + " | ".join(str(((Cz.inc2 == g) & (Cz[col] == v_)).sum()) for g in G3) + f" | {(Cz[col] == v_).sum()} |")
out.append("| Total | " + " | ".join(str((Cz.inc2 == g).sum()) for g in G3) + f" | {len(Cz)} |")
out.append(f"\nWorld Bank income groups for fiscal year 2026; the low- and lower-middle-income group contains {int((Cz.inc == 'L').sum())} low-income city-regions. Köppen zone of the majority of urban cells (TerraClimate 1991–2020). Locations in Fig. 1a.\n")
# Table 24: feasibility and coverage
out.append("**Supplementary Table 24 \\| Feasibility of each planting design and coverage at four cooling thresholds.**\n")
out.append("| Eligible land | Budget (ha) | Feasible: +10 / +20 / +30 pp cells, units, park, half and half | Coverage at 0.02 / 0.05 / 0.10 / 0.20 °C, +30 pp cells (%) | Units of ~2 ha (%) | One park (%) |")
out.append("|:--|--:|:--|:--|:--|:--|")
TH = ("02", "05", "10", "20")
for en in ("E1", "E2"):
    for B in (5, 20, 100):
        fe = " / ".join(str(P3["feasible"][f"{en}_{B}_{d}"]) for d in ("D10", "D20", "D30", "PN", "P1", "MIX"))
        cv = lambda d: " / ".join(f"{P3[f'reach{t}_{en}_R0_{B}_{d}']:.1f}" for t in TH)
        out.append(f"| {LN[en]} | {B} | {fe} | {cv('D30')} | {cv('PN')} | {cv('P1')} |")
out.append("\nFeasible: number of the 84 city-regions in which the design places the full budget on eligible land without converting more than the built-up cover of any cell. Coverage: median share of residents whose modelled residential surface cooling from the planting reaches each threshold, no displacement, on the common set of city-regions of Supplementary Table 21.\n")
# Table 25: kernel shapes
KS = json.load(open(f"{O}/kernel_shapes.json"))
out.append("**Supplementary Table 25 \\| Kernel shape and the 95% source distance (60 m re-analysis ring profiles).**\n")
out.append("| Kernel | Ring widths | Scale (m) | Tail exponent | χ² | ΔAIC | 95% source distance (m) | 95% interval (m) |")
out.append("|:--|:--|--:|--:|--:|--:|--:|:--|")
for lab_, wl in (("w60", "60 m"), ("joint", "60–180 m, joint")):
    for k_ in ("exponential", "Gaussian", "power law", "power law, b = 1.5"):
        v = KS[lab_][k_]; bb = "" if v["b"] is None else f"{v['b']:.1f}"
        out.append(f"| {k_.replace('b = 1.5', '*b* = 1.5')} | {wl} | {v['scale']:.1f} | {bb} | {v['chi2']:.1f} | {v['daic']:.1f} | {v['r95']:.0f} | {v['r95_ci'][0]:.0f}–{v['r95_ci'][1]:.0f} |")
out.append("\nRing effects beyond the first ring fitted by an amplitude (one per ring width) times the kernel mass in each annulus; weighted least squares with the pooled standard errors; grid search over scale and tail exponent (largest exponent tested, 6); ΔAIC relative to the best kernel for the same ring widths; intervals from 300–600 parametric bootstrap draws of the ring effects.\n")
out.append("\n**Supplementary Fig. 5 \\| Moisture and heat sources of the air arriving over cities.** **a**,**b**, Composite of moisture uptake (**a**) and sensible-heat gain (**b**) along 96 h ERA5 back-trajectories released at the overpass hour, in city-relative coordinates rotated so that the mean overpass wind blows from left to right (share of the attributed total per km^2^; circles at 50, 250 and 750 km). **c**, Cumulative share of attributed moisture and heat taken up within a given distance of the city (median and interquartile range over city-regions, 2020). **d**, Share of attributed moisture taken up over land (colour) and within 250 km (size). **e**, Moisture uptake around three city-regions (log scale; ±10° latitude windows; bar, 500 km; star, city centre). **f**, Canopy cooling within 120 m against the mean overpass vapour-pressure deficit (dashed, robust fit).\n")
out.append(f"![]({ROOT}/figs/SuppFig5.png){{width=16cm}}\n")
open(f"{ROOT}/txt/si_ns_tables.md", "w").write("\n".join(out)); print("\n".join(out)[:3000])
