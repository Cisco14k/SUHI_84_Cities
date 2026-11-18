"""Compute result tokens for the text and write the new Supplementary Tables (13-17) and the Supplementary Figure section."""
import json, os, numpy as np, pandas as pd, statsmodels.formula.api as smf
ROOT = "/home/claude/n84"; O = f"{ROOT}/data/out"
TOK = {}
f1 = lambda x: f"{x:.1f}"; f0 = lambda x: f"{x:.0f}"; f2 = lambda x: f"{x:.2f}"; f3 = lambda x: f"{x:.3f}"
pct = lambda x: f"{100 * x:.0f}%"
def pfmt(p): return "*P* < 0.001" if p < 0.001 else f"*P* = {p:.3f}" if p < 0.01 else f"*P* = {p:.2f}"
CT = pd.read_csv(f"{ROOT}/data/cities_inc.csv")
zmap = lambda k: "Tropical" if k[0] == "A" else "Arid" if k[0] == "B" else "Temperate" if k[0] == "C" else "Continental"
CT["zone"] = CT.koppen.map(zmap)
tabs = []
# ---------------- sensor (Table 13)
FIT = json.load(open(f"{O}/psf_fit.json"))
rows = ["**Supplementary Table 13 | Sensor-footprint forward model.**", "",
        "| Sensor model | Ring 1, *L* = 0 | Ring 2, *L* = 0 | Apparent reach, *L* = 0 (m) | Fitted *L* (m) | Reach with sensor (m) | Reach without sensor (m) |",
        "|:---|---:|---:|---:|:---|:---|:---|"]
for s, f in FIT.items():
    rows.append(f"| {s} | {f['zero_L_ring1']:.3f} | {f['zero_L_ring2']:+.3f} | {f['zero_L_reach']:.0f} | {f['L']:.0f} ({f['L_ci'][0]:.0f}–{f['L_ci'][1]:.0f}) | "
                f"{f['reach_apparent']:.0f} ({f['reach_app_ci'][0]:.0f}–{f['reach_app_ci'][1]:.0f}) | {f['reach_true']:.0f} ({f['reach_true_ci'][0]:.0f}–{f['reach_true_ci'][1]:.0f}) |")
PS = json.load(open(f"{O}/psf_sens.json"))
Ls_ = [v["L"] for v in PS.values()]
TOK["PSF_SENS"] = f"*L* = {min(Ls_):.0f}–{max(Ls_):.0f} m for TIRS across eight scenarios"
SN = {"base": "Main assumptions", "canopy_-3K": "Canopy contrast −3 K", "canopy_-9K": "Canopy contrast −9 K", "green_-1.5_water_-12": "Open green −1.5 K, water −12 K",
      "noise_0.4K": "Noise s.d. 0.4 K", "noise_1.6K": "Noise s.d. 1.6 K", "noise_corr_50m": "Noise correlation 50 m", "noise_corr_400m": "Noise correlation 400 m"}
rows += ["", "| Sensitivity (TIRS) | Ring 1, *L* = 0 | Ring 2, *L* = 0 | Fitted *L* (m) |", "|:---|---:|---:|:---|"]
for k_, v in PS.items():
    rows.append(f"| {SN[k_]} | {v['ring1_L0']:.3f} | {v['ring2_L0']:+.3f} | {v['L']:.0f} ({v['L_ci'][0]:.0f}–{v['L_ci'][1]:.0f}) |")
rows += ["", "Ring values in °C per 10 pp from the forward model with no physical lateral cooling (mean of 12 city-regions). *L* fitted to observed rings 2–6 with free amplitude; 95% intervals from 300 bootstraps of city-regions and observed sampling error. Reach at *δ* = 0.05 °C per 10 pp.", ""]
tabs.append("\n".join(rows))
# ---------------- Lagrangian (Table 14)
L = pd.read_csv(f"{O}/lagr_city.csv"); L20 = L[L.year == 2020].merge(CT[["city", "zone", "inc2", "cool_0_120", "reach_interp", "aridity", "canopy_pct"]], on="city")
TOK["TRAJ_N"] = f"{int(L20.n.sum()):,}"
med = L20.median(numeric_only=True)
TOK["Q_LAND"] = pct(med.land_q); TOK["Q250"] = pct(med.q_250); TOK["Q50"] = pct(med.q_50); TOK["DQ"] = f"{med.d_q:,.0f}"
TOK["AGEQ"] = f0(med.age_q); TOK["H50"] = pct(med.h_50); TOK["AGEH"] = f0(med.age_h); TOK["FQ"] = pct(med.f_attr_q)
TOK["H50_O"] = pct(med.h50_o); TOK["LANDH_O"] = pct(med.land_h_o); TOK["LANDH"] = pct(med.land_h)
TOK["H_LOCAL"] = f"50 km ({pct(med.h_50)} of its heat) but over {med.d_q:,.0f} km for its moisture"
L20["lvpd"] = np.log(L20.vpd_ov); L20["lar"] = np.log(L20.aridity)
m1 = smf.ols("cool_0_120 ~ lvpd + ws_ov + land_q + canopy_pct", L20).fit(cov_type="HC3")
m2 = smf.ols("reach_interp ~ lvpd + ws_ov + land_q + canopy_pct", L20[L20.reach_interp > 0]).fit(cov_type="HC3")
def coef(m, v, sc=1, unit=""):
    b, (lo, hi) = m.params[v] * sc, m.conf_int().loc[v] * sc
    return f"{b:+.3f} ({lo:+.3f} to {hi:+.3f}){unit}, {pfmt(m.pvalues[v])}"
def cc(m, v, sc=1, d=3, u=""):
    b = m.params[v] * sc; lo, hi = m.conf_int().loc[v] * sc
    return f"{b:+.{d}f} {u}; 95% CI {lo:+.{d}f} to {hi:+.{d}f}; {pfmt(m.pvalues[v])}"
from statsmodels.stats.multitest import multipletests
_pv = [m1.pvalues[v] for v in ("lvpd", "ws_ov", "land_q", "canopy_pct")] + [m2.pvalues[v] for v in ("lvpd", "ws_ov", "land_q", "canopy_pct")]
_ph = multipletests(_pv, method="holm")[1]; TOK["HOLM_LAND"] = f"{_ph[6]:.2f}"
TOK["META"] = (f"cooling within 120 m was not related to overpass vapour-pressure deficit ({cc(m1, 'lvpd', u='°C per 10 pp per log unit')}), "
               f"overpass wind speed ({cc(m1, 'ws_ov', u='°C per m s^−1^')}) or the share of moisture from land ({cc(m1, 'land_q', 0.1, u='°C per 10 pp of share')}); "
               f"the practical reach was slightly longer where more moisture came from land ({cc(m2, 'land_q', 0.1, 1, u='m per 10 pp of share')}; Holm-adjusted *P* = {_ph[6]:.2f} across the eight tests) "
               f"but not related to vapour-pressure deficit or wind ({pfmt(m2.pvalues['lvpd'])} and {pfmt(m2.pvalues['ws_ov'])})")
rows = ["**Supplementary Table 14 | Lagrangian moisture and heat attribution of the overpass air, by city-region (2020).**", "",
        "| City-region | Zone | *q* attributed | *q* from land | *q* ≤ 50 km | *q* ≤ 250 km | *q* distance (km) | *q* age (h) | Heat ≤ 50 km | Heat age (h) | VPD (kPa) | Wind (m s^−1^) |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for _, r in L20.sort_values(["zone", "city"]).iterrows():
    rows.append(f"| {r.city} | {r.zone[:4] + ('.' if len(r.zone) > 4 else '')} | {pct(r.f_attr_q)} | {pct(r.land_q)} | {pct(r.q_50)} | {pct(r.q_250)} | {r.d_q:,.0f} | {r.age_q:.0f} | {pct(r.h_50)} | {r.age_h:.0f} | {r.vpd_ov:.2f} | {r.ws_ov:.1f} |")
rows += ["", "Shares are of the attributed moisture (*q*) or heat; distances and ages are contribution-weighted means; VPD and wind are overpass-hour means at the city centre. Trop., tropical; Temp., temperate; Cont., continental. Meta-regressions across city-regions (HC3 errors):", "",
         "| Outcome | log VPD | Wind speed | Land share of *q* | Canopy cover (pp) | *n* | *R*^2^ |", "|:---|:---|:---|:---|:---|---:|---:|"]
for nm, m in (("Cooling 0–120 m (°C per 10 pp)", m1), ("Practical reach (m)", m2)):
    dd = 3 if "Cooling" in nm else 1
    cs = [f"{m.params[v]:+.{dd}f} ({m.conf_int().loc[v][0]:+.{dd}f} to {m.conf_int().loc[v][1]:+.{dd}f})" for v in ("lvpd", "ws_ov", "land_q", "canopy_pct")]
    rows.append(f"| {nm} | " + " | ".join(cs) + f" | {int(m.nobs)} | {m.rsquared:.2f} |")
rows.append("")
tabs.append("\n".join(rows))
# ---------------- equity (Table 15)
E = pd.read_csv(f"{O}/equity_income.csv")
for g, k in (("Low & lower-middle", "EQ_LLM"), ("Upper-middle", "EQ_UM"), ("High", "EQ_H")):
    TOK[k] = f0(E.set_index("inc2").loc[g, "res240_pw"])
CT["la"] = np.log(CT.aridity)
r1 = smf.ols("res240 ~ C(inc2, Treatment('High')) + la + np.abs(lat)", CT).fit(cov_type="HC3")
r2 = smf.ols("res240 ~ C(inc2, Treatment('High')) + canopy_pct", CT).fit(cov_type="HC3")
r3 = smf.ols("cool_0_120 ~ C(inc2, Treatment('High')) + la", CT).fit(cov_type="HC3")
r4 = smf.ols("reach_interp ~ C(inc2, Treatment('High')) + la", CT[CT.reach_interp > 0]).fit(cov_type="HC3")
_v = [x for x in r3.params.index if "Low" in x][0]; _w = [x for x in r3.params.index if "Upper" in x][0]
TOK["INC_COOL"] = (f"{r3.params[_v]:+.3f} °C per 10 pp for low- and lower-middle-income (95% CI {r3.conf_int().loc[_v][0]:+.3f} to {r3.conf_int().loc[_v][1]:+.3f}) "
                   f"and {r3.params[_w]:+.3f} °C (95% CI {r3.conf_int().loc[_w][0]:+.3f} to {r3.conf_int().loc[_w][1]:+.3f}) for upper-middle-income relative to high-income city-regions")
rows = ["**Supplementary Table 15 | Access to canopy, canopy cover and cooling by World Bank income group.**", "",
        "| Income group | City-regions | Residents (million) | Canopy cover, median (%) | Residents ≤ 240 m, population-weighted (%) | Residents ≤ 240 m, median city-region (%) | Cooling 0–120 m, median | Practical reach, median (m) |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|"]
for _, r in E.iterrows():
    rows.append(f"| {r.inc2} | {int(r.k)} | {r['pop']:.1f} | {r.canopy_med:.1f} | {r.res240_pw:.0f} | {r.res240_med:.0f} | {r.cool_med:.2f} | {r.reach_med:.0f} |")
rows += ["", "| Model (outcome; covariates) | Low and lower-middle vs high | Upper-middle vs high |", "|:---|:---|:---|"]
for nm, m in (("Residents ≤ 240 m (pp); log aridity, |latitude|", r1), ("Residents ≤ 240 m (pp); canopy cover", r2), ("Cooling 0–120 m (°C per 10 pp); log aridity", r3), ("Practical reach (m); log aridity", r4)):
    v = [x for x in m.params.index if "Low" in x][0]; w = [x for x in m.params.index if "Upper" in x][0]
    dd = 3 if "Cooling" in nm else 1
    rows.append(f"| {nm} | {m.params[v]:+.{dd}f} ({m.conf_int().loc[v][0]:+.{dd}f} to {m.conf_int().loc[v][1]:+.{dd}f}) | {m.params[w]:+.{dd}f} ({m.conf_int().loc[w][0]:+.{dd}f} to {m.conf_int().loc[w][1]:+.{dd}f}) |")
rows += ["", "Least squares with HC3 errors; 95% CIs. Income groups are World Bank FY2026; Venezuela (unclassified) is grouped as upper-middle income.", ""]
tabs.append("\n".join(rows))
# ---------------- high-resolution (Table 16)
H = json.load(open(f"{O}/hr_pool.json"))
r60 = H["r60"]; mids = np.array(r60["mid"]); mu = np.array(r60["mu"]); se = np.array(r60["se"])
TOK["HR_K"] = str(len(H["kernel"]["cities"]))
J = [json.load(open(f"{O}/hr/{f}")) for f in sorted(os.listdir(f"{O}/hr")) if f.endswith(".json")]
TOK["HR_S"] = str(sum(len(j["scenes"]) for j in J if len(j["scenes"]) >= 3))
ci_ = lambda q: f"{mu[q]:.3f} °C (95% CI {mu[q] - 1.96 * se[q]:.3f} to {mu[q] + 1.96 * se[q]:.3f})"
TOK["HR_R60"] = f"{ci_(0)} at 0–60 m, {ci_(1)} at 60–120 m, {ci_(2)} at 120–180 m and {ci_(3)} at 180–240 m"
k = H["kernel"]; TOK["LHAT_ABS"] = str(int(5 * round(k["Lhat"] / 5))); TOK["LHAT"] = f0(k["Lhat"]); TOK["LCI"] = f"{k['Lci'][0]:.0f}–{k['Lci'][1]:.0f}"; TOK["R95"] = f0(k["reach95"])
w = H["r120"]
def wtxt(d): return f"*F*~{d['q']},{len(J) - d['q']}~ = {d['F']:.1f}, {pfmt(d['p'])}"
nC = len(k["cities"])
TOK["W23"] = f"difference {w['wald_r2_r3']['est'][0]:+.3f} °C, *F*~1,{nC - 1}~ = {w['wald_r2_r3']['F']:.1f}, {pfmt(w['wald_r2_r3']['p'])}"
_o = np.array(w["mu"][3:]); _s = np.array(w["se"][3:])
TOK["WOUT_CI"] = f"{(_o - 1.96 * _s).min():+.3f} to {(_o + 1.96 * _s).max():+.3f}"
TOK["WOUT"] = f"*F*~6,{nC - 6}~ = {w['wald_outer0']['F']:.2f}, {pfmt(w['wald_outer0']['p'])}"

fe = H.get("fetch", {})
if fe:
    f120 = fe["120"]
    TOK["FE_N"] = f"{f120['n_scenes']} scenes in {f120['n_cities']} city-regions"
    TOK["FE_DIFF"] = f"{f120['diff_mean']:+.3f} °C per 10 pp (95% CI {f120['diff_mean'] - 1.96 * f120['diff_se']:+.3f} to {f120['diff_mean'] + 1.96 * f120['diff_se']:+.3f})"
    TOK["FE_SLOPE"] = f"{f120['slope_ws']:+.3f} °C per m s^−1^, {pfmt(f120['slope_p'])}"
    TOK["FE_BETTER"] = f"fetch better in {pct(np.mean([fe[x]['frac_fetch_better'] for x in fe]))}"
    TOK["FE_SUMMARY"] = f"upwind minus downwind {TOK['FE_DIFF']} at *L*~*a*~ = 120 m; slope on wind speed {TOK['FE_SLOPE']}"
rows = ["**Supplementary Table 16 | High-resolution re-analysis: 60 m rings, joint tests, kernel length and fetch.**", "",
        "| 60 m canopy ring (m) | Cooling (°C per 10 pp) | 95% CI |", "|:---|---:|:---|"] + [
        f"| {int(J[0]['E60'][q])}–{int(J[0]['E60'][q + 1])} | {mu[q]:.3f} | {mu[q] - 1.96 * se[q]:.3f} to {mu[q] + 1.96 * se[q]:.3f} |" for q in range(len(mu))] + [
        "", "| Joint test (120 m rings, city-region means) | Estimate | Statistic |", "|:---|:---|:---|",
        f"| Ring 1 = ring 2 | {w['wald_r1_r2']['est'][0]:+.3f} | *F* = {w['wald_r1_r2']['F']:.2f}, {pfmt(w['wald_r1_r2']['p'])} |",
        f"| Ring 2 = ring 3 | {w['wald_r2_r3']['est'][0]:+.3f} | *F* = {w['wald_r2_r3']['F']:.2f}, {pfmt(w['wald_r2_r3']['p'])} |",
        f"| Rings 4–9 = 0 (jointly) | — | *F* = {w['wald_outer0']['F']:.2f}, {pfmt(w['wald_outer0']['p'])} |",
        f"| Monotone decline rings 2→3→4 | — | *F* = {w['wald_decline']['F']:.2f}, {pfmt(w['wald_decline']['p'])} |",
        "", f"Kernel length by profile likelihood: *L̂* = {k['Lhat']:.0f} m (city bootstrap 95% CI {k['Lci'][0]:.0f}–{k['Lci'][1]:.0f} m); 95% of the kernel lies within {k['reach95']:.0f} m ({k['reach95_ci'][0]:.0f}–{k['reach95_ci'][1]:.0f} m).", ""]
if fe:
    rows += ["| Fetch length *L*~*a*~ (m) | Scenes | Upwind | Downwind | Upwind − downwind (95% CI) | Slope on wind speed (per m s^−1^) | Scenes where fetch fits better | ΔRSS, fetch − isotropic |",
             "|---:|---:|---:|---:|:---|:---|---:|---:|"]
    for La in sorted(fe, key=int):
        f = fe[La]
        rows.append(f"| {La} | {f['n_scenes']} | {f['up']:.3f} | {f['down']:.3f} | {f['diff_mean']:+.3f} ({f['diff_mean'] - 1.96 * f['diff_se']:+.3f} to {f['diff_mean'] + 1.96 * f['diff_se']:+.3f}) | {f['slope_ws']:+.4f} ({pfmt(f['slope_p'])}) | {pct(f['frac_fetch_better'])} | {100 * f['dsse_rel']:+.2f}% |")
    rows += ["", "Cooling in °C per 10 pp; upwind and downwind are means of city-region means; CIs from the scatter of city-region means; slopes from weighted least squares with city-clustered errors.", ""]
tabs.append("\n".join(rows))
# ---------------- projections (Table 17)
R = pd.read_csv(f"{O}/proj_summary.csv"); M = pd.read_csv(f"{O}/proj_citymed.csv"); F = pd.read_csv(f"{O}/proj_city.csv")
TOK["NMOD"] = str(F.model.nunique())
g = lambda s, p, grp, c: float(R[(R.scenario == s) & (R.period == p) & (R.group == grp)][c].iloc[0])
TOK["P245"] = f1(g("ssp245", "2081-2100", "All", "dtx_med")); TOK["P585"] = f1(g("ssp585", "2081-2100", "All", "dtx_med"))
d_inc = {grp: g("ssp245", "2081-2100", grp, "dtx_med") for grp in ("Low & lower-middle", "Upper-middle", "High")}
TOK["PINC"] = ("similar across income groups" if max(d_inc.values()) - min(d_inc.values()) < 0.3 else
               f"larger in high-income ({d_inc['High']:.1f} °C) than in low- and lower-middle-income city-regions ({d_inc['Low & lower-middle']:.1f} °C) under SSP2-4.5"
               if d_inc["High"] > d_inc["Low & lower-middle"] else f"larger in low- and lower-middle-income ({d_inc['Low & lower-middle']:.1f} °C) than in high-income city-regions ({d_inc['High']:.1f} °C) under SSP2-4.5")
TOK["SNR_LLM"] = f1(g("ssp245", "2081-2100", "Low & lower-middle", "snr_med")); TOK["SNR_H"] = f1(g("ssp245", "2081-2100", "High", "snr_med"))
TOK["DHOT_LLM"] = f0(g("ssp245", "2081-2100", "Low & lower-middle", "dhot_pp")); TOK["DHOT_H"] = f0(g("ssp245", "2081-2100", "High", "dhot_pp"))
TOK["SNR_ALL"] = f1(g("ssp245", "2081-2100", "All", "snr_med"))
TOK["OFF245"] = pct(g("ssp245", "2081-2100", "All", "offset_med")); TOK["OFF585"] = pct(g("ssp585", "2081-2100", "All", "offset_med"))
om = g("ssp245", "2041-2060", "All", "offset_med"); TOK["OFFMID"] = pct(om); TOK["OFFMID_SC"] = f"{100 * 0.25 * om:.0f}–{100 * 0.5 * om:.0f}%"
o245 = g("ssp245", "2081-2100", "All", "offset_med"); TOK["OFF245_HALF"] = pct(0.5 * o245); TOK["OFF245_QUART"] = pct(0.25 * o245); TOK["OFF245_RANGE"] = f"{100 * 0.25 * o245:.0f}–{100 * 0.5 * o245:.0f}%"
TOK["NEED245"] = f0(g("ssp245", "2081-2100", "All", "need_med")); TOK["NEED30"] = pct(g("ssp245", "2081-2100", "All", "share_need_gt30"))
TOK["PROJ_ABS"] = f"by 2081–2100, 10 pp more canopy within reach would offset at most about {pct(g('ssp245', '2081-2100', 'All', 'offset_med'))} of projected warming under SSP2-4.5."
SL = {"ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}
GA = {"All": "All", "Low & lower-middle": "Low/LM", "Upper-middle": "UM", "High": "High", "Tropical": "Trop.", "Arid": "Arid", "Temperate": "Temp.", "Continental": "Cont."}
rows = []
for si_, sc_ in enumerate(("ssp245", "ssp585")):
    rows += [f"**Supplementary Table 17{'' if si_ == 0 else ' (continued)'} | Projected warm-season warming and the canopy offset, {SL[sc_]} (NEX-GDDP-CMIP6).**", "",
             "| Period | Group | *n* | Δ*T*~max~ (°C) | 10th–90th (°C) | S/N | Δ days > 35 °C (pp) | Offset | Need (pp) | Need > 30 pp |",
             "|:---|:---|---:|---:|:---|---:|---:|---:|---:|---:|"]
    for _, r in R[(R.scenario == sc_) & R.group.isin(list(GA))].iterrows():
        rows.append(f"| {r.period.replace('-', '–')} | {GA[r.group]} | {r.k} | {r.dtx_med:.1f} | {r.dtx_lo:.1f}–{r.dtx_hi:.1f} | {r.snr_med:.1f} | {r.dhot_pp:+.1f} | {pct(r.offset_med)} | {r.need_med:.0f} | {pct(r.share_need_gt30)} |")
    rows.append("")
rows += [f"Medians over city-regions of multi-model medians ({TOK['NMOD']} models). S/N, change divided by the interannual standard deviation of 1995–2014; offset, share of warming offset at the surface by +10 pp canopy within 240 m (an upper bound for air temperature; halve or quarter it if air-temperature cooling is half or a quarter of surface cooling); need, canopy increase within 240 m that would offset it, capped at 100 pp. Low/LM, low and lower-middle income; UM, upper-middle income; Trop., tropical; Temp., temperate; Cont., continental.", ""]
tabs.append("\n".join(rows))
open(f"{ROOT}/txt/si_tables.md", "w").write(__import__("re").sub(r"(?<=[ (|])-(?=\d)", "−", "\n\n".join(tabs)))
open(f"{ROOT}/txt/si_figs.md", "w").write(
    "**Supplementary Figures**\n\n![](figs/SuppFig1.png){width=16.5cm}\n\n**Supplementary Fig. 1 | Directional fetch tests with ERA5 winds.** **a**, Upwind and downwind canopy cooling and their difference against fetch length (60 m re-analysis, overpass winds of 2016, 2017 and 2020; city-clustered 95% CI). **b**, Upwind minus downwind canopy cooling at *L*~*a*~ = 120 m against overpass wind speed (scenes; red, weighted fit). **c**, Share of scenes in which the fetch kernel fits surface temperature better than an isotropic kernel of the same length. **d**, Relative change of the residual sum of squares summed over scenes.\n\n"
    "![](figs/SuppFig2.png){width=16.5cm}\n\n**Supplementary Fig. 2 | Ring geometry in the forward model.** Apparent (solid) and physical (dotted) TIRS profiles with standard ring edges and with edges shifted outward by 60 m, for *L* = 0, 60 and 120 m.\n\n"
    "![](figs/SuppFig3.png){width=12cm}\n\n**Supplementary Fig. 3 | City-specific kernel lengths.** Profile-likelihood estimates of *L* in each re-analysis city-region, coloured by zone; dashed, pooled estimate; band, its 95% CI.\n\n"
    "![](figs/SuppFig4.png){width=16.5cm}\n\n**Supplementary Fig. 4 | Moisture and heat sources of the overpass air in eight city-regions.** Attributed moisture uptake (**a**) and sensible-heat gain (**b**) along 96 h ERA5 back-trajectories released at the overpass hour on warm-season days (2020, and 2016, 2017 and 2020 for re-analysis city-regions), per 0.25° cell (log colour scale, each panel scaled to its own 99.5th percentile; grey lines, coastlines from the ERA5 sea mask; star, city centre). Windows span ±10° of latitude.\n")
import re as _re
TOK = {k: _re.sub(r"(?<![\w–~])-(?=\d)", "−", v) for k, v in TOK.items()}
json.dump(TOK, open(f"{O}/tokens.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(TOK, indent=1, ensure_ascii=False))
