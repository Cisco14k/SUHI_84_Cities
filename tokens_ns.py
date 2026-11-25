"""Collect every number quoted in the Nature Sustainability manuscript into data/out/tokens_ns.json."""
import json, numpy as np, pandas as pd
ROOT = "/home/claude/n84ns"; O = f"{ROOT}/data/out"
T = json.load(open(f"{O}/tokens.json"))
CF = json.load(open(f"{O}/cityfx.json")); HW = json.load(open(f"{O}/hrw_pool.json")); HP = json.load(open(f"{O}/hr_pool.json"))
F21 = json.load(open(f"{O}/forecast21.json")); H21 = pd.read_csv(f"{O}/heat21.csv")
f0 = lambda x: f"{x:.0f}"; f2 = lambda x: f"{x:.2f}"; f1 = lambda x: f"{x:.1f}"
neg = lambda s: s.replace("-", "−")
ci = lambda a, b, f=f2: neg(f"{f(a)}–{f(b)}") if a >= 0 else neg(f"{f(a)} to {f(b)}")
k = HP["kernel"]; v = HW["variants"]
T.update(LHAT=f0(k["Lhat"]), LCI=f"{k['Lci'][0]:.0f}–{k['Lci'][1]:.0f}", R90=f0(3.89 * k["Lhat"]), R95=f0(4.744 * k["Lhat"]),
         R95CI=f"{4.744 * k['Lci'][0]:.0f}–{4.744 * k['Lci'][1]:.0f}",
         LV0=f"{v['V0']['Lhat']:.0f}", LV1=f"{v['V1']['Lhat']:.0f}", LV2=f"{v['V2']['Lhat']:.0f}",
         HRW_K=str(HW["n_cities"]), HRW_S=f"{HW['n_scenes']:,}")
for w in ("60", "90", "120", "180"):
    r = HW["widths"][w]["reach"]["0.05"]; T[f"RW{w}"] = f"{r[0]:.0f} m ({r[1]:.0f}–{r[2]:.0f} m)"
    T[f"KP{w}"] = f0(HW["kernel_pred"]["reach"][w]["0.05"])
Z = CF["zone_reach"]["zones"]
T.update(Z_TROP=f0(Z["Tropical"]["median"]), Z_ARID=f0(Z["Arid"]["median"]), Z_TEMP=f0(Z["Temperate"]["median"]), Z_CONT=f0(Z["Continental"]["median"]),
         ZKW_P=f2(CF["zone_reach"]["kruskal_p"]), Z_ARID_CI=ci(*Z["Arid"]["diff90"], f=f0))
L = CF["loro"]; T["LORO_REACH"] = f"{min(x['median_reach'] for x in L.values()):.0f}–{max(x['median_reach'] for x in L.values()):.0f}"
S = CF["income_summary"]; I = CF["income"]
T.update(DT_LLM=f2(S["Low & lower-middle"]["dT_pw"]), DT_UM=f2(S["Upper-middle"]["dT_pw"]), DT_H=f2(S["High"]["dT_pw"]),
         DT05_LLM=f0(S["Low & lower-middle"]["dT05_pw"]), DT05_H=f0(S["High"]["dT05_pw"]),
         GAP30_LLM=f2(S["Low & lower-middle"]["gap30_pw"]), GAP30_H=f2(S["High"]["gap30_pw"]),
         REPL_RHO=f2(CF["replication"]["spearman"]))
b = I["dT_pw"]["aridity_lat"]["LLM"]; T["DT_REG"] = neg(f"{-b[0]:.2f} °C (95% CI {-b[2]:.2f}–{-b[1]:.2f})")
b = I["dT_pw"]["canopy"]["LLM"]; T["DT_REG_CAN"] = neg(f"{b[0]:+.2f}; 95% CI {b[1]:.2f} to {b[2]:+.2f}")
lc = CF["loco_range"]["benefit"]; T["LOCO_DT"] = neg(f"gap {-lc[1]:.2f}–{-lc[0]:.2f}")
T["LORO_DT_AFR"] = f2(-L["Africa"]["benefit"])
m = H21.median(numeric_only=True)
T.update(HOT_BEYOND=f0(100 * m.hot_beyond), ALL_BEYOND=f0(100 * m.all_beyond), EXC_DIFF=f1((H21.excess_beyond - H21.excess_within).median()),
         DT_HOT_RATIO=f0(100 * (H21.dT_hot / H21.dT_cool).median()), RHO_MED=f2(m.rho_dT_excess))
P = CF["planting"]
T.update(RATIO5=f1(P["5"]["ratio_med"]), RATIO20=f1(P["20"]["ratio_med"]), RATIO100=f1(P["100"]["ratio_med"]),
         REACH_D100=f0(P["100"]["reach_disp"]), REACH_P100=f1(P["100"]["reach_park"]))
d1 = F21["invest_fit-invest450"]; d2 = F21["empirical-invest_fit"]
T.update(DCOOL21=f0(F21["dcool_fitted_median"]), F21_D1=neg(f"{d1[0]:.2f} (95% CI {d1[1]:.2f}–{d1[2]:.2f})"),
         F21_D2=neg(f"difference {d2[0]:+.2f}; 95% CI {d2[1]:.2f} to {d2[2]:+.2f}"))
# revision 2: planting designs under common rules, weighting, alternative heat indicators
P2 = json.load(open(f"{O}/planting2.json"))
rr = lambda key: P2[key]["median"]
rci = lambda key: f"{P2[key]['median']:.2f} (95% CI {P2[key]['ci'][0]:.2f}–{P2[key]['ci'][1]:.2f})"
for en in ("E1", "E2"):
    for rn in ("R0", "R1"):
        for B in (5, 20, 100):
            for a in ("D10", "D20", "D30", "Pn", "MIX"):
                T[f"PL_{en}_{rn}_{B}_{a}"] = f"{rr(f'{en}_{rn}_{B}_{a}_vs_P1'):.2f}"
                T[f"PLC_{en}_{rn}_{B}_{a}"] = rci(f"{en}_{rn}_{B}_{a}_vs_P1")
            for a in ("D30", "P1", "Pn"):
                T[f"PLR_{en}_{rn}_{B}_{a}"] = f1(P2[f"reach_{en}_{rn}_{B}_{a}"])
T["E2_SHARE"] = f0(100 * P2["e2_share"]); T["NP1_E2_100"] = str(P2["nP1_E2_100"])
IW = CF["income_weighted"]["dT_pw"]
T["DT_REG_W"] = neg(f"{-IW['aridity_lat'][0]:+.2f} °C; 95% CI {-IW['aridity_lat'][2]:.2f} to {-IW['aridity_lat'][1]:+.2f}")
ua = CF["unadjusted_LLM"]; T["DT_UNADJ"] = neg(f"{-ua[0]:.2f} °C (95% CI {-ua[2]:.2f}–{-ua[1]:.2f})")
lw = CF["loco_weighted"]; T["LOCO_W"] = neg(f"{-lw[1]:+.2f} to {-lw[0]:+.2f}")
import os
if os.path.exists(f"{O}/heat21b.csv"):
    H2 = pd.read_csv(f"{O}/heat21b.csv"); m2 = H2.median(numeric_only=True)
    T.update(HOT_BEYOND_CF=f0(100 * m2.hot_beyond_cf), HOT_BEYOND_B=f0(100 * m2.hot_beyond_b),
             DT_HOT_RATIO_CF=f0(100 * (H2.dT_hot_cf / H2.dT_cool_cf).median()), DT_HOT_RATIO_B=f0(100 * (H2.dT_hot_b / H2.dT_cool_b).median()),
             EXCF_DIFF=f1((H2.excf_beyond - H2.excf_within).median()),
             HOT_GT_ALL_CF=str(int((H2.hot_beyond_cf > H2.all_beyond).sum())), HOT_GT_ALL_B=str(int((H2.hot_beyond_b > H2.all_beyond).sum())),
             HOT_GT_ALL=str(int((H2.hot_beyond > H2.all_beyond).sum())))
for kk in [k_ for k_ in T if "_Pn" in k_]: T[kk.replace("_Pn", "_PN")] = T[kk]
if os.path.exists(f"{O}/planting3.json"):
    P3 = json.load(open(f"{O}/planting3.json"))
    for en in ("E1", "E2"):
        for rn in ("R0", "R1"):
            for B in (5, 20, 100):
                for a in ("D10", "D20", "D30", "PN", "MIX"):
                    vv = P3[f"{en}_{rn}_{B}_{a}_vs_P1"]
                    T[f"P3_{en}_{rn}_{B}_{a}"] = f"{vv['median']:.2f}"
                    T[f"P3C_{en}_{rn}_{B}_{a}"] = f"{vv['median']:.2f} (95% CI {vv['ci'][0]:.2f}–{vv['ci'][1]:.2f})"
                    T[f"P3S_{en}_{rn}_{B}_{a}"] = f"{vv['sat']:.2f}"
                    T[f"P3S15_{en}_{rn}_{B}_{a}"] = f"{vv.get('sat15', float('nan')):.2f}"; T[f"P3S60_{en}_{rn}_{B}_{a}"] = f"{vv.get('sat60', float('nan')):.2f}"
                for d in ("D10", "D20", "D30", "PN", "P1", "MIX"):
                    for t in ("02", "05", "10", "20"):
                        T[f"P3R{t}_{en}_{rn}_{B}_{d}"] = f1(P3[f"reach{t}_{en}_{rn}_{B}_{d}"])
        for B in (5, 20, 100):
            T[f"P3N_{en}_{B}"] = str(P3["common_n"][f"{en}_{B}"])
            T[f"P3FP1_{en}_{B}"] = str(P3["feasible"][f"{en}_{B}_P1"])
    for en in ("E1", "E2"):
        ex = P3[f"excl100_{en}"]
        T[f"EXC_HA_{en}"] = f"{ex['elig_ha_out']:,.0f}"; T[f"INC_HA_{en}"] = f"{ex['elig_ha_in']:,.0f}"; T[f"EXC_N_{en}"] = str(ex["n_excl"])
        T[f"EXC_DENS_{en}"] = f0(ex["dens_out"]); T[f"INC_DENS_{en}"] = f0(ex["dens_in"])
    T["OPEN_E1"] = f0(P3["open_mean_E1"]); T["OPEN_E2"] = f0(P3["open_mean_E2"])
if os.path.exists(f"{O}/kernel_shapes.json"):
    KSH = json.load(open(f"{O}/kernel_shapes.json"))["w60"]
    T["KS_GAUSS"] = f"{KSH['Gaussian']['r95']:.0f} ({KSH['Gaussian']['r95_ci'][0]:.0f}–{KSH['Gaussian']['r95_ci'][1]:.0f})"
    T["KS_POW"] = f"{KSH['power law']['r95']:.0f} ({KSH['power law']['r95_ci'][0]:.0f}–{KSH['power law']['r95_ci'][1]:.0f})"
    T["KS_EXP"] = f"{KSH['exponential']['r95']:.0f} ({KSH['exponential']['r95_ci'][0]:.0f}–{KSH['exponential']['r95_ci'][1]:.0f})"
    T["KS_HEAVY_DAIC"] = f"{KSH['power law, b = 1.5']['daic']:.1f}"; T["KS_HEAVY"] = f"{KSH['power law, b = 1.5']['r95']:.0f}"
    T["KS_EXP_L"] = f0(KSH["exponential"]["scale"]); T["KS_EXP_DAIC"] = f"{KSH['exponential']['daic']:.1f}"; T["KS_POW_DAIC"] = f"{KSH['power law']['daic']:.1f}"
    T["KS_RANGE"] = f"{min(KSH[k]['r95'] for k in ('Gaussian', 'power law', 'exponential')):.0f}–{max(KSH[k]['r95'] for k in ('Gaussian', 'power law', 'exponential')):.0f}"
    T["KS_POW_B"] = f"{KSH['power law']['b']:.0f}"
    KJ = json.load(open(f"{O}/kernel_shapes.json"))["joint"]
    T["KSJ_RANGE"] = f"{min(KJ[k]['r95'] for k in ('Gaussian', 'power law', 'exponential')):.0f}–{max(KJ[k]['r95'] for k in ('Gaussian', 'power law', 'exponential')):.0f}"
T["LV0CI"] = f"{v['V0']['Lci'][0]:.0f}–{v['V0']['Lci'][1]:.0f}"
T["F21_INV"] = f"+{F21['invest450'][0]:.2f}" if isinstance(F21['invest450'], list) else f"+{F21['invest450']:.2f}"
json.dump(T, open(f"{O}/tokens_ns.json", "w"), indent=1, ensure_ascii=False)
for kk in sorted(T): print(kk, "=", str(T[kk])[:110])
