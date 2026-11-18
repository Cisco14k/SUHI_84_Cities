"""Projections to 2100 (CMIP6 NEX-GDDP, bias-corrected, 0.25 deg): warm-season daily maximum air temperature at the 84 city centres.
Warm season = the city-region's three warmest months (TerraClimate 1991-2020 normals, as in the cooling analysis).
For each model, scenario and period: mean warm-season Tmax change vs 1995-2014 (dTx), the share of warm-season days with Tmax > 35 C,
and the signal-to-noise ratio dTx / sigma (interannual s.d. of warm-season mean Tmax in the historical run).
Derived planning quantities (Methods, Eq. 9-10):
  offset_10pp  = fraction of dTx offset at the surface by +10 pp canopy within 240 m of a receptor
               = (c1 + c2) / dTx, with c1 the city's 0-120 m cooling and c2 = c1 * rho_zone (ring-2 / ring-1 ratio of the zone profile)
  canopy_need  = 10 * dTx / (c1 + c2)   (pp of canopy within 240 m needed to offset dTx; capped at 100)
Outputs: data/out/proj_city.csv (city x model x scenario x period), data/out/proj_summary.csv (multi-model medians and ranges)."""
import glob, numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/claude/n84/src")
from common import warm_months
from sitab import table3
ROOT = "/home/claude/n84"
C = pd.read_csv(f"{ROOT}/data/cities_inc.csv")
zmap = lambda k: "Tropical" if k[0] == "A" else "Arid" if k[0] == "B" else "Temperate" if k[0] == "C" else "Continental"
C["zone"] = C.koppen.map(zmap)
T3 = table3(); T3 = T3[T3.source.str.contains("anopy")]
rho = {}
for z in ["Tropical", "Arid", "Temperate", "Continental"]:
    d = T3[T3.zone.str.startswith(z)].sort_values("r0"); rho[z] = float(d.cool.iloc[1] / d.cool.iloc[0])
print("ring2/ring1 by zone", {k: round(v, 2) for k, v in rho.items()})
D = pd.concat([pd.read_csv(f) for f in glob.glob(f"{ROOT}/data/cmip/*.csv")])
D.loc[D.tx.abs() > 100, "tx"] = np.nan
D["date"] = pd.to_datetime(D.year.astype(str), format="%Y") + pd.to_timedelta(D.doy - 1, unit="D")
D["month"] = D.date.dt.month
WM = {i: warm_months(C.city[i]) for i in range(len(C))}
D = D[[m in WM[i] for i, m in zip(D.idx, D.month)]]
# season year: for seasons spanning the new year (e.g. Dec-Feb) assign Dec to the next season
D["syear"] = D.year + np.where([(WM[i][0] > WM[i][-1]) and (m >= WM[i][0]) for i, m in zip(D.idx, D.month)], 1, 0)
D["period"] = np.select([D.year <= 2014, D.year <= 2060], ["1995-2014", "2041-2060"], "2081-2100")
D["hot"] = (D.tx > 35).astype(float)
S = D.groupby(["model", "scenario", "period", "idx", "syear"]).agg(tx=("tx", "mean"), hot=("hot", "mean")).reset_index()
H = S[S.scenario == "historical"].groupby(["model", "idx"]).agg(tx0=("tx", "mean"), sd0=("tx", "std"), hot0=("hot", "mean")).reset_index()
F = S[S.scenario != "historical"].groupby(["model", "scenario", "period", "idx"]).agg(tx=("tx", "mean"), hot=("hot", "mean")).reset_index()
F = F.merge(H, on=["model", "idx"])
F["dtx"] = F.tx - F.tx0; F["snr"] = F.dtx / F.sd0; F["dhot"] = F.hot - F.hot0
F = F.merge(C[["city", "country", "zone", "inc2", "region", "popm", "res240", "cool_0_120", "canopy_pct", "reach_interp", "lat", "lon"]].reset_index().rename(columns={"index": "idx"}), on="idx")
F["c240"] = F.cool_0_120 * (1 + F.zone.map(rho))
F["offset_10pp"] = F.c240 / F.dtx
F["canopy_need"] = np.minimum(10 * F.dtx / F.c240, 100)
F.to_csv(f"{ROOT}/data/out/proj_city.csv", index=False)
def q(x, p): return float(np.nanpercentile(x, p))
# multi-model median per city, then summaries
M = F.groupby(["scenario", "period", "idx"]).agg(dtx=("dtx", "median"), dtx_lo=("dtx", lambda x: q(x, 10)), dtx_hi=("dtx", lambda x: q(x, 90)),
                                                  snr=("snr", "median"), dhot=("dhot", "median"), hot0=("hot0", "median"), nmod=("dtx", "size"),
                                                  offset=("offset_10pp", "median"), need=("canopy_need", "median")).reset_index()
M = M.merge(C[["city", "zone", "inc2", "region", "popm", "res240", "canopy_pct"]].reset_index().rename(columns={"index": "idx"}), on="idx")
M.to_csv(f"{ROOT}/data/out/proj_citymed.csv", index=False)
def wavg(x, w): return float(np.sum(x * w) / np.sum(w))
rows = []
for (s, p), d in M.groupby(["scenario", "period"]):
    for g, dd in [("All", d)] + list(d.groupby("inc2")) + list(d.groupby("zone")):
        rows.append(dict(scenario=s, period=p, group=g, k=len(dd), nmod=int(dd.nmod.median()), dtx_med=dd.dtx.median(), dtx_pw=wavg(dd.dtx, dd.popm),
                         dtx_lo=dd.dtx_lo.median(), dtx_hi=dd.dtx_hi.median(), snr_med=dd.snr.median(), dhot_pp=100 * dd.dhot.median(),
                         offset_med=dd.offset.median(), need_med=dd.need.median(), share_need_gt30=float((dd.need > 30).mean()),
                         pop_beyond_m=float(np.sum(dd.popm * (100 - dd.res240) / 100)),
                         pw_dtx_beyond=wavg(dd.dtx, dd.popm * (100 - dd.res240) / 100)))
R = pd.DataFrame(rows); R.to_csv(f"{ROOT}/data/out/proj_summary.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
print(R[R.group.isin(["All", "Low & lower-middle", "Upper-middle", "High"])].round(2).to_string())
print("models:", sorted(F.model.unique()))
