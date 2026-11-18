"""Equity: access to canopy cooling (residents within 240 m of canopy), canopy cover and cooling, by World Bank FY2026 income group."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf
ROOT = "/home/claude/n84"; CT = pd.read_csv(f"{ROOT}/data/cities.csv")
G = {"L": ["Chad", "Madagascar", "Mali"],
     "LM": ["Bangladesh", "Bolivia", "Egypt", "Ghana", "Haiti", "Honduras", "India", "Ivory Coast", "Côte d'Ivoire", "Nigeria", "Papua New Guinea", "Tanzania", "Tunisia", "Zambia"],
     "UM": ["Argentina", "Azerbaijan", "Brazil", "Colombia", "Cuba", "El Salvador", "Kazakhstan", "Libya", "Mexico", "Moldova", "Peru", "Suriname", "Turkey", "Türkiye", "Ukraine", "China", "Venezuela"],
     }
def grp(c):
    for k, v in G.items():
        if c in v: return k
    return "H"
CT["inc"] = CT.country.map(grp)
CT["inc2"] = CT.inc.map({"L": "Low & lower-middle", "LM": "Low & lower-middle", "UM": "Upper-middle", "H": "High"})
print(CT.groupby("inc").country.unique())
CT["region"] = np.repeat(["Africa", "Asia", "Central America", "Europe", "North America", "Oceania", "South America"], 12)
CT["popm"] = CT["pop"] / 1e6
def wavg(x, w): return np.sum(x * w) / np.sum(w)
T = CT.groupby("inc2").apply(lambda d: pd.Series(dict(k=len(d), pop=d.popm.sum(), canopy_med=d.canopy_pct.median(), res240_med=d.res240.median(),
      res240_pw=wavg(d.res240, d.popm), cool_med=d.cool_0_120.median(), reach_med=d.reach_interp.median(),
      no_access_m=np.sum(d.popm * (100 - d.res240) / 100)))).loc[["Low & lower-middle", "Upper-middle", "High"]]
print(T.round(2)); T.to_csv(f"{ROOT}/data/out/equity_income.csv")
CT["la"] = np.log(CT.aridity)
m = smf.ols("res240 ~ C(inc2, Treatment('High')) + canopy_pct", CT).fit(cov_type="HC3"); print(m.summary().tables[1])
m2 = smf.ols("res240 ~ C(inc2, Treatment('High')) + la + np.abs(lat)", CT).fit(cov_type="HC3"); print(m2.summary().tables[1])
m3 = smf.ols("cool_0_120 ~ C(inc2, Treatment('High')) + la", CT).fit(cov_type="HC3"); print(m3.summary().tables[1])
# canopy needed: canopy-access elasticity; residents gaining access per 1 pp canopy (cross-city slope)
m4 = smf.ols("res240 ~ canopy_pct", CT).fit(cov_type="HC3"); print("slope res240 per pp canopy", m4.params.round(3).to_dict(), m4.conf_int().round(3).values.tolist())
from scipy.stats import spearmanr, kruskal
print("spearman canopy~res240", spearmanr(CT.canopy_pct, CT.res240))
print("kruskal res240 by inc2", kruskal(*[d.res240 for _, d in CT.groupby("inc2")]))
print("kruskal cool by inc2", kruskal(*[d.cool_0_120 for _, d in CT.groupby("inc2")]))
print("kruskal reach by inc2", kruskal(*[d.reach_interp for _, d in CT.groupby("inc2")]))
CT.to_csv(f"{ROOT}/data/cities_inc.csv", index=False)
