"""Nature Sustainability revision: city-level sensitivity analyses.
(1) threshold-defined practical reach of individual city-regions by climate zone (Kruskal-Wallis; bootstrap medians; two one-sided tests of
    the difference in median reach between each zone and the others against +/-60 m);
(2) leave-one-world-region-out and leave-one-city-out sensitivity of the median reach, the median first-ring cooling and the income-group
    contrasts in access (published patch measure) and in treatment-aligned benefit (modelled cooling delivered to residents, access84.py);
(3) income-group contrasts of access and benefit with HC3 errors, controlling for log aridity and absolute latitude, or canopy cover.
Output: data/out/cityfx.json"""
import json, numpy as np, pandas as pd
from scipy import stats
ROOT = "/home/claude/n84ns"; rng = np.random.default_rng(3)
C = pd.read_csv(f"{ROOT}/data/cities_inc.csv")
C["zone"] = C.koppen.str[0].map({"A": "Tropical", "B": "Arid", "C": "Temperate", "D": "Continental"})
A = pd.read_csv(f"{ROOT}/data/out/access84.csv")
C = C.merge(A.drop(columns=["city", "inc2", "region", "koppen"]), left_index=True, right_on="idx", how="left").reset_index(drop=True)
C["labslat"] = np.abs(C.lat); C["laridity"] = np.log(C.aridity)
out = {}


def hc3(y, X):
    X = np.column_stack([np.ones(len(y)), X]); b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ b
    Hh = np.einsum("ij,jk,ik->i", X, np.linalg.inv(X.T @ X), X); B = np.linalg.inv(X.T @ X)
    V = B @ (X.T * (e / (1 - Hh)) ** 2) @ X @ B; return b, np.sqrt(np.diag(V)), len(y) - X.shape[1]


def income_fx(D, ycol, cov):
    d = D.dropna(subset=[ycol] + cov)
    X = np.column_stack([(d.inc2 == "Low & lower-middle").astype(float), (d.inc2 == "Upper-middle").astype(float)] + [d[c] for c in cov])
    b, se, df = hc3(d[ycol].values, X); t = stats.t.ppf(0.975, df)
    return {"LLM": [b[1], b[1] - t * se[1], b[1] + t * se[1]], "UM": [b[2], b[2] - t * se[2], b[2] + t * se[2]]}


# (1) reach by zone
R = C[C.reach_interp > 0]
kw = stats.kruskal(*[g.reach_interp.values for _, g in R.groupby("zone")])
zz = {}
for z, g in R.groupby("zone"):
    o = R[R.zone != z].reach_interp.values; v = g.reach_interp.values
    bs = np.array([np.median(rng.choice(v, len(v))) - np.median(rng.choice(o, len(o))) for _ in range(4000)])
    lo, hi = np.percentile(bs, [5, 95])                                   # 90% interval = TOST at alpha 0.05
    meds = np.array([np.median(rng.choice(v, len(v))) for _ in range(4000)])
    zz[z] = dict(n=len(v), median=float(np.median(v)), ci=[float(x) for x in np.percentile(meds, [2.5, 97.5])],
                 diff=float(np.median(v) - np.median(o)), diff90=[float(lo), float(hi)], equivalent60=bool(lo > -60 and hi < 60))
out["zone_reach"] = dict(kruskal_H=float(kw.statistic), kruskal_p=float(kw.pvalue), zones=zz, n_excluded_zero=int((C.reach_interp <= 0).sum()))

# (2) leave-one-region-out
covs = ["laridity", "labslat"]
lor = {}
for reg in sorted(C.region.unique()):
    D = C[C.region != reg]
    lor[reg] = dict(median_reach=float(D[D.reach_interp > 0].reach_interp.median()), median_cool=float(D.cool_0_120.median()),
                    access_pub=income_fx(D, "res240", covs)["LLM"][0], benefit=income_fx(D, "dT_pw", covs)["LLM"][0],
                    share_can10=income_fx(D.assign(s=100 * D.share_can10), "s", covs)["LLM"][0])
out["loro"] = lor
loo = {k: [] for k in ("access_pub", "benefit")}
for i in range(len(C)):
    D = C.drop(index=i)
    loo["access_pub"].append(income_fx(D, "res240", covs)["LLM"][0]); loo["benefit"].append(income_fx(D, "dT_pw", covs)["LLM"][0])
out["loco_range"] = {k: [float(min(v)), float(max(v))] for k, v in loo.items()}

# (2b) population weighting: weighted least squares with residents as weights (HC3 on the transformed model)
def income_fx_w(D, ycol, cov, wcol="residents"):
    d = D.dropna(subset=[ycol, wcol] + cov); w = np.sqrt(d[wcol].values / d[wcol].mean())
    X = np.column_stack([np.ones(len(d)), (d.inc2 == "Low & lower-middle").astype(float), (d.inc2 == "Upper-middle").astype(float)] + [d[c] for c in cov])
    Xw = X * w[:, None]; yw = d[ycol].values * w; B_ = np.linalg.inv(Xw.T @ Xw); b = B_ @ Xw.T @ yw; e = yw - Xw @ b
    Hh = np.einsum("ij,jk,ik->i", Xw, B_, Xw); V = B_ @ (Xw.T * (e / (1 - Hh)) ** 2) @ Xw @ B_; se = np.sqrt(np.diag(V))
    t = stats.t.ppf(0.975, len(d) - X.shape[1]); return [b[1], b[1] - t * se[1], b[1] + t * se[1]]
out["income_weighted"] = {"dT_pw": {"aridity_lat": income_fx_w(C, "dT_pw", covs), "canopy": income_fx_w(C, "dT_pw", ["canopy_urban"])},
                          "res240": {"aridity_lat": income_fx_w(C, "res240", covs)}}
# leave-one-city-out of the weighted contrast and the unadjusted group difference
lw = [income_fx_w(C.drop(index=i), "dT_pw", covs)[0] for i in range(len(C))]
out["loco_weighted"] = [float(min(lw)), float(max(lw))]
noafr = C[C.region != "Africa"]
out["noafrica_weighted"] = income_fx_w(noafr, "dT_pw", covs)
out["unadjusted_LLM"] = income_fx(C, "dT_pw", [])["LLM"]

# (3) income contrasts on the full sample
C["acc_pct"] = 100 * C.acc240; C["can10_pct"] = 100 * C.share_can10; C["dT05_pct"] = 100 * C.share_dT05
out["income"] = {y: {"aridity_lat": income_fx(C, y, covs), "canopy": income_fx(C, y, ["canopy_urban"])}
                 for y in ("res240", "acc_pct", "can10_pct", "dT_pw", "dT05_pct")}
g = C.groupby("inc2")
pw = lambda d, c: float((d[c] * d.residents).sum() / d.residents.sum())
out["income_summary"] = {k: dict(n=len(d), residents_m=float(d.residents.sum() / 1e6), canopy_med=float(100 * d.canopy_urban.median()),
                                  acc_pw=100 * pw(d, "acc240"), can10_pw=100 * pw(d, "share_can10"), dT_pw=pw(d, "dT_pw"),
                                  dT05_pw=100 * pw(d, "share_dT05"), gap30_pw=pw(d, "dT_gap30"))
                         for k, d in g}
out["kw_income_dT"] = float(stats.kruskal(*[d.dT_pw.values for _, d in g]).pvalue)
# replication check against the published access measure
ok = C.dropna(subset=["acc240"])
out["replication"] = dict(spearman=float(stats.spearmanr(ok.res240, 100 * ok.acc240).correlation),
                          pw_new=100 * pw(ok, "acc240"), median_new=float(100 * ok.acc240.median()), median_pub=float(ok.res240.median()))
# planting design
pd_ = {}
for B in (5, 20, 100):
    r = C[f"ratio_{B}"].replace([np.inf], np.nan).dropna()
    pd_[B] = dict(ratio_med=float(r.median()), ratio_iqr=[float(r.quantile(0.25)), float(r.quantile(0.75))], n_gt1=int((r > 1).sum()), n=len(r),
                  reach_disp=float(100 * C[f"reach_disp_{B}"].median()), reach_park=float(100 * C[f"reach_park_{B}"].median()),
                  displaced=float(100 * C[f"displaced_{B}"].median()))
out["planting"] = pd_
json.dump(out, open(f"{ROOT}/data/out/cityfx.json", "w"), indent=1, default=float)
C.to_csv(f"{ROOT}/data/out/cities_merged.csv", index=False)
print(json.dumps(out, indent=1, default=float)[:6000])
