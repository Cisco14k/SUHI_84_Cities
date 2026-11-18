"""Pool the high-resolution (60 m, WorldCover exposure) re-analysis of the subset city-regions.
Outputs data/out/hr_pool.json with: R60 pooled profile, R120 pooled profile with cross-city Wald tests, kernel profile (summed SSE by L,
L-hat and city-bootstrap CI), fetch versus isotropic kernels and the upwind-downwind contrast against wind speed."""
import glob, json, numpy as np, pandas as pd
ROOT = "/home/claude/n84"; rng = np.random.default_rng(11)
J = [json.load(open(f)) for f in sorted(glob.glob(f"{ROOT}/data/out/hr/*.json"))]
J = [j for j in J if len(j["scenes"]) >= 3]
print("cities", len(J), [j["city"] for j in J], "scenes", sum(len(j["scenes"]) for j in J))


def dl_pool(m, se):
    m, se = np.asarray(m), np.asarray(se); w = 1 / se ** 2; mu0 = np.sum(w * m) / w.sum()
    Q = np.sum(w * (m - mu0) ** 2); k = len(m); tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (se ** 2 + tau2); mu = np.sum(ws * m) / ws.sum(); s = np.sqrt(1 / ws.sum())
    return float(mu), float(s), float(np.sqrt(tau2))


def city_mean(arr):
    arr = np.asarray(arr, float); n = arr.shape[0]
    return np.nanmean(arr, 0), np.nanstd(arr, 0, ddof=1) / np.sqrt(n)


out = {}
# R60 (canopy, 60 m rings)
E60 = np.array(J[0]["E60"]); mids60 = 0.5 * (E60[1:] + E60[:-1])
cm, cs = zip(*[city_mean([s["r60"] for s in j["scenes"]]) for j in J]); cm, cs = np.array(cm), np.array(cs)
cs = np.maximum(cs, 0.01)
P60 = [dl_pool(cm[:, k], cs[:, k]) for k in range(cm.shape[1])]
out["r60"] = dict(mid=mids60.tolist(), mu=[p[0] for p in P60], se=[p[1] for p in P60], tau=[p[2] for p in P60], city=cm.tolist())
d60 = mids60; mu = np.array([p[0] for p in P60])
k = np.argmax(mu < 0.025 / 2) if (mu < 0.025 / 2).any() else len(mu) - 1
# practical reach of the 60 m profile at delta = 0.05 per 10 pp of a 120 m ring is not comparable; we report the e-folding fit instead
# R120 (published-style rings; tree rings are the first 9 treatments)
names = [tuple(x) for x in J[0]["names120"]]; tr = [i for i, n in enumerate(names) if n[0] == "tree"]
C120 = np.array([np.nanmean([np.array(s["r120"])[tr] for s in j["scenes"]], 0) for j in J])
m120 = C120.mean(0); S = np.cov(C120.T) / len(C120)
def wald(A):
    A = np.atleast_2d(A); v = A @ m120; W = float(v @ np.linalg.pinv(A @ S @ A.T) @ v); q = A.shape[0]
    from scipy.stats import f as Fd
    n = len(C120); Fst = W * (n - q) / (q * (n - 1)); return dict(W=W, q=q, F=Fst, p=float(1 - Fd.cdf(Fst, q, n - q)), est=v.tolist())
A23 = np.zeros(9); A23[1], A23[2] = 1, -1
A12 = np.zeros(9); A12[0], A12[1] = 1, -1
Aout = np.zeros((6, 9)); [Aout.__setitem__((i, 3 + i), 1) for i in range(6)]
Adecl = np.zeros((2, 9)); Adecl[0, 1], Adecl[0, 2] = 1, -1; Adecl[1, 2], Adecl[1, 3] = 1, -1
out["r120"] = dict(mu=m120.tolist(), se=np.sqrt(np.diag(S)).tolist(), wald_r2_r3=wald(A23), wald_r1_r2=wald(A12), wald_outer0=wald(Aout), wald_decline=wald(Adecl))
# kernel profile likelihood
LG = J[0]["LGRID"]
SSE = np.array([[sum(s["kernel"][str(L)]["sse"] for s in j["scenes"]) for L in LG] for j in J])        # city x L
N = np.array([sum(s["n"] for s in j["scenes"]) for j in J])
def lhat(idx):
    tot = SSE[idx].sum(0); i = int(np.argmin(tot))
    if 0 < i < len(LG) - 1:                                                     # parabolic refinement on log L
        x = np.log(LG[i - 1:i + 2]); y = tot[i - 1:i + 2]; a, b, _ = np.polyfit(x, y, 2)
        if a > 0: return float(np.exp(-b / (2 * a)))
    return float(LG[i])
Lh = lhat(np.arange(len(J))); bs = [lhat(rng.integers(0, len(J), len(J))) for _ in range(1000)]
tot = SSE.sum(0); rel = (tot - tot.min()) / tot.min()
Lcity = [lhat(np.array([i])) for i in range(len(J))]
beta = [float(np.mean([s["kernel"][str(min(LG, key=lambda L: abs(L - Lh)))]["beta"] for s in j["scenes"]])) for j in J]
out["kernel"] = dict(L=LG, dsse_rel=rel.tolist(), Lhat=Lh, Lci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], Lcity=Lcity,
                     reach95=4.744 * Lh, reach95_ci=[4.744 * float(np.percentile(bs, 2.5)), 4.744 * float(np.percentile(bs, 97.5))],
                     beta_city=beta, cities=[j["city"] for j in J])
# fetch
rows = []
for j in J:
    for s in j["scenes"]:
        if "fetch" in s:
            for La, f in s["fetch"].items():
                rows.append(dict(city=j["city"], La=int(La), ws=s["wind_speed"], diff=f["diff"], se=f["diff_se"], up=f["up"], down=f["down"],
                                 dsse=f["sse_fetch"] - f["sse_iso"], sse_iso=f["sse_iso"]))
F = pd.DataFrame(rows)
if len(F):
    res = {}
    import statsmodels.formula.api as smf
    for La, d in F.groupby("La"):
        w = 1 / np.maximum(d.se, 1e-3) ** 2; mu = float(np.sum(w * d["diff"]) / w.sum())
        # cluster by city: city means then t
        cmn = d.groupby("city")["diff"].mean(); se_c = float(cmn.std(ddof=1) / np.sqrt(len(cmn)))
        m = smf.wls("diff ~ ws", d, weights=w).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.city)[0]})
        res[int(La)] = dict(n_scenes=len(d), n_cities=int(d.city.nunique()), diff_mean=float(cmn.mean()), diff_se=se_c, diff_w=mu,
                            slope_ws=float(m.params["ws"]), slope_se=float(m.bse["ws"]), slope_p=float(m.pvalues["ws"]),
                            frac_fetch_better=float((d.dsse < 0).mean()), dsse_rel=float(d.dsse.sum() / d.sse_iso.sum()),
                            up=float(d.groupby("city").up.mean().mean()), down=float(d.groupby("city").down.mean().mean()))
    out["fetch"] = res
    F.to_csv(f"{ROOT}/data/out/hr_fetch.csv", index=False)
json.dump(out, open(f"{ROOT}/data/out/hr_pool.json", "w"), indent=1, default=float)
print(json.dumps({k: (v if k != "r60" else {kk: vv for kk, vv in v.items() if kk != "city"}) for k, v in out.items()}, indent=1, default=float)[:4000])
