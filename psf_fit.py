"""Fit the physical e-folding length L of canopy cooling by matching the sensor-model profiles (psf.py) to the observed pooled
canopy profile of rings 2-6 (shape and amplitude), with uncertainty from resampling city-regions of the synthetic experiment and
the sampling error of the observed profile. Also the PSF-corrected reach. -> data/out/psf_fit.json"""
import json, numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/claude/n84/src")
from psf import reach, MID
D = pd.read_csv("/home/claude/n84/data/out/psf_profiles.csv")
obs = np.array([0.155, 0.179, 0.032, -0.003, 0.011, 0.001, -0.002, 0.014, -0.012])
lo = np.array([0.137, 0.164, 0.023, -0.013, 0.001, -0.008, -0.011, 0.001, -0.029]); hi = np.array([0.173, 0.195, 0.041, 0.006, 0.021, 0.009, 0.008, 0.028, 0.005])
se = (hi - lo) / 3.92
Ls = np.array(sorted(D.L.unique())); Lf = np.arange(30, 150.1, 1.0); K = slice(1, 6)
rng = np.random.default_rng(1); out = {}


def fit(P, PT, ob):
    # P[L, ring] apparent, PT true; returns best L, amplitude, apparent and true reach
    best = None
    for L in Lf:
        p = np.array([np.interp(L, Ls[1:], P[1:, k]) for k in range(P.shape[1])])
        pt = np.array([np.interp(L, Ls[1:], PT[1:, k]) for k in range(P.shape[1])])
        w = 1 / se[K] ** 2; a = (w * p[K] * ob[K]).sum() / (w * p[K] ** 2).sum()
        sse = (w * (a * p[K] - ob[K]) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, L, a, reach(a * p), reach(a * pt), a * p, a * pt)
    return best


for s, g in D.groupby("sensor"):
    idxs = g.idx.unique()
    def mats(sub):
        P = sub.groupby(["L", "ring"]).cool.mean().unstack().loc[Ls].values; PT = sub.groupby(["L", "ring"]).cool_true.mean().unstack().loc[Ls].values
        return P, PT
    P, PT = mats(g); b = fit(P, PT, obs)
    boots = []
    for r in range(300):
        ci = rng.choice(idxs, len(idxs)); sub = pd.concat([g[g.idx == i] for i in ci])
        Pb, PTb = mats(sub); ob = obs + rng.normal(0, se)
        bb = fit(Pb, PTb, ob); boots.append((bb[1], bb[3], bb[4]))
    B = np.array(boots)
    out[s] = dict(L=b[1], L_ci=list(np.nanpercentile(B[:, 0], [2.5, 97.5])), amplitude=b[2], reach_apparent=b[3], reach_true=b[4],
                  reach_app_ci=list(np.nanpercentile(B[:, 1], [2.5, 97.5])), reach_true_ci=list(np.nanpercentile(B[:, 2], [2.5, 97.5])),
                  wsse=b[0], prof_app=list(b[5]), prof_true=list(b[6]),
                  zero_L_ring1=float(g[g.L == 0].groupby("ring").cool.mean()[0]), zero_L_ring2=float(g[g.L == 0].groupby("ring").cool.mean()[1]),
                  zero_L_reach=reach(g[g.L == 0].groupby("ring").cool.mean().values))
    print(s, {k: (np.round(v, 3) if not isinstance(v, list) else np.round(v[:4], 3)) for k, v in out[s].items()}, flush=True)
json.dump(out, open("/home/claude/n84/data/out/psf_fit.json", "w"), indent=1, default=float)
