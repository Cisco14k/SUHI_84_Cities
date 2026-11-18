"""Sensitivity of the forward-model kernel length to the assumed cover contrasts and noise (TIRS PSF).
Scenarios change one assumption at a time; L is fitted to the observed rings 2-6 as in psf_fit.py (100 city bootstraps).
Usage: python psf_sens.py <scenario,...>  -> data/out/psf_sens_<scenario>.csv ; python psf_sens.py fit -> data/out/psf_sens.json"""
import sys, os, json, numpy as np, pandas as pd
from scipy import ndimage as ndi
from scipy.signal import fftconvolve
sys.path.insert(0, "/home/claude/n84/src")
import psf
from psf import fractions, exp_kernel, sensor, ring_shares, analyse, EDGES
OUT = "/home/claude/n84/data/out"
SC = {"base": dict(), "canopy_-3K": dict(tree=-3.0), "canopy_-9K": dict(tree=-9.0), "green_-1.5_water_-12": dict(green=-1.5, water=-12.0),
      "noise_0.4K": dict(noise=0.4), "noise_1.6K": dict(noise=1.6), "noise_corr_50m": dict(corr=5), "noise_corr_400m": dict(corr=40)}
LS = (0, 30, 45, 60, 75, 90, 120)

if sys.argv[1] != "fit":
    for sc in sys.argv[1].split(","):
        p = SC[sc]; beta = dict(psf.BETA); beta.update({k: v for k, v in p.items() if k in beta})
        rng = np.random.default_rng(42); rows = []
        for fn in sorted(os.listdir("/home/claude/n84/data/wc")):
            idx = int(fn[:2]); wc = np.load(f"/home/claude/n84/data/wc/{fn}")["wc"][:2400, :2400]
            F10 = fractions(wc, 1); F120 = fractions(wc, 12)
            base = sum(beta[k] * F10[k] for k in beta)
            noise = ndi.gaussian_filter(rng.normal(0, 1, wc.shape), p.get("corr", 15)); noise *= p.get("noise", 0.8) / noise.std()
            NDVI = 0.8 * F120["tree"] + 0.6 * F120["green"] + 0.15 * F120["built"] + 0.1 * F120["bare"] - 0.1 * F120["water"] + rng.normal(0, 0.02, F120["tree"].shape)
            R = ring_shares(F120); m = 15
            ok = (F120["built"] >= 0.25) & (F120["water"] < 0.2); ok[:m] = ok[-m:] = False; ok[:, :m] = ok[:, -m:] = False
            cand = np.argwhere(ok); sel = cand[rng.choice(len(cand), min(10000, len(cand)), replace=False)]; rec = (sel[:, 0], sel[:, 1])
            for L in LS:
                lat = 0 if L == 0 else beta["tree"] * fftconvolve(F10["tree"], exp_kernel(L), mode="same")
                T120 = sensor(base + noise + lat, 100, 100, 0.0)[:200, :200]
                prof = analyse(T120, F120, NDVI, R, rec)
                for k in range(len(EDGES) - 1):
                    rows.append(dict(idx=idx, L=L, ring=k, cool=prof[("tree", k)]))
            print(sc, idx, flush=True)
        pd.DataFrame(rows).to_csv(f"{OUT}/psf_sens_{sc}.csv", index=False)
else:
    obs = np.array([0.155, 0.179, 0.032, -0.003, 0.011, 0.001, -0.002, 0.014, -0.012])
    lo = np.array([0.137, 0.164, 0.023, -0.013, 0.001, -0.008, -0.011, 0.001, -0.029]); hi = np.array([0.173, 0.195, 0.041, 0.006, 0.021, 0.009, 0.008, 0.028, 0.005])
    se = (hi - lo) / 3.92; K = slice(1, 6); Lf = np.arange(30, 120.1, 1.0); rng = np.random.default_rng(1); out = {}
    def fit(P, ob):
        best = None
        for L in Lf:
            pr = np.array([np.interp(L, LS[1:], P[1:, k]) for k in range(P.shape[1])])
            w = 1 / se[K] ** 2; a = (w * pr[K] * ob[K]).sum() / (w * pr[K] ** 2).sum(); sse = (w * (a * pr[K] - ob[K]) ** 2).sum()
            if best is None or sse < best[0]: best = (sse, L, a)
        return best
    for sc in SC:
        fn = f"{OUT}/psf_sens_{sc}.csv"
        if not os.path.exists(fn): continue
        D = pd.read_csv(fn); idxs = D.idx.unique()
        mat = lambda sub: sub.groupby(["L", "ring"]).cool.mean().unstack().loc[list(LS)].values
        b = fit(mat(D), obs); B = []
        for r in range(100):
            ci = rng.choice(idxs, len(idxs)); B.append(fit(mat(pd.concat([D[D.idx == i] for i in ci])), obs + rng.normal(0, se))[1])
        z = D[D.L == 0].groupby("ring").cool.mean()
        out[sc] = dict(L=b[1], L_ci=list(np.percentile(B, [2.5, 97.5])), ring1_L0=float(z[0]), ring2_L0=float(z[1]))
        print(sc, out[sc])
    json.dump(out, open(f"{OUT}/psf_sens.json", "w"), indent=1)
