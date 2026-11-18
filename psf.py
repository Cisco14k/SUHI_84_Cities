"""Sensor-footprint experiment. Can the thermal sensors' point-spread function (PSF), the resampling of Landsat Collection 2
surface temperature to 30 m and the block-averaging to 120 m alone produce the apparent cooling reach?
Synthetic 10 m surface temperature is built from ESA WorldCover 2021 classes for 12 city-regions, with a known lateral cooling
kernel of canopy (e-folding length L = 0 ... 240 m), then passed through each sensor model and analysed with the ring design of the
study (receptors: built-up >= 0.25, water < 0.2; nine rings to 1.8 km; own cover fractions and NDVI held fixed).
Output: data/out/psf_profiles.csv (ring effects, apparent and true) and psf_reach.csv."""
import sys, os, numpy as np, pandas as pd
from scipy import ndimage as ndi
from scipy.signal import fftconvolve
OUT = "/home/claude/n84/data/out"; os.makedirs(OUT, exist_ok=True)
EDGES = np.array([0, 120, 240, 360, 480, 600, 840, 1080, 1440, 1800])
MID = (EDGES[:-1] + EDGES[1:]) / 2
CLS = {"tree": [10, 95], "green": [20, 30, 40, 90, 100], "water": [80], "bare": [60, 70], "built": [50]}
BETA = {"tree": -6.0, "green": -3.0, "water": -8.0, "bare": 1.5, "built": 0.0}        # K, mid-morning contrasts to built-up land
SENS = {"ETM+ (60 m)": (60, 60, 0.0), "TIRS (100 m)": (100, 100, 0.0), "TM (120 m)": (120, 120, 0.0), "TIRS, with PSF halo": (100, 100, 0.08)}
rng = np.random.default_rng(42)


def fractions(wc, f):
    n = wc.shape[0] // f * f; wc = wc[:n, :n]
    return {k: np.isin(wc, v).reshape(n // f, f, n // f, f).mean((1, 3)) for k, v in CLS.items()}


def exp_kernel(L, px=10, rmax=1800):
    r = np.arange(-rmax, rmax + px, px); X, Y = np.meshgrid(r, r); d = np.hypot(X, Y)
    k = np.exp(-d / L); k /= k.sum(); return k


def sensor(T10, fwhm, native, halo):
    s = fwhm / 2.355 / 10.0
    B = ndi.gaussian_filter(T10, s, mode="reflect")
    if halo > 0:
        B = (1 - halo) * B + halo * ndi.gaussian_filter(T10, 600 / 2.355 / 10, mode="reflect")
    st = native // 10
    nat = B[st // 2::st, st // 2::st]                                   # samples at native pixel centres
    n30 = T10.shape[0] // 3
    yy = (np.arange(n30) * 3 + 1 - st // 2) / st; xx = yy
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    T30 = ndi.map_coordinates(nat, [Y, X], order=3, mode="nearest")   # cubic resampling to the 30 m product grid
    n = n30 // 4 * 4
    return T30[:n, :n].reshape(n // 4, 4, n // 4, 4).mean((1, 3))


EDGES2 = np.array([0, 180, 300, 420, 540, 660, 900, 1140, 1500, 1800]); MID2 = (EDGES2[:-1] + EDGES2[1:]) / 2


def ring_index(edges, n_rmax_cells=15):
    r = np.arange(-n_rmax_cells, n_rmax_cells + 1); X, Y = np.meshgrid(r, r); d = np.hypot(X, Y) * 120
    return [((d > edges[k]) & (d <= edges[k + 1])).astype(np.float32) for k in range(len(edges) - 1)]


RINGS = ring_index(EDGES); RINGS2 = ring_index(EDGES2)


def ring_shares(F, rings=None):
    rings = RINGS if rings is None else rings; out = {}
    for s in ("tree", "green", "water", "bare"):
        out[s] = [fftconvolve(F[s], m / m.sum(), mode="same") for m in rings]
    return out


def analyse(T120, F, NDVI, R, rec):
    X = [np.ones(len(rec[0]))]
    for s in ("tree", "green", "water", "bare"):
        X.append(F[s][rec])
    X.append(NDVI[rec])
    names = []
    for s in ("tree", "green", "water", "bare"):
        for k in range(len(EDGES) - 1):
            X.append(R[s][k][rec]); names.append((s, k))
    X = np.column_stack(X); y = T120[rec]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    co = b[6:]
    return {nm: -0.1 * co[i] for i, nm in enumerate(names)}                  # cooling per 10 pp replacing built-up


def reach(prof, delta=0.05, mids=None):
    m = np.r_[0.0, MID if mids is None else mids]; c = np.r_[np.inf, prof]
    for k in range(1, len(c)):
        if c[k] < delta:
            if not np.isfinite(c[k - 1]): return 0.0
            return m[k - 1] + (m[k] - m[k - 1]) * (c[k - 1] - delta) / (c[k - 1] - c[k])
    return np.nan


if __name__ == "__main__":
    rows = []; rows2 = []
    for fn in sorted(os.listdir("/home/claude/n84/data/wc")):
        idx = int(fn[:2]); wc = np.load(f"/home/claude/n84/data/wc/{fn}")["wc"]
        wc = wc[:2400, :2400]
        F10 = fractions(wc, 1); F120 = fractions(wc, 12)
        base = sum(BETA[k] * F10[k] for k in BETA)
        noise = ndi.gaussian_filter(rng.normal(0, 1, wc.shape), 15); noise *= 0.8 / noise.std()
        NDVI = (0.8 * F120["tree"] + 0.6 * F120["green"] + 0.15 * F120["built"] + 0.1 * F120["bare"] - 0.1 * F120["water"])
        NDVI = NDVI + rng.normal(0, 0.02, NDVI.shape)
        R = ring_shares(F120); R2 = ring_shares(F120, RINGS2)
        m = 15; n = F120["built"].shape[0]
        ok = (F120["built"] >= 0.25) & (F120["water"] < 0.2); ok[:m] = ok[-m:] = False; ok[:, :m] = ok[:, -m:] = False
        cand = np.argwhere(ok); sel = cand[rng.choice(len(cand), min(10000, len(cand)), replace=False)]; rec = (sel[:, 0], sel[:, 1])
        for L in (0, 30, 45, 60, 75, 90, 120, 240):
            lat = 0 if L == 0 else -6.0 * fftconvolve(F10["tree"], exp_kernel(L), mode="same")
            T10 = base + noise + lat
            # true (sensor-free) profile: the 10 m field block-averaged to 120 m
            n10 = 2400; Ttrue = T10.reshape(200, 12, 200, 12).mean((1, 3))
            prof_true = analyse(Ttrue, F120, NDVI, R, rec)
            prof_true2 = analyse(Ttrue, F120, NDVI, R2, rec)
            for sname, (fw, nat, halo) in SENS.items():
                T120 = sensor(T10, fw, nat, halo)[:200, :200]
                prof = analyse(T120, F120, NDVI, R, rec)
                if sname == "TIRS (100 m)":
                    prof2 = analyse(T120, F120, NDVI, R2, rec)
                    for k in range(len(EDGES2) - 1):
                        rows2.append(dict(idx=idx, L=L, ring=k, mid=MID2[k], cool=prof2[("tree", k)], cool_true=prof_true2[("tree", k)]))
                for k in range(len(EDGES) - 1):
                    rows.append(dict(idx=idx, L=L, sensor=sname, ring=k, mid=MID[k], cool=prof[("tree", k)], cool_true=prof_true[("tree", k)],
                                     cool_water=prof[("water", k)], cool_green=prof[("green", k)]))
            print(idx, L, flush=True)
        pd.DataFrame(rows).to_csv(f"{OUT}/psf_profiles.csv", index=False); pd.DataFrame(rows2).to_csv(f"{OUT}/psf_profiles_shifted.csv", index=False)
    D = pd.DataFrame(rows)
    rr = []
    for (L, s), g in D.groupby(["L", "sensor"]):
        p = g.groupby("ring").cool.mean().values; pt = g.groupby("ring").cool_true.mean().values
        rr.append(dict(L=L, sensor=s, reach_apparent=reach(p), reach_true=reach(pt), ring1=p[0], ring2=p[1], ring3=p[2], ring1_true=pt[0], ring2_true=pt[1], ring3_true=pt[2]))
    pd.DataFrame(rr).to_csv(f"{OUT}/psf_reach.csv", index=False); print(pd.DataFrame(rr).round(3).to_string())
