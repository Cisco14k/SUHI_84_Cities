"""High-resolution re-analysis of canopy cooling for a 21-city subset (three city-regions per world region), Landsat 8, 2016-2020.

Independent of the published pipeline in three respects: exposures come from ESA WorldCover 2021 at 10 m (not from Landsat-derived
hindcast fractions), the analysis grid is 60 m (half the published 120 m) with 60 m canopy rings, and the distance dependence is also
estimated as a continuous kernel whose e-folding length is chosen by cross-fitted profile likelihood (threshold-free).
Three designs per scene, all with double machine learning (DML; spatial-block cross-fitting):
  R120  the published ring design at 120 m (nine rings to 1.8 km per source), for comparison and a joint test of the ring decline;
  R60   60 m canopy rings to 600 m, then the published outer rings; other sources in 120 m rings;
  K     an isotropic exponential kernel of canopy, exposure E_L = sum_y w_L(d) c(y) / sum_y w_L(d), w_L = exp(-d/L), d >= one cell;
  F     a Lagrangian fetch kernel along the upwind direction of the overpass-hour 100 m ERA5 wind (2016-2020), with crosswind spread,
        exposure E_F,La = sum_y exp(-s/La) exp(-n^2/(2 (0.3 s)^2)) c(y) / norm, s = upwind distance, n = crosswind offset;
        compared with the isotropic kernel of the same length and with an upwind-minus-downwind contrast.
Output: data/out/hr/<idx>.json (scene estimates) ; run: python hr.py <idx,...>"""
import os, sys, json, warnings, glob
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from scipy.signal import fftconvolve
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
warnings.filterwarnings("ignore")
ROOT = "/home/claude/n84"; OUT = f"{ROOT}/data/out/hr"; os.makedirs(OUT, exist_ok=True)
C = pd.read_csv(f"{ROOT}/data/cities.csv")
CLS = {"tree": [10, 95], "green": [20, 30, 40, 90, 100], "water": [80], "bare": [60, 70], "built": [50]}
SRC = ["tree", "green", "water", "bare"]
E120 = np.array([0, 120, 240, 360, 480, 600, 840, 1080, 1440, 1800])
E60 = np.array([0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 840, 1080, 1440, 1800])
LGRID = [20, 30, 40, 50, 60, 80, 110, 160, 240, 320]
LAGRID = [60, 120, 240]
PX = 60; NREC = int(os.environ.get("NREC", 8000)); rng = np.random.default_rng(7)


def frac(wc, f):
    n = wc.shape[0] // f * f; wc = wc[:n, :n]
    return {k: np.isin(wc, v).reshape(n // f, f, n // f, f).mean((1, 3)).astype(np.float32) for k, v in CLS.items()}


def dist_grid(rmax, px):
    r = np.arange(-int(rmax // px), int(rmax // px) + 1); X, Y = np.meshgrid(r, r)
    return np.hypot(X, Y) * px, X * px, -Y * px                       # distance, east, north offsets of source cells


def conv(a, k):
    return fftconvolve(a, k, mode="same")


def rings(F, edges, px, srcs):
    d, _, _ = dist_grid(edges[-1], px); out = {}
    for s in srcs:
        for k in range(len(edges) - 1):
            m = ((d > edges[k]) & (d <= edges[k + 1])).astype(np.float32)
            if m.sum() == 0: continue
            out[(s, k)] = conv(F[s], m / m.sum())
    return out


def kernel_exposure(Fs, L, px, rmax=1800):
    d, _, _ = dist_grid(rmax, px); w = np.exp(-d / L) * (d >= px * 0.99) * (d <= rmax); w /= w.sum()
    return conv(Fs, w)


def fetch_exposure(Fs, La, wdir_from, px, rmax=1800, sign=1):
    """canopy along the upwind (sign=1) or downwind (sign=-1) streamline. wdir_from: meteorological direction (deg) wind blows from."""
    d, E, N = dist_grid(rmax, px)
    th = np.radians(wdir_from); ux, uy = np.sin(th), np.cos(th)                # unit vector pointing to where the wind comes from
    s = sign * (E * ux + N * uy); n = -E * uy + N * ux                          # along-wind distance (upwind positive), crosswind offset
    w = np.where((s > 0) & (d <= rmax), np.exp(-s / La) * np.exp(-n ** 2 / (2 * (0.3 * np.maximum(s, px)) ** 2)), 0.0)
    w[d < px * 0.99] = 0; w /= w.sum()
    return conv(Fs, w[::-1, ::-1])                                              # correlation -> flip kernel for convolution


def blocks(ix, iy, size_cells):
    b = (ix // size_cells) * 1000 + (iy // size_cells)
    u = np.unique(b); perm = rng.permutation(len(u)); fold = {bb: perm[i] % 5 for i, bb in enumerate(u)}
    return np.array([fold[bb] for bb in b]), b


def crossfit_y(X, y, fold):
    r = np.zeros_like(y)
    for f in range(5):
        tr, te = fold != f, fold == f
        m = lgb.LGBMRegressor(n_estimators=150, num_leaves=31, learning_rate=0.06, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=2)
        m.fit(X[tr], y[tr]); r[te] = y[te] - m.predict(X[te])
    return r


def crossfit_D(X, D, fold):
    r = np.zeros_like(D)
    for f in range(5):
        tr, te = fold != f, fold == f
        m = ExtraTreesRegressor(n_estimators=120, min_samples_leaf=5, max_features=0.5, n_jobs=2, random_state=f)
        m.fit(X[tr], D[tr]); r[te] = D[te] - m.predict(X[te]).reshape(D[te].shape)
    return r


def ols_cluster(yt, Dt, cl):
    XtX = Dt.T @ Dt; b = np.linalg.solve(XtX, Dt.T @ yt); e = yt - Dt @ b
    Bi = np.linalg.inv(XtX); meat = np.zeros_like(XtX)
    for g in np.unique(cl):
        s = Dt[cl == g].T @ e[cl == g]; meat += np.outer(s, s)
    G = len(np.unique(cl)); V = Bi @ meat @ Bi * G / (G - 1)
    return b, V, e


def run(idx):
    fn = f"{OUT}/{idx:02d}.json"
    if os.path.exists(fn): return "skip"
    W = np.load(f"{ROOT}/data/wc14/{idx:02d}.npz"); L8 = np.load(f"{ROOT}/data/ls/{idx:02d}.npz", allow_pickle=True)
    import io; meta = pd.read_json(io.StringIO(str(L8["meta"])))
    wc = W["wc"][:2790, :2790]; F = frac(wc, 6)                                   # 60 m fractions (465 x 465)
    n = F["built"].shape[0]
    lst30 = L8["lst"].astype(np.float32)[:, :930, :930]; clr = L8["clear"][:, :930, :930]; ndvi30 = L8["ndvi"].astype(np.float32)[:, :930, :930]
    agg = lambda a: a.reshape(a.shape[0], n, 2, n, 2).mean((2, 4))
    lst = agg(np.where(clr, lst30, np.nan)); ndvi = np.nanmedian(agg(np.where(clr, ndvi30, np.nan)), 0)
    dem = L8["dem"][:930, :930].reshape(n, 2, n, 2).mean((1, 3))
    gy, gx = np.gradient(dem, PX); slope = np.hypot(gx, gy); asp = np.arctan2(-gx, gy)
    dwat = ndi.distance_transform_edt(F["water"] < 0.5) * PX
    far = {}
    d, _, _ = dist_grid(6000, PX)
    for s in ("tree", "water", "built"):
        for a, b in ((1800, 3000), (3000, 6000)):
            m = ((d > a) & (d <= b)).astype(np.float32); far[(s, a)] = conv(F[s], m / m.sum())
    m6 = 6000 // PX
    ok = (F["built"] >= 0.25) & (F["water"] < 0.2); ok[:m6] = ok[-m6:] = False; ok[:, :m6] = ok[:, -m6:] = False
    cand = np.argwhere(ok)
    if len(cand) < 1500: return f"too few receptors {len(cand)}"
    sel = cand[rng.choice(len(cand), min(NREC, len(cand)), replace=False)]; iy, ix = sel[:, 0], sel[:, 1]
    X = np.column_stack([F[k][iy, ix] for k in CLS] + [ndvi[iy, ix], dem[iy, ix], slope[iy, ix], np.sin(asp[iy, ix]), np.cos(asp[iy, ix]),
                         np.log1p(dwat[iy, ix]), ix * PX, iy * PX] + [far[k][iy, ix] for k in far])
    X = np.nan_to_num(X)
    fold, blk = blocks(ix, iy, 2040 // PX)
    # treatments (fixed across scenes): R120, R60, kernels
    R120 = rings(F, E120, PX, SRC); R60t = rings(F, E60, PX, ["tree"])
    other = {k: v for k, v in R120.items() if k[0] != "tree"}
    names120 = list(R120.keys()); D120 = np.column_stack([R120[k][iy, ix] for k in names120])
    names60 = [("tree60", k) for k in range(len(E60) - 1)] + list(other.keys())
    D60 = np.column_stack([R60t[("tree", k)][iy, ix] for k in range(len(E60) - 1)] + [other[k][iy, ix] for k in other])
    Do = np.column_stack([other[k][iy, ix] for k in other])
    Rd120 = crossfit_D(X, D120, fold); Rd60 = crossfit_D(X, D60, fold); Rdo = Rd120[:, [names120.index(k) for k in other]]
    KE = {L: kernel_exposure(F["tree"], L, PX)[iy, ix] for L in LGRID}
    RK = {L: crossfit_D(X, KE[L][:, None], fold) for L in LGRID}
    ws = None
    tf = sorted(glob.glob(f"{ROOT}/data/traj/{idx:02d}_*.npz"))
    if tf:
        ov = []
        for f in tf:
            z = np.load(f); ov.append(pd.DataFrame(dict(time=pd.to_datetime(z["ov_time"]), u=z["ov_u"], v=z["ov_v"])))
        ws = pd.concat(ov).set_index("time").sort_index()
    res = []
    for si in range(min(lst.shape[0], int(os.environ.get("MAXSC", 10)))):
        y_all = lst[si]
        rural = (F["built"] < 0.05) & (F["water"] < 0.1) & np.isfinite(y_all)
        if rural.sum() < 200: continue
        gam = np.polyfit(dem[rural], y_all[rural], 1)[0] if np.ptp(dem[rural]) > 50 else 0.0
        gam = float(np.clip(gam, -0.012, 0.0))
        ya = y_all - gam * (dem - np.nanmedian(dem[rural])); ya = ya - np.nanmedian(ya[rural])
        y = ya[iy, ix]; v = np.isfinite(y)
        if v.mean() < 0.5: continue
        Xv, yv, fv, bv = X[v], y[v], fold[v], blk[v]
        ry = crossfit_y(Xv, yv, fv)
        rec = dict(scene=meta.id.iloc[si], time=str(meta.time.iloc[si]), sun_az=meta.sun_az.iloc[si], sun_el=meta.sun_el.iloc[si], n=int(v.sum()))
        b, V, _ = ols_cluster(ry, Rd120[v], bv); rec["r120"] = (-0.1 * b).tolist(); rec["r120_V"] = (0.01 * V[:9, :9]).tolist()
        b, V, _ = ols_cluster(ry, Rd60[v], bv); rec["r60"] = (-0.1 * b[:len(E60) - 1]).tolist(); rec["r60_se"] = (0.1 * np.sqrt(np.diag(V)[:len(E60) - 1])).tolist()
        sse = {}
        for L in LGRID:
            Dk = np.column_stack([RK[L][v], Rdo[v]]); bk, Vk, e = ols_cluster(ry, Dk, bv)
            sse[L] = dict(sse=float((e ** 2).sum()), beta=float(-0.1 * bk[0]), se=float(0.1 * np.sqrt(Vk[0, 0])))
        rec["kernel"] = sse; rec["sse0"] = float(((ry - Rdo[v] @ np.linalg.lstsq(Rdo[v], ry, rcond=None)[0]) ** 2).sum())
        if ws is not None:
            t = pd.Timestamp(meta.time.iloc[si]).tz_localize(None)
            j = ws.index.get_indexer([t], method="nearest")[0]
            if j >= 0 and abs((ws.index[j] - t).total_seconds()) < 3 * 3600 * 24:
                u, vv = float(ws.u.iloc[j]), float(ws.v.iloc[j]); spd = float(np.hypot(u, vv)); wdir = float((np.degrees(np.arctan2(-u, -vv)) + 360) % 360)
                rec.update(wind_speed=spd, wind_dir=wdir, wind_dt_h=float((ws.index[j] - t).total_seconds() / 3600))
                if abs(rec["wind_dt_h"]) <= 1.5:
                    fe = {}
                    for La in LAGRID:
                        up = fetch_exposure(F["tree"], La, wdir, PX, sign=1)[iy, ix][v]; dn = fetch_exposure(F["tree"], La, wdir, PX, sign=-1)[iy, ix][v]
                        Ru = crossfit_D(Xv, np.column_stack([up, dn]), fv)
                        bf, Vf, ef = ols_cluster(ry, np.column_stack([Ru, Rdo[v]]), bv)
                        diff = -0.1 * (bf[0] - bf[1]); sed = 0.1 * np.sqrt(Vf[0, 0] + Vf[1, 1] - 2 * Vf[0, 1])
                        Ri = crossfit_D(Xv, kernel_exposure(F["tree"], La, PX)[iy, ix][v][:, None], fv)
                        bi, Vi, ei = ols_cluster(ry, np.column_stack([Ri, Rdo[v]]), bv)
                        Rf = crossfit_D(Xv, up[:, None], fv); bu, Vu, eu = ols_cluster(ry, np.column_stack([Rf, Rdo[v]]), bv)
                        fe[La] = dict(up=float(-0.1 * bf[0]), down=float(-0.1 * bf[1]), diff=float(diff), diff_se=float(sed),
                                      sse_fetch=float((eu ** 2).sum()), sse_iso=float((ei ** 2).sum()), beta_fetch=float(-0.1 * bu[0]), beta_iso=float(-0.1 * bi[0]))
                    rec["fetch"] = fe
        res.append(rec); print(idx, si, rec["r120"][:3], flush=True)
    json.dump(dict(idx=idx, city=C.city[idx], names120=[list(k) for k in names120], E60=E60.tolist(), LGRID=LGRID, LAGRID=LAGRID, scenes=res),
              open(fn, "w"), default=float)
    return f"ok {len(res)}"


if __name__ == "__main__":
    for i in [int(x) for x in sys.argv[1].split(",")]:
        try:
            print(i, run(i), flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print(i, "ERR", repr(e)[:300], flush=True)
    os._exit(0)
