"""Nature Sustainability revision: dependence of the threshold reach on ring width and on the confounder set (21 city-regions, 60 m).

Same inputs and learners as hr.py (Landsat 8-9 surface temperature at 60 m, ESA WorldCover 2021 exposures, DML with 2.04 km spatial-block
cross-fitting). For each scene:
  widths  canopy rings of width w = 60, 90, 120 or 180 m out to 600-720 m, then the published outer rings (840-1800 m);
          open green space, water and bare ground in the published 120 m rings; main confounder set (V0);
  V1      as the 120 m design and the canopy kernel, without the receptor's own NDVI (the own-cell control that may absorb adjacent effects);
  V2      as V1 but keeping NDVI and dropping the far-neighbourhood shares (1.8-3 km and 3-6 km);
  kernel  isotropic exponential canopy kernel exposure for L in LGRID under V0, V1 and V2 (residual sum of squares for profile likelihood).
Output: data/out/hrw/<idx>.json ; run: python hr_width.py <idx,...>"""
import os, sys, json, warnings, io
import numpy as np, pandas as pd
from scipy import ndimage as ndi
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hr as H
ROOT = H.ROOT; OUT = f"{ROOT}/data/out/hrw"; os.makedirs(OUT, exist_ok=True)
WIDTHS = [60, 90, 120, 180]
OUTER = [840, 1080, 1440, 1800]
LGRID = [20, 30, 40, 50, 60, 80, 110, 160]
PX = 60; NREC = int(os.environ.get("NREC", 6000)); MAXSC = int(os.environ.get("MAXSC", 10))
rng = np.random.default_rng(11)


def edges_for(w):
    e = list(np.arange(0, 600 + w, w))
    while e[-1] < 600: e.append(e[-1] + w)
    return np.array(e + [o for o in OUTER if o > e[-1]], dtype=float)


def run(idx):
    fn = f"{OUT}/{idx:02d}.json"
    if os.path.exists(fn): return "skip"
    W = np.load(f"{ROOT}/data/wc14/{idx:02d}.npz"); L8 = np.load(f"{ROOT}/data/ls/{idx:02d}.npz", allow_pickle=True)
    meta = pd.read_json(io.StringIO(str(L8["meta"])))
    wc = W["wc"][:2790, :2790]; F = H.frac(wc, 6); n = F["built"].shape[0]
    lst30 = L8["lst"].astype(np.float32)[:, :930, :930]; clr = L8["clear"][:, :930, :930]; ndvi30 = L8["ndvi"].astype(np.float32)[:, :930, :930]
    agg = lambda a: a.reshape(a.shape[0], n, 2, n, 2).mean((2, 4))
    lst = agg(np.where(clr, lst30, np.nan)); ndvi = np.nanmedian(agg(np.where(clr, ndvi30, np.nan)), 0)
    dem = L8["dem"][:930, :930].reshape(n, 2, n, 2).mean((1, 3))
    gy, gx = np.gradient(dem, PX); slope = np.hypot(gx, gy); asp = np.arctan2(-gx, gy)
    dwat = ndi.distance_transform_edt(F["water"] < 0.5) * PX
    far = {}; d, _, _ = H.dist_grid(6000, PX)
    for s in ("tree", "water", "built"):
        for a, b in ((1800, 3000), (3000, 6000)):
            m = ((d > a) & (d <= b)).astype(np.float32); far[(s, a)] = H.conv(F[s], m / m.sum())
    m6 = 6000 // PX
    ok = (F["built"] >= 0.25) & (F["water"] < 0.2); ok[:m6] = ok[-m6:] = False; ok[:, :m6] = ok[:, -m6:] = False
    cand = np.argwhere(ok)
    if len(cand) < 1500: return f"too few receptors {len(cand)}"
    sel = cand[rng.choice(len(cand), min(NREC, len(cand)), replace=False)]; iy, ix = sel[:, 0], sel[:, 1]
    own = [F[k][iy, ix] for k in H.CLS]
    terr = [dem[iy, ix], slope[iy, ix], np.sin(asp[iy, ix]), np.cos(asp[iy, ix]), np.log1p(dwat[iy, ix]), ix * PX, iy * PX]
    farc = [far[k][iy, ix] for k in far]
    XV = {"V0": np.nan_to_num(np.column_stack(own + [ndvi[iy, ix]] + terr + farc)),
          "V1": np.nan_to_num(np.column_stack(own + terr + farc)),
          "V2": np.nan_to_num(np.column_stack(own + [ndvi[iy, ix]] + terr))}
    fold, blk = H.blocks(ix, iy, 2040 // PX)
    R120 = H.rings(F, H.E120, PX, H.SRC)
    other = [k for k in R120 if k[0] != "tree"]
    Dother = np.column_stack([R120[k][iy, ix] for k in other])
    RDo = {v: H.crossfit_D(XV[v], Dother, fold) for v in XV}                       # residualized other sources per confounder set
    Dw = {}
    for w in WIDTHS:
        e = edges_for(w); R = H.rings(F, e, PX, ["tree"])
        Dw[w] = (e.tolist(), H.crossfit_D(XV["V0"], np.column_stack([R[("tree", k)][iy, ix] for k in range(len(e) - 1)]), fold))
    e120 = H.E120; D120t = np.column_stack([R120[("tree", k)][iy, ix] for k in range(len(e120) - 1)])
    R120v = {v: H.crossfit_D(XV[v], D120t, fold) for v in ("V1", "V2")}
    KE = {L: H.kernel_exposure(F["tree"], L, PX)[iy, ix][:, None] for L in LGRID}
    RK = {v: {L: H.crossfit_D(XV[v], KE[L], fold) for L in LGRID} for v in XV}
    res = []
    for si in range(min(lst.shape[0], MAXSC)):
        y_all = lst[si]
        rural = (F["built"] < 0.05) & (F["water"] < 0.1) & np.isfinite(y_all)
        if rural.sum() < 200: continue
        gam = np.polyfit(dem[rural], y_all[rural], 1)[0] if np.ptp(dem[rural]) > 50 else 0.0
        gam = float(np.clip(gam, -0.012, 0.0))
        ya = y_all - gam * (dem - np.nanmedian(dem[rural])); ya = ya - np.nanmedian(ya[rural])
        y = ya[iy, ix]; v = np.isfinite(y)
        if v.mean() < 0.5: continue
        rec = dict(scene=meta.id.iloc[si], n=int(v.sum()), widths={}, v120={}, kernel={})
        ry = {k: H.crossfit_y(XV[k][v], y[v], fold[v]) for k in XV}
        for w in WIDTHS:
            e, Rd = Dw[w]; Dt = np.column_stack([Rd[v], RDo["V0"][v]]); nt = Rd.shape[1]
            b, V, _ = H.ols_cluster(ry["V0"], Dt, blk[v])
            rec["widths"][w] = dict(edges=e, c=(-0.1 * b[:nt]).tolist(), se=(0.1 * np.sqrt(np.diag(V)[:nt])).tolist())
        for k in ("V1", "V2"):
            Dt = np.column_stack([R120v[k][v], RDo[k][v]]); b, V, _ = H.ols_cluster(ry[k], Dt, blk[v])
            rec["v120"][k] = dict(c=(-0.1 * b[:9]).tolist(), se=(0.1 * np.sqrt(np.diag(V)[:9])).tolist())
        for k in XV:
            ks = {}
            for L in LGRID:
                bk, Vk, e_ = H.ols_cluster(ry[k], np.column_stack([RK[k][L][v], RDo[k][v]]), blk[v])
                ks[L] = dict(sse=float((e_ ** 2).sum()), beta=float(-0.1 * bk[0]), se=float(0.1 * np.sqrt(Vk[0, 0])))
            rec["kernel"][k] = ks
        res.append(rec); print(idx, si, [round(x, 3) for x in rec["widths"][60]["c"][:4]], flush=True)
    json.dump(dict(idx=idx, city=H.C.city[idx], LGRID=LGRID, scenes=res), open(fn, "w"), default=float)
    return f"ok {len(res)}"


if __name__ == "__main__":
    for i in [int(x) for x in sys.argv[1].split(",")]:
        try:
            print(i, run(i), flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print(i, "ERR", repr(e)[:300], flush=True)
    os._exit(0)
