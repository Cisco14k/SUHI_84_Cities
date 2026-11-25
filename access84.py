"""Nature Sustainability revision: access to and benefit from canopy cooling around residents, aligned with the estimated treatment.

For each of the 84 city-regions (circle of radius R around the centre, as in the main analysis):
  * ESA WorldCover 2021 (Planetary Computer) read at 30 m and averaged to cover fractions on a 120 m local azimuthal-equidistant grid;
  * GlobPOP 2020 residents (30 arc-second counts) aggregated to 960 m blocks and allocated to the urban cells of each block
    (built-up fraction >= 0.25, water < 0.2) in proportion to built-up fraction (dasymetric), as in the main analysis;
  * access as published: share of residents within 240 m of a canopy patch (canopy fraction >= 0.5, >= 2 cells, i.e. >= 2.9 ha at 120 m);
  * benefit aligned with the treatment: modelled surface cooling delivered to each urban cell by canopy in the surrounding rings,
    dT_i = 10 * sum_k c_k * f_tree,k(i), with the pooled ring effects c_k (degC per 10 pp; Table 1) and the ring canopy fractions f_tree,k;
    the receptor's own canopy is excluded, as in the estimand;
  * planting design: a fixed canopy budget (0.5% of urban area, replacing built-up cover) either dispersed as +30 pp canopy in the most
    populous low-canopy urban cells or consolidated into one round park placed where it cools the most residents; person-degC per hectare.
Output: data/out/access84.csv, data/out/access84_cells/<idx>.npz"""
import os, sys, json, io, numpy as np, pandas as pd, rasterio, pystac_client, planetary_computer as pc
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import from_origin
from rasterio.windows import from_bounds
from scipy.signal import fftconvolve
from scipy import ndimage as ndi
from pyproj import CRS
ROOT = "/home/claude/n84ns"; OUT = f"{ROOT}/data/out"; CD = f"{OUT}/access84_cells"; os.makedirs(CD, exist_ok=True)
POP = f"{ROOT}/data/pop/GlobPOP_Count_30arc_2020_I32.tiff"
C = pd.read_csv(f"{ROOT}/data/cities_inc.csv")
CLS = {"tree": [10, 95], "green": [20, 30, 40, 90, 100], "water": [80], "bare": [60, 70], "built": [50]}
PX = 120
EDGES = np.array([0, 120, 240, 360, 480, 600, 840])
CK = np.array([0.155, 0.179, 0.032, -0.003, 0.011, 0.001])            # pooled canopy ring effects, degC per 10 pp (Table 1)


def ring_kernels(px=PX):
    r = np.arange(-int(EDGES[-1] // px), int(EDGES[-1] // px) + 1); X, Y = np.meshgrid(r, r); d = np.hypot(X, Y) * px
    ks = []
    for k in range(len(EDGES) - 1):
        m = ((d > EDGES[k]) & (d <= EDGES[k + 1])).astype(np.float64)
        ks.append(m / m.sum() if m.sum() else m)
    return ks, d


KS, DK = ring_kernels()
RESP = sum(10 * c * k for c, k in zip(CK, KS))                         # response of a receptor to unit canopy fraction at offset (degC)


def conv(a, k):
    return fftconvolve(a, k, mode="same")


def city_grid(c):
    from pyproj import Transformer
    crs = CRS.from_proj4(f"+proj=aeqd +lat_0={c.lat} +lon_0={c.lon} +datum=WGS84 +units=m")
    half = (c.R_km * 1000 // 960 + 1) * 960 + 960                         # include a margin for the outer rings
    n = int(2 * half / PX)
    return crs, from_origin(-half, half, PX, PX), n, half


def worldcover_fracs(cat, c, crs, tr, n, half):
    bb = transform_bounds(crs, 4326, -half, -half, half, half, densify_pts=41)
    items = list(cat.search(collections=["esa-worldcover"], bbox=bb, query={"esa_worldcover:product_version": {"eq": "2.0.0"}}).items())
    acc = {k: np.zeros((n, n), np.float32) for k in CLS}; wsum = np.zeros((n, n), np.float32)
    for it in items:
        with rasterio.open(it.assets["map"].href) as src:
            sb = src.bounds
            ib = (max(bb[0], sb.left), max(bb[1], sb.bottom), min(bb[2], sb.right), min(bb[3], sb.top))
            if ib[0] >= ib[2] or ib[1] >= ib[3]: continue
            win = from_bounds(*ib, transform=src.transform).round_offsets().round_lengths()
            h, w = int(win.height), int(win.width); f = 3                                   # 30 m decimated read
            cls = src.read(1, window=win, out_shape=(max(h // f, 1), max(w // f, 1)), resampling=Resampling.nearest)
            wt = src.window_transform(win) * rasterio.Affine.scale(w / cls.shape[1], h / cls.shape[0])
            valid = (cls > 0).astype(np.float32)
            tmpv = np.zeros((n, n), np.float32)
            reproject(valid, tmpv, src_transform=wt, src_crs=src.crs, dst_transform=tr, dst_crs=crs, resampling=Resampling.average)
            for k, v in CLS.items():
                tmp = np.zeros((n, n), np.float32)
                reproject(np.isin(cls, v).astype(np.float32), tmp, src_transform=wt, src_crs=src.crs, dst_transform=tr, dst_crs=crs, resampling=Resampling.average)
                acc[k] += tmp * (tmpv > 0)
            wsum += tmpv
    tot = sum(acc.values()); tot[tot == 0] = 1
    return {k: acc[k] / tot for k in CLS}, len(items)


def population(c, crs, n, half, built, urban):
    nb = n // 8; trb = from_origin(-half, half, 960, 960); blk = np.zeros((nb, nb), np.float32)
    with rasterio.open(POP) as src:
        bb = transform_bounds(crs, 4326, -half, -half, half, half, densify_pts=41)
        win = from_bounds(bb[0] - 0.02, bb[1] - 0.02, bb[2] + 0.02, bb[3] + 0.02, transform=src.transform).round_offsets().round_lengths()
        a = src.read(1, window=win).astype(np.float32); a[a < 0] = 0
        reproject(a, blk, src_transform=src.window_transform(win), src_crs=src.crs, dst_transform=trb, dst_crs=crs, resampling=Resampling.sum)
    w = np.where(urban, built, 0.0).astype(np.float64)
    wb = w.reshape(nb, 8, nb, 8).sum((1, 3))
    share = np.where(wb[:, None, :, None] > 0, w.reshape(nb, 8, nb, 8) / np.where(wb > 0, wb, 1)[:, None, :, None], 0)
    return (share * blk[:, None, :, None]).reshape(n, n)


def run(idx, cat):
    fn = f"{CD}/{idx:02d}.npz"
    c = C.iloc[idx]; crs, tr, n, half = city_grid(c)
    if os.path.exists(fn):
        z = np.load(fn); F = {k: z[k] for k in CLS}; pop = z["pop"]; nit = int(z["nit"])
    else:
        F, nit = worldcover_fracs(cat, c, crs, tr, n, half)
        r = np.arange(n) * PX - half + PX / 2; XX, YY = np.meshgrid(r, -r); inside = np.hypot(XX, YY) <= c.R_km * 1000
        urban = (F["built"] >= 0.25) & (F["water"] < 0.2) & inside
        pop = population(c, crs, n, half, F["built"], urban)
        np.savez_compressed(fn, pop=pop.astype(np.float32), nit=nit, **{k: F[k].astype(np.float16) for k in F})
    r = np.arange(n) * PX - half + PX / 2; XX, YY = np.meshgrid(r, -r); inside = np.hypot(XX, YY) <= c.R_km * 1000
    urban = (F["built"] >= 0.25) & (F["water"] < 0.2) & inside
    tree = F["tree"].astype(np.float64); P = np.where(urban, pop, 0.0); Ptot = P.sum()
    # access as published: patches of canopy >= 0.5, >= 2 connected cells, residents within 240 m
    lab, nl = ndi.label(tree >= 0.5); sz = ndi.sum(np.ones_like(tree), lab, index=np.arange(1, nl + 1))
    patch = np.isin(lab, np.arange(1, nl + 1)[sz >= 2]); dpatch = ndi.distance_transform_edt(~patch) * PX
    acc240 = P[dpatch <= 240].sum() / Ptot
    # benefit aligned with the treatment
    ring = [conv(tree, k) for k in KS]
    dT = 10 * sum(ck * rk for ck, rk in zip(CK, ring))
    can240 = conv(tree, (DK > 0) & (DK <= 240)) / ((DK > 0) & (DK <= 240)).sum()
    m = urban & (P > 0); pw = P[m] / P[m].sum()
    out = dict(idx=idx, city=c.city, inc2=c.inc2, region=c.region, koppen=c.koppen, items=nit, residents=Ptot, urban_ha=urban.sum() * 1.44,
               canopy_urban=float(tree[urban].mean()), acc240=acc240, dT_pw=float((dT[m] * pw).sum()), dT_med=float(np.median(dT[m])),
               share_dT05=float(pw[dT[m] >= 0.5].sum()), share_dT02=float(pw[dT[m] >= 0.2].sum()), share_can10=float(pw[can240[m] >= 0.10].sum()),
               share_can30=float(pw[can240[m] >= 0.30].sum()), can240_pw=float((can240[m] * pw).sum()),
               dT_gap30=float((np.maximum(0.30 - can240[m], 0) * pw).sum() * 10 * (CK[0] + CK[1]) / 1.0))
    # planting design: a canopy budget (5, 20 or 100 ha, replacing built-up cover) dispersed or consolidated into one round park.
    one = conv(P, RESP[::-1, ::-1])                                      # person-degC per unit canopy fraction added in a cell
    cand = np.argwhere(urban & (F["built"] >= 0.3) & (tree < 0.2))
    order = np.argsort(-one[cand[:, 0], cand[:, 1]])
    for B in (5, 20, 100):
        # dispersed: +0.30 canopy fraction in single cells where one such planting cools most residents (best case)
        ncell = max(int(round(B / (0.30 * 1.44))), 1)
        add = np.zeros_like(tree); sel = cand[order[:ncell]]; add[sel[:, 0], sel[:, 1]] = 0.30
        gain_d = conv(add, RESP); pc_d = float((P * gain_d).sum())
        # consolidated: one round park of the same area, fully converted; residents of the park cells are displaced, so the park is
        # placed where it cools most residents outside it (best case)
        area_cells = B / 1.44; rad = np.sqrt(area_cells / np.pi) * PX
        rr = np.arange(-int(rad // PX) - 1, int(rad // PX) + 2); XP, YP = np.meshgrid(rr, rr)
        dmask = (np.hypot(XP, YP) * PX <= max(rad, PX / 2)).astype(float); disc = dmask * area_cells / dmask.sum()
        big = conv(disc, RESP)
        pad = (big.shape[0] - dmask.shape[0]) // 2; inside_big = np.zeros_like(big); inside_big[pad:pad + dmask.shape[0], pad:pad + dmask.shape[1]] = dmask
        big_out = big * (inside_big < 0.5)
        score = conv(P, big_out[::-1, ::-1])
        j = np.unravel_index(np.argmax(np.where(urban, score, -1)), score.shape)
        parkmap = np.zeros_like(tree); parkmap[j] = 1; gain_p = conv(parkmap, big)
        inpark = conv(parkmap, dmask) > 0.5; Pp = np.where(inpark, 0.0, P); pc_p = float((Pp * gain_p).sum())
        out.update({f"pc_disp_{B}": pc_d, f"pc_park_{B}": pc_p, f"ratio_{B}": pc_d / pc_p if pc_p > 0 else np.nan,
                    f"reach_disp_{B}": float(P[gain_d >= 0.05].sum() / Ptot), f"reach_park_{B}": float(Pp[gain_p >= 0.05].sum() / Ptot),
                    f"displaced_{B}": float(P[inpark].sum() / Ptot)})
    return out


if __name__ == "__main__":
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    rows = []
    ids = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else range(len(C))
    for i in ids:
        for attempt in range(3):
            try:
                rows.append(run(i, cat)); print(i, C.city[i], {k: round(v, 3) for k, v in rows[-1].items() if isinstance(v, float)}, flush=True); break
            except Exception as e:
                print(i, "retry", repr(e)[:200], flush=True)
    fo = f"{OUT}/access84.csv"
    old = pd.read_csv(fo) if os.path.exists(fo) else pd.DataFrame()
    new = pd.DataFrame(rows)
    allr = pd.concat([old[~old.idx.isin(new.idx)] if len(old) else old, new]).sort_values("idx")
    allr.to_csv(fo, index=False)
    os._exit(0)
