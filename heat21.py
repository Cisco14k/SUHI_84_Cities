"""Nature Sustainability revision: does canopy cooling reach the residents who are most exposed to surface heat? (21 city-regions, 60 m)

For each re-analysis city-region: mid-morning surface-temperature excess of urban cells over the rural background (median over the clear
Landsat 8-9 scenes, elevation-adjusted as in equation (1)); residents from GlobPOP 2020 allocated to urban cells in proportion to built-up
fraction within each 1 km cell; modelled cooling delivered by surrounding canopy, dT_i = 10 * sum_k c_k f_tree,k(i), with the 60 m ring
effects of the re-analysis (pooled, degC per 10 pp); distance to the nearest canopy patch (canopy >= 0.5, >= 2 ha).
Outputs per city: population-weighted heat excess of residents beyond vs within 240 m of canopy; share of the hottest third of residents
who live beyond 240 m; rank correlation of delivered cooling with heat excess. Output: data/out/heat21.csv"""
import os, sys, io, json, numpy as np, pandas as pd, rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from scipy import ndimage as ndi, stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hr as H
ROOT = H.ROOT; PX = 60; POP = f"{ROOT}/data/pop/GlobPOP_Count_30arc_2020_I32.tiff"


def run(idx, c60):
    W = np.load(f"{ROOT}/data/wc14/{idx:02d}.npz"); L8 = np.load(f"{ROOT}/data/ls/{idx:02d}.npz", allow_pickle=True)
    wc = W["wc"][:2790, :2790]; F = H.frac(wc, 6); n = F["built"].shape[0]
    lst30 = L8["lst"].astype(np.float32)[:, :930, :930]; clr = L8["clear"][:, :930, :930]
    lst = lst30.reshape(lst30.shape[0], n, 2, n, 2); cl = clr.reshape(clr.shape[0], n, 2, n, 2)
    lst = np.nanmean(np.where(cl, lst, np.nan), (2, 4))
    dem = L8["dem"][:930, :930].reshape(n, 2, n, 2).mean((1, 3))
    ex = []
    for si in range(lst.shape[0]):
        y = lst[si]; rural = (F["built"] < 0.05) & (F["water"] < 0.1) & np.isfinite(y)
        if rural.sum() < 200 or np.isfinite(y).mean() < 0.5: continue
        gam = np.polyfit(dem[rural], y[rural], 1)[0] if np.ptp(dem[rural]) > 50 else 0.0; gam = float(np.clip(gam, -0.012, 0.0))
        ya = y - gam * (dem - np.nanmedian(dem[rural])); ex.append(ya - np.nanmedian(ya[rural]))
    EX = np.nanmedian(np.stack(ex), 0)
    # population on the 60 m grid
    tr = from_origin(float(W["x0"]), float(W["y0"]), PX, PX); dens = np.zeros((n, n), np.float32)
    with rasterio.open(POP) as src:
        from rasterio.warp import transform_bounds
        from rasterio.windows import from_bounds
        bb = transform_bounds(int(W["epsg"]), 4326, float(W["x0"]), float(W["y0"]) - n * PX, float(W["x0"]) + n * PX, float(W["y0"]), densify_pts=21)
        win = from_bounds(bb[0] - 0.02, bb[1] - 0.02, bb[2] + 0.02, bb[3] + 0.02, transform=src.transform).round_offsets().round_lengths()
        a = src.read(1, window=win).astype(np.float32); a[a < 0] = 0
        reproject(a, dens, src_transform=src.window_transform(win), src_crs=src.crs, dst_transform=tr, dst_crs=f"EPSG:{int(W['epsg'])}",
                  resampling=Resampling.nearest)                       # count per 30" cell, assigned to every 60 m cell inside it
    cell30 = 926.6 ** 2 * np.cos(np.radians(H.C.lat[idx]))               # area of a 30 arc-second cell (m2)
    m6 = 6000 // PX; urban = (F["built"] >= 0.25) & (F["water"] < 0.2)
    keep = np.zeros_like(urban); keep[m6:-m6, m6:-m6] = True; urban &= keep & np.isfinite(EX)
    bl = ndi.uniform_filter(np.where(urban, F["built"], 0.0), size=17)  # mean built-up weight over ~1 km
    P = np.where(urban & (bl > 0), dens * (PX * PX / cell30) * F["built"] / np.maximum(bl, 1e-6), 0.0)
    tree = F["tree"].astype(np.float64)
    d, _, _ = H.dist_grid(600, PX); dT = np.zeros_like(tree)
    E60 = np.arange(0, 660, 60)
    for k in range(len(E60) - 1):
        m = ((d > E60[k]) & (d <= E60[k + 1])).astype(float); dT += 10 * c60[k] * H.conv(tree, m / m.sum())
    lab, nl = ndi.label(tree >= 0.5); sz = ndi.sum(np.ones_like(tree), lab, index=np.arange(1, nl + 1))
    patch = np.isin(lab, np.arange(1, nl + 1)[sz * PX * PX / 1e4 >= 2]); dp = ndi.distance_transform_edt(~patch) * PX
    m = urban & (P > 0); p = P[m]; e = EX[m]; t = dT[m]; near = dp[m] <= 240
    hot = e >= np.quantile(e, 2 / 3)
    wmean = lambda v, w: float((v * w).sum() / w.sum())
    # alternative heat indicators that existing canopy cannot lower: (a) counterfactual excess without the modelled cooling of the
    # surrounding canopy, EX + dT; (b) built-up fraction within 240 m (exposure proxy independent of surface temperature)
    ecf = e + t
    b240 = H.conv(F["built"].astype(float), ((d > 0) & (d <= 240)).astype(float) / ((d > 0) & (d <= 240)).sum())[m]
    hot_cf = ecf >= np.quantile(ecf, 2 / 3); hot_b = b240 >= np.quantile(b240, 2 / 3)
    extra = dict(hot_beyond_cf=float(p[hot_cf & ~near].sum() / p[hot_cf].sum()), hot_beyond_b=float(p[hot_b & ~near].sum() / p[hot_b].sum()),
                 dT_hot_cf=wmean(t[hot_cf], p[hot_cf]), dT_cool_cf=wmean(t[~hot_cf], p[~hot_cf]),
                 dT_hot_b=wmean(t[hot_b], p[hot_b]), dT_cool_b=wmean(t[~hot_b], p[~hot_b]),
                 excf_within=wmean(ecf[near], p[near]) if near.any() else np.nan, excf_beyond=wmean(ecf[~near], p[~near]))
    return dict(idx=idx, city=H.C.city[idx], residents=float(p.sum()), within240=float(p[near].sum() / p.sum()),
                excess_within=wmean(e[near], p[near]) if near.any() else np.nan, excess_beyond=wmean(e[~near], p[~near]),
                hot_beyond=float(p[hot & ~near].sum() / p[hot].sum()), all_beyond=float(p[~near].sum() / p.sum()),
                rho_dT_excess=float(stats.spearmanr(t, e).correlation), dT_hot=wmean(t[hot], p[hot]), dT_cool=wmean(t[~hot], p[~hot]), **extra)


if __name__ == "__main__":
    c60 = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None
    rows = [run(int(i), c60) for i in sys.argv[1].split(",")]
    pd.DataFrame(rows).to_csv(sys.argv[3] if len(sys.argv) > 3 else f"{ROOT}/data/out/heat21.csv", index=False); print(pd.DataFrame(rows).round(3).to_string())
    os._exit(0)
