"""Lagrangian moisture (WaterSip) and heat attribution for the overpass air over each city-region.

For each back-trajectory, specific humidity q and potential temperature theta are differenced over 3 h steps
(moving backward in time the step k -> k-1 is forward in time). An increase dq > dq_min is an uptake; a decrease
reduces all earlier uptakes in proportion (Sodemann et al. 2008). Uptakes surviving to the release time are attributed to the
grid cell (land or ocean) and distance from the city at which they occurred. The same bookkeeping applied to theta gives the
sensible heat gained by the air (Schumacher et al. 2019 style), with diabatic cooling discounting earlier gains.
Outputs: data/out/lagr_city.csv (one row per city-year), data/out/lagr_maps.npz (uptake densities on a 0.5 deg grid relative to
each city, per city), data/out/lagr_rel.npz (pooled density in city-relative km coordinates).
"""
import glob, os, sys, numpy as np, pandas as pd
ROOT = "/home/claude/n84"; C = pd.read_csv(f"{ROOT}/data/cities.csv")
DGRID = [10, 25, 50, 100, 175, 250, 500, 750, 1000, 1500, 2000]
STEP = 3; DQMIN = 0.1; DTHMIN = 0.25           # g/kg and K per 3 h
R = 6371.0


def hav(la1, lo1, la2, lo2):
    p = np.radians; a = np.sin(p(la2 - la1) / 2) ** 2 + np.cos(p(la1)) * np.cos(p(la2)) * np.sin(p(lo2 - lo1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def watersip(x, thr):
    """x: (n, T) along-track series ordered from release (0) backwards. Returns contribution matrix (n, T-1 steps) in units of x,
    the attributed fraction of x at release, and the index of each step (oldest last)."""
    xs = x[:, ::-STEP][:, ::-1]                  # forward in time, 3-hourly, last column = release
    xs = x[:, ::STEP][:, ::-1]
    n, T = xs.shape; contrib = np.zeros((n, T - 1)); fr = np.zeros(n)
    for j in range(1, T):
        d = xs[:, j] - xs[:, j - 1]; ok = np.isfinite(d)
        up = ok & (d > thr); dn = ok & (d < 0)
        # loss: discount existing contributions in proportion
        if dn.any():
            tot = np.where(xs[:, j - 1] > 0, xs[:, j - 1], np.nan)
            f = np.clip(1 + d / tot, 0, 1); f = np.where(dn & np.isfinite(f), f, 1.0)
            contrib[:, :j] *= f[:, None]
        contrib[up, j - 1] = d[up]
    last = xs[:, -1]
    frac = np.nansum(contrib, 1) / np.where(last > 0, last, np.nan)
    return contrib, frac


def anomaly_theta(th):
    # heat gain relative to the domain mean of the release time removes the large-scale gradient (theta is not conserved-zero based)
    return th - np.nanmin(th, 1, keepdims=True)


rows = []; maps = {}; REL = np.zeros((161, 161)); RELH = np.zeros((161, 161)); NREL = 0
for fn in sorted(glob.glob(f"{ROOT}/data/traj/*.npz")):
    idx, yr = map(int, os.path.basename(fn)[:-4].split("_")); c = C.iloc[idx]
    d = np.load(fn, allow_pickle=True)
    la, lo, q, th = d["lat"].astype(float), d["lon"].astype(float), d["q"].astype(float), d["th"].astype(float)
    gl, go, oc = d["grid_lat"], d["grid_lon"], d["ocean"]
    n = la.shape[0]
    # positions at the 3-hourly steps (forward order, last = release); step j-1 -> j is attributed to the midpoint location
    LA = la[:, ::STEP][:, ::-1]; LO = lo[:, ::STEP][:, ::-1]
    mla = 0.5 * (LA[:, 1:] + LA[:, :-1]); mlo = 0.5 * (LO[:, 1:] + LO[:, :-1])
    cq, fq = watersip(q, DQMIN)
    thp = anomaly_theta(th); ch, fh = watersip(thp, DTHMIN)
    # variant: heat gain relative to the oldest point of each trajectory (origin-anchored), negative values floored at 0.1 K
    oldest = np.array([r[np.isfinite(r)][-1] if np.isfinite(r).any() else np.nan for r in th])
    ch_o, fh_o = watersip(np.maximum(th - oldest[:, None], 0.1), DTHMIN)
    dist = hav(c.lat, c.lon, mla, mlo)
    iy = np.clip(np.round((mla - gl[0]) / (gl[1] - gl[0])).astype(int), 0, len(gl) - 1)
    ix = np.clip(np.round((mlo - go[0]) / (go[1] - go[0])).astype(int), 0, len(go) - 1)
    isoc = np.where(np.isfinite(mla), oc[iy, ix], False)
    q0 = np.nansum(cq); h0 = np.nansum(ch)
    age = np.arange(cq.shape[1])[::-1] * STEP + STEP / 2           # hours before release
    def share(m, cc, tot): return float(np.nansum(np.where(m, cc, 0)) / tot) if tot > 0 else np.nan
    qrel = q[:, 0]
    rows.append(dict(idx=idx, city=c.city, year=yr, n=n, q_rel=np.nanmean(qrel), th_rel=np.nanmean(th[:, 0]),
                     f_attr_q=np.nanmean(fq), f_attr_h=np.nanmean(fh),
                     land_q=share(~isoc, cq, q0), land_h=share(~isoc, ch, h0),
                     q_50=share(dist <= 50, cq, q0), q_250=share(dist <= 250, cq, q0), q_1000=share(dist <= 1000, cq, q0),
                     h_50=share(dist <= 50, ch, h0), h50_o=share(dist <= 50, ch_o, np.nansum(ch_o)), land_h_o=share(~isoc, ch_o, np.nansum(ch_o)),
                     h250_o=share(dist <= 250, ch_o, np.nansum(ch_o)), age_h_o=float(np.nansum(ch_o * age) / np.nansum(ch_o)) if np.nansum(ch_o) > 0 else np.nan, h_250=share(dist <= 250, ch, h0), h_1000=share(dist <= 1000, ch, h0),
                     age_q=float(np.nansum(cq * age) / q0) if q0 > 0 else np.nan, age_h=float(np.nansum(ch * age) / h0) if h0 > 0 else np.nan,
                     d_q=float(np.nansum(np.where(np.isfinite(dist), cq * dist, 0)) / q0) if q0 > 0 else np.nan,
                     d_h=float(np.nansum(np.where(np.isfinite(dist), ch * dist, 0)) / h0) if h0 > 0 else np.nan,
                     ws_ov=float(np.nanmean(np.hypot(d["ov_u"], d["ov_v"]))),
                     vpd_ov=float(np.nanmean(0.6108 * (np.exp(17.27 * (d["ov_T2"] - 273.15) / (d["ov_T2"] - 35.85)) - np.exp(17.27 * (d["ov_D2"] - 273.15) / (d["ov_D2"] - 35.85))))),
                     T_ov=float(np.nanmean(d["ov_T2"] - 273.15)),
                     **{f"cq_{r}": share(dist <= r, cq, q0) for r in DGRID}, **{f"ch_{r}": share(dist <= r, ch, h0) for r in DGRID}))
    # uptake density on the city's own 0.25 deg grid (per release, g/kg and K)
    Mq = np.zeros(oc.shape); Mh = np.zeros(oc.shape); ok = np.isfinite(mla)
    np.add.at(Mq, (iy[ok], ix[ok]), cq[ok]); np.add.at(Mh, (iy[ok], ix[ok]), ch[ok])
    key = f"{idx:02d}"
    if key in maps:
        maps[key][0] += Mq; maps[key][1] += Mh; maps[key][2] += n
    else:
        maps[key] = [Mq, Mh, n, gl, go, oc]
    # city-relative km grid (+-800 km, 10 km cells)
    yk = (mla - c.lat) * 111.2; xk = (mlo - c.lon) * 111.2 * np.cos(np.radians(c.lat))
    # rotate so that the mean overpass wind blows from left (-x) to right (+x): upwind sources appear at negative x
    U, V = np.nanmean(d["ov_u"]), np.nanmean(d["ov_v"]); ang = np.arctan2(V, U)
    xk, yk = xk * np.cos(-ang) - yk * np.sin(-ang), xk * np.sin(-ang) + yk * np.cos(-ang)
    jy = np.round(yk / 10 + 80).astype(int, copy=False) if False else np.round(np.nan_to_num(yk, nan=1e9) / 10 + 80)
    jx = np.round(np.nan_to_num(xk, nan=1e9) / 10 + 80)
    g = (jy >= 0) & (jy < 161) & (jx >= 0) & (jx < 161)
    np.add.at(REL, (jy[g].astype(int), jx[g].astype(int)), cq[g] / max(qrel.mean(), 1e-6) / n)
    np.add.at(RELH, (jy[g].astype(int), jx[g].astype(int)), ch[g] / max(np.nanmean(ch.sum(1)), 1e-6) / n)
    NREL += 1
    print(idx, c.city, yr, f"landq={rows[-1]['land_q']:.2f} q250={rows[-1]['q_250']:.2f} h250={rows[-1]['h_250']:.2f} fq={rows[-1]['f_attr_q']:.2f}", flush=True)
D = pd.DataFrame(rows); D.to_csv(f"{ROOT}/data/out/lagr_city.csv", index=False)
np.savez_compressed(f"{ROOT}/data/out/lagr_rel.npz", q=REL / max(NREL, 1), h=RELH / max(NREL, 1), n=NREL)
np.savez_compressed(f"{ROOT}/data/out/lagr_maps.npz", **{f"{k}_q": v[0] / v[2] for k, v in maps.items()}, **{f"{k}_h": v[1] / v[2] for k, v in maps.items()},
                    **{f"{k}_lat": v[3] for k, v in maps.items()}, **{f"{k}_lon": v[4] for k, v in maps.items()}, **{f"{k}_oc": v[5] for k, v in maps.items()})
print(D.describe().T[["mean", "50%"]].round(3))
