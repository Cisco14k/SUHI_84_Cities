"""Regional Lagrangian moisture and heat tracking for the 84 city-regions (ERA5, Planetary Computer era5-pds, 0.25 deg, hourly).
For every day of each city-region's three warmest months (TerraClimate 1991-2020 normals), parcels are released at the hour nearest
10:00 local solar time (the Landsat overpass) from the city centre and from 8 points 10 km away, and integrated 96 h backwards in the
100 m wind (RK2, 1 h steps). Specific humidity q (2 m dewpoint, surface pressure) and potential temperature theta (2 m) are sampled
along each path. Overpass-hour 2 m temperature, dewpoint and 10 m wind at the centre are stored for every warm-season day.
Usage: python lagr84.py <city index list, comma separated or all> <year0> <year1>   ->  data/traj/<idx>_<year>.npz"""
import os, sys, time, warnings
import numpy as np, pandas as pd, xarray as xr, dask
import pystac_client, planetary_computer as pc
warnings.filterwarnings("ignore")
dask.config.set(scheduler="threads", num_workers=4)
ROOT = "/home/claude/n84"; OUT = f"{ROOT}/data/traj"; os.makedirs(OUT, exist_ok=True)
NB = 96; R = 6371e3
VARS = {"u": "eastward_wind_at_100_metres", "v": "northward_wind_at_100_metres", "t": "air_temperature_at_2_metres",
        "d": "dew_point_temperature_at_2_metres"}
C = pd.read_csv(f"{ROOT}/data/cities.csv")


sys.path.insert(0, '/home/claude/n84/src')
from common import warm_months


def box(lat, lon):
    hl = 10.0; hw = min(10.0 / max(np.cos(np.radians(lat)), 0.5), 16.0)
    return max(lat - hl, -89.5), min(lat + hl, 89.5), lon - hw, lon + hw


def open_var(cat, y, m, var):
    for k in range(6):
        try:
            it = next(i for i in cat.search(collections=["era5-pds"], datetime=f"{y}-{m:02d}").items() if i.id.endswith("-an"))
            a = it.assets[var]
            return xr.open_dataset(a.href, **a.extra_fields["xarray:open_kwargs"])[var]
        except Exception:
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"open {y}-{m} {var}")


def sub(da, la0, la1, lo0, lo1):
    lo0m, lo1m = lo0 % 360, lo1 % 360
    if lo0m < lo1m:
        x = da.sel(lat=slice(la1, la0), lon=slice(lo0m, lo1m))
    else:
        x = xr.concat([da.sel(lat=slice(la1, la0), lon=slice(lo0m, 360)), da.sel(lat=slice(la1, la0), lon=slice(0, lo1m))], "lon")
    for k in range(6):
        try:
            return x.load()
        except Exception:
            time.sleep(5 * (k + 1))
    raise RuntimeError("load")


def fields(cat, c, y, ms):
    la0, la1, lo0, lo1 = box(c.lat, c.lon)
    # season months as (year, month); seasons crossing the new year start in y-1; 5 days of the preceding month for 96 h histories
    cross = ms[0] > ms[-1]
    seq = [((y - 1) if (cross and m >= ms[0]) else y, m) for m in ms]
    pm = ms[0] - 1 if ms[0] > 1 else 12; pre = (seq[0][0] if ms[0] > 1 else seq[0][0] - 1, pm)
    F = {k: [] for k in VARS}; times = []
    for j, (yy, m) in enumerate([pre] + seq):
        for k, v in VARS.items():
            da = open_var(cat, yy, m, v)
            if j == 0:
                da = da.isel(time=slice(-5 * 24, None))
            a = sub(da, la0, la1, lo0, lo1); F[k].append(a.values.astype(np.float32))
        times.append(a.time.values)
    lat = a.lat.values; lon = a.lon.values
    sp = sub(open_var(cat, seq[0][0], seq[0][1], "surface_air_pressure").isel(time=slice(0, 744, 24)), la0, la1, lo0, lo1).values.mean(0)
    sst = sub(open_var(cat, seq[0][0], seq[0][1], "sea_surface_temperature").isel(time=0), la0, la1, lo0, lo1).values
    F = {k: np.concatenate(v) for k, v in F.items()}; t = np.concatenate(times)
    lonu = np.where(lon > lo1 % 360 + 1e-6, lon - 360, lon) if lo0 % 360 > lo1 % 360 else lon
    lonu = np.where(lonu > 180, lonu - 360, lonu)
    if lonu[0] > lonu[-1]:
        o = np.argsort(lonu); lonu = lonu[o]; F = {k: v[:, :, o] for k, v in F.items()}; sp = sp[:, o]; sst = sst[:, o]
    lat = lat[::-1]; F = {k: v[:, ::-1, :] for k, v in F.items()}; sp = sp[::-1]; sst = sst[::-1]
    e = 6.112 * np.exp(17.67 * (F["d"] - 273.15) / (F["d"] - 29.65))
    q = 622.0 * e / (sp[None] / 100.0 - 0.378 * e)
    th = F["t"] * (1e5 / sp[None]) ** 0.2854
    return dict(t=t, lat=lat, lon=lonu, u=F["u"], v=F["v"], q=q.astype(np.float32), th=th.astype(np.float32), T2=F["t"], D2=F["d"],
                ocean=np.isfinite(sst), seq=seq)


def interp(Fv, lat, lon, ti, la, lo):
    i0 = np.clip(np.floor(ti).astype(int), 0, Fv.shape[0] - 2); ft = np.clip(ti - i0, 0, 1)
    y = (la - lat[0]) / (lat[1] - lat[0]); x = (lo - lon[0]) / (lon[1] - lon[0])
    y0 = np.clip(np.floor(y).astype(int), 0, len(lat) - 2); x0 = np.clip(np.floor(x).astype(int), 0, len(lon) - 2)
    fy = np.clip(y - y0, 0, 1); fx = np.clip(x - x0, 0, 1); out = 0
    for it, wt in ((i0, 1 - ft), (i0 + 1, ft)):
        out = out + wt * (Fv[it, y0, x0] * (1 - fy) * (1 - fx) + Fv[it, y0 + 1, x0] * fy * (1 - fx) + Fv[it, y0, x0 + 1] * (1 - fy) * fx + Fv[it, y0 + 1, x0 + 1] * fy * fx)
    return out


def run(idx, y, cat):
    c = C.iloc[idx]; fn = f"{OUT}/{idx:02d}_{y}.npz"
    if os.path.exists(fn):
        return "skip"
    ms = warm_months(c.city); F = fields(cat, c, y, ms); t = pd.to_datetime(F["t"])
    hr = int(round((10.0 - c.lon / 15.0) % 24)) % 24                            # UTC hour nearest 10:00 local solar time
    inseason = np.array([(tt.month in ms) for tt in t])
    rel = [i for i, tt in enumerate(t) if tt.hour == hr and inseason[i] and i >= NB]
    dk = 10.0 / 111.2; dl = dk / np.cos(np.radians(c.lat))
    pts = [(c.lat, c.lon)] + [(c.lat + dk * np.cos(a), c.lon + dl * np.sin(a)) for a in np.radians(np.arange(0, 360, 45))]
    P = len(pts); n = len(rel) * P
    la = np.tile(np.array([p[0] for p in pts]), len(rel)); lo = np.tile(np.array([p[1] for p in pts]), len(rel))
    ti = np.repeat(np.array(rel, float), P)
    LA = np.full((n, NB + 1), np.nan, np.float32); LO = LA.copy(); Q = LA.copy(); TH = LA.copy(); alive = np.ones(n, bool)
    la0, la1, lo0, lo1 = F["lat"][0], F["lat"][-1], F["lon"][0], F["lon"][-1]
    for s in range(NB + 1):
        LA[alive, s] = la[alive]; LO[alive, s] = lo[alive]
        Q[alive, s] = interp(F["q"], F["lat"], F["lon"], ti[alive], la[alive], lo[alive])
        TH[alive, s] = interp(F["th"], F["lat"], F["lon"], ti[alive], la[alive], lo[alive])
        if s == NB:
            break
        a = alive
        u = interp(F["u"], F["lat"], F["lon"], ti[a], la[a], lo[a]); v = interp(F["v"], F["lat"], F["lon"], ti[a], la[a], lo[a])
        lam = la[a] - 0.5 * v * 3600 / R * 180 / np.pi; lom = lo[a] - 0.5 * u * 3600 / (R * np.cos(np.radians(la[a]))) * 180 / np.pi
        um = interp(F["u"], F["lat"], F["lon"], ti[a] - 0.5, lam, lom); vm = interp(F["v"], F["lat"], F["lon"], ti[a] - 0.5, lam, lom)
        la[a] = la[a] - vm * 3600 / R * 180 / np.pi; lo[a] = lo[a] - um * 3600 / (R * np.cos(np.radians(la[a]))) * 180 / np.pi
        ti[a] = ti[a] - 1
        alive = alive & (la > la0) & (la < la1) & (lo > lo0) & (lo < lo1) & (ti >= 0)
    # overpass-hour weather at the centre for every season day
    ov = [i for i in range(len(t)) if t[i].hour == hr and inseason[i]]
    tt = np.array(ov, float); P0la = np.full(len(ov), c.lat); P0lo = np.full(len(ov), c.lon)
    w = {k: interp(F[k], F["lat"], F["lon"], tt, P0la, P0lo) for k in ("T2", "D2", "u", "v")}
    np.savez_compressed(fn, lat=LA, lon=LO, q=Q.astype(np.float16), th=TH, release=np.array([str(t[i]) for i in rel]), npt=P,
                        grid_lat=F["lat"], grid_lon=F["lon"], ocean=F["ocean"], months=np.array(ms),
                        ov_time=np.array([str(t[i]) for i in ov]), **{f"ov_{k}": v.astype(np.float32) for k, v in w.items()})
    return f"ok {n}"


if __name__ == "__main__":
    idxs = range(len(C)) if sys.argv[1] == "all" else [int(x) for x in sys.argv[1].split(",")]
    yrs = range(int(sys.argv[2]), int(sys.argv[3]) + 1)
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    for y in yrs:
        for i in idxs:
            t0 = time.time()
            try:
                r = run(i, y, cat)
            except Exception as e:
                r = "ERR " + repr(e)[:300]
            print(i, C.city[i], y, r, round(time.time() - t0), "s", flush=True)
    os._exit(0)
