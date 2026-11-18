"""CMIP6 projections (NASA NEX-GDDP-CMIP6, bias-corrected and downscaled to 0.25 deg, Planetary Computer).
Daily tasmax is read for every 3rd day of every 2nd year per 20-year period, by fetching only the needed HDF5 chunks (one chunk = one day,
global) with byte-range requests, and sampled at the 84 city centres (nearest cell, plus the 3x3 mean).
Periods: historical 1995-2014, 2041-2060 and 2081-2100 under SSP1-2.6, SSP2-4.5 and SSP5-8.5.
Usage: python cmip.py <model> -> data/cmip/<model>.csv  (city, scenario, year, doy, tasmax_C)
"""
import os, sys, zlib, time, numpy as np, pandas as pd, requests, h5py, fsspec
from concurrent.futures import ThreadPoolExecutor
import pystac_client, planetary_computer as pc
ROOT = "/home/claude/n84"; OUT = f"{ROOT}/data/cmip"; os.makedirs(OUT, exist_ok=True)
C = pd.read_csv(f"{ROOT}/data/cities.csv")
MODEL = sys.argv[1]
RUNS = [("historical", y) for y in range(1995, 2015, 2)] + [(s, y) for s in ("ssp126", "ssp245", "ssp585") for y in list(range(2041, 2061, 2)) + list(range(2081, 2101, 2))]
STEP = 3
iy = np.clip(np.round((C.lat.values + 59.875) / 0.25).astype(int), 1, 598)
ix = np.clip(np.round((C.lon.values % 360 - 0.125) / 0.25).astype(int) % 1440, 1, 1438)
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
fn = f"{OUT}/{MODEL}.csv"; done = set()
if os.path.exists(fn):
    old = pd.read_csv(fn); done = set(zip(old.scenario, old.year))
S = requests.Session()


def get(url, off, size):
    for k in range(6):
        try:
            r = S.get(url, headers={"Range": f"bytes={off}-{off + size - 1}"}, timeout=120); r.raise_for_status(); return r.content
        except Exception:
            time.sleep(3 * (k + 1))
    raise RuntimeError("range")


for scen, yr in RUNS:
    if (scen, yr) in done:
        continue
    t0 = time.time()
    it = next(iter(cat.search(collections=["nasa-nex-gddp-cmip6"], query={"cmip6:model": {"eq": MODEL}, "cmip6:scenario": {"eq": scen}, "cmip6:year": {"eq": yr}}).items()), None)
    if it is None:
        print(MODEL, scen, yr, "missing", flush=True); continue
    url = it.assets["tasmax"].href
    with fsspec.open(url, block_size=2 ** 20).open() as f:
        h = h5py.File(f); d = h["tasmax"]; nt = d.shape[0]
        lat = h["lat"][:]; lon = h["lon"][:]
        assert abs(lat[0] + 59.875) < 1e-3 and abs(lon[0] - 0.125) < 1e-3, (lat[0], lon[0])
        days = list(range(2, nt, STEP))
        info = [d.id.get_chunk_info_by_coord((t, 0, 0)) for t in days]
        shape = d.chunks; dt = d.dtype; filt = d.compression; shuf = d.shuffle
    def one(k):
        raw = get(url, info[k].byte_offset, info[k].size); b = zlib.decompress(raw)
        if shuf:
            a = np.frombuffer(b, np.uint8).reshape(dt.itemsize, -1).T.copy().view(dt).ravel()
        else:
            a = np.frombuffer(b, dt)
        a = a.reshape(shape[1:])
        v0 = a[iy, ix]
        v3 = np.nanmean(np.stack([a[iy + i, ix + j] for i in (-1, 0, 1) for j in (-1, 0, 1)]), 0)
        return v0, v3
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, range(len(days))))
    rows = []
    for k, t in enumerate(days):
        v0, v3 = res[k]
        for c in range(len(C)):
            rows.append((MODEL, scen, yr, t + 1, c, v0[c] - 273.15, v3[c] - 273.15))
    pd.DataFrame(rows, columns=["model", "scenario", "year", "doy", "idx", "tx", "tx3"]).to_csv(fn, mode="a", header=not os.path.exists(fn), index=False)
    print(MODEL, scen, yr, len(days), f"{time.time() - t0:.0f} s", flush=True)
os._exit(0)
