"""TerraClimate monthly 2001-2020 at the 84 city centres (vectorised point reads, one year chunk at a time) -> data/clim/tc84_{Y0}.csv"""
import sys, time, numpy as np, pandas as pd, xarray as xr, pystac_client, planetary_computer as pc, os
C = pd.read_csv("/home/claude/n84/data/cities.csv")
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
z = cat.get_collection("terraclimate").assets["zarr-abfs"]
ds = xr.open_dataset(z.href, **z.extra_fields["xarray:open_kwargs"])
li = np.abs(ds.lat.values[:, None] - C.lat.values[None]).argmin(0); lo = np.abs(ds.lon.values[:, None] - C.lon.values[None]).argmin(0)
V = ["tmax", "tmin", "ppt", "vap", "vpd", "soil", "pet", "aet", "ws"]
t0i = int(np.where(pd.to_datetime(ds.time.values).year == 2001)[0][0])
rows = []
Y0 = int(sys.argv[1]); NY = int(sys.argv[2])
for y in range(Y0 - 2001, Y0 - 2001 + NY):
    sl = slice(t0i + 12 * y, t0i + 12 * y + 12)
    d = {}
    for v in V:
        for k in range(6):
            try:
                d[v] = ds[v].isel(time=sl).isel(lat=xr.DataArray(li, dims="p"), lon=xr.DataArray(lo, dims="p")).load().values; break
            except Exception as e:
                print("retry", y, v, repr(e)[:80], flush=True); time.sleep(10)
    tt = pd.to_datetime(ds.time.values[sl])
    for i in range(len(C)):
        for m in range(12):
            rows.append(dict(city=C.city[i], time=tt[m], **{v: float(d[v][m, i]) for v in V}))
    pd.DataFrame(rows).to_csv("/home/claude/n84/data/clim/tc84_{Y0}.csv", index=False); print("year", 2001 + y, flush=True)
print("done", flush=True)
os._exit(0)
