"""Landsat 8 Collection 2 Level-2 warm-season scenes 2016-2020 for the high-resolution subset, read onto the 30 m UTM grid of the
WorldCover window (data/wc14/<idx>.npz). Per scene: surface temperature (deg C), NDVI and the clear mask (QA_PIXEL fill, dilated cloud,
cirrus, cloud, shadow, snow). Also Copernicus GLO-30 elevation. -> data/ls/<idx>.npz"""
import sys, os, numpy as np, pandas as pd, rasterio, pystac_client, planetary_computer as pc
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
C = pd.read_csv("/home/claude/n84/data/cities.csv")
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
NMAX = int(os.environ.get("NMAX", 14))


sys.path.insert(0, '/home/claude/n84/src')
from common import warm_months


def read(href, tr, crs, n, rs=Resampling.bilinear, dtype="float32"):
    for k in range(5):
        try:
            with rasterio.open(href) as src, WarpedVRT(src, crs=crs, transform=tr, width=n, height=n, resampling=rs) as v:
                return v.read(1).astype(dtype)
        except Exception as e:
            err = e
    raise err


for idx in [int(x) for x in sys.argv[1].split(",")]:
    fn = f"/home/claude/n84/data/ls/{idx:02d}.npz"
    if os.path.exists(fn): continue
    c = C.iloc[idx]; W = np.load(f"/home/claude/n84/data/wc14/{idx:02d}.npz"); crs = f"EPSG:{int(W['epsg'])}"
    n = 930; tr = from_origin(float(W["x0"]), float(W["y0"]), 30, 30)
    bb = transform_bounds(crs, 4326, float(W["x0"]), float(W["y0"]) - n * 30, float(W["x0"]) + n * 30, float(W["y0"]))
    ms = warm_months(c.city)
    items = [i for i in cat.search(collections=["landsat-c2-l2"], bbox=bb, datetime=os.environ.get("DT", "2016-01-01/2020-12-31"),
                                   query={"eo:cloud_cover": {"lt": float(os.environ.get("CC", 40))}, "platform": {"in": os.environ.get("PLAT", "landsat-8").split(",")}}).items()
             if pd.Timestamp(i.datetime).month in ms]
    items.sort(key=lambda i: i.properties["eo:cloud_cover"])
    LST, NDV, MASK, META = [], [], [], []
    for it in items:
        if len(LST) >= NMAX: break
        try:
            qa = read(it.assets["qa_pixel"].href, tr, crs, n, Resampling.nearest, "uint16")
            bad = (qa & (1 << 0)) | (qa & (1 << 1)) | (qa & (1 << 2)) | (qa & (1 << 3)) | (qa & (1 << 4)) | (qa & (1 << 5))
            clear = (bad == 0) & (qa > 0)
            if clear[n // 4:3 * n // 4, n // 4:3 * n // 4].mean() < float(os.environ.get("CLR", 0.6)): continue
            st = read(it.assets["lwir11"].href, tr, crs, n); st = np.where(st > 0, st * 0.00341802 + 149.0 - 273.15, np.nan)
            r = read(it.assets["red"].href, tr, crs, n) * 2.75e-5 - 0.2; ni = read(it.assets["nir08"].href, tr, crs, n) * 2.75e-5 - 0.2
            ndvi = (ni - r) / (ni + r + 1e-6)
        except Exception as e:
            print("skip", it.id, repr(e)[:80], flush=True); continue
        LST.append(st.astype(np.float16)); NDV.append(ndvi.astype(np.float16)); MASK.append(clear)
        META.append(dict(id=it.id, time=str(it.datetime), sun_az=it.properties.get("view:sun_azimuth"), sun_el=it.properties.get("view:sun_elevation"),
                         cloud=it.properties["eo:cloud_cover"]))
    dem_items = list(cat.search(collections=["cop-dem-glo-30"], bbox=bb).items())
    dem = np.full((n, n), np.nan, np.float32)
    for it in dem_items:
        d = read(it.assets["data"].href, tr, crs, n); dem = np.where(np.isnan(dem) & (d != 0), d, dem)
    np.savez_compressed(fn, lst=np.array(LST), ndvi=np.array(NDV), clear=np.array(MASK), dem=dem, meta=pd.DataFrame(META).to_json(), months=np.array(ms))
    print(idx, c.city, ms, len(LST), "scenes of", len(items), flush=True)
os._exit(0)
