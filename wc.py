"""ESA WorldCover 2021 (10 m) for a square window around selected city centres -> data/wc/<idx>.npz (class codes, UTM transform)."""
import sys, os, numpy as np, pandas as pd, rasterio, pystac_client, planetary_computer as pc
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import from_origin
from pyproj import CRS
C = pd.read_csv("/home/claude/n84/data/cities.csv")
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
HALF = float(os.environ.get("HALF", 12000))
for idx in [int(x) for x in sys.argv[1].split(",")]:
    fn = f"{os.environ.get('WCDIR', '/home/claude/n84/data/wc')}/{idx:02d}.npz"
    if os.path.exists(fn): continue
    c = C.iloc[idx]; zone = int((c.lon + 180) // 6) + 1
    crs = CRS.from_epsg((32600 if c.lat >= 0 else 32700) + zone)
    from pyproj import Transformer
    x0, y0 = Transformer.from_crs(4326, crs, always_xy=True).transform(c.lon, c.lat)
    n = int(2 * HALF / 10); tr = from_origin(x0 - HALF, y0 + HALF, 10, 10)
    bb = transform_bounds(crs, 4326, x0 - HALF, y0 - HALF, x0 + HALF, y0 + HALF)
    items = list(cat.search(collections=["esa-worldcover"], bbox=bb, query={"esa_worldcover:product_version": {"eq": "2.0.0"}}).items())
    out = np.zeros((n, n), np.uint8)
    for it in items:
        with rasterio.open(it.assets["map"].href) as src:
            tmp = np.zeros((n, n), np.uint8)
            reproject(rasterio.band(src, 1), tmp, dst_transform=tr, dst_crs=crs, resampling=Resampling.nearest)
            out = np.where(out == 0, tmp, out)
    np.savez_compressed(fn, wc=out, x0=x0 - HALF, y0=y0 + HALF, epsg=crs.to_epsg())
    print(idx, c.city, len(items), np.bincount(out.ravel(), minlength=101)[[0, 10, 20, 30, 40, 50, 60, 80]], flush=True)
