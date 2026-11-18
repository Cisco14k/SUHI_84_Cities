import glob, pandas as pd
ROOT = "/home/claude/n84"


def tc():
    fs = sorted(glob.glob(f"{ROOT}/data/clim/tc84*.csv"))
    return pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(["city", "time"])


def warm_months(city, T=None):
    T = tc() if T is None else T; T = T[T.city == city]
    m = T.groupby(pd.to_datetime(T.time).dt.month).tmax.mean()
    best = max(range(1, 13), key=lambda s: sum(m[((s + i - 1) % 12) + 1] for i in range(3)))
    return [((best + i - 1) % 12) + 1 for i in range(3)]
