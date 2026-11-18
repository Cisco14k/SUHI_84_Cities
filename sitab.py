"""Load the published supplementary tables (parsed to CSV) as numeric frames."""
import glob, re, numpy as np, pandas as pd
D = "/home/claude/n84/si"
num = lambda s: float(str(s).replace("−", "-").replace("+", "").replace("—", "nan").replace("P < 0.001", "0.0005"))


def ci(s):
    s = str(s).replace("−", "-")
    m = re.match(r"\s*([+-]?[\d.]+)\s*\(([+-]?[\d.]+)\s*(?:–|to)\s*([+-]?[\d.]+)\)", s)
    return (float(m.group(1)), float(m.group(2)), float(m.group(3))) if m else (np.nan,) * 3


def rng(s):
    m = re.search(r"([+-]?\d*\.?\d+)\s*(?:–|to)\s*([+-]?\d*\.?\d+)", str(s).replace("−", "-"))
    return (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)


def t(n):
    return pd.read_csv(glob.glob(f"{D}/T{n:02d}_*.csv")[0])


def table3():
    T = t(3); T.columns = ["source", "zone", "ring", "cool", "ci", "tau", "pi", "k"]
    T["source"] = T.source.replace("​", np.nan).ffill() if T.source.dtype == object else T.source
    T["cool"] = T.cool.map(num)
    T["lo"], T["hi"] = zip(*T.ci.map(rng)); T["pilo"], T["pihi"] = zip(*T.pi.map(rng))
    T["r0"] = T.ring.map(lambda s: float(re.split("–", s)[0])); T["r1"] = T.ring.map(lambda s: float(re.split("–", s)[1]))
    T["mid"] = (T.r0 + T.r1) / 2
    return T
