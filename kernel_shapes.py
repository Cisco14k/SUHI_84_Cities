"""Nature Sustainability revision 4: how much does the source distance depend on the kernel shape?

Ring effects of the 60 m re-analysis (pooled cooling per 10 pp, with standard errors, for ring widths 60, 90, 120 and 180 m; hrw_pool.json)
are fitted beyond the first ring by A x (kernel mass in each annulus) for three isotropic 2D kernel families:
  exponential  K ~ exp(-d/L)                  mass within r: 1 - (1 + r/L) exp(-r/L)
  Gaussian     K ~ exp(-d^2 / (2 s^2))         mass within r: 1 - exp(-r^2 / (2 s^2))
  power law    K ~ (1 + d^2/s^2)^(-b), b > 1   mass within r: 1 - (1 + r^2/s^2)^(1 - b)   (b free, and b = 1.5)
Amplitude by weighted least squares for each scale; scale (and b) by grid search minimizing chi^2; fits per ring width and jointly
over widths (each width with its own amplitude). The 95% source distance r95 solves mass(r95) = 0.95. Intervals by parametric bootstrap
of the ring effects (2,000 draws). Model comparison by AIC (chi^2 + 2k).
Output: data/out/kernel_shapes.json"""
import json, numpy as np
from scipy.optimize import brentq
ROOT = "/home/claude/n84ns"; HW = json.load(open(f"{ROOT}/data/out/hrw_pool.json"))
rng = np.random.default_rng(21)


def mass(r, fam, s, b=None):
    r = np.asarray(r, float)
    if fam == "exp": return 1 - (1 + r / s) * np.exp(-r / s)
    if fam == "gauss": return 1 - np.exp(-r ** 2 / (2 * s ** 2))
    return 1 - (1 + r ** 2 / s ** 2) ** (1 - b)


def r95(fam, s, b=None):
    if fam == "exp": return 4.744 * s
    if fam == "gauss": return s * np.sqrt(2 * np.log(20))
    return s * np.sqrt(20 ** (1 / (b - 1)) - 1)


GRID = {"exp": np.exp(np.linspace(np.log(10), np.log(400), 160)), "gauss": np.exp(np.linspace(np.log(10), np.log(500), 160)),
        "pow": np.exp(np.linspace(np.log(5), np.log(500), 120))}
BGRID = np.array([1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0])
WIDTHS = ("60", "90", "120", "180")
DATA = {}
for w in WIDTHS:
    d = HW["widths"][w]; e = np.array(d["edges"]); mu = np.array(d["mu"]); se = np.array(d["se"])
    k = np.arange(1, len(mu))                                             # beyond the first ring
    DATA[w] = (e[k], e[k + 1], mu[k], se[k])


def chi2_fit(fam, s, b, widths, mus):
    tot = 0.0
    for w in widths:
        a, bb, _, se = DATA[w]; mu = mus[w]
        m = mass(bb, fam, s, b) - mass(a, fam, s, b); wt = 1 / se ** 2
        A = (wt * m * mu).sum() / max((wt * m * m).sum(), 1e-12)
        tot += (wt * (mu - A * m) ** 2).sum()
    return tot


def fit(fam, widths, mus, bfix=None):
    best = (np.inf, None, None)
    bs = [None] if fam != "pow" else ([bfix] if bfix else BGRID)
    for b in bs:
        for s in GRID[fam]:
            c = chi2_fit(fam, s, b, widths, mus)
            if c < best[0]: best = (c, s, b)
    return best


out = {}
for label, widths in [("w60", ["60"]), ("joint", list(WIDTHS))]:
    mus0 = {w: DATA[w][2] for w in WIDTHS}
    n_obs = sum(len(DATA[w][2]) for w in widths)
    res = {}
    for fam, bfix, nm in (("exp", None, "exponential"), ("gauss", None, "Gaussian"), ("pow", None, "power law"), ("pow", 1.5, "power law, b = 1.5")):
        c, s, b = fit(fam, widths, mus0, bfix)
        kpar = len(widths) + 1 + (1 if (fam == "pow" and bfix is None) else 0)
        boots = []
        for _ in range(300 if fam == "pow" and bfix is None else 600):
            mus = {w: DATA[w][2] + rng.normal(0, DATA[w][3]) for w in WIDTHS}
            cb, sb, bb = fit(fam, widths, mus, bfix); boots.append(r95(fam, sb, bb))
        res[nm] = dict(chi2=float(c), k=kpar, aic=float(c + 2 * kpar), scale=float(s), b=None if b is None else float(b),
                       r95=float(r95(fam, s, b)), r95_ci=[float(x) for x in np.percentile(boots, [2.5, 97.5])])
    amin = min(v["aic"] for v in res.values())
    for v in res.values(): v["daic"] = v["aic"] - amin
    res["n_obs"] = n_obs; out[label] = res
    print(label, json.dumps({k: (round(v["scale"], 1), v["b"], round(v["r95"]), [round(x) for x in v["r95_ci"]], round(v["daic"], 1)) for k, v in res.items() if isinstance(v, dict)}))
json.dump(out, open(f"{ROOT}/data/out/kernel_shapes.json", "w"), indent=1)
