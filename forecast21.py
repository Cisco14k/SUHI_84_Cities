"""Nature Sustainability revision: paired uncertainty of held-out forecast skill (21 re-analysis city-regions, 120 m canopy rings).
Each city-region in turn is held out; its nine ring means of canopy cooling are forecast from the other 20 with (i) the InVEST kernel,
weight exp(-d/d_cool) truncated at d_cool, with d_cool = 450 m and amplitude fitted; (ii) the same kernel with d_cool also fitted (30 m grid);
(iii) the empirical curve (weighted mean ring profile of the training cities). Ring forecasts of a kernel are its area mean over the ring.
Skill = 1 - sum w (mu - f)^2 / sum w (mu - f_c)^2 with f_c the training mean over rings, pooled over held-out cities; paired differences
of skill between forecasts use 2,000 bootstrap samples of city-regions. Output: data/out/forecast21.json"""
import glob, json, numpy as np
ROOT = "/home/claude/n84ns"; rng = np.random.default_rng(5)
J = [json.load(open(f)) for f in sorted(glob.glob(f"{ROOT}/data/out/hr/*.json"))]; J = [j for j in J if len(j["scenes"]) >= 3]
names = [tuple(x) for x in J[0]["names120"]]; tr = [i for i, n in enumerate(names) if n[0] == "tree"]
M = np.array([np.nanmean([np.array(s["r120"])[tr] for s in j["scenes"]], 0) for j in J])
S = np.array([np.nanstd([np.array(s["r120"])[tr] for s in j["scenes"]], 0, ddof=1) / np.sqrt(len(j["scenes"])) for j in J])
W = 1 / np.maximum(S, 0.005) ** 2
E = np.array([0, 120, 240, 360, 480, 600, 840, 1080, 1440, 1800.])
r = np.linspace(0, 1800, 3601)[1:]


def ringmean(dcool):
    w = np.where(r <= dcool, np.exp(-r / dcool), 0.0)
    return np.array([np.trapezoid((w * r)[(r > a) & (r <= b)], r[(r > a) & (r <= b)]) / np.trapezoid(r[(r > a) & (r <= b)], r[(r > a) & (r <= b)]) for a, b in zip(E[:-1], E[1:])])


GRID = np.arange(60, 1501, 30); RM = {d: ringmean(d) for d in GRID}


def fit_amp(mu, w, shape):
    return float((w * mu * shape).sum() / max((w * shape * shape).sum(), 1e-12))


def forecasts(train):
    mu = M[train]; w = W[train]; mbar = (w * mu).sum(0) / w.sum(0)
    f450 = fit_amp(mu, w, RM[450][None, :]) * RM[450]
    best = min(GRID, key=lambda d: ((w * (mu - fit_amp(mu, w, RM[d][None, :]) * RM[d]) ** 2).sum()))
    ffit = fit_amp(mu, w, RM[best][None, :]) * RM[best]
    fc = np.full(9, (w * mu).sum() / w.sum())
    return dict(invest450=f450, invest_fit=ffit, empirical=mbar, const=fc), best


n = len(J); F = {}; dbest = []
for i in range(n):
    f, b = forecasts(np.array([k for k in range(n) if k != i])); dbest.append(int(b))
    for k, v in f.items(): F.setdefault(k, []).append(v)
F = {k: np.array(v) for k, v in F.items()}


def skill(idx, k):
    e = (W[idx] * (M[idx] - F[k][idx]) ** 2).sum(); e0 = (W[idx] * (M[idx] - F["const"][idx]) ** 2).sum(); return 1 - e / e0


out = {"n": n, "dcool_fitted_median": float(np.median(dbest)), "dcool_fitted_range": [int(min(dbest)), int(max(dbest))]}
allidx = np.arange(n); bs = [rng.integers(0, n, n) for _ in range(2000)]
for k in ("invest450", "invest_fit", "empirical"):
    v = [skill(b, k) for b in bs]; out[k] = [float(skill(allidx, k))] + [float(x) for x in np.percentile(v, [2.5, 97.5])]
for a, b in (("invest_fit", "invest450"), ("empirical", "invest450"), ("empirical", "invest_fit")):
    v = [skill(s, a) - skill(s, b) for s in bs]
    out[f"{a}-{b}"] = [float(skill(allidx, a) - skill(allidx, b))] + [float(x) for x in np.percentile(v, [2.5, 97.5])] + [float(np.mean(np.array(v) <= 0))]
json.dump(out, open(f"{ROOT}/data/out/forecast21.json", "w"), indent=1); print(json.dumps(out, indent=1))
