"""Pool the ring-width and confounder-set re-analysis (hr_width.py).
For each ring width: random-effects pooled canopy profile (city means over scenes, DerSimonian-Laird across cities) and the threshold reach
R_delta (equation (5)) for delta = 0.025, 0.05, 0.10 degC per 10 pp with 400 city bootstraps. For each confounder set (V0 main, V1 without the
receptor's NDVI, V2 without the far-neighbourhood shares): kernel length by profile likelihood (1,000 city bootstraps) and, for V1 and V2,
the pooled 120 m profile and its reach. Kernel prediction: with the pooled L and an amplitude fitted to the 120 m profile beyond the first
ring, the effect of 10 pp of canopy in a ring is the amplitude times the kernel mass in that ring; the predicted reach for each width follows
from equation (5). Output: data/out/hrw_pool.json"""
import glob, json, numpy as np
ROOT = "/home/claude/n84ns"; rng = np.random.default_rng(17)
J = [json.load(open(f)) for f in sorted(glob.glob(f"{ROOT}/data/out/hrw/*.json"))]; J = [j for j in J if len(j["scenes"]) >= 3]
WID = ["60", "90", "120", "180"]; DEL = [0.025, 0.05, 0.10]


def dl_pool(m, se):
    m, se = np.asarray(m), np.maximum(np.asarray(se), 0.01); w = 1 / se ** 2; mu0 = np.sum(w * m) / w.sum()
    Q = np.sum(w * (m - mu0) ** 2); k = len(m); tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (se ** 2 + tau2); return float(np.sum(ws * m) / ws.sum()), float(np.sqrt(1 / ws.sum()))


def reach(mid, mu, d):
    k = np.argmax(mu < d) if (mu < d).any() else None
    if k is None: return float(mid[-1])
    if k == 0: return 0.0
    return float(mid[k - 1] + (mid[k] - mid[k - 1]) * (mu[k - 1] - d) / (mu[k - 1] - mu[k]))


def city_stats(key_fn):
    cm, cs = [], []
    for j in J:
        a = np.array([key_fn(s) for s in j["scenes"]], float); cm.append(np.nanmean(a, 0)); cs.append(np.nanstd(a, 0, ddof=1) / np.sqrt(len(a)))
    return np.array(cm), np.array(cs)


def pooled(cm, cs, idx=None):
    idx = np.arange(len(cm)) if idx is None else idx
    return np.array([dl_pool(cm[idx, k], cs[idx, k])[0] for k in range(cm.shape[1])]), np.array([dl_pool(cm[idx, k], cs[idx, k])[1] for k in range(cm.shape[1])])


out = {"n_cities": len(J), "n_scenes": int(sum(len(j["scenes"]) for j in J)), "widths": {}, "variants": {}}
BS = [rng.integers(0, len(J), len(J)) for _ in range(400)]
for w in WID:
    e = np.array(J[0]["scenes"][0]["widths"][w]["edges"]); mid = 0.5 * (e[1:] + e[:-1])
    cm, cs = city_stats(lambda s: s["widths"][w]["c"]); mu, se = pooled(cm, cs)
    r = {str(d): [reach(mid, mu, d)] + list(np.percentile([reach(mid, pooled(cm, cs, b)[0], d) for b in BS], [2.5, 97.5])) for d in DEL}
    out["widths"][w] = dict(edges=e.tolist(), mid=mid.tolist(), mu=mu.tolist(), se=se.tolist(), reach=r)
# kernel profile likelihood per confounder set
for v in ("V0", "V1", "V2"):
    LG = J[0]["LGRID"]
    SSE = np.array([[sum(s["kernel"][v][str(L)]["sse"] for s in j["scenes"]) for L in LG] for j in J])

    def lhat(idx):
        tot = SSE[idx].sum(0); i = int(np.argmin(tot))
        if 0 < i < len(LG) - 1:
            x = np.log(LG[i - 1:i + 2]); y = tot[i - 1:i + 2]; a, b, _ = np.polyfit(x, y, 2)
            if a > 0: return float(np.exp(-b / (2 * a)))
        return float(LG[i])
    Lh = lhat(np.arange(len(J))); bs = [lhat(rng.integers(0, len(J), len(J))) for _ in range(1000)]
    tot = SSE.sum(0)
    out["variants"][v] = dict(L=LG, dsse_rel=((tot - tot.min()) / tot.min()).tolist(), Lhat=Lh, Lci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    if v != "V0":
        cm, cs = city_stats(lambda s: s["v120"][v]["c"]); mu, se = pooled(cm, cs); mid = np.array(out["widths"]["120"]["mid"])
        out["variants"][v].update(mu120=mu.tolist(), se120=se.tolist(), reach120=reach(mid, mu, 0.05))
# kernel-predicted ring effects and reach for each width
L = out["variants"]["V0"]["Lhat"]; rr = np.linspace(0.5, 3000, 6000)
mass = lambda a, b: (np.exp(-a / L) * (1 + a / L) - np.exp(-b / L) * (1 + b / L))       # 2-D exponential kernel mass in an annulus
w120 = out["widths"]["120"]; e = np.array(w120["edges"]); mu = np.array(w120["mu"]); se = np.array(w120["se"])
M = np.array([mass(a, b) for a, b in zip(e[:-1], e[1:])]); sel = slice(1, 6)
A = float(np.sum(M[sel] * mu[sel] / se[sel] ** 2) / np.sum(M[sel] ** 2 / se[sel] ** 2))
pred = {}
for w in [60, 90, 120, 150, 180, 240]:
    ee = np.arange(0, 1200 + w, w); mm = 0.5 * (ee[1:] + ee[:-1]); p = A * np.array([mass(a, b) for a, b in zip(ee[:-1], ee[1:])])
    pred[str(w)] = {str(d): reach(mm, p, d) for d in DEL}
out["kernel_pred"] = dict(L=L, amplitude=A, reach=pred, r90=float(3.89 * L), r95=float(4.744 * L),
                          r95_ci=[float(4.744 * x) for x in out["variants"]["V0"]["Lci"]])
json.dump(out, open(f"{ROOT}/data/out/hrw_pool.json", "w"), indent=1)
print(json.dumps({k: (v if k != "widths" else {w: v[w]["reach"] for w in v}) for k, v in out.items() if k != "variants"}, indent=1))
print({v: (out["variants"][v]["Lhat"], out["variants"][v]["Lci"], out["variants"][v].get("reach120")) for v in out["variants"]})
