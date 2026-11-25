"""Nature Sustainability revision 3: planting designs with feasible footprints, feasibility flags, coverage thresholds and a saturating
response (reviewer comments on budget scaling, common city sets, coverage thresholds and the additive model).

Changes from planting2.py:
  * conversion never exceeds the built-up cover available in a cell (canopy replaces built-up cover, as in the estimand);
  * park footprints are sized so that their eligible built-up cover can absorb the budget: for each footprint size (area factors 1, 1.4,
    2 and 2.8 times B / 0.6) only locations whose convertible built-up area reaches the target are allowed, the most efficient
    feasible size and location are kept, and conversion is then scaled DOWN to exactly the target (never up);
  * a design is feasible in a city-region if it can place the full budget; ratios are later compared on the common set of
    city-regions in which every compared design is feasible;
  * coverage is reported at 0.02, 0.05, 0.10 and 0.20 degC;
  * a saturating sensitivity: the cooling from a ring rises with its canopy fraction as h(f) = (1 - exp(-f/a)) / (1 - exp(-1/a)),
    a = 0.15, 0.3 or 0.6 (strong, moderate, weak saturation), so that additional canopy cools less where much canopy is already present;
  * WorldCover open land (open green + bare) in eligible cells, as a check of the less-occupied proxy.
Designs: D10, D20, D30 (+10/+20/+30 pp in single cells), PN (units of about 2 ha), P1 (one park), MIX (half park, half +30 pp).
Output: data/out/planting3/<idx>.npz and part_*.csv"""
import os, sys, numpy as np, pandas as pd
from scipy.signal import fftconvolve
from scipy import ndimage as ndi
ROOT = "/home/claude/n84ns"; OUT = f"{ROOT}/data/out"; CD = f"{OUT}/access84_cells"; PD = f"{OUT}/planting3"; os.makedirs(PD, exist_ok=True)
C = pd.read_csv(f"{ROOT}/data/cities_inc.csv")
PX = 120; HA = PX * PX / 1e4
EDGES = np.array([0, 120, 240, 360, 480, 600, 840])
CK = np.array([0.155, 0.179, 0.032, -0.003, 0.011, 0.001])
r = np.arange(-7, 8); X, Y = np.meshgrid(r, r); D = np.hypot(X, Y) * PX
KN = []
for k in range(6):
    m = ((D > EDGES[k]) & (D <= EDGES[k + 1])).astype(float); KN.append(m / m.sum())
RESP = sum(10 * c * k for c, k in zip(CK, KN))
conv = lambda a, k: fftconvolve(a, k, mode="same")
BUD = (5, 20, 100); THR = (0.02, 0.05, 0.10, 0.20); FACT = (1.0, 1.4, 2.0, 2.8); ASATS = (0.15, 0.3, 0.6)
hs = lambda f, a: (1 - np.exp(-np.clip(f, 0, 1) / a)) / (1 - np.exp(-1 / a))


def disc(area_ha):
    rad = np.sqrt(area_ha / HA / np.pi) * PX; n = int(rad // PX) + 1; rr = np.arange(-n, n + 1); XP, YP = np.meshgrid(rr, rr)
    return (np.hypot(XP, YP) * PX <= max(rad, PX / 2)).astype(bool)


def stamp(mask, j, st, val=True):
    """set mask[j + offsets of structuring element st] = val, clipped at the edges"""
    hh = st.shape[0] // 2; i0, j0 = j[0] - hh, j[1] - hh
    a0, b0 = max(i0, 0), max(j0, 0); a1, b1 = min(i0 + st.shape[0], mask.shape[0]), min(j0 + st.shape[1], mask.shape[1])
    sub = st[a0 - i0:a1 - i0, b0 - j0:b1 - j0]
    mask[a0:a1, b0:b1][sub] = val


def place(head, one, E, target, n, shape):
    """best feasible placement of n non-overlapping units each absorbing `target` ha; returns (add map, feasible)"""
    best = (None, -np.inf); Ef = E.astype(float); ho = head * one
    for f in FACT:
        dm = disc(f * target / 0.6)
        frac = conv(Ef, dm / dm.sum()) >= 0.75
        num = conv(ho, dm.astype(float)); den = conv(head, dm.astype(float))
        score = np.where(frac & (den * HA >= target), num / np.maximum(den, 1e-9), -np.inf)
        hd = dm.shape[0] // 2; excl = np.ones((2 * dm.shape[0] + 1, 2 * dm.shape[0] + 1), bool)
        add = np.zeros(shape); k = 0; tot = 0.0
        flat = np.argsort(-score, axis=None); ptr = 0; taken = np.zeros(shape, bool)
        while k < n and ptr < flat.size:
            j = np.unravel_index(flat[ptr], shape); ptr += 1
            if not np.isfinite(score[j]): break
            if taken[j]: continue
            fp = np.zeros(shape, bool); stamp(fp, j, dm)
            a = head * fp; a *= target / (a.sum() * HA)                  # scale down to exactly the target
            add += a; tot += score[j] * target; k += 1
            stamp(taken, j, excl)                                         # no overlap, gap of about one cell
        if k == n and tot > best[1]: best = (add, tot)
    return (best[0], True) if best[0] is not None else (None, False)


def run(idx):
    c = C.iloc[idx]; z = np.load(f"{CD}/{idx:02d}.npz")
    F = {k: z[k].astype(np.float64) for k in ("tree", "green", "water", "bare", "built")}; pop = z["pop"].astype(np.float64)
    n = F["built"].shape[0]; half = n * PX / 2
    rr = np.arange(n) * PX - half + PX / 2; XX, YY = np.meshgrid(rr, -rr); inside = np.hypot(XX, YY) <= c.R_km * 1000
    urban = (F["built"] >= 0.25) & (F["water"] < 0.2) & inside
    tree = F["tree"]; built = F["built"]; P = np.where(urban, pop, 0.0); Ptot = P.sum(); shape = tree.shape
    E1 = urban & (built >= 0.3) & (tree < 0.2)
    E2 = E1 & (P <= np.median(P[urban & (P > 0)]))
    one = conv(P, RESP[::-1, ::-1])
    Fk = np.stack([conv(tree, k) for k in KN])                            # ring canopy fractions before planting
    res = dict(idx=idx, city=c.city, inc2=c.inc2, region=c.region, residents=Ptot, e1_ha=float(E1.sum() * HA), e2_ha=float(E2.sum() * HA))
    open_ = F["green"] + F["bare"]
    for en, E in (("E1", E1), ("E2", E2)):
        res[f"open_mean_{en}"] = float(open_[E].mean()); res[f"open20_{en}"] = float((open_[E] >= 0.2).mean())
    G = {}
    for en, E in (("E1", E1), ("E2", E2)):
        cand = np.argwhere(E); order = np.argsort(-one[cand[:, 0], cand[:, 1]])
        head = built * E
        for B in BUD:
            designs = {}
            for dpp in (0.1, 0.2, 0.3):
                nc = int(np.ceil(B / (dpp * HA)))
                if nc > len(cand): designs[f"D{int(dpp * 100)}"] = None; continue
                sel = cand[order[:nc]]; add = np.zeros(shape); add[sel[:, 0], sel[:, 1]] = dpp; add *= B / (add.sum() * HA)
                designs[f"D{int(dpp * 100)}"] = add
            designs["P1"], _ = place(head, one, E, B, 1, shape)
            npk = max(2, int(round(B / 2))); designs["PN"], _ = place(head, one, E, B / npk, npk, shape)
            pk, ok = place(head, one, E, B / 2, 1, shape)
            if ok:
                nc = int(np.ceil(B / 2 / (0.3 * HA))); okc = cand[order]; okc = okc[pk[okc[:, 0], okc[:, 1]] == 0][:nc]
                if len(okc) == nc:
                    dd = np.zeros(shape); dd[okc[:, 0], okc[:, 1]] = 0.3; dd *= (B / 2) / (dd.sum() * HA); designs["MIX"] = pk + dd
                else: designs["MIX"] = None
            else: designs["MIX"] = None
            for dn, add in designs.items():
                key0 = f"{en}_{B}_{dn}"; res[f"feas_{key0}"] = add is not None
                if add is None: continue
                assert np.all(add <= built * (1 + 1e-6) + 1e-9), "conversion exceeds built-up cover"
                rem = np.clip(add / np.maximum(built, 1e-6), 0, 1)
                comp = np.stack([conv(add, k) for k in KN])               # change of ring canopy fraction
                gain = 10 * np.tensordot(CK, comp, 1)
                gsats = {a: 10 * np.tensordot(CK, hs(Fk + comp, a) - hs(Fk, a), 1) for a in ASATS}
                for rn, Pe in (("R0", P), ("R1", P * (1 - rem))):
                    key = f"{en}_{rn}_{B}_{dn}"
                    G[key] = 10 * (comp * Pe[None]).sum((1, 2))
                    res[f"pc_{key}"] = float(CK @ G[key]); res[f"pcsat_{key}"] = float((Pe * gsats[0.3]).sum()); res[f"pcsat15_{key}"] = float((Pe * gsats[0.15]).sum()); res[f"pcsat60_{key}"] = float((Pe * gsats[0.6]).sum())
                    for t in THR: res[f"reach{int(t * 100):02d}_{key}"] = float(Pe[gain >= t].sum() / Ptot)
                    res[f"disp_{key}"] = float((P - Pe).sum() / Ptot)
    np.savez_compressed(f"{PD}/{idx:02d}.npz", keys=np.array(list(G)), G=np.stack(list(G.values())))
    return res


if __name__ == "__main__":
    ids = [int(x) for x in sys.argv[1].split(",")]
    rows = []
    for i in ids:
        try:
            rows.append(run(i)); print(i, C.city[i], flush=True)
        except Exception as e:
            print(i, "FAIL", repr(e)[:300], flush=True)
    pd.DataFrame(rows).to_csv(f"{PD}/part_{ids[0]:02d}.csv", index=False)
