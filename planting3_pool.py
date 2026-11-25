"""Pool planting3 results on common sets of city-regions. Output: data/out/planting3.json"""
import glob, json, numpy as np, pandas as pd
ROOT = "/home/claude/n84ns"; PD = f"{ROOT}/data/out/planting3"
R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{PD}/part_*.csv"))]).sort_values("idx").reset_index(drop=True)
R.to_csv(f"{ROOT}/data/out/planting3.csv", index=False)
CK = np.array([0.155, 0.179, 0.032, -0.003, 0.011, 0.001]); SE = np.array([0.036, 0.031, 0.018, 0.019, 0.020, 0.017]) / 3.92
G = {}
for i in R.idx:
    z = np.load(f"{PD}/{int(i):02d}.npz"); G[int(i)] = dict(zip(z["keys"], z["G"]))
rng = np.random.default_rng(12); NB = 2000; draws = rng.normal(CK, SE, size=(NB, 6))
DES = ("D10", "D20", "D30", "PN", "P1", "MIX"); THR = ("02", "05", "10", "20")
out = {"n": int(len(R)), "feasible": {}, "common_n": {}}
for en in ("E1", "E2"):
    for B in (5, 20, 100):
        for d in DES:
            out["feasible"][f"{en}_{B}_{d}"] = int(R[f"feas_{en}_{B}_{d}"].fillna(False).astype(bool).sum())
        common = R[np.all([R[f"feas_{en}_{B}_{d}"].fillna(False).astype(bool) for d in DES], axis=0)].idx.astype(int).values
        out["common_n"][f"{en}_{B}"] = int(len(common))
        Rc = R[R.idx.isin(common)]
        for rn in ("R0", "R1"):
            for a in ("D10", "D20", "D30", "PN", "MIX"):
                ka, kb = f"{en}_{rn}_{B}_{a}", f"{en}_{rn}_{B}_P1"
                GA = np.stack([G[i][ka] for i in common]); GB = np.stack([G[i][kb] for i in common])
                pt = (GA @ CK) / (GB @ CK)
                bs = [np.median((GA[s] @ draws[d]) / (GB[s] @ draws[d])) for d, s in enumerate(rng.integers(0, len(common), (NB, len(common))))]
                sat = (Rc[f"pcsat_{ka}"] / Rc[f"pcsat_{kb}"]).values
                s15 = float(np.median(Rc[f"pcsat15_{ka}"] / Rc[f"pcsat15_{kb}"])) if f"pcsat15_{ka}" in Rc else np.nan
                s60 = float(np.median(Rc[f"pcsat60_{ka}"] / Rc[f"pcsat60_{kb}"])) if f"pcsat60_{ka}" in Rc else np.nan
                sbs = [np.median(sat[s]) for s in rng.integers(0, len(sat), (NB, len(sat)))]
                out[f"{ka}_vs_P1"] = dict(median=float(np.median(pt)), ci=[float(x) for x in np.percentile(bs, [2.5, 97.5])],
                                          n_gt1=int((pt > 1).sum()), n=int(len(pt)),
                                          sat=float(np.median(sat)), sat15=s15, sat60=s60, sat_ci=[float(x) for x in np.percentile(sbs, [2.5, 97.5])], sat_gt1=int((sat > 1).sum()))
            for d in DES:
                for t in THR:
                    out[f"reach{t}_{en}_{rn}_{B}_{d}"] = float(100 * Rc[f"reach{t}_{en}_{rn}_{B}_{d}"].median())
                out[f"disp_{en}_{rn}_{B}_{d}"] = float(100 * Rc[f"disp_{en}_{rn}_{B}_{d}"].median())
            # coverage ratio dispersed / park at each threshold, with city bootstrap
            for t in THR:
                a_ = Rc[f"reach{t}_{en}_{rn}_{B}_D30"].values; b_ = Rc[f"reach{t}_{en}_{rn}_{B}_P1"].values
                out[f"covdiff{t}_{en}_{rn}_{B}"] = dict(median=float(100 * np.median(a_ - b_)), n_gt=int((a_ > b_).sum()), n=int(len(a_)))
for en in ("E1", "E2"):
    out[f"open_mean_{en}"] = float(100 * R[f"open_mean_{en}"].median()); out[f"open20_{en}"] = float(100 * R[f"open20_{en}"].median())
out["e2_share"] = float((R.e2_ha / R.e1_ha).median())
A = pd.read_csv(f"{ROOT}/data/out/access84.csv")[["idx", "urban_ha"]]; M = R.merge(A, on="idx"); M["dens"] = M.residents / M.urban_ha
for en in ("E1", "E2"):
    ok = np.all([M[f"feas_{en}_100_{d}"].fillna(False).astype(bool) for d in DES], axis=0)
    out[f"excl100_{en}"] = dict(n_excl=int((~ok).sum()), elig_ha_in=float(M[f"{en.lower()}_ha"][ok].median()), elig_ha_out=float(M[f"{en.lower()}_ha"][~ok].median()),
                               dens_in=float(M.dens[ok].median()), dens_out=float(M.dens[~ok].median()), res_in=float(M.residents[ok].median()), res_out=float(M.residents[~ok].median()),
                               cities_out=sorted(M.city[~ok].tolist()))
json.dump(out, open(f"{ROOT}/data/out/planting3.json", "w"), indent=1)
for en in ("E1", "E2"):
    for rn in ("R0", "R1"):
        for B in (5, 20, 100):
            s = " ".join(f"{a}:{out[f'{en}_{rn}_{B}_{a}_vs_P1']['median']:.2f}/{out[f'{en}_{rn}_{B}_{a}_vs_P1']['sat']:.2f}" for a in ("D10", "D30", "PN", "MIX"))
            cov = " ".join(f"{t}:{out[f'reach{t}_{en}_{rn}_{B}_D30']:.1f}/{out[f'reach{t}_{en}_{rn}_{B}_PN']:.1f}/{out[f'reach{t}_{en}_{rn}_{B}_P1']:.1f}" for t in THR)
            print(en, rn, B, "n", out["common_n"][f"{en}_{B}"], s, "|", cov)
print(out["feasible"]); print({k: v for k, v in out.items() if k.startswith("open")}, out["e2_share"])
