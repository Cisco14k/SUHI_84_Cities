"""Match the 84 city-regions of Supplementary Table 1 to Natural Earth populated places (coordinates, population)."""
import json, unicodedata, pandas as pd, sys
T = pd.read_csv("/home/claude/n84/si_table1.csv")
G = json.load(open(sys.argv[1]))
norm = lambda s: unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().replace("-", " ").replace("'", "").strip()
cc = {"Ivory Coast": "Ivory Coast"}
alias = {"gold coast (australia)": "gold coast", "quebec city": "quebec", "chattogram": "chittagong", "bengaluru": "bangalore", "cap haitien": "cap haitien",
         "sekondi takoradi": "sekondi", "newcastle upon tyne": "newcastle", "valencia": "valencia", "hamilton": "hamilton", "david": "david", "san miguel": "san miguel"}
P = pd.DataFrame([dict(name=f["properties"]["name"], nameascii=f["properties"].get("nameascii") or f["properties"]["name"], adm0=f["properties"]["adm0name"],
                       pop=f["properties"]["pop_max"], lat=f["geometry"]["coordinates"][1], lon=f["geometry"]["coordinates"][0]) for f in G["features"]])
P["k"] = P.nameascii.map(norm)
out = []
for _, r in T.iterrows():
    k = alias.get(norm(r.city), norm(r.city))
    c = P[P.k == k]
    if len(c) == 0: c = P[P.k.str.startswith(k)]
    cn = norm(r.country)
    c2 = c[c.adm0.map(norm).str.contains(cn.split()[0])] if len(c) > 1 else c
    if len(c2) == 0: c2 = c
    if len(c2) == 0: print("MISS", r.city, r.country); continue
    b = c2.sort_values("pop", ascending=False).iloc[0]
    out.append(dict(city=r.city, country=r.country, lat=round(b.lat, 4), lon=round(b.lon, 4), pop=int(b["pop"]), ne=b["name"], ne_country=b.adm0))
O = pd.DataFrame(out); O.to_csv("/home/claude/n84/data/cities_ne.csv", index=False)
print(O.to_string())
