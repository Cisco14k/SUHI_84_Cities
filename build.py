"""Assemble manuscript, SI and response; fill result tokens; number citations; convert to docx with native equations."""
import re, json, sys, os, subprocess
sys.path.insert(0, "/home/claude/n84/src")
from refs import REFS
ROOT = "/home/claude/n84"; T = f"{ROOT}/txt"; B = f"{ROOT}/build"
TOK = json.load(open(f"{ROOT}/data/out/tokens.json"))


def fill(s):
    def rep(m):
        k = m.group(1)
        if k not in TOK: raise KeyError(f"missing token {k}")
        return str(TOK[k])
    return re.sub(r"⟦([A-Z0-9_]+)⟧", rep, s)


def cite(s, order):
    def num(keys):
        ns = []
        for k in keys:
            if k not in order: order[k] = len(order) + 1
            ns.append(order[k])
        ns = sorted(set(ns)); out = []; i = 0
        while i < len(ns):
            j = i
            while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1: j += 1
            out.append(f"{ns[i]}–{ns[j]}" if j - i >= 2 else (f"{ns[i]},{ns[j]}" if j == i + 1 else f"{ns[i]}"))
            i = j + 1
        return "^" + ",".join(out) + "^"
    return re.sub(r"\{\{([a-z0-9_,]+)\}\}", lambda m: num(m.group(1).split(",")), s)


# ---------------- manuscript
head = open(f"{T}/ms_head.md").read(); body = open(f"{T}/ms_body.md").read(); tail = open(f"{T}/ms_tail.md").read()
body = fill(body); tail = fill(tail)
order = {}
body = cite(body, order)
legends_idx = tail.index("# Figure legends")
tail = cite(tail, order)
reflist = "\n\n".join(f"{n}. {REFS[k]}" for k, n in sorted(order.items(), key=lambda x: x[1]))
# split tail: acknowledgements etc., tables, legends; insert figures after legends
pre, legends = tail[:tail.index("# Figure legends")], tail[tail.index("# Figure legends"):]
FIG = {f"Fig. {i}": f"{ROOT}/figs/Fig{i}.png" for i in range(1, 6)}
ED = {f"Extended Data Fig. {i}": f"{ROOT}/dl/docx/ms/word/media/image{i + 5}.png" for i in range(1, 9)}
ED["Extended Data Fig. 9"] = f"{ROOT}/figs/EDFig9.png"; ED["Extended Data Fig. 10"] = f"{ROOT}/figs/EDFig10.png"
paras = [p for p in legends.split("\n\n") if p.strip()]
figs_md = ["# Figures"]; ed_md = ["# Extended Data"]
for p in paras[1:]:
    m = re.match(r"\*\*((?:Extended Data )?Fig\. \d+) \|", p)
    if not m: continue
    key = m.group(1); path = FIG.get(key) or ED.get(key)
    blk = f"![]({path}){{width=16.5cm}}\n\n{p}\n\n\\newpage"
    (ed_md if key.startswith("Extended") else figs_md).append(blk)
md = head + "\n\n" + body + "\n\n# References\n\n" + reflist + "\n\n" + pre + "\n\n" + "\n\n".join(figs_md) + "\n\n" + "\n\n".join(ed_md) + "\n"
assert "⟦" not in md and "{{" not in md
open(f"{B}/manuscript.md", "w").write(md)
# check no bullet lists
for i, line in enumerate(md.splitlines()):
    if re.match(r"^\s*([-*+]|\d+\.)\s", line) and not re.match(r"^\d+\. [A-Z]", line.strip()):
        print("possible list line", i, line[:60])
subprocess.run(["pandoc", f"{B}/manuscript.md", "-o", f"{B}/Manuscript_NatureCities_revised.docx", f"--reference-doc={B}/ref2.docx", "--resource-path", ROOT], check=True)
print("refs", len(order))

# ---------------- response
resp = fill(open(f"{T}/response.md").read())
open(f"{B}/response.md", "w").write(resp)
subprocess.run(["pandoc", f"{B}/response.md", "-o", f"{B}/Response_to_Reviewers.docx", f"--reference-doc={B}/ref2.docx"], check=True)
resp2 = fill(open(f"{T}/response2.md").read()); open(f"{B}/response2.md", "w").write(resp2)
resp3 = fill(open(f"{T}/response3.md").read()); open(f"{B}/response3.md", "w").write(resp3)
subprocess.run(["pandoc", f"{B}/response3.md", "-o", f"{B}/Response_to_Reviewers_round3.docx", f"--reference-doc={B}/ref2.docx"], check=True)
subprocess.run(["pandoc", f"{B}/response2.md", "-o", f"{B}/Response_to_Reviewers_round2.docx", f"--reference-doc={B}/ref2.docx"], check=True)
print("built")

# ---------------- table formatting in the manuscript
import docx
from docx.shared import Pt
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
d = docx.Document(f"{B}/Manuscript_NatureCities_revised.docx")
for t in d.tables:
    tblPr = t._tbl.tblPr
    w = tblPr.find(qn("w:tblW"))
    if w is None: w = OxmlElement("w:tblW"); tblPr.append(w)
    w.set(qn("w:type"), "pct"); w.set(qn("w:w"), "5000")
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                pf = p.paragraph_format; pf.line_spacing = 1.0; pf.space_after = Pt(1); pf.space_before = Pt(1)
                for r in p.runs: r.font.size = Pt(9); r.font.name = "Arial"
d.save(f"{B}/Manuscript_NatureCities_revised.docx")
print("tables formatted")
