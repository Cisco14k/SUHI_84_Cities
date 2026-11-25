"""Supplementary Information for the Nature Sustainability submission: the latest SI with text updates, new Supplementary Notes 15-19,
Supplementary Tables 18-22 and Supplementary Fig. 5 inserted before the Supplementary References.
Usage: python build_si_ns.py <latest_SI.docx>"""
import sys, re, json, copy, subprocess
import docx
from docx.oxml.ns import qn
from docxcompose.composer import Composer
ROOT = "/home/claude/n84ns"; B = f"{ROOT}/build"; SRC = sys.argv[1]
TOK = json.load(open(f"{ROOT}/data/out/tokens_ns.json"))
fill = lambda s: re.sub(r"⟦([A-Z0-9_]+)⟧", lambda m: str(TOK[m.group(1)]), s)
md = fill(open(f"{ROOT}/txt/si_ns_notes.md").read()) + "\n\n" + fill(open(f"{ROOT}/txt/si_ns_tables.md").read())
assert "⟦" not in md
open(f"{B}/si_ns_new.md", "w").write(md)
subprocess.run(["pandoc", f"{B}/si_ns_new.md", "-o", f"{B}/si_ns_new.docx", f"--reference-doc={SRC}", "--resource-path", ROOT, "--columns=2000"], check=True)

REPL = [("Tree canopy cools urban land surfaces within about 300 metres across climates and decades",
         "Urban tree-canopy surface cooling is concentrated within about 240 m across 84 city-regions"),
        ("The share of scenes with confident single-scene cooling, used as a secondary target in the previous version, depends on measurement precision as much as on the effect and is no longer used.",
         "The share of scenes with confident single-scene cooling depends on measurement precision as much as on the effect and is not used as a target. The InVEST cooling equations are given in Supplementary Note 19 (equation (S4)), and paired skill differences for the 21 re-analysis city-regions in Supplementary Table 22."),
        ("(Methods, equation (9))", "(Supplementary Note 19, equation (S1))"),
        ("The weather-modulation analysis (Fig. 4e)", "The weather-modulation analysis (Extended Data Fig. 9e)"),
        ("maps for representative city-regions are in Supplementary Fig. 4.", "maps for representative city-regions are in Supplementary Fig. 4 and composites over all city-regions in Supplementary Fig. 5; the attribution equations are given in Supplementary Note 19 (equations (S2) and (S3))."),
        ]


def replace_in_paragraph(p, old, new):
    full = "".join(r.text for r in p.runs)
    if old not in full: return False
    # put the whole text in the first run that starts the match region, keep formatting of the first run
    start = full.index(old); end = start + len(old); pos = 0; first = None
    for r in p.runs:
        L = len(r.text); rs, re_ = pos, pos + L
        if re_ <= start or rs >= end:
            pos = re_; continue
        if first is None:
            first = r; r.text = r.text[:start - rs] + new + (r.text[end - rs:] if re_ > end else "")
        else:
            r.text = r.text[end - rs:] if re_ > end else ""
        pos = re_
    return True


d = docx.Document(SRC); done = set()
for p in d.paragraphs:
    for i, (o, n) in enumerate(REPL):
        if replace_in_paragraph(p, o, n): done.add(i)
for p in d.paragraphs:
    for r in p.runs:
        if "land-surface" in r.text: r.text = r.text.replace("land-surface", "land surface")
for t in d.tables:
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for r in p.runs:
                    if "land-surface" in r.text: r.text = r.text.replace("land-surface", "land surface")
print("replacements done", sorted(done), "of", len(REPL))
# split at the Supplementary References heading
body = d.element.body; kids = list(body.iterchildren())
idx = next(i for i, el in enumerate(kids) if el.tag == qn("w:p") and "".join(t.text or "" for t in el.iter(qn("w:t"))).strip() == "Supplementary References")
d.save(f"{B}/_si_full.docx")
A = docx.Document(f"{B}/_si_full.docx"); kids = list(A.element.body.iterchildren())
for el in kids[idx:]:
    if el.tag != qn("w:sectPr"): A.element.body.remove(el)
A.save(f"{B}/_si_a.docx")
Cd = docx.Document(f"{B}/_si_full.docx"); kids = list(Cd.element.body.iterchildren())
for el in kids[:idx]: Cd.element.body.remove(el)
Cd.save(f"{B}/_si_c.docx")
comp = Composer(docx.Document(f"{B}/_si_a.docx"))
new = docx.Document(f"{B}/si_ns_new.docx")
# style the new tables like the existing ones: full width, 8 pt
from docx.shared import Pt
from docx.oxml import OxmlElement
from docx.shared import Pt as _Pt
TW = 9026
def _el(tag, **a):
    e = OxmlElement(tag)
    for k, v in a.items(): e.set(qn("w:" + k), v)
    return e
def _font(run, size, bold=None):
    rPr = run._r.get_or_add_rPr()
    for f in rPr.findall(qn("w:rFonts")): rPr.remove(f)
    rPr.insert(0, _el("w:rFonts", ascii="Lucida Sans", hAnsi="Lucida Sans", cs="Lucida Sans"))
    run.font.size = _Pt(size)
    if bold is not None: run.font.bold = bold
for p in new.paragraphs:
    pPr = p._p.get_or_add_pPr()
    for ps in pPr.findall(qn("w:pStyle")): pPr.remove(ps)
    txt = p.text
    cap = txt.startswith("Supplementary Table") or txt.startswith("Supplementary Note") or txt.startswith("Supplementary Fig")
    for r in p.runs: _font(r, 10, True if txt.startswith("Supplementary Note") and "|" in txt[:30] else None)
    if txt.startswith("Supplementary Note") and "|" in txt[:30]:
        pPr.append(_el("w:keepNext"))
    if txt.startswith("Supplementary Table"): pPr.append(_el("w:keepNext"))
    pPr.append(_el("w:spacing", after="80", line="360", lineRule="auto"))
for t in new.tables:
    tbl = t._tbl; tblPr = tbl.tblPr
    for tag in ("w:tblStyle", "w:jc", "w:tblW", "w:tblBorders", "w:tblLayout"):
        for e in tblPr.findall(qn(tag)): tblPr.remove(e)
    tblPr.insert(0, _el("w:tblW", w=str(TW), type="dxa"))
    tblPr.append(_el("w:jc", val="center"))
    b = _el("w:tblBorders")
    b.append(_el("w:top", val="single", sz="6", space="0", color="000000"))
    b.append(_el("w:bottom", val="single", sz="6", space="0", color="000000"))
    tblPr.append(b)
    tblPr.append(_el("w:tblLayout", type="fixed"))
    mar = _el("w:tblCellMar")
    mar.append(_el("w:left", w="90", type="dxa")); mar.append(_el("w:right", w="90", type="dxa"))
    tblPr.append(mar)
    for lk in tblPr.findall(qn("w:tblLook")):
        tblPr.remove(lk); tblPr.append(lk)
    ncol = len(t.columns)
    # width weights: first column wider
    import re as _re
    def _hw(ci):
        tx = t.rows[0].cells[ci].text
        return max([len(w) for w in tx.split()] + [4])
    def _bw(ci):
        return max([min(len(r.cells[ci].text.strip()), 14) for r in list(t.rows)[1:]] + [4])
    floor = [100 * _hw(ci) + 260 for ci in range(ncol)]
    need = [max(floor[ci], 100 * _bw(ci) + 260) for ci in range(ncol)]
    if sum(need) <= TW:
        extra = TW - sum(need)
        ws = [int(n + extra * n / sum(need)) for n in need]
    else:
        slack = [n - f for n, f in zip(need, floor)]
        over = sum(need) - TW
        if sum(slack) >= over:
            ws = [int(n - over * sl / sum(slack)) for n, sl in zip(need, slack)]
        else:
            ws = [int(f * TW / sum(floor)) for f in floor]
    grid = tbl.tblGrid
    for gc, w in zip(grid.findall(qn("w:gridCol")), ws): gc.set(qn("w:w"), str(w))
    for ri, row in enumerate(t.rows):
        trPr = row._tr.get_or_add_trPr()
        if trPr.find(qn("w:cantSplit")) is None: trPr.insert(0, _el("w:cantSplit"))
        if ri == 0 and trPr.find(qn("w:tblHeader")) is None: trPr.append(_el("w:tblHeader"))
        for ci, c in enumerate(row.cells):
            tcPr = c._tc.get_or_add_tcPr()
            for e in tcPr.findall(qn("w:tcW")): tcPr.remove(e)
            tcPr.insert(0, _el("w:tcW", w=str(ws[ci]), type="dxa"))
            if ri == 0:
                tb = _el("w:tcBorders"); tb.append(_el("w:bottom", val="single", sz="4", space="0", color="000000")); tcPr.append(tb)
            for p in c.paragraphs:
                pPr = p._p.get_or_add_pPr()
                for ps in pPr.findall(qn("w:pStyle")): pPr.remove(ps)
                pPr.append(_el("w:spacing", before="20", after="20", line="240", lineRule="auto"))
                for r in p.runs: _font(r, 8)
for t in new.tables:
    nxt = t._tbl.getnext()
    if nxt is not None and nxt.tag == qn("w:p"):
        pPr = nxt.find(qn("w:pPr"))
        sp = pPr.find(qn("w:spacing")) if pPr is not None else None
        if sp is not None: sp.set(qn("w:before"), "80")
new.save(f"{B}/si_ns_new_fmt.docx")
comp.append(docx.Document(f"{B}/si_ns_new_fmt.docx"))
comp.append(docx.Document(f"{B}/_si_c.docx"))
comp.save(f"{B}/Supplementary_Information_NatureSustainability.docx")
print("saved")
