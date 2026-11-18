"""Revised SI: the original SI document with new Supplementary Notes 9-14 inserted after Note 8 and new Tables 13-17 and
Supplementary Figs 1-3 inserted before the references. New material is generated with pandoc using the original SI as the style
reference, restyled to match the original tables (Lucida Sans 8 pt, rules above and below the header and at the bottom)."""
import subprocess, copy, re, sys, os
import docx
from docx.shared import Pt, Cm
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docxcompose.composer import Composer
ROOT = "/home/claude/n84"; B = f"{ROOT}/build"; ORIG = sys.argv[1]
sys.path.insert(0, f"{ROOT}/src")
import json
TOK = json.load(open(f"{ROOT}/data/out/tokens.json"))
fill = lambda s: re.sub(r"⟦([A-Z0-9_]+)⟧", lambda m: TOK[m.group(1)], s)
open(f"{B}/si_notes.md", "w").write(fill(open(f"{ROOT}/txt/si_new.md").read()))
open(f"{B}/si_tabs.md", "w").write(fill(open(f"{ROOT}/txt/si_tables.md").read()) + "\n\n" + open(f"{ROOT}/txt/si_figs.md").read())
for nm in ("si_notes", "si_tabs"):
    subprocess.run(["pandoc", f"{B}/{nm}.md", "-o", f"{B}/{nm}.docx", f"--reference-doc={ORIG}", "--resource-path", ROOT, "--columns=2000"], check=True)

orig = docx.Document(ORIG)
def _txt(el): return "".join(t.text or "" for t in el.iter(qn("w:t")))
_body = list(orig.element.body.iterchildren()); TPL = {}
for i, el in enumerate(_body):
    if el.tag != qn("w:p"): continue
    tx = _txt(el)
    if tx.startswith("The 84 city-regions comprise"): TPL["body"] = el
    if tx.startswith("Supplementary Table 12 |"): TPL["cap"] = el
    if tx.startswith("Medians over city-region"): TPL["leg"] = el
def apply(p_el, kind):
    t = TPL[kind]; tp = t.find(qn("w:pPr")); trun = t.find(qn("w:r")); tr = trun.find(qn("w:rPr")) if trun is not None else None
    old = p_el.find(qn("w:pPr"))
    if old is not None: p_el.remove(old)
    if tp is not None: p_el.insert(0, copy.deepcopy(tp))
    for r in p_el.iter(qn("w:r")):
        orp = r.find(qn("w:rPr")); keep = []
        if orp is not None:
            keep = [copy.deepcopy(e) for e in orp if e.tag in (qn("w:b"), qn("w:i"), qn("w:vertAlign"))]
            r.remove(orp)
        if tr is not None:
            nr = copy.deepcopy(tr)
            for e in list(nr):
                if e.tag in (qn("w:b"), qn("w:i"), qn("w:vertAlign"), qn("w:bCs"), qn("w:iCs")): nr.remove(e)
            for e in keep: nr.append(e)
            r.insert(0, nr)


def restyle(path):
    d = docx.Document(path)
    els = list(d.element.body.iterchildren()); prev = None
    for el in els:
        if el.tag == qn("w:p"):
            p = docx.text.paragraph.Paragraph(el, d)
            tx = p.text
            if p.style.name.startswith("Heading"):
                pass
            elif el.find(".//" + qn("w:drawing")) is not None:
                p.style = d.styles["Normal"]
            elif tx.startswith("Supplementary Table") or tx.startswith("Supplementary Fig") or tx.startswith("Supplementary Figures"):
                p.style = d.styles["Normal"]; apply(el, "cap")
            elif prev == "tbl" or prev == "leg":
                p.style = d.styles["Normal"]; apply(el, "leg"); prev = "leg"; continue
            else:
                p.style = d.styles["Normal"]; apply(el, "body")
            prev = "p"
        elif el.tag == qn("w:tbl"):
            prev = "tbl"
    for t in d.tables:
        t.autofit = True
        tblPr = t._tbl.tblPr
        for el in tblPr.findall(qn("w:tblStyle")): tblPr.remove(el)
        w = tblPr.find(qn("w:tblW"))
        if w is None: w = OxmlElement("w:tblW"); tblPr.append(w)
        w.set(qn("w:type"), "pct"); w.set(qn("w:w"), "5000")
        b = OxmlElement("w:tblBorders")
        for side in ("top", "bottom"):
            e = OxmlElement(f"w:{side}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "6"); e.set(qn("w:color"), "000000"); b.append(e)
        for el in tblPr.findall(qn("w:tblBorders")): tblPr.remove(el)
        tblPr.append(b)
        for ri, row in enumerate(t.rows):
            for c in row.cells:
                if ri == 0:
                    tcPr = c._tc.get_or_add_tcPr(); bd = OxmlElement("w:tcBorders"); e = OxmlElement("w:bottom")
                    e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:color"), "000000"); bd.append(e); tcPr.append(bd)
                for p in c.paragraphs:
                    p.style = d.styles["Normal"]; pf = p.paragraph_format; pf.space_before = Pt(1); pf.space_after = Pt(1); pf.line_spacing = 1.0
                    for r in p.runs:
                        r.font.size = Pt(8); r.font.name = "Lucida Sans"
                        if ri == 0: r.font.bold = False
    d.save(path)


restyle(f"{B}/si_notes.docx"); restyle(f"{B}/si_tabs.docx")
# title
for p in orig.paragraphs[:3]:
    if p.text.startswith("Short-range surface cooling"):
        for r in p.runs[1:]: r.text = ""
        p.runs[0].text = "Tree canopy cools urban land surfaces within about 300 metres across climates and decades"
body = orig.element.body
def idx_of(prefix):
    for i, el in enumerate(body.iterchildren()):
        if el.tag == qn("w:p"):
            txt = "".join(t.text or "" for t in el.iter(qn("w:t")))
            if txt.startswith(prefix): return i
    raise KeyError(prefix)
comp = Composer(orig)
comp.insert(idx_of("Supplementary References"), docx.Document(f"{B}/si_tabs.docx"))
comp.insert(idx_of("Supplementary Table 1 |"), docx.Document(f"{B}/si_notes.docx"))
comp.save(f"{B}/Supplementary_Information_revised.docx")
print("ok")
