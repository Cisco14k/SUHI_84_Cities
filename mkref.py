"""Reference docx: Times New Roman 12 pt, 1.5 line spacing, headings in Arial, Box style with border and shading, line numbering."""
import docx, sys
from docx.shared import Pt, RGBColor, Cm
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
d = docx.Document(sys.argv[1])
STY = {s.name: s for s in d.styles}
class _S(dict):
    def __getitem__(self, k): return STY[k]

def font(st, name, size, bold=None, color=None):
    st.font.name = name; st.font.size = Pt(size)
    rpr = st.element.get_or_add_rPr(); rf = rpr.find(qn("w:rFonts"))
    if rf is None: rf = OxmlElement("w:rFonts"); rpr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"): rf.set(qn(a), name)
    if bold is not None: st.font.bold = bold
    if color: st.font.color.rgb = RGBColor.from_string(color)
for nm in ("Normal", "Body Text", "First Paragraph", "Compact", "Abstract", "Bibliography", "Block Text"):
    try: st = STY[nm]
    except KeyError: continue
    font(st, "Times New Roman", 12); pf = st.paragraph_format; pf.line_spacing = 1.5; pf.space_after = Pt(6); pf.space_before = Pt(0)
for nm, sz in (("Title", 16), ("Heading 1", 14), ("Heading 2", 12), ("Heading 3", 12)):
    st = STY[nm]; font(st, "Arial", sz, True, "000000"); st.paragraph_format.space_before = Pt(12); st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.keep_with_next = True
STY["Heading 3"].font.italic = True
for nm in ("Caption", "Image Caption", "Table Caption"):
    try: st = STY[nm]; font(st, "Times New Roman", 11); st.font.italic = False; st.paragraph_format.line_spacing = 1.15
    except KeyError: pass
# Box style
try: bs = STY["Box"]
except KeyError: bs = d.styles.add_style("Box", WD_STYLE_TYPE.PARAGRAPH); bs.base_style = STY["Normal"]; STY["Box"] = bs
font(bs, "Arial", 10); bs.paragraph_format.line_spacing = 1.15; bs.paragraph_format.space_after = Pt(4)
bs.paragraph_format.left_indent = Cm(0.3); bs.paragraph_format.right_indent = Cm(0.3)
ppr = bs.element.get_or_add_pPr()
bdr = OxmlElement("w:pBdr")
for side in ("top", "left", "bottom", "right"):
    e = OxmlElement(f"w:{side}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "6"); e.set(qn("w:space"), "4"); e.set(qn("w:color"), "3C5488"); bdr.append(e)
ppr.append(bdr)
shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "EEF3F8"); ppr.append(shd)
# table style font
try: font(STY["Table"], "Arial", 9)
except KeyError: pass
d.save(sys.argv[2])
