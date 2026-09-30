"""
lib_plot.py — a minimal vector plotting backend that emits PDF directly.

matplotlib is not available in this environment. Rather than ship no figures, or
raster ones, this writes PDF content streams by hand using the base-14 Helvetica
fonts, so no font embedding and no external dependency is needed. pdflatex takes
the output directly via \\includegraphics.

Only what the report needs is implemented: axes with ticks, lines, step curves,
bars, points with error bars, and text. Coordinates are in points (1/72 inch).
"""

from __future__ import annotations

import zlib
from pathlib import Path

# Helvetica advance widths (units/1000 em) for the printable ASCII range.
_W = {
    " ": 278, "!": 278, '"': 355, "#": 556, "$": 556, "%": 889, "&": 667,
    "'": 191, "(": 333, ")": 333, "*": 389, "+": 584, ",": 278, "-": 333,
    ".": 278, "/": 278, "0": 556, "1": 556, "2": 556, "3": 556, "4": 556,
    "5": 556, "6": 556, "7": 556, "8": 556, "9": 556, ":": 278, ";": 278,
    "<": 584, "=": 584, ">": 584, "?": 556, "@": 1015, "A": 667, "B": 667,
    "C": 722, "D": 722, "E": 667, "F": 611, "G": 778, "H": 722, "I": 278,
    "J": 500, "K": 667, "L": 556, "M": 833, "N": 722, "O": 778, "P": 667,
    "Q": 778, "R": 722, "S": 667, "T": 611, "U": 722, "V": 667, "W": 944,
    "X": 667, "Y": 667, "Z": 611, "[": 278, "\\": 278, "]": 278, "^": 469,
    "_": 556, "`": 333, "a": 556, "b": 556, "c": 500, "d": 556, "e": 556,
    "f": 278, "g": 556, "h": 556, "i": 222, "j": 222, "k": 500, "l": 222,
    "m": 833, "n": 556, "o": 556, "p": 556, "q": 556, "r": 333, "s": 500,
    "t": 278, "u": 556, "v": 500, "w": 722, "x": 500, "y": 500, "z": 500,
    "{": 334, "|": 260, "}": 334, "~": 584,
}


def text_width(s: str, size: float, bold: bool = False) -> float:
    w = sum(_W.get(c, 556) for c in str(s))
    return w / 1000.0 * size * (1.03 if bold else 1.0)


def _esc(s: str) -> str:
    return (str(s).replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)"))


class Canvas:
    """A single-page PDF drawing surface."""

    def __init__(self, width=468, height=300):
        self.w, self.h = float(width), float(height)
        self.ops: list[str] = []

    # ── primitives ───────────────────────────────────────────────────────────
    def _c(self, rgb):
        r, g, b = rgb
        return f"{r:.3f} {g:.3f} {b:.3f}"

    def line(self, x1, y1, x2, y2, w=0.8, color=(0, 0, 0), dash=None):
        d = f"[{dash}] 0 d " if dash else "[] 0 d "
        self.ops.append(
            f"q {d}{w:.2f} w {self._c(color)} RG "
            f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S Q")

    def polyline(self, pts, w=1.2, color=(0, 0, 0), dash=None):
        if len(pts) < 2:
            return
        d = f"[{dash}] 0 d " if dash else "[] 0 d "
        p = f"{pts[0][0]:.2f} {pts[0][1]:.2f} m " + " ".join(
            f"{x:.2f} {y:.2f} l" for x, y in pts[1:])
        self.ops.append(f"q {d}{w:.2f} w {self._c(color)} RG {p} S Q")

    def rect(self, x, y, w, h, fill=None, stroke=None, lw=0.8, alpha=1.0):
        seg = ["q"]
        if alpha < 1.0:
            seg.append(f"/GS{int(alpha * 100)} gs")
        if fill:
            seg.append(f"{self._c(fill)} rg")
        if stroke:
            seg.append(f"{self._c(stroke)} RG {lw:.2f} w")
        seg.append(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re")
        seg.append("B" if (fill and stroke) else ("f" if fill else "S"))
        seg.append("Q")
        self.ops.append(" ".join(seg))

    def circle(self, cx, cy, r, fill=(0, 0, 0), stroke=None, lw=0.8):
        k = 0.5523 * r
        p = (f"{cx + r:.2f} {cy:.2f} m "
             f"{cx + r:.2f} {cy + k:.2f} {cx + k:.2f} {cy + r:.2f} {cx:.2f} {cy + r:.2f} c "
             f"{cx - k:.2f} {cy + r:.2f} {cx - r:.2f} {cy + k:.2f} {cx - r:.2f} {cy:.2f} c "
             f"{cx - r:.2f} {cy - k:.2f} {cx - k:.2f} {cy - r:.2f} {cx:.2f} {cy - r:.2f} c "
             f"{cx + k:.2f} {cy - r:.2f} {cx + r:.2f} {cy - k:.2f} {cx + r:.2f} {cy:.2f} c")
        seg = ["q"]
        if fill:
            seg.append(f"{self._c(fill)} rg")
        if stroke:
            seg.append(f"{self._c(stroke)} RG {lw:.2f} w")
        seg.append(p)
        seg.append("B" if (fill and stroke) else ("f" if fill else "S"))
        seg.append("Q")
        self.ops.append(" ".join(seg))

    def text(self, x, y, s, size=8, color=(0, 0, 0), bold=False,
             anchor="start", rotate=0):
        font = "F2" if bold else "F1"
        w = text_width(s, size, bold)
        if anchor == "middle":
            x -= w / 2
        elif anchor == "end":
            x -= w
        if rotate:
            import math
            a = math.radians(rotate)
            ca, sa = math.cos(a), math.sin(a)
            tm = f"{ca:.4f} {sa:.4f} {-sa:.4f} {ca:.4f} {x:.2f} {y:.2f} Tm"
        else:
            tm = f"1 0 0 1 {x:.2f} {y:.2f} Tm"
        self.ops.append(
            f"q BT /{font} {size:.1f} Tf {self._c(color)} rg {tm} "
            f"({_esc(s)}) Tj ET Q")

    # ── output ───────────────────────────────────────────────────────────────
    def save(self, path: Path) -> Path:
        path = Path(path)
        content = "\n".join(self.ops).encode("latin-1", "replace")
        comp = zlib.compress(content)

        ext = " ".join(f"/GS{i} << /ca {i / 100:.2f} /CA {i / 100:.2f} >>"
                       for i in (20, 30, 40, 50, 60, 70, 80, 90))
        objs = [
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.w:.1f} {self.h:.1f}] "
             f"/Resources << /Font << /F1 5 0 R /F2 6 0 R >> "
             f"/ExtGState << {ext} >> >> /Contents 4 0 R >>"),
            None,  # stream
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold "
            "/Encoding /WinAnsiEncoding >>",
        ]

        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for i, o in enumerate(objs, start=1):
            offsets.append(len(out))
            if o is None:
                out += (f"{i} 0 obj\n<< /Length {len(comp)} /Filter /FlateDecode >>\n"
                        "stream\n").encode()
                out += comp + b"\nendstream\nendobj\n"
            else:
                out += f"{i} 0 obj\n{o}\nendobj\n".encode()

        xref = len(out)
        out += f"xref\n0 {len(objs) + 1}\n".encode()
        out += b"0000000000 65535 f \n"
        for off in offsets[1:]:
            out += f"{off:010d} 00000 n \n".encode()
        out += (f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
                f"startxref\n{xref}\n%%EOF\n").encode()

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(out))
        return path


class Axes:
    """A linear x/y plotting region inside a Canvas."""

    GRID = (0.88, 0.88, 0.90)
    AXIS = (0.25, 0.25, 0.28)
    TEXT = (0.15, 0.15, 0.18)

    def __init__(self, cv: Canvas, x, y, w, h, xlim, ylim,
                 xlabel="", ylabel="", title=""):
        self.cv, self.x, self.y, self.w, self.h = cv, x, y, w, h
        self.x0, self.x1 = xlim
        self.y0, self.y1 = ylim
        if self.x1 == self.x0:
            self.x1 = self.x0 + 1
        if self.y1 == self.y0:
            self.y1 = self.y0 + 1
        self.xlabel, self.ylabel, self.title = xlabel, ylabel, title

    def px(self, v):
        return self.x + (float(v) - self.x0) / (self.x1 - self.x0) * self.w

    def py(self, v):
        return self.y + (float(v) - self.y0) / (self.y1 - self.y0) * self.h

    def frame(self, xticks=None, yticks=None, xfmt="{:g}", yfmt="{:g}",
              grid=True, xtick_labels=None, xrot=0):
        cv = self.cv
        if grid and yticks is not None:
            for t in yticks:
                cv.line(self.x, self.py(t), self.x + self.w, self.py(t),
                        w=0.4, color=self.GRID)
        cv.line(self.x, self.y, self.x + self.w, self.y, w=0.8, color=self.AXIS)
        cv.line(self.x, self.y, self.x, self.y + self.h, w=0.8, color=self.AXIS)
        if yticks is not None:
            for t in yticks:
                cv.line(self.x - 3, self.py(t), self.x, self.py(t), w=0.7,
                        color=self.AXIS)
                cv.text(self.x - 5, self.py(t) - 2.5, yfmt.format(t), size=7,
                        anchor="end", color=self.TEXT)
        if xticks is not None:
            labs = xtick_labels or [xfmt.format(t) for t in xticks]
            for t, lab in zip(xticks, labs):
                cv.line(self.px(t), self.y - 3, self.px(t), self.y, w=0.7,
                        color=self.AXIS)
                if xrot:
                    cv.text(self.px(t) + 2, self.y - 8, lab, size=6.5,
                            anchor="end", color=self.TEXT, rotate=xrot)
                else:
                    cv.text(self.px(t), self.y - 11, lab, size=7,
                            anchor="middle", color=self.TEXT)
        if self.xlabel:
            cv.text(self.x + self.w / 2, self.y - 24, self.xlabel, size=8,
                    anchor="middle", color=self.TEXT)
        if self.ylabel:
            cv.text(self.x - 32, self.y + self.h / 2, self.ylabel, size=8,
                    anchor="middle", color=self.TEXT, rotate=90)
        if self.title:
            cv.text(self.x, self.y + self.h + 8, self.title, size=8.5,
                    bold=True, color=self.TEXT)

    def step(self, xs, ys, color=(0.2, 0.35, 0.6), w=1.3, dash=None):
        pts = []
        for i in range(len(xs)):
            if i:
                pts.append((self.px(xs[i]), self.py(ys[i - 1])))
            pts.append((self.px(xs[i]), self.py(ys[i])))
        self.cv.polyline(pts, w=w, color=color, dash=dash)

    def hline(self, v, color=(0.5, 0.5, 0.55), w=0.7, dash="2 2"):
        self.cv.line(self.x, self.py(v), self.x + self.w, self.py(v),
                     w=w, color=color, dash=dash)

    def errbar_h(self, y, lo, hi, est, color=(0.2, 0.35, 0.6), r=2.2):
        lo = max(lo, self.x0); hi = min(hi, self.x1)
        yy = self.py(y)
        self.cv.line(self.px(lo), yy, self.px(hi), yy, w=1.1, color=color)
        for e in (lo, hi):
            self.cv.line(self.px(e), yy - 2.2, self.px(e), yy + 2.2,
                         w=1.1, color=color)
        self.cv.circle(self.px(max(min(est, self.x1), self.x0)), yy, r,
                       fill=color)


PALETTE = {
    "blue": (0.20, 0.36, 0.60),
    "orange": (0.85, 0.52, 0.20),
    "red": (0.72, 0.24, 0.24),
    "green": (0.25, 0.55, 0.38),
    "grey": (0.45, 0.45, 0.50),
    "purple": (0.45, 0.34, 0.60),
    "light": (0.80, 0.84, 0.90),
}


def nice_ticks(lo, hi, n=5):
    """Human-readable tick positions covering [lo, hi]."""
    import math
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / max(n, 2)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    for m in (1, 2, 2.5, 5, 10):
        if raw / mag <= m:
            step = m * mag
            break
    else:
        step = 10 * mag
    start = math.floor(lo / step) * step
    ticks, t = [], start
    while t <= hi + step * 1e-9:
        if t >= lo - step * 1e-9:
            ticks.append(round(t, 10))
        t += step
    return ticks
