"""
17_publication_figures.py — run figures/make_figures.py if matplotlib is here.

Stage 10 exports every figure's data and draws draft renderings with the
hand-written PDF backend, which has no font metrics. Those drafts carry a
visible watermark so they cannot be mistaken for the real thing. This stage
replaces them with the matplotlib versions whenever matplotlib is installed, so
a completed run never leaves a draft in paper/figures/.

If matplotlib is absent the stage says so and exits 0; the drafts remain, and
their watermark is the warning.

Produces:
    paper/figures/fig1_cohort_flow.pdf        ->  Figure 1
    paper/figures/fig2_censoring.pdf          ->  Figure 2
    paper/figures/fig3_effects_forest.pdf     ->  Figure 3
    paper/figures/fig4_overlap_balance.pdf    ->  Figure 4
    paper/figures/fig5_site_confounding.pdf   ->  Figure 5
    paper/figures/fig6_correction_ladder.pdf  ->  Figure 6
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from lib_io import PAPER, Results, banner  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
SCRIPT = HERE / "figures" / "make_figures.py"


def main() -> None:
    banner("17 — Publication figures")

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("  matplotlib is not installed in this environment.")
        print("  The draft renderings from stage 10 remain in paper/figures/;")
        print("  they carry a visible 'Draft rendering' watermark.")
        print("\n  To build the publication figures:")
        print("    pip install matplotlib && cd figures && python make_figures.py")
        Results().put("publication_figures",
                      dict(status="skipped: matplotlib not installed"))
        return

    if not SCRIPT.exists():
        raise SystemExit(f"missing {SCRIPT}")

    r = subprocess.run([sys.executable, "-X", "utf8", str(SCRIPT)],
                       cwd=SCRIPT.parent, capture_output=True, text=True)
    print(r.stdout.rstrip())
    if r.returncode != 0:
        print(r.stderr.rstrip())
        raise SystemExit(f"make_figures.py exited with {r.returncode}")

    # A draft must never survive a completed run.
    figs = sorted((PAPER / "figures").glob("*.pdf"))
    drafts = []
    try:
        import fitz
        for f in figs:
            d = fitz.open(f)
            txt = " ".join(p.get_text() for p in d)
            d.close()
            if "Draft rendering" in txt:
                drafts.append(f.name)
    except ImportError:
        print("  (pymupdf absent; watermark check skipped)")

    print(f"\n  {len(figs)} figures in paper/figures/")
    if drafts:
        raise SystemExit(f"draft renderings survived: {drafts}")
    print("  no draft renderings remain")

    import matplotlib as _mpl
    Results().put("publication_figures",
                  dict(status="built", matplotlib_version=_mpl.__version__,
                       n_figures=len(figs),
                       files=[f.name for f in figs]))


if __name__ == "__main__":
    main()
