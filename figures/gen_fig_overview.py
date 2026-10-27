"""Figure 2: overview of the method, drawn in TikZ (figures/fig_overview.tex) with the fonts of the paper.

    python -m figures.gen_fig_overview

Compiles the TikZ source with latexmk in a temporary directory and writes figures/fig_overview.pdf (no
creation date, so an unchanged source gives the same file), then figures/fig_overview.png at 300 dpi with
Ghostscript when it is installed. Needs a TeX installation with TikZ; the numbers in the drawing are the
paper's (Sections 2 to 5).
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

from figures.style import OUT


def main():
    latexmk = shutil.which("latexmk") or "/Library/TeX/texbin/latexmk"
    src, pdf = OUT / "fig_overview.tex", OUT / "fig_overview.pdf"
    with tempfile.TemporaryDirectory() as tmp:
        run = subprocess.run([latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"-outdir={tmp}",
                              str(src)], capture_output=True, text=True)
        if run.returncode:
            raise SystemExit(run.stdout[-3000:])
        shutil.copy(Path(tmp) / "fig_overview.pdf", pdf)
    gs = shutil.which("gs")
    if gs:
        subprocess.run([gs, "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png16m", "-r300", "-dTextAlphaBits=4",
                        "-dGraphicsAlphaBits=4", f"-sOutputFile={OUT / 'fig_overview.png'}", str(pdf)], check=True)
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
