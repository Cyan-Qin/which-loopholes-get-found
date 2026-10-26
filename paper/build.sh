#!/usr/bin/env bash
# Build the paper locally: bash paper/build.sh  ->  two PDFs, intermediate files in paper/build/:
#   paper/main-review.pdf    the anonymous submission with line numbers (acl.sty "review", what Overleaf builds)
#   paper/main-preprint.pdf  the reading version: authors shown, no line numbers (acl.sty "preprint")
#
# Why a script: TeX Live 2026 ships lineno v5.6 or later. When the LaTeX format is 2025-06-01 or later, its new
# twocolumn code records the column through the build/column/before hook, and with acl.sty's review mode the
# right column's line numbers then land inside the left column. Overleaf is not affected. For local builds this
# script compiles against a copy of the installed lineno.sty without that branch, so lineno keeps its earlier
# \@LN@makecol path; the copy lives in a temporary directory and no file of the paper changes.
set -euo pipefail
cd "$(dirname "$0")"
OUT=${OUT:-build}
LATEXMK=$(command -v latexmk || echo /Library/TeX/texbin/latexmk)
KPSEWHICH=$(command -v kpsewhich || echo /Library/TeX/texbin/kpsewhich)
mkdir -p "$OUT"
FIX=$(mktemp -d)
trap 'rm -rf "$FIX"' EXIT

python3 - "$("$KPSEWHICH" lineno.sty)" "$FIX/lineno.sty" <<'EOF'
import sys
src = open(sys.argv[1]).read()
marker = "\\ifdefined\\IfFormatAtLeastTF\n   \\IfFormatAtLeastTF{2025-06-01}{"
if marker in src:
    start = src.index(marker)
    end = src.index("\\fi\n", start) + 3
    assert "build/column/before" in src[start:end], "unexpected lineno.sty layout"
    src = src[:start] + "% local build: earlier \\@LN@makecol path (see paper/build.sh)\n" + src[end:]
    print("lineno.sty: using the pre-v5.6 twocolumn path for this build")
open(sys.argv[2], "w").write(src)
EOF

build() {  # jobname, then optional TeX code read before main.tex
  local args=(-pdf -interaction=nonstopmode -halt-on-error -outdir="$OUT" -jobname="$1")
  if [ -n "${2:-}" ]; then args+=(-usepretex="$2"); fi
  TEXINPUTS="$FIX:" "$LATEXMK" "${args[@]}" main.tex
  cp "$OUT/$1.pdf" "$1.pdf"
  echo "wrote paper/$1.pdf"
}
build main-review
build main-preprint '\def\aclpreprint{}'
