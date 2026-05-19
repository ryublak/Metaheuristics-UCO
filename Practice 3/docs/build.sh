#!/usr/bin/env bash
# Compile the LaTeX report and clean up auxiliary files.
# Usage:  bash build.sh

set -euo pipefail

cd "$(dirname "$0")"

echo "==> Compiling report_en.tex ..."
pdflatex -interaction=nonstopmode report_en.tex
pdflatex -interaction=nonstopmode report_en.tex

echo "==> Cleaning auxiliary files ..."
rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz

echo "==> Done: report_en.pdf"
