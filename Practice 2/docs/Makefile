# Compilación de los reportes
DOC_ES=report_es
DOC_EN=report_en

all: pdf

pdf:
	pdflatex -interaction=nonstopmode $(DOC_ES).tex
	pdflatex -interaction=nonstopmode $(DOC_ES).tex
	pdflatex -interaction=nonstopmode $(DOC_EN).tex
	pdflatex -interaction=nonstopmode $(DOC_EN).tex

clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
