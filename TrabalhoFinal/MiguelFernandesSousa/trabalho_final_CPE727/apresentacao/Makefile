# Makefile for LaTeX documents
# Compiler
LATEX = pdflatex
LATEX_FLAGS = -interaction=nonstopmode -halt-on-error

# Directories
RELATORIO_DIR = relatorio
SLIDE_DIR = slide

# Source files
RELATORIO_TEX = $(RELATORIO_DIR)/RELATORIO.tex
RELATORIO_MAIN_TEX = $(RELATORIO_DIR)/main.tex
SLIDE_MAIN_TEX = $(SLIDE_DIR)/main.tex

# PDF outputs
RELATORIO_PDF = $(RELATORIO_DIR)/RELATORIO.pdf
RELATORIO_MAIN_PDF = $(RELATORIO_DIR)/main.pdf
SLIDE_MAIN_PDF = $(SLIDE_DIR)/main.pdf

.PHONY: all clean relatorio-RELATORIO relatorio-main slide-main

# Default target
all: relatorio-RELATORIO slide-main

# Compile relatorio/RELATORIO.tex
relatorio-RELATORIO:
	cd $(RELATORIO_DIR) && $(LATEX) $(LATEX_FLAGS) RELATORIO.tex
	cd $(RELATORIO_DIR) && $(LATEX) $(LATEX_FLAGS) RELATORIO.tex

# Compile relatorio/main.tex
relatorio-main:
	cd $(RELATORIO_DIR) && $(LATEX) $(LATEX_FLAGS) main.tex
	cd $(RELATORIO_DIR) && $(LATEX) $(LATEX_FLAGS) main.tex

# Compile slide/main.tex
slide-main:
	cd $(SLIDE_DIR) && $(LATEX) $(LATEX_FLAGS) main.tex
	cd $(SLIDE_DIR) && $(LATEX) $(LATEX_FLAGS) main.tex

# Clean auxiliary files
clean:
	find . -type f \( -name "*.aux" -o -name "*.log" -o -name "*.out" -o -name "*.toc" -o -name "*.nav" -o -name "*.snm" -o -name "*.vrb" -o -name "*.fls" -o -name "*.fdb_latexmk" -o -name "*.synctex.gz" \) -delete

# Clean everything including PDFs
clean-all: clean
	find . -type f -name "*.pdf" -delete

