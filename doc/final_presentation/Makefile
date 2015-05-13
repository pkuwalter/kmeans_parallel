PROJ_NAME=presentation
MAIN_LATEX=$(PROJ_NAME).tex
OUT_PDF=$(PROJ_NAME).pdf
OUT_PS=$(patsubst %.pdf,%.ps,$(OUT_PDF))
OUT_DVI=$(patsubst %.pdf,%.dvi,$(OUT_PDF))
RUBBER=rubber -vv -pd

DEPS=*.tex *.bib

all: $(OUT_PDF)

$(OUT_PDF): $(DEPS)
	$(RUBBER) $(PROJ_NAME)
	@grep --color undefined $(PROJ_NAME).log && notify-send --urgency=critical "WARNING: UNDEFINED REF" || true

clean:
	rubber -v -pd --clean $(PROJ_NAME)
