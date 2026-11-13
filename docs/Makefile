# Build the documentation: `make html` (notebooks executed) or `make quick` (not executed).
SPHINXBUILD ?= sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build

.PHONY: html quick clean

html:
	$(SPHINXBUILD) -W --keep-going -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"

quick:
	NB_EXECUTION_MODE=off $(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"

clean:
	rm -rf "$(BUILDDIR)" api/generated notebooks/_generated
