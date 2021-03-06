# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =


# Internal variables.
BUILDROOT       = ../build
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDROOT)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .
.PHONY: help clean html web pickle htmlhelp latex changes linkcheck

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html      to make standalone HTML files"
	@echo "  pickle    to make pickle files (usable by e.g. sphinx-web)"
	@echo "  htmlhelp  to make HTML files and a HTML help project"
	@echo "  latex     to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  changes   to make an overview over all changed/added/deprecated items"
	@echo "  linkcheck to check all external links for integrity"

clean:
	-rm -rf $(BUILDROOT)/html
	-rm -rf $(BUILDROOT)/latex

distclean: clean
	-rm -rf $(BUILDROOT)/doctrees

html:
	mkdir -p $(BUILDROOT)/html $(BUILDROOT)/doctrees
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDROOT)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDROOT)/html."

pickle:
	mkdir -p $(BUILDROOT)/pickle $(BUILDROOT)/doctrees
	$(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) $(BUILDROOT)/pickle
	@echo
	@echo "Build finished; now you can process the pickle files or run"
	@echo "  sphinx-web $(BUILDROOT)/pickle"
	@echo "to start the sphinx-web server."

web: pickle

htmlhelp:
	mkdir -p $(BUILDROOT)/htmlhelp $(BUILDROOT)/doctrees
	$(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) $(BUILDROOT)/htmlhelp
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in $(BUILDROOT)/htmlhelp."

latex:
	mkdir -p $(BUILDROOT)/latex $(BUILDROOT)/doctrees
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDROOT)/latex
	@echo
	@echo "Build finished; the LaTeX files are in $(BUILDROOT)/latex."
	@echo "Run \`make all-pdf' or \`make all-ps' in that directory to" \
	      "run these through (pdf)latex."

changes:
	mkdir -p $(BUILDROOT)/changes $(BUILDROOT)/doctrees
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) $(BUILDROOT)/changes
	@echo
	@echo "The overview file is in $(BUILDROOT)/changes."

linkcheck:
	mkdir -p $(BUILDROOT)/linkcheck $(BUILDROOT)/doctrees
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(BUILDROOT)/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDROOT)/linkcheck/output.txt."
