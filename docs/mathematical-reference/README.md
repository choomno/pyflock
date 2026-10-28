# Mathematical reference

Read the [compiled reference](mathematical-reference.pdf) or its
[LaTeX source](main.tex). It defines the model and boundaries, derives the
local spectral coefficients, and states the hypotheses behind the path-wave
and heterogeneous comparisons.

From the repository root, run `python ci/build_docs.py`. The build needs
`pdflatex` with the standard AMS, Latin Modern, geometry, hyperref and microtype
packages. Auxiliary files stay in a temporary directory. The helper rejects
unresolved references and overfull boxes before replacing the PDF.
CI compiles the same source and retains the result as a workflow artifact.

The build uses the source revision's timestamp when Git history is available,
or a fixed epoch for a source archive. `SOURCE_DATE_EPOCH` overrides that value.
Different TeX or font versions can
still produce different bytes; mathematical content is reviewed from the source
and the rendered pages.
