# Sources and provenance

This reconstruction grew from Choomno Moos's flocking project. Its source
archive contains papers, drafts, MATLAB and C++ simulations, handwritten
calculations, and later Python experiments. The archive filenames below
identify that material; the PDFs are not required to run this repository.
Page references use the archived manuscript pagination.

## Mathematical basis

- **C. E. Cantos, J. J. P. Veerman, D. K. Hammond,
  “Signal velocity in oscillator arrays,”** *European Physical Journal
  Special Topics* **225**, 1115–1126 (2016).
  [DOI](https://doi.org/10.1140/epjst/e2016-02665-0).
  Archive: `signalvelocity.pdf`. Theorem 1 (p. 5) characterizes stability
  for the nearest-neighbour circle family. Theorem 2 (p. 10) establishes
  the two travelling waves under stated decay assumptions on the initial
  Fourier coefficients. The point-impulse illustration following it is
  numerical evidence outside those assumptions.

- **C. E. Cantos, D. K. Hammond, J. J. P. Veerman,
  “Transients in the synchronization of asymmetrically coupled oscillator arrays,”** *European
  Physical Journal Special Topics* **225**, 1199–1209 (2016).
  [DOI](https://doi.org/10.1140/epjst/e2016-02658-y).
  Archive: `Transients.pdf`. Definition 1 (p. 4) gives the
  two terminal boundary conditions. Conjectures 1–3 (p. 6) connect circle
  dynamics to the path and prescribe effective reflection conditions.
  Theorem 3 (p. 8), derived using those conjectures, gives the leading tail
  amplitudes and timing. The parameter-family experiment in §4 tests their
  convergence as the number of followers grows.

- **J. Herbrych, A. G. Chazirakis, N. Christakis, J. J. P. Veerman,
  “Dynamics of locally coupled agents with next nearest neighbor
  interaction,”** *Differential Equations and Dynamical Systems* **29**,
  487–509 (2021; published online 2017).
  [DOI](https://doi.org/10.1007/s12591-017-0377-3).
  Archive: `NNN-agents.pdf`. Proposition 4.1 (pp. 11–12) gives the local
  spectral expansion; Theorem 5.1 (pp. 12–13) treats waves on the circle.
  The path classifications, including reflectionless motion, use the
  circle-to-path conjectures (§3): Theorems 5.2 and 5.3 (pp. 14, 16).

- **F. M. Tangerman, J. J. P. Veerman, B. D. Stošić,
  “Asymmetric decentralized flocks,”** *IEEE Transactions on Automatic
  Control* **57**(11), 2844–2853 (2012).
  Archive: `flocks_final.pdf`. This earlier model uses the same Laplacian
  for position and velocity. Theorems 2.3, 3.5, 4.8 and 4.9 distinguish
  finite-system stability from amplification growing exponentially with
  system size. Its asymmetry conclusions belong to that model class.

## Project context and open calculations

**Choomno Moos and J. J. P. Veerman, *Flocks, Digraphs, and Oscillator
Arrays*, April 4, 2022** (`mcnair_paper.pdf`) is the project's expository
entry point: graph Laplacians, circulants, and nearest-neighbour circle
stability (§§4–7). Its bibliography supplies the signal-velocity DOI above.

**J. J. P. Veerman, *Some Flocking Ideas*, December 14, 2021**
(`Flocks-etc.pdf`) records the research agenda: test the conjectures beyond
symmetry, improve the amplitude approximation, and compare discrete and
continuous models. These are questions motivating experiments.

**De Dios, He Hao and J. P. P. Veerman, *Controlling systems of decentralized
oscillators with Symmetric spacial feedback and variable velocity
feedback*** (`DeDios2/DeDios.pdf`, `DeDios2.pdf` and handwritten notes) are
unpublished working calculations. The typed draft has inconsistent endpoint
matrices: on p. 2 the position and velocity terminal rows differ even at
the purported common unperturbed parameter. Its displayed Laplacian
eigenvalue also misses a factor of two. Small-asymmetry stability estimates
in the previous Python implementation therefore require an independent
derivation and numerical study; they are exploratory formulas.

For nonnormal transient growth, the archive includes **L. N. Trefethen,
“Pseudospectra of linear operators,”** *SIAM Review* **39**(3), 383–406
(1997), and **L. N. Trefethen and M. Embree, *Spectra and Pseudospectra:
The Behavior of Nonnormal Matrices and Operators*** (Princeton, 2005).
They explain why a stable spectrum can coexist with large transient growth
and why computed eigenvalues may be sensitive to perturbations.

The original programs record parameter choices and implementation history.
Their boundary rules and numerical outputs are comparison material; the
equations and independent numerical checks determine the reconstruction.
