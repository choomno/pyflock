# Experiment coverage

The six experiments were selected to exercise propagation, reflection, boundary
sensitivity, the local spectral expansion, and finite-size approximation error.
They form an initial demonstration and validation suite. They do not reproduce
all experiments in the source papers. Selection was qualitative; no systematic
paper-by-paper coverage criterion was used in the first implementation.

Here, **adapted** means that a source experiment supplied parameters or a question,
but its size, initialization, observable, or measurement procedure changed.
**New diagnostic** means a calculation motivated by the theory rather than a
reconstruction of a published figure. Figure references below use the archived
manuscripts listed in [Sources](sources.md).

| Source experiment | Repository status | Material differences |
| --- | --- | --- |
| *Transients*, Fig. 1: trajectories and definitions of extrema | Adapted: `propagation` | Different parameters; spacing field and selected trajectories; schematic omitted. |
| *Transients*, Fig. 3: velocity-asymmetry comparison | Adapted: `reflection` | Same gains and weights; 160 followers instead of 400; tail response normalized by size replaces all-agent trajectories. |
| *Transients*, Fig. 2 and §4: convergence over 30 parameter pairs, six sizes, two boundaries | Published grid completed: `convergence --profile paper` | All 360 cases computed. Period uses the first-to-third-extremum interval; the source averages successive same-sign intervals. This is an adapted numerical comparison. |
| *Signal velocity*, Figs. 1–2: stable spectra and two instability mechanisms | Not implemented | `dispersion` is a new Taylor-error diagnostic for a different stencil. |
| *Signal velocity*, Fig. 3(a): phase speeds and damping by mode | Not implemented | Signal-speed markers alone do not reproduce this calculation. |
| *Signal velocity*, Fig. 3(b): periodic velocity impulse | Adapted: `pulse` | Smooth Gaussian instead of an impulse; different gains, velocity weights, and size (1,024 instead of 200). |
| *NNN-agents*, Fig. 3: stable circle and unstable path | Adapted: `boundary` | Same stencil and fixed-interaction boundary; 80 followers instead of 200; additional spectral and growth diagnostics. |
| *NNN-agents*, Figs. 5–7: Types I, II, III | Not implemented | In particular, the gallery lacks the stable reflectionless Type II example. |
| *NNN-agents*, Fig. 8 and §6: 500 Type I and 500 Type II configurations | Not implemented | Sizes $25\cdot2^n$, $n=0,\ldots,11$, and both boundary rules are absent. The nearest-neighbour convergence run does not cover this study. |
| *Asymmetric decentralized flocks*, Figs. 2–3; *Symmetry and stability*, Figs. 2–4 | Not implemented | Matched position/velocity coupling, amplification with size, and frequency responses need their own experiments. |

The quick convergence profile covers six parameter pairs and three smaller sizes.
It is a convenient demonstration, distinct from the completed published-grid run.
New propagators, extrema refinement, and sampling checks are documented in the
[gallery](gallery.md); agreement of curves alone is not a claim of procedural
replication.

## Next experiments

1. Recover the published nearest-neighbour stability, phase-speed, and impulse
   examples. Compare impulse and smooth initial data under the same couplings.
2. Add the NNN Type I/II/III representatives, followed by the random-configuration
   study once its selection rule and random provenance are explicit.
3. Recover the matched-Laplacian amplification comparison and frequency responses.
   Keep the input, output, norm, and dependence on system size explicit.
4. Compare the force-disturbance energy bounds in Herman–Knorn–Ahlén with a
   normalized interior force pulse. Specify the boundary and input/output norm;
   this observable differs from the leader-step tail response.
5. Extend to Lyons's heterogeneous-agent examples, including paired periodic/path
   spectra and reversal of velocity asymmetry with the same saved gain arrays.

This audit covers the central papers above. Additional archive papers, working
notes, and theses still require experiment inventories. Path-wave comparisons
retain the conjectural and asymptotic qualifications in [Model](model.md).
