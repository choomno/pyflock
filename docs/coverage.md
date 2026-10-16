# Experiment coverage

**Source parameters** retain the stated couplings and sizes; **adapted** changes
an input, observable or measurement; **new** denotes an additional investigation.
Sizes refer to `--profile paper`. Figure numbers use the archived manuscripts in
[Sources](sources.md). See the [original](gallery.md) and
[expanded](gallery-expanded.md) galleries for the figures.

| Source figures or question | Named experiments | Coverage and departures |
| --- | --- | --- |
| *Transients*, Fig. 1 | `propagation` | Adapted trajectories and spacing field, with 160 followers and different couplings; schematic omitted. |
| *Transients*, Fig. 3 | `reflection` | Source parameters and 400 followers; normalized tail response and measured/predicted extrema replace all-agent trajectories. |
| *Transients*, Fig. 2 and §4 | `convergence` | Completed 360-case published grid: 30 pairs, six sizes, two boundaries. Adapted period estimator uses first-to-third extrema instead of averaging successive same-sign intervals. |
| *Signal velocity*, local expansion and smooth waves | `dispersion`, `pulse` | New Taylor-error diagnostic and smooth Gaussian control at 1,024 agents. |
| *Signal velocity*, Figs. 1–2 | `nn_stable_spectra`, `nn_instability` | Source parameters and 500 agents: stable spectra and both instability mechanisms. |
| *Signal velocity*, Fig. 3(a) | `nn_phase_speeds` | Source parameters and 500 agents; speed and damping use separate axes. |
| *Signal velocity*, Fig. 3(b) | `nn_pulse_comparison` | Source 200-agent impulse parameters; agent index replaces physical position. Added smooth input has equal total velocity. |
| *NNN-agents*, Fig. 3 | `boundary` | Source stencil and 200 followers; added spectra and growth diagnostics. |
| *NNN-agents*, Figs. 5–7 | `nnn-types` | Source Type I/II/III sizes and parameters; documented correction to the printed Type III coefficient. |
| *NNN-agents*, Fig. 8 and §6 | `nnn-ensemble` | New restricted stable families: 500 configurations/class × two boundaries × twelve sizes, $25$–$51\,200$ (24,000 cases). Type II gains vary. Historical member files and selection grid remain unavailable. |
| *Asymmetric decentralized flocks*, Fig. 2; *Symmetry and stability*, Figs. 2–3 | `matched_amplification` | Source trajectory parameters; adapted harmonic-frequency sampling, resonance refinement and size sweep. |
| *Asymmetric decentralized flocks*, Fig. 3; *Symmetry and stability*, Fig. 4 | `matched_transfer_roots`, `matched_spectra` | Source transfer parameters with added diagnostics; finite spectra provide a numerical companion to the historical schematic. |
| Lyons thesis, Figs. 8–17 | `lyons_boundary`, `lyons_response` | Boundary arrays of 400/1,600 agents; response arrays of 400, three samples/family. Thesis gain distributions with new saved samples. Spectra sample the origin only. Negative-asymmetry wave overlays formally extrapolate outside the stability hypothesis. |
| Lyons thesis, Figs. 12–14 | `lyons_stability_reversal` | New explicit repeated two-agent cell at 64 agents recovers the reported unstable-circle/stable-path behavior. Full finite spectra and a path Lyapunov check; original gain arrays remain unavailable. |
| Lyons, Remark 2.8.1 and Appendix E | `lyons_ordering` | New repeated-cell, ordering and equivalent-homogeneous comparisons; one realization. |
| Herman–Knorn–Ahlén, Eqs. (3)–(9), Lemma 6 | `energy` | Source audit, singular-value scaling and force response. A numerical counterexample to Eq. (7)'s finite coefficient is verified in the original ODE and independent coordinates; this does not refute the scaling result. |
| Lyons, Appendix H | `continuum` | New controlled symmetric periodic specialization at fixed physical wavelength; it does not establish the general heuristic PDE limit. |

Records separate numerical sensitivity, sample variation and asymptotic error.
Harmonic peaks are sampled; partial spectra do not establish full stability.
NNN measurement windows and the Type II identity $T_1=A/v_0$ are explicit
adaptations. See [numerical notes](numerical-notes.md).

Remaining gaps include unavailable historical ensembles and seeds, Cantos's
master's full text, omitted schematics, the general PDE problem, and experiments
in other papers outside the agreed expanded scope. Path-wave comparisons retain
the hypotheses and conjectural steps in [Model](model.md).
