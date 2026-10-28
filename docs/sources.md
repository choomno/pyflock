# Demo sources

The [coverage map](coverage.md) gives figure numbers, parameters and departures
from the cited papers and dissertation. Source PDFs are not needed to run the
package.

| Work | Demos |
| --- | --- |
| Cantos, Veerman and Hammond, *Signal velocity in oscillator arrays* (2016), [DOI](https://doi.org/10.1140/epjst/e2016-02665-0). | `nn_stable_spectra`, `nn_instability`, `nn_phase_speeds`, `nn_pulse_comparison`; `dispersion` and `pulse` are numerical controls. |
| Cantos, Hammond and Veerman, *Transients in the synchronization of asymmetrically coupled oscillator arrays* (2016), [DOI](https://doi.org/10.1140/epjst/e2016-02658-y). | `propagation`, `reflection`, `convergence`. |
| Herbrych, Chazirakis, Christakis and Veerman, *Dynamics of locally coupled agents with next nearest neighbor interaction* (2021; online 2017), [DOI](https://doi.org/10.1007/s12591-017-0377-3). | `boundary`, `nnn-types`; `nnn-ensemble` uses new restricted parameter families. |
| Tangerman, Veerman and Stošić, *Asymmetric decentralized flocks* (2012), IEEE TAC 57(11), 2844–2853. Veerman, *Symmetry and stability of homogeneous flocks (a position paper)*, supplies additional figure comparisons. | `matched_transfer_roots`, `matched_spectra`, `matched_amplification`. |
| Lyons, *Linear Nearest Neighbor Flocks With All Distinct Agents* (2022), [PSU dissertation](https://pdxscholar.library.pdx.edu/open_access_etds/5903/). | `lyons_boundary`, `lyons_response`; constructed comparisons `lyons_stability_reversal`, `lyons_ordering`, and the symmetric `continuum` specialization. |
| Herman, Knorn and Ahlén, *Disturbance scaling in bidirectional vehicle platoons with different asymmetry in position and velocity coupling* (2017), [DOI](https://doi.org/10.1016/j.automatica.2017.04.010). | `energy`: source damping matrices, force response and a numerical bound diagnostic. |

*Transients*, Definition 1, specifies the terminal boundaries. Its tail-extremum
formulas use the circle-to-path and reflection conjectures; see
[Model](model.md). The NNN path classifications also use circle-to-path
conjectures. Finite spectra and response computations are independent checks of
these comparisons, rather than proofs of the asymptotic formulas.

Corrections needed for the demos are listed in [Numerical notes](numerical-notes.md).
Saved experiment records contain the complete coefficients, initial data,
boundary rules and numerical methods.
