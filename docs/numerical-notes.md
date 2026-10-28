# Reproduction notes

## NNN coefficients and boundaries

*NNN-agents*, Figure 7, prints
$\rho_x=(-2,-15/4,1,-21/4,5/2)$, whose sum is $-15/2$.
The surviving MATLAB case uses $+15/4$, restoring zero row sum and zero first
position moment. `nnn-types` uses this correction and records it in its JSON.

The paper's *fixed mass* boundary clamps a missing neighbour to the endpoint.
It differs from the package's historical `fold` rule, which also rescales the
final row. The NNN study implements the clamp separately. Its velocity endpoint
uses velocity weights, correcting the printed position-weight subscript typo.

Original NNN configuration files and Lyons's random seeds were unavailable.
The new ensembles save their complete coefficient arrays and selection rules;
the galleries identify reconstructed and constructed examples.

## Force response

`energy` uses the source's matrices and energy observable, with unit masses and
gains. It adds a distributed-force numerical diagnostic of equation (7), checked
by independent ODE integration. That forcing is a constructed comparison; the
figure also shows the separate single-site pulse. It does not determine the
asymptotic scaling in flock size.

## Precision and interpretation

The matched-coupling example at $\rho=.55$, $N=100$ has a narrow low-frequency
resonance. Sensitive peak neighbourhoods use a 70-digit recurrence. Reported
peaks are refined samples over the saved band, not certified supremum norms.
Partial spectra near the origin do not establish full spectral stability.
The older Gaussian amplitude correction and small-asymmetry helpers remain
exploratory; the reproduced demos do not use them as correctness criteria.
