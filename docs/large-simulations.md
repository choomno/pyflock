# Large-array simulations

Status: stub for future work on hundreds of thousands or millions of agents.
The reserved entry point is [examples/large_simulation.py](../examples/large_simulation.py).
It provides `--help`; running it exits with an unimplemented message.

## Proposed scope

- Matrix-free operators with forward and adjoint actions and an explicit trace.
  Geometry, boundary rules, gain conventions and initial data belong to the model.
- Streaming observations: selected agents, aggregate quantities and sampled
  spatial fields, with an explicit buffer limit.
- Local and heterogeneous coupling first. Dense operators should expose their
  structure, such as convolution or low rank, rather than require a full array.
- A CPU reference implementation, followed by GPU or distributed execution
  where measured performance warrants it.

Periodic homogeneous arrays admit Fourier mode evolution. Compatible Dirichlet
operators admit sine transforms; differing position and velocity asymmetries
generally require another propagator. Boundary closures determine which
specialization is valid.

## Validation and benchmarks

Compare small systems with the existing matrix-exponential, Fourier and ODE
methods. Then measure runtime, peak memory and numerical error at ten thousand,
one hundred thousand and one million agents, separating fixed-duration runs
from runs that follow a disturbance across the flock. Record output and buffer
sizes with each measurement.

Research cases should compare endpoint closures, heterogeneous gain orderings,
localized defects and approach to wave-speed degeneracies. A continuum study
must specify how spacing, gains and initial data change with refinement.
Reuse the [experiment records](reproduction.md) for parameters and provenance.
