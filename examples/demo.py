"""End-to-end demo: dispersion figures, a line transient, and theory-vs-measurement."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import pyflock as pf
from pyflock.plotting import dispersion_figures, trajectory_figure

OUT = "figures"
import os; os.makedirs(OUT, exist_ok=True)

# --- flock.m's four figures, for the symmetric NN flock it was set up with
st = pf.Stencil([0.0, -0.5, 1.0, -0.5, 0.0], [-1.0, -1.0, 1.0, 1.0, 0.0], -3.0, -3.0,
                name="flock.m defaults")
for k, f in dispersion_figures(st, N=100).items():
    f.savefig(f"{OUT}/dispersion_{k}.png", dpi=130)
print(f"wrote 4 dispersion figures to {OUT}/")

# --- a line transient with a leader
st2 = pf.nn_stencil(-1.0, -1.0, -0.5, -0.3)
cp, cm = pf.signal_velocities(st2)
N = 200
P = N * (1.0 / cp - 1.0 / cm)
tmax = N / cp + 3.5 * P
res = pf.simulate_line(st2, N, tmax=tmax, nt=int(25 * tmax), bc="regular")
trajectory_figure(res, stride=4).savefig(f"{OUT}/trajectories.png", dpi=130)
print(f"wrote {OUT}/trajectories.png   ({res.collisions()})")

# --- measured vs leading-order theory
meas = pf.transient_summary(res.t, res.last_relative, nmax=4)
th = pf.theory_amplitudes(st2, N, kmax=4)
print(f"\nN={N}  c_+={cp:.6f}  c_-={cm:.6f}   D_+={pf.coefficients(st2)['+']['D']:.6f}")
print(f"{'k':>2} {'|A_k| measured':>15} {'|A_k| Thm 3':>13} {'rel. err':>10}")
for k, (m, t) in enumerate(zip(meas["A"], th["A"]), start=1):
    print(f"{k:2d} {abs(m):15.5f} {abs(t):13.5f} {(abs(t)-abs(m))/abs(m):+10.4f}")
print(f"\ngamma: measured {meas.get('gamma', float('nan')):.5f}   "
      f"Thm 3 {th['gamma']:.5f}")
