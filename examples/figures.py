"""Generate the full figure set for a couple of stencils."""
import os, matplotlib
matplotlib.use("Agg")
import pyflock as pf
from pyflock import plotting as P

OUT = "figures"; os.makedirs(OUT, exist_ok=True)
st = pf.nn_stencil(-1.0, -1.0, -0.5, -0.3, name="NN, q=0.4")
cp, cm = pf.signal_velocities(st)
N = 150
Pn = N * (1/cp - 1/cm)
res = pf.simulate_line(st, N, tmax=N/cp + 2.2*Pn, nt=int(12*(N/cp+2.2*Pn)), bc="regular")

for nm, f in [
    ("spectrum",      P.spectrum_figure(st)),
    ("circle_vs_line",P.circle_vs_line_figure(st, N=120, bc="regular")),
    ("spacetime",     P.spacetime_figure(res)),
    ("trajectories",  P.trajectory_figure(res, stride=2)),
    ("transient",     P.transient_figure(res)),
    ("packet",        P.packet_figure(st, N=4096, times=(50,100,200,400))),
    ("convergence",   P.convergence_figure(st, Ns=(50,100,200,400))),
    ("overview",      P.overview_figure(st, N=150)),
]:
    f.savefig(f"{OUT}/{nm}.png", dpi=140, bbox_inches="tight")
    print("wrote", f"{OUT}/{nm}.png")

# the pathological case, for contrast
stI = pf.case("type_I")
P.overview_figure(stI, N=120, bc="drop").savefig(f"{OUT}/overview_type_I.png",
                                                 dpi=140, bbox_inches="tight")
print("wrote", f"{OUT}/overview_type_I.png")


# stability over (q, N) for the velocity-asymmetry family, against the theory
import numpy as np
fam = pf.velocity_asymmetry_family(-1.0, -3.0)
sm = pf.stability_map(fam, np.linspace(-0.1, 0.1, 31), (40, 60, 90, 130, 190, 280),
                      bc="regular", theory=pf.least_stable_eigenvalue, label="$g_x$=-1, $g_v$=-3")
P.stability_map_figure(sm).savefig(f"{OUT}/stability_map.png", dpi=140, bbox_inches="tight")
print("wrote", f"{OUT}/stability_map.png")
