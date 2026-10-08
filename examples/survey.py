"""Classify every curated case: circle vs line stability, signal velocities, diffusion."""
import numpy as np, pyflock as pf

hdr = (f"{'case':<28}{'a':>9}{'circle maxRe*':>14}{'line maxRe':>12}"
       f"{'c_+':>9}{'c_-':>9}{'D_+':>9}{'D_-':>9}")
print(hdr); print("-" * len(hdr))
for name in pf.case_names():
    st = pf.case(name)
    ev_c = np.linalg.eigvals(pf.system_matrix_circle(st, 120)).real
    circ = np.sort(ev_c)[-3]                     # exclude the double zero
    rel = pf.eigen_reliability(pf.system_matrix_line_leaderless(st, 60, bc="drop"))
    line = rel["eigenvalue"].real
    if st.a > 0:
        cp, cm = pf.signal_velocities(st)
        co = pf.coefficients(st)
        cols = f"{cp:9.4f}{cm:9.4f}{co['+']['D']:9.4f}{co['-']['D']:9.4f}"
    else:
        cols = f"{'--':>9}{'--':>9}{'--':>9}{'--':>9}"
    print(f"{name:<28}{st.a:9.4f}{circ:14.2e}{line:12.2e}{cols}")
print("\n* third-largest real part on the circle (the double zero is excluded)")
