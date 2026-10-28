"""Stencil normalization, Fourier coordinates, and invalid domains."""

import numpy as np
import pytest

import pyflock as pf


@pytest.mark.parametrize("weights", [[], [0, 1, 0], [0] * 6])
def test_stencil_requires_five_offsets(weights):
    with pytest.raises(ValueError, match="length 5"):
        pf.Stencil(weights, [0, -0.5, 1, -0.5, 0], -1, -1)


def test_stencil_even_odd_coordinates_recover_the_original_weights():
    rx = np.array([2, -3, 5, -7, 11], dtype=float)
    rv = np.array([13, -17, 19, -23, 29], dtype=float)
    st = pf.Stencil(rx, rv, -2, -3)
    assert (st.ax0, st.ax1, st.ax2, st.bx1, st.bx2) == (5, -10, 13, -4, 9)
    assert (st.av0, st.av1, st.av2, st.bv1, st.bv2) == (19, -40, 42, -6, 16)
    assert st.row_sums == (8, 21)
    assert st.moment_x(0) == -16
    assert st.moment_v(0) == -63
    assert not st.is_decentralized()
    with pytest.raises(ValueError, match=r"sum\(rx\)=8, sum\(rv\)=21"):
        st.require_decentralized()
    assert st.require_decentralized(tol=22) is st


def test_row_sum_check_includes_both_stencils():
    st = pf.Stencil([0, -0.5, 1, -0.5, 0], [0, -0.5, 2, -0.5, 0], -1, -1)
    assert not st.is_decentralized()
    with pytest.raises(ValueError, match=r"sum\(rx\)=0, sum\(rv\)=1"):
        st.require_decentralized()


def test_normalization_preserves_both_gain_weight_products():
    st = pf.Stencil([1, -3, 4, -5, 3], [-2, 7, -6, 4, -3], -2, -3, "example", "note")
    normalized = st.normalised()
    assert normalized.ax0 == normalized.av0 == 1
    assert normalized.gx == -8
    assert normalized.gv == 18
    assert normalized.gx * normalized.rx == pytest.approx(st.gx * st.rx, abs=0)
    assert normalized.gv * normalized.rv == pytest.approx(st.gv * st.rv, abs=0)
    assert normalized.name == st.name and normalized.notes == st.notes


@pytest.mark.parametrize("which", ["rx", "rv"])
def test_normalization_rejects_zero_diagonal(which):
    weights = {"rx": [0, -0.5, 1, -0.5, 0], "rv": [0, -0.5, 1, -0.5, 0]}
    weights[which][2] = 0
    with pytest.raises(ValueError, match="zero diagonal"):
        pf.Stencil(**weights, gx=-1, gv=-1).normalised()


def test_velocity_asymmetry_is_invariant_under_stencil_scaling():
    st = pf.Stencil([0, -2, 4, -2, 0], [0, -3, 4, -1, 0], -1, -2)
    assert st.is_nearest_neighbour
    assert st.q_asym == 0.5
    assert st.normalised().q_asym == st.q_asym
    with pytest.raises(ValueError, match="nearest-neighbour"):
        _ = pf.Stencil([1, -1, 0, 0, 0], st.rv, -1, -1).q_asym


@pytest.mark.parametrize("name", ["", "sample"])
def test_representation_reports_name_gains_and_offset_weights(name):
    st = pf.nn_stencil(-2, -3, name=name)
    representation = repr(st)
    assert representation.startswith("<Stencil") and representation.endswith(">")
    assert "gx=-2 gv=-3" in representation
    assert "rx=" in representation and "rv=" in representation
    assert (repr(name) in representation) == bool(name)
