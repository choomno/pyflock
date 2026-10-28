"""Named examples retain decentralized couplings and public API identity."""

import numpy as np
import pytest

import pyflock as pf
from pyflock import cases


def test_names_are_sorted_complete_and_return_original_immutable_stencils():
    names = pf.case_names()
    assert names == sorted(pf.CASES)
    assert len(names) == 13
    for name in names:
        stencil = pf.case(name)
        assert stencil is pf.CASES[name]["stencil"]
        assert stencil.name == name
        np.testing.assert_allclose(stencil.row_sums, 0, atol=2e-15)
        assert pf.CASES[name]["tlim"] > 0
        assert not stencil.rx.flags.writeable
        assert not stencil.rv.flags.writeable
    names.clear()
    assert len(pf.case_names()) == 13


def test_unknown_case_error_lists_available_examples():
    with pytest.raises(KeyError, match="unknown case") as error:
        pf.case("nonexistent")
    for name in pf.case_names():
        assert name in str(error.value)


def test_parameterized_families_preserve_their_defining_moments():
    for name in (
        "different_reflections_m1",
        "different_reflections_0",
        "different_reflections_p1",
    ):
        stencil = pf.case(name)
        assert stencil.rx.tolist() == [-0.25, -0.25, 1.0, -0.25, -0.25]
        assert stencil.moment_x(1) == 0
    for sign in (-1, 1):
        rx, rv, gx, gv = cases._neg_pos_equal_velocities(0.4, sign, eps=0.03)
        assert np.sum(rx) == pytest.approx(0)
        assert np.sum(rv) == pytest.approx(0)
        assert np.dot(rx, np.arange(-2, 3)) == pytest.approx(0)
        assert gx == gv == -2
    rx, rv, gx, gv = cases._workflow()
    np.testing.assert_allclose(rx, pf.case("workflow_i3").rx)
    np.testing.assert_allclose(rv, pf.case("workflow_i3").rv)
    assert gx == gv == -2


def test_public_exports_are_defined_and_unique():
    assert len(pf.__all__) == len(set(pf.__all__))
    for name in pf.__all__:
        assert hasattr(pf, name)
    assert pf.__version__ == "0.2.0"
