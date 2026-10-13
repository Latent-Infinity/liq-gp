"""Tests for the scrambled Sobol initial design over the unit cube.

The module is imported inside each case, so an absent module reports its
absence per case rather than as one collection error.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import sys
import warnings
from types import ModuleType

import numpy as np
import pytest

_MODULE = "liq.gp.evolution.array_design"
_ALLOWED_IMPORT_ROOTS = {"__future__", "numpy", "scipy", "liq"}

# SHA-256 over the concatenated ``float64`` bytes of the designs drawn for
# seeds 32..63, one ``numpy.random.default_rng(seed)`` per design. These are the
# digests recorded for the scrambled Sobol arm by the operator ablation that
# selected this design, so a match means the library draws the measured points.
_MEASURED_DIGESTS = {
    (16, 4): "daf0fa4aac6f2a8efce40eece67bfa525a96d5b6b2a16e0ca3e24c0c9c43aa23",
    (50, 4): "b69677e5b27b7404cd007ba6b05a46685b85eafef935e05ea51817bef02742fb",
    (64, 4): "d9e0c9e5be6c521c0bac753421873649c706631500b3518cdba376d79e61354a",
    (50, 130): "fbf5f3d0d089093acca0d1896edada456c38b0db900e69eb8dfa419ef6c82807",
}
_MEASURED_SEEDS = range(32, 64)


def _module() -> ModuleType:
    return importlib.import_module(_MODULE)


def _design(n: int, d: int, seed: int) -> np.ndarray:
    return _module().sobol_design(n, d, np.random.default_rng(seed))


# --- shape, range, determinism ---------------------------------------------


@pytest.mark.parametrize(("n", "d"), [(1, 1), (16, 4), (50, 3), (64, 130)])
def test_returns_n_points_of_dimension_d_in_the_unit_cube(n: int, d: int) -> None:
    points = _design(n, d, seed=7)
    assert points.shape == (n, d)
    assert points.dtype == np.float64
    assert np.all(points >= 0.0)
    assert np.all(points < 1.0)


def test_same_seed_gives_identical_design() -> None:
    np.testing.assert_array_equal(_design(50, 5, seed=3), _design(50, 5, seed=3))


def test_different_seeds_give_different_designs() -> None:
    assert not np.array_equal(_design(32, 5, seed=3), _design(32, 5, seed=4))


def test_draws_from_the_passed_generator() -> None:
    rng = np.random.default_rng(11)
    first = _module().sobol_design(16, 3, rng)
    second = _module().sobol_design(16, 3, rng)
    assert not np.array_equal(first, second)


@pytest.mark.parametrize(("n", "d"), sorted(_MEASURED_DIGESTS))
def test_draws_the_design_the_ablation_measured(n: int, d: int) -> None:
    digest = hashlib.sha256()
    for seed in _MEASURED_SEEDS:
        digest.update(_design(n, d, seed).tobytes())
    assert digest.hexdigest() == _MEASURED_DIGESTS[(n, d)]


# --- stratification -----------------------------------------------------------


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_column_has_one_point_per_stratum_at_powers_of_two(
    m: int, seed: int
) -> None:
    """At n = 2**m each column puts exactly one point in each [k/n, (k+1)/n)."""
    n = 2**m
    points = _design(n, 8, seed)
    strata = np.floor(points * n).astype(np.int64)
    for column in strata.T:
        assert sorted(column.tolist()) == list(range(n))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_two_dimensional_elementary_box_is_balanced(seed: int) -> None:
    """At n = 2**m every pair of the first dimensions is a (0, m, 2)-net.

    Each box of 2**a by 2**(m - a) equal cells holds exactly one point, the
    balance property a Sobol' net promises for its leading pair of coordinates.
    """
    m = 6
    n = 2**m
    points = _design(n, 2, seed)
    for a in range(m + 1):
        rows = np.floor(points[:, 0] * 2**a).astype(np.int64)
        cols = np.floor(points[:, 1] * 2 ** (m - a)).astype(np.int64)
        cells = rows * 2 ** (m - a) + cols
        assert np.bincount(cells, minlength=n).tolist() == [1] * n


def test_balance_is_not_promised_off_a_power_of_two() -> None:
    """At n = 50 the leading prefix of a 64-point net is not one-per-stratum."""
    n = 50
    unbalanced = 0
    for seed in range(8):
        strata = np.floor(_design(n, 4, seed) * n).astype(np.int64)
        unbalanced += sum(
            sorted(column.tolist()) != list(range(n)) for column in strata.T
        )
    assert unbalanced > 0


# --- warnings -------------------------------------------------------------------


@pytest.mark.parametrize("n", [16, 50])
def test_emits_no_warning(n: int) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _design(n, 4, seed=0)


def test_does_not_suppress_unrelated_warnings_globally() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _design(50, 4, seed=0)
        warnings.warn("unrelated", UserWarning, stacklevel=1)
    assert [str(w.message) for w in caught] == ["unrelated"]


# --- validation -------------------------------------------------------------------


@pytest.mark.parametrize(("n", "d"), [(0, 3), (-1, 3), (4, 0), (4, -2)])
def test_rejects_non_positive_sizes(n: int, d: int) -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        _module().sobol_design(n, d, np.random.default_rng(0))


@pytest.mark.parametrize(("n", "d"), [(2.0, 3), (4, 3.0), (True, 3), (4, False)])
def test_rejects_non_integer_sizes(n: object, d: object) -> None:
    with pytest.raises(TypeError, match="must be an int"):
        _module().sobol_design(n, d, np.random.default_rng(0))  # type: ignore[arg-type]


def test_rejects_a_seed_in_place_of_a_generator() -> None:
    with pytest.raises(TypeError, match="numpy.random.Generator"):
        _module().sobol_design(4, 3, 0)  # type: ignore[arg-type]


# --- dependency scope ----------------------------------------------------------------


def test_imports_only_numpy_scipy_and_liq_gp() -> None:
    tree = ast.parse(inspect.getsource(_module()))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
                if alias.name.split(".")[0] == "liq":
                    assert alias.name.startswith("liq.gp")
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative imports are not allowed"
            assert node.module is not None
            roots.add(node.module.split(".")[0])
            if node.module.split(".")[0] == "liq":
                assert node.module.startswith("liq.gp")
    assert roots <= _ALLOWED_IMPORT_ROOTS


def test_is_not_exported_from_the_package_root() -> None:
    import liq.gp

    module = _module()
    assert module.__name__ in sys.modules
    assert "sobol_design" not in liq.gp.__all__
    assert not hasattr(liq.gp, "sobol_design")
