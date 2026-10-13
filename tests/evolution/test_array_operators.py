"""Tests for the array variation family: reflection, a self-adaptive log-normal
scale step, uniform recombination and bounded Gaussian mutation with bit flips.

The module is imported inside each case, so an absent module reports its
absence per case rather than as one collection error.

The digests below were computed from the operator ablation's measured
reference implementation (``ops.py``, SHA-256 ``a3df9737…ba53``) on the
arrays, states and seeds declared here; a match means the library performs
exactly the arithmetic, and consumes exactly the draws, that were measured.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import math
import sys
from types import ModuleType

import numpy as np
import pytest

_MODULE = "liq.gp.evolution.array_operators"
_ALLOWED_IMPORT_ROOTS = {"__future__", "dataclasses", "math", "numpy", "scipy", "liq"}
_PUBLIC = {
    "Mutation",
    "ScaleStep",
    "gaussian_mutation",
    "lognormal_scale_step",
    "reflect_unit",
    "uniform_crossover",
}

# Declared problem: eight positions, the first three held at {0, 1}.
_N = 8
_N_BITS = 3
_LOW = np.array([0.0, 0.0, 0.0, -0.5, -0.5, -0.5, -1.0, 0.9])
_HIGH = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 0.9999])
_BITS = np.arange(_N) < _N_BITS
_SCALE_LOW, _SCALE_HIGH = 1e-3, 0.5
_SEEDS = range(64)
_SCALE_STATES = [
    (0.1, 0.1),
    (0.1, 0.3),
    (0.002, 0.001),
    (0.0011, 0.0011),
    (0.5, 0.5),
    (0.45, 0.2),
]

_SCALE_DIGEST = "b1171be6ea5f1d37fe6f5cab437e82230cdf8a9c5471c053de764fbb4aec0193"
_SCALE_FIRINGS = 114
_CROSSOVER_DIGEST = "89f2bc24ef01164bbcdc489b9158f2eaa6f4e18d9f7c975106586dd14b401418"
_MUTATION_DIGEST = "2769fde0216350951f75ef72a928169046aa20961284a33ac107178cabd8c1e4"
_MUTATION_FIRINGS = [0, 0, 0, 7, 9, 15, 8, 6]
_COMPOSED_DIGEST = "42f808005bca7959b219ee11a7a0ad53b73e5f37726561f9693ab57251aee95a"


def _module() -> ModuleType:
    return importlib.import_module(_MODULE)


def _parents(seed: int, *, same_bits: bool) -> tuple[np.ndarray, np.ndarray]:
    g = np.random.default_rng([seed, 4242])
    out = []
    for _ in range(2):
        x = _LOW + g.random(_N) * (_HIGH - _LOW)
        x[:_N_BITS] = (g.random(_N_BITS) < 0.5).astype(float)
        out.append(x)
    if same_bits:
        out[1][:_N_BITS] = out[0][:_N_BITS]
    return out[0], out[1]


def _scale_step(s1: float, s2: float, n: int, rng: np.random.Generator):
    return _module().lognormal_scale_step(
        s1, s2, n, rng, low=_SCALE_LOW, high=_SCALE_HIGH
    )


def _mutate(x: np.ndarray, scale: float, rng: np.random.Generator):
    return _module().gaussian_mutation(x, _LOW, _HIGH, scale, rng, bits=_BITS)


def _fold(v: float) -> float:
    """Reflection restated independently, scalar ``math`` only."""
    v = math.fmod(v, 2.0)
    if v < 0.0:
        v += 2.0
    return 2.0 - v if v > 1.0 else v


# --- reflection ----------------------------------------------------------------


def test_reflection_folds_out_of_range_values_to_the_predicted_value() -> None:
    values, fired = _module().reflect_unit(
        np.array([-0.25, 1.25, 2.5, -1.5, 3.5, 0.5, 0.0, 1.0])
    )
    np.testing.assert_array_equal(values, [0.25, 0.75, 0.5, 0.5, 0.5, 0.5, 0.0, 1.0])
    np.testing.assert_array_equal(
        fired, [True, True, True, True, True, False, False, False]
    )


def test_reflection_changes_every_out_of_range_value() -> None:
    raw = np.array([-0.3, 1.7, 0.4])
    values, _ = _module().reflect_unit(raw)
    assert values[0] != raw[0] and values[1] != raw[1]
    assert np.all((values >= 0.0) & (values <= 1.0))


def test_reflection_matches_the_scalar_restatement_far_from_the_range() -> None:
    raw = np.random.default_rng(5).normal(0.5, 4.0, size=500)
    values, fired = _module().reflect_unit(raw)
    expected = [v if 0.0 <= v <= 1.0 else _fold(v) for v in raw.tolist()]
    np.testing.assert_array_equal(values, expected)
    np.testing.assert_array_equal(fired, (raw < 0.0) | (raw > 1.0))
    assert np.all((values >= 0.0) & (values <= 1.0))


def test_reflection_leaves_its_input_untouched() -> None:
    raw = np.array([-0.25, 0.5])
    _module().reflect_unit(raw)
    np.testing.assert_array_equal(raw, [-0.25, 0.5])


@pytest.mark.parametrize(
    "bad", [np.array([np.nan]), np.array([np.inf]), np.ones((2, 2))]
)
def test_reflection_refuses_non_finite_or_non_vector_input(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        _module().reflect_unit(bad)


# --- self-adaptive log-normal scale step -------------------------------------------


def test_scale_step_produces_the_declared_value_inside_the_bounds() -> None:
    step = _scale_step(0.1, 0.1, 8, np.random.default_rng(3))
    z = np.random.default_rng(3).standard_normal()
    assert step.parent_scale == 0.1
    assert step.scale == 0.1 * math.exp(z / math.sqrt(8))
    assert step.scale != 0.1
    assert step.fired is False


def test_scale_step_starts_from_the_parents_geometric_mean() -> None:
    step = _scale_step(0.1, 0.4, 8, np.random.default_rng(3))
    assert step.parent_scale == math.sqrt(0.1 * 0.4)
    z = np.random.default_rng(3).standard_normal()
    assert step.scale == math.sqrt(0.1 * 0.4) * math.exp(z / math.sqrt(8))


def _reflected_scale(raw: float) -> float:
    lo, hi = math.log(_SCALE_LOW), math.log(_SCALE_HIGH)
    v = _fold((math.log(raw) - lo) / (hi - lo))
    if v == 0.0:
        return _SCALE_LOW
    if v == 1.0:
        return _SCALE_HIGH
    return math.exp(lo + v * (hi - lo))


@pytest.mark.parametrize("state", [0.5, 0.0011])
def test_scale_step_reflects_in_log_space_at_either_bound(state: float) -> None:
    hits = 0
    for seed in range(200):
        step = _scale_step(state, state, 8, np.random.default_rng(seed))
        z = np.random.default_rng(seed).standard_normal()
        raw = state * math.exp(z / math.sqrt(8))
        assert _SCALE_LOW <= step.scale <= _SCALE_HIGH
        if _SCALE_LOW <= raw <= _SCALE_HIGH:
            assert step.fired is False
            assert step.scale == raw
        else:
            hits += 1
            assert step.fired is True
            assert step.scale == pytest.approx(_reflected_scale(raw), rel=1e-12)
    assert hits > 0


class _FixedNormal(np.random.Generator):
    """A Generator whose normal draw is declared, to reach exact fold endpoints."""

    def __init__(self, z: float) -> None:
        super().__init__(np.random.PCG64(0))
        self._z = z

    def standard_normal(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return self._z


@pytest.mark.parametrize(("z", "expected"), [(2.0, 1.0), (3.0, math.e)])
def test_scale_step_returns_the_bound_itself_when_the_fold_lands_on_it(
    z: float, expected: float
) -> None:
    # With bounds [1, e] the log-space position is log(raw) exactly: z = 2 folds
    # to 0 (the lower bound) and z = 3 folds to 1 (the upper bound).
    step = _module().lognormal_scale_step(
        1.0, 1.0, 1, _FixedNormal(z), low=1.0, high=math.e
    )
    assert step.fired is True
    assert step.scale == expected


def test_scale_step_consumes_exactly_one_normal_draw() -> None:
    rng, twin = np.random.default_rng(9), np.random.default_rng(9)
    _scale_step(0.1, 0.2, 8, rng)
    twin.standard_normal()
    assert rng.random() == twin.random()


def test_scale_step_matches_the_measured_arithmetic() -> None:
    digest = hashlib.sha256()
    firings = 0
    for n in (8, 130):
        for s1, s2 in _SCALE_STATES:
            for seed in _SEEDS:
                rng = np.random.default_rng([seed, 0, 7])
                step = _scale_step(s1, s2, n, rng)
                firings += step.fired
                digest.update(
                    np.array(
                        [step.parent_scale, step.scale, float(step.fired)]
                    ).tobytes()
                )
                digest.update(np.array([rng.random()]).tobytes())
    assert firings == _SCALE_FIRINGS
    assert digest.hexdigest() == _SCALE_DIGEST


def test_scale_step_is_reproducible_under_a_seed() -> None:
    a = _scale_step(0.2, 0.3, 8, np.random.default_rng(1))
    b = _scale_step(0.2, 0.3, 8, np.random.default_rng(1))
    assert a == b


@pytest.mark.parametrize(
    ("s1", "s2", "n", "low", "high"),
    [
        (0.0, 0.1, 8, 1e-3, 0.5),
        (0.1, -0.1, 8, 1e-3, 0.5),
        (math.nan, 0.1, 8, 1e-3, 0.5),
        (0.6, 0.1, 8, 1e-3, 0.5),
        (0.1, 1e-4, 8, 1e-3, 0.5),
        (0.1, 0.1, 0, 1e-3, 0.5),
        (0.1, 0.1, 8, 0.0, 0.5),
        (0.1, 0.1, 8, 0.5, 0.5),
        (0.1, 0.1, 8, 1e-3, math.inf),
    ],
)
def test_scale_step_refuses_an_invalid_state(
    s1: float, s2: float, n: int, low: float, high: float
) -> None:
    with pytest.raises(ValueError):
        _module().lognormal_scale_step(
            s1, s2, n, np.random.default_rng(0), low=low, high=high
        )


@pytest.mark.parametrize("n", [8.0, True, "8"])
def test_scale_step_refuses_a_non_integer_count(n: object) -> None:
    with pytest.raises(TypeError):
        _module().lognormal_scale_step(
            0.1,
            0.1,
            n,  # ty: ignore[invalid-argument-type]
            np.random.default_rng(0),
            low=_SCALE_LOW,
            high=_SCALE_HIGH,
        )


def test_scale_step_refuses_a_seed_in_place_of_a_generator() -> None:
    with pytest.raises(TypeError):
        _module().lognormal_scale_step(
            0.1,
            0.1,
            8,
            3,  # ty: ignore[invalid-argument-type]
            low=_SCALE_LOW,
            high=_SCALE_HIGH,
        )


# --- uniform recombination -------------------------------------------------------


def test_recombination_takes_every_position_from_one_parent() -> None:
    p1, p2 = _parents(0, same_bits=False)
    for seed in range(50):
        child = _module().uniform_crossover(p1, p2, np.random.default_rng(seed))
        assert np.all((child == p1) | (child == p2))


def test_recombination_of_distinct_parents_differs_from_each_parent() -> None:
    p1, p2 = _parents(1, same_bits=False)
    differs = 0
    for seed in range(50):
        child = _module().uniform_crossover(p1, p2, np.random.default_rng(seed))
        differs += (not np.array_equal(child, p1)) and (not np.array_equal(child, p2))
    assert differs >= 40


def test_recombination_picks_each_parent_with_probability_one_half() -> None:
    p1, p2 = np.zeros(10_000), np.ones(10_000)
    child = _module().uniform_crossover(p1, p2, np.random.default_rng(2))
    assert abs(child.mean() - 0.5) < 0.02


def test_recombination_consumes_one_uniform_draw_per_position() -> None:
    p1, p2 = _parents(2, same_bits=False)
    rng, twin = np.random.default_rng(4), np.random.default_rng(4)
    child = _module().uniform_crossover(p1, p2, rng)
    np.testing.assert_array_equal(child, np.where(twin.random(_N) < 0.5, p2, p1))
    assert rng.random() == twin.random()


def test_recombination_leaves_its_parents_untouched() -> None:
    p1, p2 = _parents(3, same_bits=False)
    c1, c2 = p1.copy(), p2.copy()
    child = _module().uniform_crossover(p1, p2, np.random.default_rng(0))
    np.testing.assert_array_equal(p1, c1)
    np.testing.assert_array_equal(p2, c2)
    assert child is not p1 and child is not p2


def test_recombination_matches_the_measured_arithmetic() -> None:
    digest = hashlib.sha256()
    for seed in _SEEDS:
        p1, p2 = _parents(seed, same_bits=False)
        rng = np.random.default_rng([seed, 1, 7])
        child = _module().uniform_crossover(p1[_N_BITS:], p2[_N_BITS:], rng)
        digest.update(child.tobytes())
        digest.update(np.array([rng.random()]).tobytes())
    assert digest.hexdigest() == _CROSSOVER_DIGEST


def test_recombination_is_reproducible_under_a_seed() -> None:
    p1, p2 = _parents(5, same_bits=False)
    np.testing.assert_array_equal(
        _module().uniform_crossover(p1, p2, np.random.default_rng(8)),
        _module().uniform_crossover(p1, p2, np.random.default_rng(8)),
    )


@pytest.mark.parametrize(
    ("p1", "p2"),
    [
        (np.zeros(3), np.zeros(4)),
        (np.zeros((2, 2)), np.zeros((2, 2))),
        (np.array([np.nan, 0.0]), np.zeros(2)),
    ],
)
def test_recombination_refuses_mismatched_or_non_finite_parents(
    p1: np.ndarray, p2: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        _module().uniform_crossover(p1, p2, np.random.default_rng(0))


def test_recombination_refuses_a_seed_in_place_of_a_generator() -> None:
    with pytest.raises(TypeError):
        _module().uniform_crossover(
            np.zeros(2),
            np.ones(2),
            0,  # ty: ignore[invalid-argument-type]
        )


# --- bounded Gaussian mutation with bit flips -----------------------------------------


def test_mutation_always_changes_its_input() -> None:
    for seed in range(200):
        x, _ = _parents(seed, same_bits=False)
        result = _mutate(x, 0.1, np.random.default_rng(seed))
        assert result.selected.any()
        assert not np.array_equal(result.values, x)


def test_mutation_keeps_every_position_within_its_bounds() -> None:
    for seed in range(300):
        x, _ = _parents(seed, same_bits=False)
        result = _mutate(x, 0.5, np.random.default_rng(seed))
        assert np.all(result.values >= _LOW)
        assert np.all(result.values <= _HIGH)


def test_mutation_flips_exactly_the_selected_bit_positions() -> None:
    for seed in range(200):
        x, _ = _parents(seed, same_bits=False)
        result = _mutate(x, 0.3, np.random.default_rng(seed))
        sel = result.selected[:_N_BITS]
        np.testing.assert_array_equal(
            result.values[:_N_BITS], np.where(sel, 1.0 - x[:_N_BITS], x[:_N_BITS])
        )
        assert not result.fired[:_N_BITS].any()


def test_mutation_moves_and_reflects_exactly_as_predicted() -> None:
    """Restates the declared rule from a twin generator: select each position with
    probability 1/n (one forced if none), flip selected bits, step the rest by
    ``scale * z`` in unit coordinates and fold what leaves [0, 1]."""
    scale = 0.4
    total = np.zeros(_N, dtype=np.int64)
    for seed in range(300):
        x, _ = _parents(seed, same_bits=False)
        result = _mutate(x, scale, np.random.default_rng(seed))
        twin = np.random.default_rng(seed)
        sel = twin.random(_N) < 1.0 / _N
        if not sel.any():
            sel[twin.integers(_N)] = True
        np.testing.assert_array_equal(result.selected, sel)
        moved = np.flatnonzero(sel & ~_BITS)
        z = twin.standard_normal(moved.size)
        expected_fired = np.zeros(_N, dtype=bool)
        for i, zi in zip(moved.tolist(), z.tolist(), strict=True):
            width = _HIGH[i] - _LOW[i]
            u = (x[i] - _LOW[i]) / width + scale * zi
            if not 0.0 <= u <= 1.0:
                expected_fired[i] = True
                u = _fold(u)
            assert result.values[i] == pytest.approx(_LOW[i] + u * width, abs=1e-15)
        np.testing.assert_array_equal(result.fired, expected_fired)
        untouched = ~sel
        np.testing.assert_array_equal(result.values[untouched], x[untouched])
        total += result.fired
    assert total[_N_BITS:].sum() > 0
    assert total[:_N_BITS].sum() == 0


def test_mutation_without_bit_positions_moves_every_selected_position() -> None:
    low, high = np.zeros(4), np.ones(4)
    x = np.array([0.2, 0.4, 0.6, 0.8])
    result = _module().gaussian_mutation(x, low, high, 0.1, np.random.default_rng(6))
    assert np.all(result.values[result.selected] != x[result.selected])


def test_mutation_leaves_its_input_untouched() -> None:
    x, _ = _parents(4, same_bits=False)
    copy = x.copy()
    _mutate(x, 0.3, np.random.default_rng(0))
    np.testing.assert_array_equal(x, copy)


def test_mutation_matches_the_measured_arithmetic() -> None:
    digest = hashlib.sha256()
    total = np.zeros(_N, dtype=np.int64)
    for scale in (0.05, 0.3, 0.5):
        for seed in _SEEDS:
            x, _ = _parents(seed, same_bits=False)
            rng = np.random.default_rng([seed, 2, 7])
            result = _mutate(x, scale, rng)
            total += result.fired
            digest.update(result.values.tobytes())
            digest.update(result.selected.astype(np.uint8).tobytes())
            digest.update(result.fired.astype(np.uint8).tobytes())
            digest.update(np.array([rng.random()]).tobytes())
    assert total.tolist() == _MUTATION_FIRINGS
    assert digest.hexdigest() == _MUTATION_DIGEST


def test_mutation_is_reproducible_under_a_seed() -> None:
    x, _ = _parents(6, same_bits=False)
    a = _mutate(x, 0.3, np.random.default_rng(12))
    b = _mutate(x, 0.3, np.random.default_rng(12))
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(a.selected, b.selected)
    np.testing.assert_array_equal(a.fired, b.fired)


def _bad_mutation_inputs() -> list[tuple[str, dict]]:
    x, _ = _parents(0, same_bits=False)
    outside = x.copy()
    outside[4] = 0.75
    half_bit = x.copy()
    half_bit[0] = 0.5
    return [
        ("short low", {"low": _LOW[:-1]}),
        ("two-dimensional", {"values": np.tile(x, (2, 1))}),
        ("inverted bounds", {"low": _HIGH, "high": _LOW}),
        ("equal bounds", {"high": _LOW}),
        ("non-finite value", {"values": np.where(np.arange(_N) == 5, np.nan, x)}),
        ("value outside its bounds", {"values": outside}),
        ("bit not in {0, 1}", {"values": half_bit}),
        ("zero scale", {"scale": 0.0}),
        ("non-finite scale", {"scale": math.inf}),
        ("short bit layout", {"bits": _BITS[:-1]}),
        ("non-boolean bit layout", {"bits": _BITS.astype(int)}),
    ]


@pytest.mark.parametrize(
    "override",
    [o for _, o in _bad_mutation_inputs()],
    ids=[i for i, _ in _bad_mutation_inputs()],
)
def test_mutation_refuses_invalid_input(override: dict) -> None:
    x, _ = _parents(0, same_bits=False)
    kwargs = {"values": x, "low": _LOW, "high": _HIGH, "scale": 0.1, "bits": _BITS}
    kwargs.update(override)
    with pytest.raises(ValueError):
        _module().gaussian_mutation(
            kwargs["values"],
            kwargs["low"],
            kwargs["high"],
            kwargs["scale"],
            np.random.default_rng(0),
            bits=kwargs["bits"],
        )


def test_mutation_refuses_a_seed_in_place_of_a_generator() -> None:
    with pytest.raises(TypeError):
        _module().gaussian_mutation(
            np.array([0.5]),
            np.zeros(1),
            np.ones(1),
            0.1,
            0,  # ty: ignore[invalid-argument-type]
        )


# --- composition: scale step, recombination, mutation ------------------------------


def test_composed_variation_matches_the_measured_arithmetic() -> None:
    """Scale step, then recombination of the non-bit positions, then mutation, on
    one generator: the order and arithmetic of the measured birth. The parents
    share their bit positions, so no inheritance rule for those positions acts."""
    digest = hashlib.sha256()
    for seed in _SEEDS:
        p1, p2 = _parents(seed, same_bits=True)
        s1, s2 = _SCALE_STATES[seed % len(_SCALE_STATES)]
        rng = np.random.default_rng([seed, seed, 7])
        step = _scale_step(s1, s2, _N, rng)
        child = p1.copy()
        child[_N_BITS:] = _module().uniform_crossover(p1[_N_BITS:], p2[_N_BITS:], rng)
        result = _mutate(child, step.scale, rng)
        digest.update(result.values.tobytes())
        digest.update(np.array([step.parent_scale, step.scale]).tobytes())
        digest.update(result.fired.astype(np.uint8).tobytes())
        digest.update(np.array([rng.random()]).tobytes())
    assert digest.hexdigest() == _COMPOSED_DIGEST


# --- dependency scope and surface ----------------------------------------------------


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


def test_exposes_only_the_selected_primitives() -> None:
    assert set(_module().__all__) == _PUBLIC


def test_is_not_exported_from_the_package_root() -> None:
    import liq.gp

    module = _module()
    assert module.__name__ in sys.modules
    for name in _PUBLIC:
        assert name not in liq.gp.__all__
        assert not hasattr(liq.gp, name)
