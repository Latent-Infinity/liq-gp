"""Edge-case and error-path tests for the quality-diversity archive."""

from __future__ import annotations

import math

import numpy as np
import pytest

from liq.gp.evolution.qd_archive import (
    QDArchive,
    QDEntry,
    _parse_objective_directions,
)
from liq.gp.program.ast import Program, TerminalNode
from liq.gp.types import Series


def _program(name: str) -> Program:
    return TerminalNode(name=name, output_type=Series)


def _single_bin_archive(
    *,
    directions: list[str],
    bin_capacity: int = 1,
) -> QDArchive:
    return QDArchive(
        n_dims=1,
        bins_per_dim=1,
        descriptor_bounds=((0.0, 1.0),),
        objective_directions=directions,  # type: ignore[arg-type]
        bin_capacity=bin_capacity,
    )


def test_parse_objective_directions_rejects_empty() -> None:
    """An empty direction list has no defensible default and is rejected."""
    with pytest.raises(ValueError, match="objective_directions must be non-empty"):
        _parse_objective_directions([], 2)


def test_parse_objective_directions_pads_missing_dimensions_with_maximize() -> None:
    """Directions shorter than the objective vector default to maximize."""
    assert _parse_objective_directions(["minimize"], 3) == [
        "minimize",
        "maximize",
        "maximize",
    ]


def test_missing_direction_defaults_to_maximize_during_insert() -> None:
    """Padding is observable through dominance on the undeclared dimension.

    The archive declares one ``minimize`` direction but receives two-objective
    vectors.  ``(1.0, 5.0)`` dominates ``(2.0, 4.0)`` only if the second
    dimension defaults to ``maximize``.
    """
    archive = _single_bin_archive(directions=["minimize"])
    assert archive.insert(_program("a"), (1.0, 5.0), (0.5,))
    assert not archive.insert(_program("b"), (2.0, 4.0), (0.5,))
    elites = archive.elites()
    assert [p.name for p in elites] == ["a"]  # type: ignore[attr-defined]


def test_descriptor_above_upper_bound_is_rejected() -> None:
    """Descriptors outside the declared bounds cannot be binned."""
    archive = _single_bin_archive(directions=["maximize"])
    with pytest.raises(ValueError, match=r"outside bounds \[0.0, 1.0\]"):
        archive.insert(_program("a"), (1.0,), (1.5,))


def test_descriptor_below_lower_bound_is_rejected() -> None:
    """The bounds check is two-sided."""
    archive = _single_bin_archive(directions=["maximize"])
    with pytest.raises(ValueError, match=r"outside bounds \[0.0, 1.0\]"):
        archive.insert(_program("a"), (1.0,), (-0.001,))


def test_crowding_distance_degenerate_population_sizes() -> None:
    """Fewer than three occupants means every occupant is a boundary point."""
    archive = _single_bin_archive(directions=["maximize"], bin_capacity=4)
    one = QDEntry(individual=_program("a"), objectives=(1.0,), descriptors=(0.5,))
    two = QDEntry(individual=_program("b"), objectives=(2.0,), descriptors=(0.5,))

    assert archive._crowding_distances([]) == []
    assert archive._crowding_distances([one]) == [math.inf]
    assert archive._crowding_distances([one, two]) == [math.inf, math.inf]


def test_crowding_eviction_under_minimize_directions() -> None:
    """Minimising objectives still evict the most crowded interior occupant.

    Four mutually non-dominated points sit on a straight front; the two
    extremes have infinite crowding distance and the interior pair ties, so
    the lower-indexed interior occupant is removed.
    """
    archive = _single_bin_archive(directions=["minimize", "minimize"], bin_capacity=3)
    archive.insert(_program("a"), (0.0, 3.0), (0.5,))
    archive.insert(_program("b"), (1.0, 2.0), (0.5,))
    archive.insert(_program("c"), (2.0, 1.0), (0.5,))
    archive.insert(_program("d"), (3.0, 0.0), (0.5,))

    names = {p.name for p in archive.elites()}  # type: ignore[attr-defined]
    assert names == {"a", "c", "d"}


def test_sample_rejects_negative_count() -> None:
    """Negative sample counts are a caller error."""
    archive = _single_bin_archive(directions=["maximize"])
    with pytest.raises(ValueError, match="n must be >= 0"):
        archive.sample(-1, np.random.default_rng(0))


def test_sample_rejects_out_of_range_coverage_weight() -> None:
    """coverage_weight is a probability and must stay in [0, 1]."""
    archive = _single_bin_archive(directions=["maximize"])
    with pytest.raises(ValueError, match=r"coverage_weight must be in \[0.0, 1.0\]"):
        archive.sample(1, np.random.default_rng(0), coverage_weight=1.5)
    with pytest.raises(ValueError, match=r"coverage_weight must be in \[0.0, 1.0\]"):
        archive.sample(1, np.random.default_rng(0), coverage_weight=-0.5)


def test_sample_zero_returns_empty_without_consuming_rng() -> None:
    """Sampling zero individuals short-circuits before any draw."""
    archive = _single_bin_archive(directions=["maximize"])
    archive.insert(_program("a"), (1.0,), (0.5,))
    rng = np.random.default_rng(3)
    assert archive.sample(0, rng) == []
    # The generator must be untouched, so the next draw matches a fresh stream.
    assert rng.random() == np.random.default_rng(3).random()


def test_fitness_pressure_follows_first_objective_direction() -> None:
    """Under fitness pressure a minimize archive picks the smallest value."""
    bounds = ((0.0, 1.0),)
    minimizing = QDArchive(
        n_dims=1,
        bins_per_dim=2,
        descriptor_bounds=bounds,
        objective_directions=["minimize"],
        bin_capacity=1,
    )
    maximizing = QDArchive(
        n_dims=1,
        bins_per_dim=2,
        descriptor_bounds=bounds,
        objective_directions=["maximize"],
        bin_capacity=1,
    )
    for archive in (minimizing, maximizing):
        archive.insert(_program("low"), (1.0,), (0.1,))
        archive.insert(_program("high"), (5.0,), (0.9,))

    # Every bin is at capacity, so coverage pressure never applies.
    low = minimizing.sample(3, np.random.default_rng(11), coverage_weight=1.0)
    high = maximizing.sample(3, np.random.default_rng(11), coverage_weight=1.0)
    assert [p.name for p in low] == ["low"] * 3  # type: ignore[attr-defined]
    assert [p.name for p in high] == ["high"] * 3  # type: ignore[attr-defined]


def test_to_dict_falls_back_to_repr_for_unnamed_individuals() -> None:
    """Individuals without a string ``name`` serialize via ``repr``."""
    archive = _single_bin_archive(directions=["maximize"])
    individual = object()
    archive.insert(individual, (1.0,), (0.5,))
    payload = archive.to_dict()
    entries = payload["entries"]
    assert isinstance(entries, list)
    assert entries[0]["individual_repr"] == repr(individual)


def test_to_dict_falls_back_to_repr_for_non_string_name() -> None:
    """A non-string ``name`` attribute is not trusted as the serialized form."""

    class _Numbered:
        name = 7

        def __repr__(self) -> str:
            return "<numbered>"

    archive = _single_bin_archive(directions=["maximize"])
    archive.insert(_Numbered(), (1.0,), (0.5,))
    entries = archive.to_dict()["entries"]
    assert isinstance(entries, list)
    assert entries[0]["individual_repr"] == "<numbered>"


def test_from_dict_without_restorer_keeps_serialized_strings() -> None:
    """The default restorer yields the stored repr strings verbatim."""
    archive = QDArchive(
        n_dims=1,
        bins_per_dim=2,
        descriptor_bounds=((0.0, 1.0),),
        objective_directions=["maximize"],
        bin_capacity=1,
    )
    archive.insert(_program("a"), (1.0,), (0.1,))
    archive.insert(_program("b"), (2.0,), (0.9,))

    restored = QDArchive.from_dict(archive.to_dict())
    assert restored.elites() == ["a", "b"]
    assert restored.coverage_report() == archive.coverage_report()


def test_sample_tolerates_entries_without_objectives() -> None:
    """``insert`` accepts an empty objectives tuple, so ``sample`` must too.

    Every fitness comparison in ``sample`` already guards for it; only the
    direction lookup did not, which made an accepted entry unsamplable.
    """
    archive = QDArchive(
        n_dims=1,
        bins_per_dim=2,
        descriptor_bounds=((0.0, 1.0),),
        objective_directions=["maximize"],
        bin_capacity=1,
    )
    individual = _program("x")
    assert archive.insert(individual, (), (0.5,)) is True

    rng = np.random.default_rng(0)
    assert archive.sample(1, rng, coverage_weight=0.0) == [individual]
    assert archive.sample(1, rng, coverage_weight=1.0) == [individual]
