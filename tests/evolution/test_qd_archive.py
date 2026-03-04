"""Tests for the domain-agnostic quality-diversity archive."""

from __future__ import annotations

from liq.gp.evolution.qd_archive import QDArchive
from liq.gp.program.ast import Program, TerminalNode
from liq.gp.types import Series


def _program(name: str) -> Program:
    return TerminalNode(name=name, output_type=Series)


def _make_archive() -> QDArchive:
    return QDArchive(
        n_dims=2,
        bins_per_dim=(3, 3),
        descriptor_bounds=((0.0, 1.0), (0.0, 1.0)),
        objective_directions=["maximize", "maximize"],
        bin_capacity=2,
    )


def test_dominating_newcomer_replaces_existing_occupants() -> None:
    """A strict Pareto dominator should replace dominated occupants."""
    archive = _make_archive()
    archive.insert(_program("weak"), (1.0, 1.0), (0.1, 0.1))
    # Dominated by (1.0, 1.0) => rejected.
    assert not archive.insert(_program("dominated"), (0.5, 0.8), (0.1, 0.1))
    # Dominates (1.0, 1.0) => replaces it.
    assert archive.insert(_program("dominator"), (2.0, 3.0), (0.1, 0.1))
    elites = archive.elites()
    assert len(elites) == 1
    assert elites[0].name == "dominator"  # type: ignore[attr-defined]


def test_non_dominated_newcomers_evicted_by_crowding_distance() -> None:
    """When a bin overfills, the worst crowding-distance entrant is removed."""
    archive = _make_archive()
    archive.insert(_program("a"), (1.0, 0.0), (0.1, 0.1))
    archive.insert(_program("b"), (0.0, 1.0), (0.1, 0.1))
    # Non-dominated newcomer triggers eviction when capacity is exceeded.
    archive.insert(_program("c"), (0.5, 0.5), (0.1, 0.1))
    elites = archive.elites()
    names = {p.name for p in elites}  # type: ignore[attr-defined]
    assert "c" not in names
    assert names == {"a", "b"}


def test_coverage_report_reports_fill_ratio_and_histograms() -> None:
    """coverage_report counts filled bins and per-dimension occupancy."""
    archive = QDArchive(
        n_dims=2,
        bins_per_dim=(2, 2),
        descriptor_bounds=((0.0, 1.0), (0.0, 1.0)),
        objective_directions=["maximize", "maximize"],
        bin_capacity=1,
    )
    archive.insert(_program("p1"), (1.0,), (0.1, 0.1))
    archive.insert(_program("p2"), (2.0,), (0.9, 0.9))
    report = archive.coverage_report()
    assert report["filled_bins"] == 2
    assert report["total_bins"] == 4
    assert report["fill_ratio"] == 0.5
    assert report["dimension_histograms"] == [[1, 1], [1, 1]]


def test_under_filled_bins_are_sampled_under_coverage_pressure() -> None:
    """Coverage-first sampling selects candidates from partially filled bins."""
    archive = _make_archive()
    archive.insert(_program("full_a"), (1.0,), (0.1, 0.1))
    archive.insert(_program("full_b"), (1.0,), (0.1, 0.1))
    archive.insert(_program("underfilled"), (1.0,), (0.9, 0.9))
    sampled = archive.sample(3, rng=__import__("numpy").random.default_rng(7), coverage_weight=1.0)
    assert sampled
    assert all(p.name == "underfilled" for p in sampled)  # type: ignore[attr-defined]


def test_sample_empty_archive_is_empty() -> None:
    """Sampling an empty archive returns an empty list."""
    archive = _make_archive()
    sampled = archive.sample(3, rng=__import__("numpy").random.default_rng(1))
    assert sampled == []


def test_coverage_report_empty_archive() -> None:
    """coverage_report is still valid for an empty archive."""
    archive = _make_archive()
    report = archive.coverage_report()
    assert report["filled_bins"] == 0
    assert report["total_bins"] == 9
    assert report["fill_ratio"] == 0.0
    assert report["dimension_histograms"] == [[0, 0, 0], [0, 0, 0]]


def test_roundtrip_serialization_is_recoverable() -> None:
    """Serializing and restoring an archive preserves reportable state."""
    archive = _make_archive()
    archive.insert(_program("a"), (1.0, 2.0), (0.1, 0.4))
    archive.insert(_program("b"), (2.0, 1.0), (0.2, 0.5))
    payload = archive.to_dict()
    restored = QDArchive.from_dict(
        payload,
        restore_individual=lambda name: _program(name),
    )
    assert restored.coverage_report() == archive.coverage_report()
    assert {repr(p) for p in restored.elites()} == {repr(_program("a")), repr(_program("b"))}


def test_single_objective_singleton_bin_replacement() -> None:
    """With one objective, dominance degenerates to scalar comparison."""
    archive = QDArchive(
        n_dims=1,
        bins_per_dim=2,
        descriptor_bounds=((0.0, 1.0),),
        objective_directions=["maximize"],
        bin_capacity=1,
    )
    archive.insert(_program("low"), (1.0,), (0.1,))
    assert archive.insert(_program("higher"), (3.0,), (0.2,))
    assert not archive.insert(_program("lowest"), (0.5,), (0.2,))
    elites = archive.elites()
    assert len(elites) == 1
    assert elites[0].name == "higher"  # type: ignore[attr-defined]


def test_archive_is_bounded_by_bin_capacity() -> None:
    """Elite count cannot exceed n_bins × bin_capacity."""
    archive = QDArchive(
        n_dims=2,
        bins_per_dim=(4, 4),
        descriptor_bounds=((0.0, 1.0), (0.0, 1.0)),
        objective_directions=["maximize", "maximize"],
        bin_capacity=2,
    )
    total_bins = archive.total_bins

    for idx in range(300):
        descriptor_x = (idx % 40) / 39.0
        descriptor_y = ((idx * 7) % 40) / 39.0
        archive.insert(
            _program(f"p{idx}"),
            (float(idx % 10), float((idx * 2) % 10)),
            (descriptor_x, descriptor_y),
        )

    assert len(archive.elites()) <= total_bins * archive.bin_capacity
    assert archive.coverage_report()["filled_bins"] <= archive.total_bins


def test_deterministic_sampling_given_seed() -> None:
    """Fixed seed + same archive produces identical samples."""
    archive_a = _make_archive()
    archive_b = _make_archive()
    for i in range(4):
        archive_a.insert(_program(f"x{i}"), (float(i),), (0.1, 0.2))
        archive_b.insert(_program(f"x{i}"), (float(i),), (0.1, 0.2))
    rng_a = __import__("numpy").random.default_rng(12)
    rng_b = __import__("numpy").random.default_rng(12)
    s_a = archive_a.sample(5, rng_a, coverage_weight=0.3)
    s_b = archive_b.sample(5, rng_b, coverage_weight=0.3)
    assert [p.name for p in s_a] == [p.name for p in s_b]  # type: ignore[attr-defined]
