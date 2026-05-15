"""Quality-diversity archive primitives (MAP-Elites style).

The archive is domain-agnostic. It bins individuals by behavior descriptors,
stores per-bin occupants with fitness vectors, and performs Pareto-aware
replacement with size-aware crowding eviction when a bin overfills.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, cast

ObjectiveDirection = Literal["maximize", "minimize"]


@dataclass(frozen=True)
class QDEntry:
    """Single occupant for a QD bin."""

    individual: object
    objectives: tuple[float, ...]
    descriptors: tuple[float, ...]


def _parse_bounds(descriptor_bounds: tuple[tuple[float, float], ...]) -> None:
    if len(descriptor_bounds) == 0:
        raise ValueError("descriptor_bounds must be non-empty")
    for lower, upper in descriptor_bounds:
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError("descriptor bounds must be finite")
        if upper <= lower:
            raise ValueError("descriptor bound lower must be < upper")


def _parse_objective_directions(
    objective_directions: list[ObjectiveDirection],
    dimensions: int,
) -> list[ObjectiveDirection]:
    if not objective_directions:
        raise ValueError("objective_directions must be non-empty")
    if len(objective_directions) < dimensions:
        # Missing directions default to maximize, matching existing engine defaults.
        default_directions: list[ObjectiveDirection] = [
            "maximize" for _ in range(dimensions - len(objective_directions))
        ]
        return list(objective_directions) + default_directions
    return list(objective_directions[:dimensions])


def _dominates(
    left: tuple[float, ...],
    right: tuple[float, ...],
    directions: list[ObjectiveDirection],
) -> bool:
    """Pareto dominance with directional objectives.

    ``left`` dominates ``right`` when it is no worse on every dimension and
    strictly better on at least one.
    """
    if len(left) != len(right):
        return False

    improved = False
    for lv, rv, direction in zip(left, right, directions, strict=True):
        if not math.isfinite(lv) or not math.isfinite(rv):
            return False
        if direction == "maximize":
            if lv < rv:
                return False
            improved = improved or lv > rv
        else:
            if lv > rv:
                return False
            improved = improved or lv < rv
    return improved


class QDArchive:
    """Domain-agnostic quality-diversity archive with binning and Pareto replacement."""

    def __init__(
        self,
        *,
        n_dims: int,
        bins_per_dim: int | tuple[int, ...],
        descriptor_bounds: tuple[tuple[float, float], ...],
        objective_directions: list[ObjectiveDirection],
        bin_capacity: int = 1,
    ) -> None:
        if n_dims < 1:
            raise ValueError("n_dims must be >= 1")
        if isinstance(bins_per_dim, int):
            if bins_per_dim < 1:
                raise ValueError("bins_per_dim must be >= 1")
            bins_per_dim_tuple = (bins_per_dim,) * n_dims
        else:
            if len(bins_per_dim) != n_dims:
                raise ValueError("bins_per_dim length must equal n_dims")
            if any(value < 1 for value in bins_per_dim):
                raise ValueError("bins_per_dim values must be >= 1")
            bins_per_dim_tuple = bins_per_dim

        if len(descriptor_bounds) != n_dims:
            raise ValueError("descriptor_bounds length must equal n_dims")
        if bin_capacity < 1:
            raise ValueError("bin_capacity must be >= 1")

        _parse_bounds(descriptor_bounds)
        if not objective_directions:
            raise ValueError("objective_directions must be non-empty")

        self.n_dims = n_dims
        self.bins_per_dim = bins_per_dim_tuple
        self.descriptor_bounds = descriptor_bounds
        self.objective_directions = objective_directions
        self.bin_capacity = bin_capacity
        self._bins: dict[tuple[int, ...], list[QDEntry]] = {}

    @property
    def total_bins(self) -> int:
        total = 1
        for value in self.bins_per_dim:
            total *= value
        return total

    @property
    def filled_bins(self) -> int:
        return len(self._bins)

    @staticmethod
    def _as_float_tuple(values: tuple[float, ...] | list[float]) -> tuple[float, ...]:
        return tuple(float(value) for value in values)

    def _bin_index(
        self, value: float, *, min_value: float, max_value: float, bins: int
    ) -> int:
        if not (min_value <= value <= max_value):
            raise ValueError(
                f"descriptor {value!r} outside bounds [{min_value}, {max_value}]"
            )
        span = max_value - min_value
        raw = (value - min_value) / span * bins
        index = int(raw)
        if index >= bins:
            index = bins - 1
        return index

    def _to_bin(self, descriptors: tuple[float, ...]) -> tuple[int, ...]:
        if len(descriptors) != self.n_dims:
            raise ValueError(
                f"descriptors length {len(descriptors)} must equal n_dims {self.n_dims}"
            )
        return tuple(
            self._bin_index(
                value=descriptor,
                min_value=low,
                max_value=high,
                bins=bins,
            )
            for descriptor, (low, high), bins in zip(
                descriptors,
                self.descriptor_bounds,
                self.bins_per_dim,
                strict=True,
            )
        )

    @staticmethod
    def _objective_values(entry: QDEntry) -> tuple[float, ...]:
        return entry.objectives

    def _crowding_distances(self, entries: list[QDEntry]) -> list[float]:
        if not entries:
            return []
        if len(entries) == 1:
            return [math.inf]
        if len(entries) == 2:
            return [math.inf, math.inf]

        dims = len(entries[0].objectives)
        distances = [0.0 for _ in entries]
        directions = _parse_objective_directions(self.objective_directions, dims)
        for dim in range(dims):
            ordered = sorted(
                range(len(entries)),
                key=lambda i: QDArchive._objective_values(entries[i])[dim],
            )
            if len(ordered) <= 1:
                continue

            lo_index = ordered[0]
            hi_index = ordered[-1]
            distances[lo_index] = math.inf
            distances[hi_index] = math.inf

            lo = QDArchive._objective_values(entries[lo_index])[dim]
            hi = QDArchive._objective_values(entries[hi_index])[dim]
            span = hi - lo
            if span == 0.0:
                continue
            if directions[dim] == "minimize":
                # distance uses raw values; span sign is handled by absolute.
                span = abs(span)
            for i in range(1, len(ordered) - 1):
                lower = ordered[i - 1]
                upper = ordered[i + 1]
                current = ordered[i]
                lower_value = QDArchive._objective_values(entries[lower])[dim]
                upper_value = QDArchive._objective_values(entries[upper])[dim]
                distances[current] += abs(upper_value - lower_value) / span
        return distances

    def _evict_worst(self, entries: list[QDEntry]) -> int:
        distances = self._crowding_distances(entries)
        # Tie-break by stable lower index to preserve deterministic behavior.
        worst_index = min(range(len(entries)), key=lambda i: (distances[i], i))
        return worst_index

    def insert(
        self,
        individual: object,
        objectives: tuple[float, ...] | list[float],
        descriptors: tuple[float, ...] | list[float],
    ) -> bool:
        """Insert an individual and return True if it is kept.

        Replacement follows Pareto dominance:
        - Replace dominated occupants.
        - Keep dominated occupants if newcomer is dominated.
        - Add non-dominated newcomer; if bin full, evict lowest crowding distance.
        """
        normalized_objectives = self._as_float_tuple(objectives)
        normalized_descriptors = self._as_float_tuple(descriptors)
        entry = QDEntry(
            individual=individual,
            objectives=normalized_objectives,
            descriptors=normalized_descriptors,
        )
        key = self._to_bin(normalized_descriptors)
        bucket = self._bins.setdefault(key, [])
        directions = _parse_objective_directions(
            self.objective_directions, len(normalized_objectives)
        )

        dominated_indices: list[int] = []
        for index, current in enumerate(bucket):
            if _dominates(current.objectives, normalized_objectives, directions):
                # Existing objective dominates newcomer
                return False
            if _dominates(normalized_objectives, current.objectives, directions):
                dominated_indices.append(index)

        for index in reversed(dominated_indices):
            bucket.pop(index)

        bucket.append(entry)
        if len(bucket) > self.bin_capacity:
            victim = self._evict_worst(bucket)
            bucket.pop(victim)
        return True

    def elites(self) -> list[object]:
        """Return all occupants from all non-empty bins."""
        occupants: list[object] = []
        for key in sorted(self._bins):
            occupants.extend(entry.individual for entry in self._bins[key])
        return occupants

    def sample(
        self,
        n: int,
        rng,
        *,
        coverage_weight: float = 0.5,
    ) -> list[object]:
        """Sample with alternate pressure between fitness and coverage bins."""
        if n < 0:
            raise ValueError("n must be >= 0")
        if not 0.0 <= coverage_weight <= 1.0:
            raise ValueError("coverage_weight must be in [0.0, 1.0]")

        if n == 0:
            return []

        if not self._bins:
            return []

        selected: list[object] = []
        all_entries = [entry for entries in self._bins.values() for entry in entries]
        if not all_entries:
            return []

        directions = _parse_objective_directions(
            self.objective_directions,
            len(all_entries[0].objectives),
        )
        max_first: bool = directions[0] == "maximize"

        for _ in range(n):
            underfilled = [
                (bin_key, bucket)
                for bin_key, bucket in self._bins.items()
                if 0 < len(bucket) < self.bin_capacity
            ]
            use_coverage = bool(underfilled) and rng.random() < coverage_weight

            if use_coverage:
                underfilled.sort(key=lambda item: (len(item[1]), item[0]))
                bucket_entries = underfilled[0][1]
            else:
                candidate_entries = [
                    entry for bucket in self._bins.values() for entry in bucket
                ]
                if max_first:
                    best_value = max(
                        entry.objectives[0] if len(entry.objectives) > 0 else 0.0
                        for entry in candidate_entries
                    )
                    top = [
                        entry
                        for entry in candidate_entries
                        if (entry.objectives[0] if len(entry.objectives) > 0 else 0.0)
                        == best_value
                    ]
                else:
                    best_value = min(
                        entry.objectives[0] if len(entry.objectives) > 0 else 0.0
                        for entry in candidate_entries
                    )
                    top = [
                        entry
                        for entry in candidate_entries
                        if (entry.objectives[0] if len(entry.objectives) > 0 else 0.0)
                        == best_value
                    ]
                bucket_entries = [top[int(rng.integers(len(top)))]]

            selected.append(
                bucket_entries[int(rng.integers(len(bucket_entries)))].individual
            )

        return selected

    def coverage_report(self) -> dict[str, object]:
        """Return fill ratio and per-dimension histogram counts."""
        hist: list[list[int]] = [[0 for _ in range(n)] for n in self.bins_per_dim]
        for bin_key, entries in self._bins.items():
            for dim, index in enumerate(bin_key):
                hist[dim][index] += len(entries)
        fill_ratio = 0.0 if self.total_bins == 0 else self.filled_bins / self.total_bins
        return {
            "filled_bins": self.filled_bins,
            "total_bins": self.total_bins,
            "fill_ratio": fill_ratio,
            "dimension_histograms": hist,
        }

    def to_dict(self) -> dict[str, object]:
        """Serialize all entries for checkpoint/restart."""
        entries: list[dict[str, object]] = []
        for bin_key in sorted(self._bins):
            for entry in self._bins[bin_key]:
                individual = entry.individual
                serialized = getattr(individual, "name", None)
                if not isinstance(serialized, str):
                    serialized = repr(individual)
                entries.append(
                    {
                        "individual_repr": serialized,
                        "objectives": list(entry.objectives),
                        "descriptors": list(entry.descriptors),
                        "bin": list(bin_key),
                    }
                )
        return {
            "n_dims": self.n_dims,
            "bins_per_dim": list(self.bins_per_dim),
            "descriptor_bounds": [list(item) for item in self.descriptor_bounds],
            "objective_directions": list(self.objective_directions),
            "bin_capacity": self.bin_capacity,
            "entries": entries,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
        *,
        restore_individual: Callable[[str], object] | None = None,
    ) -> QDArchive:
        """Reconstruct a serializable archive payload."""
        bins_per_dim = cast(int | tuple[int, ...] | list[int], payload["bins_per_dim"])
        descriptor_bounds = cast(
            Sequence[Sequence[float]], payload["descriptor_bounds"]
        )
        objective_directions = cast(
            list[ObjectiveDirection], payload["objective_directions"]
        )
        archive = cls(
            n_dims=int(cast(int, payload["n_dims"])),
            bins_per_dim=tuple(bins_per_dim)
            if not isinstance(bins_per_dim, int)
            else bins_per_dim,
            descriptor_bounds=tuple(
                (float(lower), float(upper)) for lower, upper in descriptor_bounds
            ),
            objective_directions=objective_directions,
            bin_capacity=int(cast(int, payload["bin_capacity"])),
        )
        if restore_individual is None:
            restore_individual = str
        entries = cast(Sequence[dict[str, object]], payload["entries"])
        for raw in entries:
            archive.insert(
                individual=restore_individual(str(raw["individual_repr"])),
                objectives=tuple(
                    float(value) for value in cast(Sequence[float], raw["objectives"])
                ),
                descriptors=tuple(
                    float(value) for value in cast(Sequence[float], raw["descriptors"])
                ),
            )
        return archive
