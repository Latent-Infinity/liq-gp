"""Tests for semantic deduplication (FR-8)."""

from __future__ import annotations

import numpy as np
import pytest

from liq.gp.config import GPConfig
from liq.gp.evolution.diversity import (
    compute_fingerprint,
    deduplicate_population,
    sample_reference_context,
)
from liq.gp.primitives.registry import PrimitiveInfo, PrimitiveRegistry
from liq.gp.program.ast import (
    ConstantNode,
    FunctionNode,
    Program,
    TerminalNode,
)
from liq.gp.types import BoolSeries, Series

# --- helpers ---------------------------------------------------------------


def _add_info() -> PrimitiveInfo:
    return PrimitiveInfo(
        name="add",
        category="numeric",
        arity=2,
        input_types=(Series, Series),
        output_type=Series,
        callable=lambda a, b: a + b,
    )


def _mul_info() -> PrimitiveInfo:
    return PrimitiveInfo(
        name="mul",
        category="numeric",
        arity=2,
        input_types=(Series, Series),
        output_type=Series,
        callable=lambda a, b: a * b,
    )


def _neg_info() -> PrimitiveInfo:
    return PrimitiveInfo(
        name="neg",
        category="numeric",
        arity=1,
        input_types=(Series,),
        output_type=Series,
        callable=lambda a: -a,
    )


def _make_registry() -> PrimitiveRegistry:
    """Build a minimal registry for testing diversity."""
    reg = PrimitiveRegistry()
    reg.register("close", lambda: None, input_types=(), output_type=Series)
    reg.register("volume", lambda: None, input_types=(), output_type=Series)
    reg.register(
        "add",
        lambda a, b: a + b,
        category="numeric",
        input_types=(Series, Series),
        output_type=Series,
    )
    reg.register(
        "neg",
        lambda a: -a,
        category="numeric",
        input_types=(Series,),
        output_type=Series,
    )
    reg.register(
        "mul",
        lambda a, b: a * b,
        category="numeric",
        input_types=(Series, Series),
        output_type=Series,
    )
    return reg


def _make_context(n: int = 100) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "close": rng.standard_normal(n).astype(np.float64),
        "volume": rng.standard_normal(n).astype(np.float64),
    }


def _make_config(**overrides: object) -> GPConfig:
    """Build a GPConfig with sensible test defaults."""
    defaults: dict[str, object] = {
        "population_size": 20,
        "max_depth": 4,
        "generations": 1,
        "seed": 42,
    }
    defaults.update(overrides)
    return GPConfig(**defaults)  # type: ignore[arg-type]


# --- compute_fingerprint ---------------------------------------------------


class TestComputeFingerprint:
    """Fingerprinting: evaluate + round + tobytes (FR-8.1)."""

    def test_identical_programs_produce_identical_fingerprints(self) -> None:
        """FR-8.1: Identical programs must produce identical fingerprints."""
        ctx = _make_context()
        ref = sample_reference_context(ctx, ref_size=10, rng=np.random.default_rng(0))
        node = TerminalNode(name="close", output_type=Series)
        fp1 = compute_fingerprint(node, ref, precision=6)
        fp2 = compute_fingerprint(node, ref, precision=6)
        assert fp1 == fp2

    def test_different_programs_produce_different_fingerprints(self) -> None:
        """Different semantic behaviour should (usually) yield different fingerprints."""
        ctx = _make_context()
        ref = sample_reference_context(ctx, ref_size=10, rng=np.random.default_rng(0))
        close_node = TerminalNode(name="close", output_type=Series)
        volume_node = TerminalNode(name="volume", output_type=Series)
        fp_close = compute_fingerprint(close_node, ref, precision=6)
        fp_volume = compute_fingerprint(volume_node, ref, precision=6)
        assert fp_close != fp_volume

    def test_semantically_equivalent_programs_same_fingerprint(self) -> None:
        """add(x, 0) is semantically equivalent to x (FR-8.1)."""
        ctx = _make_context()
        ref = sample_reference_context(ctx, ref_size=20, rng=np.random.default_rng(0))
        close_node = TerminalNode(name="close", output_type=Series)
        zero = ConstantNode(value=0.0)
        add_zero = FunctionNode(primitive=_add_info(), children=(close_node, zero))

        fp_x = compute_fingerprint(close_node, ref, precision=6)
        fp_add_zero = compute_fingerprint(add_zero, ref, precision=6)
        assert fp_x == fp_add_zero

    def test_fingerprint_returns_bytes(self) -> None:
        """Fingerprint is a bytes object."""
        ctx = _make_context()
        ref = sample_reference_context(ctx, ref_size=5, rng=np.random.default_rng(0))
        node = TerminalNode(name="close", output_type=Series)
        fp = compute_fingerprint(node, ref, precision=6)
        assert isinstance(fp, bytes)

    def test_fingerprint_nan_handling(self) -> None:
        """NaN values are replaced with 0.0 before fingerprinting so that
        two programs that both produce NaN at the same positions get the
        same fingerprint."""
        ref = {
            "close": np.array([1.0, np.nan, 3.0], dtype=np.float64),
            "volume": np.array([4.0, 5.0, 6.0], dtype=np.float64),
        }
        node = TerminalNode(name="close", output_type=Series)
        fp1 = compute_fingerprint(node, ref, precision=6)
        fp2 = compute_fingerprint(node, ref, precision=6)
        assert fp1 == fp2
        # The fingerprint should be deterministic even with NaN
        assert isinstance(fp1, bytes)

    def test_precision_affects_fingerprint(self) -> None:
        """Different precision values can produce different fingerprints for
        values that differ only at high precision."""
        ref = {
            "close": np.array([1.0000001, 2.0000002], dtype=np.float64),
        }
        node = TerminalNode(name="close", output_type=Series)
        fp_high = compute_fingerprint(node, ref, precision=10)
        fp_low = compute_fingerprint(node, ref, precision=2)
        # At precision=2, 1.0000001 rounds to 1.00 and at precision=10 it
        # remains 1.0000001000, so the bytes representation differs.
        assert fp_high != fp_low


# --- sample_reference_context -----------------------------------------------


class TestSampleReferenceContext:
    """Reference dataset sampling (FR-8.2)."""

    def test_reference_context_has_correct_size(self) -> None:
        ctx = _make_context(100)
        ref = sample_reference_context(ctx, ref_size=10, rng=np.random.default_rng(0))
        for arr in ref.values():
            assert len(arr) == 10

    def test_reference_context_preserves_keys(self) -> None:
        ctx = _make_context(100)
        ref = sample_reference_context(ctx, ref_size=10, rng=np.random.default_rng(0))
        assert set(ref.keys()) == set(ctx.keys())

    def test_reference_context_deterministic_with_seed(self) -> None:
        """Same rng seed produces the same reference context (FR-8.2)."""
        ctx = _make_context(100)
        ref1 = sample_reference_context(ctx, ref_size=10, rng=np.random.default_rng(42))
        ref2 = sample_reference_context(ctx, ref_size=10, rng=np.random.default_rng(42))
        for key in ref1:
            np.testing.assert_array_equal(ref1[key], ref2[key])

    def test_reference_context_different_seed_differs(self) -> None:
        """Different rng seeds produce different reference contexts."""
        ctx = _make_context(100)
        ref1 = sample_reference_context(ctx, ref_size=10, rng=np.random.default_rng(1))
        ref2 = sample_reference_context(ctx, ref_size=10, rng=np.random.default_rng(2))
        any_differ = False
        for key in ref1:
            if not np.array_equal(ref1[key], ref2[key]):
                any_differ = True
                break
        assert any_differ

    def test_ref_size_larger_than_context_clamps(self) -> None:
        """When ref_size >= context length, use the full context."""
        ctx = _make_context(5)
        ref = sample_reference_context(ctx, ref_size=100, rng=np.random.default_rng(0))
        for key in ref:
            assert len(ref[key]) == 5
            np.testing.assert_array_equal(ref[key], ctx[key])

    def test_sampled_rows_are_from_context(self) -> None:
        """Each row in the reference context is an actual row from the source."""
        ctx = _make_context(50)
        ref = sample_reference_context(ctx, ref_size=10, rng=np.random.default_rng(0))
        close_vals = set(ctx["close"].tolist())
        for v in ref["close"]:
            assert v in close_vals


# --- deduplicate_population ------------------------------------------------


class TestDeduplicatePopulation:
    """Duplicate replacement and unique ratio (FR-8.4, FR-8.5)."""

    def test_all_unique_returns_same_population(self) -> None:
        """When all programs are unique, population is unchanged."""
        ctx = _make_context()
        ref = sample_reference_context(ctx, ref_size=10, rng=np.random.default_rng(0))
        reg = _make_registry()
        config = _make_config(semantic_dedup_enabled=True, semantic_precision=6)
        # Build a small population with distinct semantics
        programs: list[Program] = [
            TerminalNode(name="close", output_type=Series),
            TerminalNode(name="volume", output_type=Series),
            FunctionNode(
                primitive=_neg_info(),
                children=(TerminalNode(name="close", output_type=Series),),
            ),
        ]
        rng = np.random.default_rng(0)
        new_pop, ratio = deduplicate_population(programs, ref, reg, config, rng)
        assert len(new_pop) == len(programs)
        assert ratio == pytest.approx(1.0)
        # Original programs preserved
        for orig, new in zip(programs, new_pop, strict=True):
            assert orig == new

    def test_duplicates_are_replaced(self) -> None:
        """FR-8.4: Duplicate programs are replaced with new random individuals."""
        ctx = _make_context()
        ref = sample_reference_context(ctx, ref_size=10, rng=np.random.default_rng(0))
        reg = _make_registry()
        config = _make_config(semantic_dedup_enabled=True, semantic_precision=6)
        close_node = TerminalNode(name="close", output_type=Series)
        # All duplicates
        programs: list[Program] = [close_node, close_node, close_node, close_node]
        rng = np.random.default_rng(0)
        new_pop, ratio = deduplicate_population(programs, ref, reg, config, rng)
        assert len(new_pop) == 4
        # First occurrence kept, rest replaced
        assert new_pop[0] == close_node
        # Unique ratio before replacement
        assert ratio == pytest.approx(1 / 4)

    def test_unique_ratio_metric(self) -> None:
        """FR-8.5: unique_semantics_ratio = unique fingerprints / pop size."""
        ctx = _make_context()
        ref = sample_reference_context(ctx, ref_size=10, rng=np.random.default_rng(0))
        reg = _make_registry()
        config = _make_config(semantic_dedup_enabled=True, semantic_precision=6)
        close_node = TerminalNode(name="close", output_type=Series)
        volume_node = TerminalNode(name="volume", output_type=Series)
        # 2 unique out of 4
        programs: list[Program] = [close_node, volume_node, close_node, volume_node]
        rng = np.random.default_rng(0)
        _, ratio = deduplicate_population(programs, ref, reg, config, rng)
        assert ratio == pytest.approx(2 / 4)

    def test_disabled_dedup_returns_original(self) -> None:
        """FR-8.6: When dedup is disabled, population is unchanged."""
        ctx = _make_context()
        ref = sample_reference_context(ctx, ref_size=10, rng=np.random.default_rng(0))
        reg = _make_registry()
        config = _make_config(semantic_dedup_enabled=False, semantic_precision=6)
        close_node = TerminalNode(name="close", output_type=Series)
        programs: list[Program] = [close_node, close_node, close_node]
        rng = np.random.default_rng(0)
        new_pop, ratio = deduplicate_population(programs, ref, reg, config, rng)
        assert len(new_pop) == 3
        # All programs are the same, unchanged
        for p in new_pop:
            assert p == close_node
        # Ratio is still computed even when disabled
        assert ratio == pytest.approx(1 / 3)

    def test_population_size_preserved_after_dedup(self) -> None:
        """Population size is unchanged after deduplication."""
        ctx = _make_context()
        ref = sample_reference_context(ctx, ref_size=10, rng=np.random.default_rng(0))
        reg = _make_registry()
        config = _make_config(semantic_dedup_enabled=True, semantic_precision=6)
        close_node = TerminalNode(name="close", output_type=Series)
        volume_node = TerminalNode(name="volume", output_type=Series)
        programs: list[Program] = [
            close_node,
            close_node,
            volume_node,
            volume_node,
            close_node,
        ]
        rng = np.random.default_rng(42)
        new_pop, _ = deduplicate_population(programs, ref, reg, config, rng)
        assert len(new_pop) == len(programs)

    def test_replacement_programs_are_valid(self) -> None:
        """Replacement programs should have valid output types."""
        ctx = _make_context()
        ref = sample_reference_context(ctx, ref_size=10, rng=np.random.default_rng(0))
        reg = _make_registry()
        config = _make_config(semantic_dedup_enabled=True, semantic_precision=6)
        close_node = TerminalNode(name="close", output_type=Series)
        programs: list[Program] = [close_node] * 10
        rng = np.random.default_rng(0)
        new_pop, _ = deduplicate_population(programs, ref, reg, config, rng)
        for prog in new_pop:
            assert prog.output_type is Series

    def test_semantically_equivalent_counted_as_duplicate(self) -> None:
        """add(x, 0) and x are semantically equivalent and counted as duplicates."""
        ctx = _make_context()
        ref = sample_reference_context(ctx, ref_size=20, rng=np.random.default_rng(0))
        reg = _make_registry()
        config = _make_config(semantic_dedup_enabled=True, semantic_precision=6)
        close_node = TerminalNode(name="close", output_type=Series)
        zero = ConstantNode(value=0.0)
        add_zero = FunctionNode(primitive=_add_info(), children=(close_node, zero))
        # x and add(x, 0) are semantically equivalent
        programs: list[Program] = [close_node, add_zero]
        rng = np.random.default_rng(0)
        _, ratio = deduplicate_population(programs, ref, reg, config, rng)
        assert ratio == pytest.approx(1 / 2)

    def test_deduplicate_population_accepts_precomputed_fingerprints(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FR-8.4: caller-provided fingerprints are consumed directly."""
        ctx = _make_context()
        ref = sample_reference_context(ctx, ref_size=10, rng=np.random.default_rng(0))
        reg = _make_registry()
        config = _make_config(semantic_dedup_enabled=True, semantic_precision=6)
        close_node = TerminalNode(name="close", output_type=Series)
        volume_node = TerminalNode(name="volume", output_type=Series)
        programs: list[Program] = [close_node, volume_node]
        fingerprints = [
            compute_fingerprint(close_node, ref, precision=6),
            compute_fingerprint(volume_node, ref, precision=6),
        ]

        def _unexpected_compute(*_args: object, **_kwargs: object) -> bytes:
            raise RuntimeError("compute_fingerprint should not be called")

        import liq.gp.evolution.diversity as diversity

        monkeypatch.setattr(diversity, "compute_fingerprint", _unexpected_compute)
        new_pop, _ = deduplicate_population(
            programs,
            ref,
            reg,
            config,
            np.random.default_rng(0),
            fingerprints=fingerprints,
        )
        assert new_pop == programs

    def test_deduplicate_population_validates_fingerprint_length(self) -> None:
        """FR-8.4: fingerprint length must match population length."""
        ctx = _make_context()
        ref = sample_reference_context(ctx, ref_size=10, rng=np.random.default_rng(0))
        reg = _make_registry()
        config = _make_config(semantic_dedup_enabled=True, semantic_precision=6)
        close_node = TerminalNode(name="close", output_type=Series)
        bad_fingerprints = [compute_fingerprint(close_node, ref, precision=6)]
        with pytest.raises(ValueError, match="must be provided for every individual"):
            deduplicate_population(
                [close_node, close_node],
                ref,
                reg,
                config,
                np.random.default_rng(0),
                fingerprints=bad_fingerprints,
            )

    def test_dedup_preserves_output_type_for_bool_series(self) -> None:
        """Replacement programs must match the population's output_type."""
        ctx = _make_context()
        ref = sample_reference_context(ctx, ref_size=10, rng=np.random.default_rng(0))
        # Registry with BoolSeries primitives
        reg = PrimitiveRegistry()
        reg.register("close", lambda: None, input_types=(), output_type=Series)
        reg.register("volume", lambda: None, input_types=(), output_type=Series)
        reg.register(
            "gt",
            lambda a, b: np.where(a > b, 1.0, 0.0),
            category="comparison",
            input_types=(Series, Series),
            output_type=BoolSeries,
        )
        config = _make_config(semantic_dedup_enabled=True, semantic_precision=6)

        # Two identical BoolSeries programs (duplicates)
        gt_info = reg.get("gt")
        bool_prog = FunctionNode(
            gt_info,
            (
                TerminalNode("close", output_type=Series),
                TerminalNode("volume", output_type=Series),
            ),
        )
        programs: list[Program] = [bool_prog, bool_prog, bool_prog]
        rng = np.random.default_rng(42)
        new_pop, _ = deduplicate_population(programs, ref, reg, config, rng)
        assert len(new_pop) == 3
        # First kept; replacements must be BoolSeries, not Series
        for prog in new_pop:
            assert prog.output_type == BoolSeries, (
                f"Expected BoolSeries, got {prog.output_type}"
            )
