"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source
License, use of this software will be governed by the Apache License, version 2.0.

Compares polarctic scan_arcticdb() against direct ArcticDB lib.read() and prints a
speedup/overhead ratio table.

Benchmarking policy:
- One-time setup (LMDB init, LazyFrame construction, QB construction) is excluded.
- Timed sections only run collect() or lib.read() for apples-to-apples timing.

Usage:
    uv run --extra dev python tests/bench_compare.py [--rounds N]
"""

import argparse
import gc
import statistics
import tempfile
import timeit
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
from arcticdb import Arctic, OutputFormat, QueryBuilder
from arcticdb.version_store.library import Library

import polarctic.polarctic as polarctic_module

_SIMPLE_FILTER_EXPR = pl.col("a") > 500
_COMPOUND_FILTER_EXPR = (pl.col("a") > 200) & (pl.col("b") < 700.0)
_TWO_COLUMN_PROJECTION = ["a", "b"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _median_ms(fn: Callable[[], object], rounds: int) -> float:
    """Return the median wall-clock time of `rounds` calls, in milliseconds."""
    times = timeit.repeat(fn, number=1, repeat=rounds)
    return statistics.median(times) * 1_000


def _qb_simple() -> QueryBuilder:
    qb: Any = QueryBuilder()
    qb = qb[qb["a"] > 500]
    return cast(QueryBuilder, qb)


def _qb_compound() -> QueryBuilder:
    qb: Any = QueryBuilder()
    qb = qb[(qb["a"] > 200) & (qb["b"] < 700.0)]
    return cast(QueryBuilder, qb)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def _setup(tmp_dir: Path) -> dict[str, Library]:
    lmdb_dir = tmp_dir / "lmdb"
    lmdb_dir.mkdir(parents=True)
    uri = f"lmdb://{lmdb_dir}"

    ac = Arctic(uri)
    lib: Library = ac.create_library("bench")

    rng = np.random.default_rng(42)

    def _df(n: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "a": rng.integers(0, 1000, size=n).astype(np.int64),
                "b": rng.uniform(0.0, 1000.0, size=n).astype(np.float64),
                "label": [f"item_{i % 50}" for i in range(n)],
                "ts": pd.date_range("2020-01-01", periods=n, freq="s"),
            }
        )

    lib.write("medium", _df(10_000))
    lib.write("large", _df(100_000))

    return {"lib": lib}


# ---------------------------------------------------------------------------
# Benchmark pairs
# ---------------------------------------------------------------------------


def _build_pairs(lib: Library) -> list[tuple[str, Callable[[], object], Callable[[], object]]]:
    """Return (label, polarctic_fn, baseline_fn) triples."""
    qb_simple = _qb_simple()
    qb_compound = _qb_compound()

    lazy_medium = polarctic_module.scan_arcticdb(lib, "medium")
    lazy_large = polarctic_module.scan_arcticdb(lib, "large")

    lazy_filter_simple_medium = lazy_medium.filter(_SIMPLE_FILTER_EXPR)
    lazy_filter_simple_large = lazy_large.filter(_SIMPLE_FILTER_EXPR)
    lazy_filter_compound_medium = lazy_medium.filter(_COMPOUND_FILTER_EXPR)
    lazy_filter_compound_large = lazy_large.filter(_COMPOUND_FILTER_EXPR)

    lazy_select_two_columns_medium = lazy_medium.select(*_TWO_COLUMN_PROJECTION)
    lazy_select_two_columns_large = lazy_large.select(*_TWO_COLUMN_PROJECTION)

    return [
        (
            "Full scan - medium (10k rows)",
            lazy_medium.collect,
            lambda: lib.read("medium", output_format=OutputFormat.PANDAS).data,
        ),
        (
            "Full scan - large (100k rows)",
            lazy_large.collect,
            lambda: lib.read("large", output_format=OutputFormat.PANDAS).data,
        ),
        (
            "Filter simple (a > 500) - medium",
            lazy_filter_simple_medium.collect,
            lambda: lib.read(
                "medium",
                query_builder=qb_simple,
                output_format=OutputFormat.PANDAS,
            ).data,
        ),
        (
            "Filter simple (a > 500) - large",
            lazy_filter_simple_large.collect,
            lambda: lib.read(
                "large",
                query_builder=qb_simple,
                output_format=OutputFormat.PANDAS,
            ).data,
        ),
        (
            "Filter compound (a>200 & b<700) - medium",
            lazy_filter_compound_medium.collect,
            lambda: lib.read(
                "medium",
                query_builder=qb_compound,
                output_format=OutputFormat.PANDAS,
            ).data,
        ),
        (
            "Filter compound (a>200 & b<700) - large",
            lazy_filter_compound_large.collect,
            lambda: lib.read(
                "large",
                query_builder=qb_compound,
                output_format=OutputFormat.PANDAS,
            ).data,
        ),
        (
            "Select 2 columns - medium",
            lazy_select_two_columns_medium.collect,
            lambda: lib.read(
                "medium",
                columns=_TWO_COLUMN_PROJECTION,
                output_format=OutputFormat.PANDAS,
            ).data,
        ),
        (
            "Select 2 columns - large",
            lazy_select_two_columns_large.collect,
            lambda: lib.read(
                "large",
                columns=_TWO_COLUMN_PROJECTION,
                output_format=OutputFormat.PANDAS,
            ).data,
        ),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_COL_LABEL = 42
_COL_NUM = 10


def _row(label: str, polar_ms: float, base_ms: float) -> str:
    ratio = base_ms / polar_ms
    overhead_ms = polar_ms - base_ms
    sign = "+" if overhead_ms >= 0 else "-"
    return (
        f"  {label:<{_COL_LABEL}}"
        f"  {polar_ms:{_COL_NUM}.2f} ms"
        f"  {base_ms:{_COL_NUM}.2f} ms"
        f"  {ratio:{_COL_NUM}.2f}x"
        f"  {sign}{abs(overhead_ms):.2f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=20, help="Timing rounds per benchmark")
    args = parser.parse_args()

    rounds: int = args.rounds

    with tempfile.TemporaryDirectory(prefix="polarctic_bench_") as tmp:
        store = _setup(Path(tmp))
        lib = store["lib"]

        pairs = _build_pairs(lib)

        # Warm-up pass (not timed)
        print("Warming up ...")
        for _, polar_fn, base_fn in pairs:
            polar_fn()
            base_fn()
        gc.collect()

        # Timed pass
        results: list[tuple[str, float, float]] = []
        for label, polar_fn, base_fn in pairs:
            print(f"  timing: {label}")
            polar_ms = _median_ms(polar_fn, rounds)
            base_ms = _median_ms(base_fn, rounds)
            results.append((label, polar_ms, base_ms))

        # Table
        header_label = "Scenario"
        header = (
            f"\n  {header_label:<{_COL_LABEL}}"
            f"  {'polarctic':>{_COL_NUM + 3}}"
            f"  {'arcticdb':>{_COL_NUM + 3}}"
            f"  {'ratio':>{_COL_NUM + 1}}"
            f"  overhead"
        )
        sep = "  " + "-" * (_COL_LABEL + 2 * (_COL_NUM + 5) + (_COL_NUM + 3) + 12)

        print(f"\nResults ({rounds} rounds, median wall-clock time)")
        print(sep)
        print(header)
        print(sep)
        for label, polar_ms, base_ms in results:
            print(_row(label, polar_ms, base_ms))
        print(sep)
        print(
            "  ratio > 1x -> polarctic faster than raw ArcticDB  |"
            "  ratio < 1x -> overhead (positive overhead column)"
        )


if __name__ == "__main__":
    main()
