"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source
License, use of this software will be governed by the Apache License, version 2.0.

Execution-focused benchmarks for PolarsToArcticDBTranslator.translate().

Construction-only benchmarks live in tests/bench_query_translation_construction.py.

Run with:
    pytest tests/bench_query_translation.py -v --benchmark-only
"""

from typing import Any

import polars as pl
import pytest
from arcticdb import QueryBuilder

from polarctic.polarctic import PolarsToArcticDBTranslator


@pytest.fixture(scope="module")
def translator() -> PolarsToArcticDBTranslator:
    return PolarsToArcticDBTranslator()


@pytest.fixture(scope="module")
def exprs() -> dict[str, pl.Expr]:
    """Build expressions once so timed sections focus on translation work."""
    return {
        "greater": pl.col("col1") > 2,
        "greater_equal": pl.col("col1") >= 2,
        "less": pl.col("col1") < 2,
        "less_equal": pl.col("col1") <= 2,
        "equal": pl.col("col1") == 2,
        "not_equal": pl.col("col1") != 2,
        "add": pl.col("col1") + pl.col("col2"),
        "subtract": pl.col("col1") - pl.col("col2"),
        "multiply": pl.col("col1") * pl.col("col2"),
        "divide": pl.col("col1") / pl.col("col2"),
        "and": (pl.col("col1") > 2) & (pl.col("col2") < 3),
        "or": (pl.col("col1") > 2) | (pl.col("col2") < 3),
        "bitwise_and": (pl.col("col1") & 1) == 0,
        "bitwise_or": (pl.col("col1") | 1) == 3,
        "bitwise_xor": (pl.col("col1") ^ 1) == 3,
        "not": ~pl.col("col1"),
        "negation": -pl.col("col1"),
        "abs": pl.col("col1").abs(),
        "is_null": pl.col("col1").is_null(),
        "is_not_null": ~pl.col("col1").is_null(),
        "str_contains": pl.col("col1").str.contains("e+"),
        "isin": pl.col("col1").is_in([24, 42]),
        "complex_nested": (
            (pl.col("col1") > 0)
            & (pl.col("col2") < 100)
            & ((pl.col("col3") == 5) | (pl.col("col3") == 10))
        ),
    }


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------


def bench_greater(
    benchmark: Any,
    translator: PolarsToArcticDBTranslator,
    exprs: dict[str, pl.Expr],
) -> None:
    benchmark(lambda: translator.translate(exprs["greater"], QueryBuilder()))


def bench_greater_equal(
    benchmark: Any,
    translator: PolarsToArcticDBTranslator,
    exprs: dict[str, pl.Expr],
) -> None:
    benchmark(lambda: translator.translate(exprs["greater_equal"], QueryBuilder()))


def bench_less(
    benchmark: Any,
    translator: PolarsToArcticDBTranslator,
    exprs: dict[str, pl.Expr],
) -> None:
    benchmark(lambda: translator.translate(exprs["less"], QueryBuilder()))


def bench_less_equal(
    benchmark: Any,
    translator: PolarsToArcticDBTranslator,
    exprs: dict[str, pl.Expr],
) -> None:
    benchmark(lambda: translator.translate(exprs["less_equal"], QueryBuilder()))


def bench_equal(
    benchmark: Any,
    translator: PolarsToArcticDBTranslator,
    exprs: dict[str, pl.Expr],
) -> None:
    benchmark(lambda: translator.translate(exprs["equal"], QueryBuilder()))


def bench_not_equal(
    benchmark: Any,
    translator: PolarsToArcticDBTranslator,
    exprs: dict[str, pl.Expr],
) -> None:
    benchmark(lambda: translator.translate(exprs["not_equal"], QueryBuilder()))


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


def bench_add(
    benchmark: Any,
    translator: PolarsToArcticDBTranslator,
    exprs: dict[str, pl.Expr],
) -> None:
    benchmark(lambda: translator.translate(exprs["add"], QueryBuilder()))


def bench_subtract(
    benchmark: Any,
    translator: PolarsToArcticDBTranslator,
    exprs: dict[str, pl.Expr],
) -> None:
    benchmark(lambda: translator.translate(exprs["subtract"], QueryBuilder()))


def bench_multiply(
    benchmark: Any,
    translator: PolarsToArcticDBTranslator,
    exprs: dict[str, pl.Expr],
) -> None:
    benchmark(lambda: translator.translate(exprs["multiply"], QueryBuilder()))


def bench_divide(
    benchmark: Any,
    translator: PolarsToArcticDBTranslator,
    exprs: dict[str, pl.Expr],
) -> None:
    benchmark(lambda: translator.translate(exprs["divide"], QueryBuilder()))


# ---------------------------------------------------------------------------
# Boolean / bitwise
# ---------------------------------------------------------------------------


def bench_and(
    benchmark: Any,
    translator: PolarsToArcticDBTranslator,
    exprs: dict[str, pl.Expr],
) -> None:
    benchmark(lambda: translator.translate(exprs["and"], QueryBuilder()))


def bench_or(
    benchmark: Any,
    translator: PolarsToArcticDBTranslator,
    exprs: dict[str, pl.Expr],
) -> None:
    benchmark(lambda: translator.translate(exprs["or"], QueryBuilder()))


def bench_bitwise_and(
    benchmark: Any,
    translator: PolarsToArcticDBTranslator,
    exprs: dict[str, pl.Expr],
) -> None:
    benchmark(lambda: translator.translate(exprs["bitwise_and"], QueryBuilder()))


def bench_bitwise_or(
    benchmark: Any,
    translator: PolarsToArcticDBTranslator,
    exprs: dict[str, pl.Expr],
) -> None:
    benchmark(lambda: translator.translate(exprs["bitwise_or"], QueryBuilder()))


def bench_bitwise_xor(
    benchmark: Any,
    translator: PolarsToArcticDBTranslator,
    exprs: dict[str, pl.Expr],
) -> None:
    benchmark(lambda: translator.translate(exprs["bitwise_xor"], QueryBuilder()))


def bench_not(
    benchmark: Any,
    translator: PolarsToArcticDBTranslator,
    exprs: dict[str, pl.Expr],
) -> None:
    benchmark(lambda: translator.translate(exprs["not"], QueryBuilder()))


def bench_negation(
    benchmark: Any,
    translator: PolarsToArcticDBTranslator,
    exprs: dict[str, pl.Expr],
) -> None:
    benchmark(lambda: translator.translate(exprs["negation"], QueryBuilder()))


# ---------------------------------------------------------------------------
# Unary / null checks
# ---------------------------------------------------------------------------


def bench_abs(
    benchmark: Any,
    translator: PolarsToArcticDBTranslator,
    exprs: dict[str, pl.Expr],
) -> None:
    benchmark(lambda: translator.translate(exprs["abs"], QueryBuilder()))


def bench_is_null(
    benchmark: Any,
    translator: PolarsToArcticDBTranslator,
    exprs: dict[str, pl.Expr],
) -> None:
    benchmark(lambda: translator.translate(exprs["is_null"], QueryBuilder()))


def bench_is_not_null(
    benchmark: Any,
    translator: PolarsToArcticDBTranslator,
    exprs: dict[str, pl.Expr],
) -> None:
    benchmark(lambda: translator.translate(exprs["is_not_null"], QueryBuilder()))


# ---------------------------------------------------------------------------
# String / membership
# ---------------------------------------------------------------------------


def bench_str_contains(
    benchmark: Any,
    translator: PolarsToArcticDBTranslator,
    exprs: dict[str, pl.Expr],
) -> None:
    benchmark(lambda: translator.translate(exprs["str_contains"], QueryBuilder()))


def bench_isin(
    benchmark: Any,
    translator: PolarsToArcticDBTranslator,
    exprs: dict[str, pl.Expr],
) -> None:
    benchmark(lambda: translator.translate(exprs["isin"], QueryBuilder()))


# ---------------------------------------------------------------------------
# Complex / nested expressions
# ---------------------------------------------------------------------------


def bench_complex_nested(
    benchmark: Any,
    translator: PolarsToArcticDBTranslator,
    exprs: dict[str, pl.Expr],
) -> None:
    """Three-clause AND/OR expression to stress the recursive tree walk."""
    benchmark(lambda: translator.translate(exprs["complex_nested"], QueryBuilder()))
