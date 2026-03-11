"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source
License, use of this software will be governed by the Apache License, version 2.0.

Construction-focused benchmarks for translation inputs and helper objects.

These benchmarks intentionally do not call translate(); they isolate input/setup
construction overhead.

Run with:
    pytest tests/bench_query_translation_construction.py -v --benchmark-only
"""

from typing import Any, cast

import polars as pl
from arcticdb import QueryBuilder
from pytest_benchmark.fixture import BenchmarkFixture

from polarctic.polarctic import PolarsToArcticDBTranslator


def _build_qb_simple() -> QueryBuilder:
    qb: Any = QueryBuilder()
    qb = qb[qb["col1"] > 2]
    return cast(QueryBuilder, qb)


def _build_qb_compound() -> QueryBuilder:
    qb: Any = QueryBuilder()
    qb = qb[(qb["col1"] > 2) & (qb["col2"] < 3)]
    return cast(QueryBuilder, qb)


def bench_construct_translator_instance(benchmark: BenchmarkFixture) -> None:
    benchmark(PolarsToArcticDBTranslator)


def bench_construct_querybuilder_empty(benchmark: BenchmarkFixture) -> None:
    benchmark(QueryBuilder)


def bench_construct_querybuilder_simple(benchmark: BenchmarkFixture) -> None:
    benchmark(_build_qb_simple)


def bench_construct_querybuilder_compound(benchmark: BenchmarkFixture) -> None:
    benchmark(_build_qb_compound)


def bench_construct_expr_simple(benchmark: BenchmarkFixture) -> None:
    benchmark(lambda: pl.col("col1") > 2)


def bench_construct_expr_compound(benchmark: BenchmarkFixture) -> None:
    benchmark(lambda: (pl.col("col1") > 2) & (pl.col("col2") < 3))


def bench_construct_expr_complex_nested(benchmark: BenchmarkFixture) -> None:
    benchmark(
        lambda: (
            (pl.col("col1") > 0)
            & (pl.col("col2") < 100)
            & ((pl.col("col3") == 5) | (pl.col("col3") == 10))
        )
    )
