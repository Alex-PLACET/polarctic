"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source
License, use of this software will be governed by the Apache License, version 2.0.

Construction-focused benchmarks for polarctic scan/planning paths.

These benchmarks intentionally avoid collect() so timing isolates plan/source
creation overhead.

Run with:
    pytest tests/bench_scan_arcticdb_construction.py -v --benchmark-only
"""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
import pytest
from arcticdb import Arctic, OutputFormat, QueryBuilder
from pytest_benchmark.fixture import BenchmarkFixture

import polarctic.polarctic as polarctic_module

_SIMPLE_FILTER_EXPR = pl.col("a") > 500
_COMPOUND_FILTER_EXPR = (pl.col("a") > 200) & (pl.col("b") < 700.0)
_TWO_COLUMN_PROJECTION = ("a", "b")


def _build_simple_filter_qb() -> QueryBuilder:
    qb: Any = QueryBuilder()
    qb = qb[qb["a"] > 500]
    return cast(QueryBuilder, qb)


def _build_compound_filter_qb() -> QueryBuilder:
    qb: Any = QueryBuilder()
    qb = qb[(qb["a"] > 200) & (qb["b"] < 700.0)]
    return cast(QueryBuilder, qb)


@pytest.fixture(scope="module")
def arcticdb_store(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    lmdb_dir: Path = tmp_path_factory.mktemp("arctic_bench_construct") / "lmdb"
    lmdb_dir.mkdir(parents=True, exist_ok=True)
    uri = f"lmdb://{lmdb_dir}"
    lib_name = "bench_lib_construct"

    ac = Arctic(uri)
    lib = ac.create_library(lib_name)

    rng = np.random.default_rng(42)

    def _make_df(n: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "a": rng.integers(0, 1000, size=n).astype(np.int64),
                "b": rng.uniform(0.0, 1000.0, size=n).astype(np.float64),
                "label": [f"item_{i % 50}" for i in range(n)],
                "ts": pd.date_range("2020-01-01", periods=n, freq="s"),
            }
        )

    lib.write("medium", _make_df(10_000))

    return {
        "uri": uri,
        "lib_name": lib_name,
        "lib": lib,
        "symbol": "medium",
    }


@pytest.fixture(scope="module")
def prebuilt_inputs(arcticdb_store: dict[str, Any]) -> dict[str, Any]:
    lib = arcticdb_store["lib"]
    symbol = arcticdb_store["symbol"]

    return {
        "lazy": polarctic_module.scan_arcticdb(lib, symbol),
        "arctic_lazy": lib.read(symbol, lazy=True, output_format=OutputFormat.PYARROW),
        "qb_simple": _build_simple_filter_qb(),
        "qb_compound": _build_compound_filter_qb(),
    }


# ---------------------------------------------------------------------------
# polarctic construction path
# ---------------------------------------------------------------------------


def bench_construct_scan_library_source(
    benchmark: BenchmarkFixture,
    arcticdb_store: dict[str, Any],
) -> None:
    lib = arcticdb_store["lib"]
    symbol = arcticdb_store["symbol"]
    benchmark(lambda: polarctic_module.scan_arcticdb(lib, symbol))


def bench_construct_scan_uri_source(
    benchmark: BenchmarkFixture,
    arcticdb_store: dict[str, Any],
) -> None:
    uri = arcticdb_store["uri"]
    lib_name = arcticdb_store["lib_name"]
    symbol = arcticdb_store["symbol"]
    benchmark(lambda: polarctic_module.scan_arcticdb(uri, lib_name, symbol))


def bench_construct_wrap_lazy_dataframe_source(
    benchmark: BenchmarkFixture,
    prebuilt_inputs: dict[str, Any],
) -> None:
    benchmark(lambda: polarctic_module.scan_arcticdb(prebuilt_inputs["arctic_lazy"]))


def bench_construct_filter_plan_simple(
    benchmark: BenchmarkFixture,
    prebuilt_inputs: dict[str, Any],
) -> None:
    lazy: pl.LazyFrame = prebuilt_inputs["lazy"]
    benchmark(lambda: lazy.filter(_SIMPLE_FILTER_EXPR))


def bench_construct_filter_plan_compound(
    benchmark: BenchmarkFixture,
    prebuilt_inputs: dict[str, Any],
) -> None:
    lazy: pl.LazyFrame = prebuilt_inputs["lazy"]
    benchmark(lambda: lazy.filter(_COMPOUND_FILTER_EXPR))


def bench_construct_select_plan(
    benchmark: BenchmarkFixture,
    prebuilt_inputs: dict[str, Any],
) -> None:
    lazy: pl.LazyFrame = prebuilt_inputs["lazy"]
    benchmark(lambda: lazy.select(*_TWO_COLUMN_PROJECTION))


# ---------------------------------------------------------------------------
# ArcticDB baseline construction path
# ---------------------------------------------------------------------------


def bench_construct_querybuilder_simple(benchmark: BenchmarkFixture) -> None:
    benchmark(_build_simple_filter_qb)


def bench_construct_querybuilder_compound(benchmark: BenchmarkFixture) -> None:
    benchmark(_build_compound_filter_qb)
