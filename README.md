# polarctic

polarctic connects ArcticDB symbols to Polars through a native `LazyFrame` interface.
It lets you scan data from ArcticDB and continue working with standard Polars queries,
while pushing supported filters and column projections down into ArcticDB where possible.

## Why polarctic?

If your data already lives in ArcticDB, `polarctic` gives you a direct bridge into the
Polars lazy engine.

- Read ArcticDB symbols as `polars.LazyFrame`
- Keep using normal Polars expressions and query composition
- Push supported predicates into ArcticDB instead of filtering only after materialization
- Reuse existing ArcticDB `LazyDataFrame` reads, including pre-applied query builders or row ranges

## Requirements

- Python 3.10+
- `polars>=0.20.0`
- `arcticdb>=4.0.0`
- `pyarrow>=21`

## Installation

From a local checkout:

```bash
pip install -e .
```

From GitHub:

```bash
pip install git+https://github.com/man-group/polarctic.git
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

The public API is a single function:

```python
from polarctic import scan_arcticdb
```

`scan_arcticdb(...)` always returns a `polars.LazyFrame`, so nothing is read until you
call `.collect()`.

### Preferred form: scan from an ArcticDB library

Use this form when you already have a `Library` object and expect to query it repeatedly.

```python
from arcticdb import Arctic
import polars as pl

from polarctic import scan_arcticdb

ac = Arctic("lmdb:///tmp/arctic")
lib = ac.get_library("market_data")

result = (
	scan_arcticdb(lib, "prices")
	.filter((pl.col("close") > 200) & (pl.col("venue") == "XNYS"))
	.select("ts", "symbol", "close")
	.collect()
)
```

### URI form

This is convenient for one-off reads:

```python
import polars as pl

from polarctic import scan_arcticdb

lf = scan_arcticdb("lmdb:///tmp/arctic", "market_data", "prices")
df = lf.filter(pl.col("close") >= 100).collect()
```

This form opens an Arctic connection inside the call, so the `Library` form is usually
better for repeated access.

### LazyDataFrame form

If you already have an ArcticDB lazy read, you can pass it directly to `scan_arcticdb`.
This keeps existing ArcticDB-level operations and lets Polars add more on top.

```python
from arcticdb import OutputFormat, QueryBuilder
import polars as pl

from polarctic import scan_arcticdb

qb = QueryBuilder()
qb = qb[qb["close"] > 200]

lazy_df = lib.read(
	"prices",
	query_builder=qb,
	lazy=True,
	output_format=OutputFormat.PYARROW,
)

result = (
	scan_arcticdb(lazy_df)
	.filter(pl.col("volume") > 1_000_000)
	.select("ts", "symbol", "close", "volume")
	.collect()
)
```

## Supported Calling Forms

`scan_arcticdb` supports three signatures:

```python
scan_arcticdb(uri, lib_name, symbol, *, as_of=None)
scan_arcticdb(lib, symbol, *, as_of=None)
scan_arcticdb(lazy_df)
```

The `as_of` argument accepts the same kinds of version selectors ArcticDB does,
including integer versions, snapshot names, and datetimes.

## Pushdown Behavior

`polarctic` translates a subset of Polars expressions into ArcticDB `QueryBuilder`
operations. When translation succeeds, filtering happens closer to the storage layer.

The current translator supports:

- Comparisons: `==`, `!=`, `<`, `<=`, `>`, `>=`
- Boolean composition: `&`, `|`, `.and_()`, `.or_()`, unary `~`
- Arithmetic expressions: `+`, `-`, `*`, `/`, unary `-`, `.abs()`
- Null checks: `.is_null()`
- Membership tests: `.is_in([...])`
- String regex matching: `.str.contains(...)`
- Integer bitwise expressions such as `(pl.col("flags") & 1) == 0`
- Column projection through `select(...)`

If an expression cannot be translated, the query still remains usable from Polars,
but that part of the plan may execute outside ArcticDB. In practice that means the
result is still correct, but less data may be pushed down.

For example, modulo is not currently translated:

```python
import polars as pl

lf = scan_arcticdb(lib, "prices")
df = lf.filter((pl.col("sequence") % 2) == 0).collect()
```

## Notes on Schema and Execution

- Schema is derived from the ArcticDB symbol before execution, so Polars can plan lazily.
- Reads happen in batches internally rather than materializing the full symbol up front.
- When you pass an ArcticDB `LazyDataFrame`, existing `query_builder`, `row_range`, and
  schema-changing ArcticDB preprocessing are preserved.

## Development

Run the test and validation suite with:

```bash
pytest
ruff check polarctic tests
ruff format --check polarctic tests
mypy polarctic tests
```

## License

This project is distributed under the Business Source License 1.1. See [LICENSE](LICENSE)
for the full terms and the change date after which the project transitions to Apache 2.0.
