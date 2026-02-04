# TPC-H Data Generator for DuckDB

A simple Python script to generate TPC-H benchmark data at any scale factor using DuckDB's built-in TPC-H extension.

## Installation

Install the required dependency:

```bash
pip install -r requirements.txt
```

## Usage

### Generate Parquet Files

Generate TPC-H data as separate Parquet files (one per table):

```bash
# Scale factor 1 (~1GB of data)
python generate_tpch.py --scale-factor 1

# Scale factor 0.1 (~100MB, good for testing)
python generate_tpch.py --scale-factor 0.1

# Custom output directory
python generate_tpch.py --scale-factor 1 --output-dir ./my_data
```

### Generate DuckDB Database

Generate TPC-H data directly into a DuckDB database file:

```bash
# Default database name: tpch_sf1.duckdb
python generate_tpch.py --scale-factor 1 --format database

# Custom database file name
python generate_tpch.py --scale-factor 10 --format database --db-file my_tpch.duckdb
```

## Scale Factors

Common scale factors and their approximate sizes:

| Scale Factor | Approximate Size | Use Case |
|--------------|------------------|----------|
| 0.01 | ~10 MB | Quick testing |
| 0.1 | ~100 MB | Development |
| 1 | ~1 GB | Standard benchmark |
| 10 | ~10 GB | Large dataset testing |
| 100 | ~100 GB | Enterprise scale |

## Generated Tables

The script generates all 8 TPC-H tables:

- `customer` - Customer information
- `lineitem` - Line items in orders (largest table)
- `nation` - Nations
- `orders` - Customer orders
- `part` - Parts information
- `partsupp` - Parts/supplier relationships
- `region` - Regions
- `supplier` - Supplier information

## Examples

### Example 1: Quick Test Dataset

```bash
python generate_tpch.py --scale-factor 0.01 --output-dir ./test_data
```

### Example 2: Standard Benchmark Database

```bash
python generate_tpch.py --scale-factor 1 --format database
```

### Example 3: Large Dataset for Performance Testing

```bash
python generate_tpch.py --scale-factor 10 --output-dir ./tpch_sf10
```

## Querying the Data

### With Parquet Files

```python
import duckdb

conn = duckdb.connect()
result = conn.execute("""
    SELECT * FROM read_parquet('tpch_data/customer.parquet')
    LIMIT 10
""").fetchall()
print(result)
```

### With Database File

```python
import duckdb

conn = duckdb.connect('tpch_sf1.duckdb')
result = conn.execute("SELECT * FROM customer LIMIT 10").fetchall()
print(result)
```

## Command-Line Options

```
--scale-factor, -sf    TPC-H scale factor (required)
--output-dir, -o       Output directory for Parquet files (default: tpch_data)
--format, -f           Output format: 'parquet' or 'database' (default: parquet)
--db-file              Database file path (for database format only)
```

## License

This is a utility script for generating TPC-H data using DuckDB's built-in extension.
