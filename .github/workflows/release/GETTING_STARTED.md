## Getting started

A quick path from nothing installed to a GPU-accelerated query. Full details, requirements, and the Python API:
[docs/README.md](https://github.com/{{REPO}}/blob/main/docs/README.md).

**Requirements**: Linux (amd64 or arm64), NVIDIA GPU Turing or newer (compute capability 7.5+), and either CUDA 12.x
(driver >=525.60.13) or CUDA 13.x (driver >=580.65.06) matching the file you download below.

### 1. Install DuckDB {{DUCKDB_VERSION}}

Sirius extensions must match the exact DuckDB version they were built against:

```bash
curl https://install.duckdb.org | DUCKDB_VERSION={{DUCKDB_VERSION_BARE}} sh
```

### 2. Download the Sirius extension for your platform

{{DOWNLOAD_LINKS}}

### 3. Allow unsigned extensions

Sirius isn't published through DuckDB's official signed extension repository yet, so you need to explicitly allow
unsigned extensions before loading it:

```bash
duckdb -unsigned
```

For client APIs, pass the same config at connect time instead, e.g. Python:
`duckdb.connect(config={"allow_unsigned_extensions": "true"})`.

**Warning**: unsigned extensions execute native code with the host process's privileges.

### 4. Load the extension and try it

```sql
LOAD './{{FIRST_FILE}}';

-- Generate TPC-H data on the fly (scale factor 1 = ~1 GB) and query it, on GPU,
-- automatically. No query rewrites needed; unsupported operators fall back to CPU.
INSTALL tpch;
LOAD tpch;
CALL dbgen(sf=1);

SELECT l_returnflag, sum(l_quantity)
FROM lineitem
GROUP BY l_returnflag
ORDER BY l_returnflag;
```
