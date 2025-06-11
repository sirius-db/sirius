# This file is included by DuckDB's build system. It specifies which extension to load

# Extension from this repo
duckdb_extension_load(sirius
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
    LOAD_TESTS
)

duckdb_extension_load(json)
duckdb_extension_load(tpcds)
duckdb_extension_load(tpch)
duckdb_extension_load(parquet)
duckdb_extension_load(icu)

# Any extra extensions that should be built
duckdb_extension_load(substrait)
