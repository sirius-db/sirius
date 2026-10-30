duckdb_extension_load(
  sirius_gateway
  SOURCE_DIR
  ${CMAKE_CURRENT_LIST_DIR}
  INCLUDE_DIR
  ${CMAKE_CURRENT_LIST_DIR}/src/include
  TEST_DIR
  ${CMAKE_CURRENT_LIST_DIR}/test/sql
  LOAD_TESTS)

# Same pins DuckDB itself uses. Quack serves and calls over httpfs's HTTP client
# and crypto.
include(
  ${CMAKE_CURRENT_LIST_DIR}/../sirius-duckdb/duckdb/.github/config/extensions/httpfs.cmake
)
include(
  ${CMAKE_CURRENT_LIST_DIR}/../sirius-duckdb/duckdb/.github/config/extensions/quack.cmake
)

duckdb_extension_statically_link(httpfs quack sirius_gateway)
