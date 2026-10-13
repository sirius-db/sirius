set(SIRIUS_DUCKDB_EXTENSION_NAME sirius_cuda12)
set(SIRIUS_DUCKDB_LINKAGE
    static
    CACHE STRING "Sirius library linkage")
duckdb_extension_load(sirius_cuda12 SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..
                      INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../src/include)
