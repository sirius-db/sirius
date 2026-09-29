# DuckDB remains a source dependency until its package exposes the required
# internal headers and extension libraries. Keep that contract in one place.
set(SIRIUS_DUCKDB_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/duckdb"
    CACHE PATH "DuckDB source tree used to build Sirius")

if(NOT EXISTS "${SIRIUS_DUCKDB_SOURCE_DIR}/src/include/duckdb.hpp")
  message(
    FATAL_ERROR
      "Initialize the DuckDB submodule or set SIRIUS_DUCKDB_SOURCE_DIR")
endif()

add_library(sirius_duckdb_dependency INTERFACE IMPORTED)
add_library(sirius::duckdb_dependency ALIAS sirius_duckdb_dependency)
get_directory_property(_duckdb_headers DIRECTORY "${SIRIUS_DUCKDB_SOURCE_DIR}"
                                                 INCLUDE_DIRECTORIES)
get_directory_property(
  _duckdb_definitions DIRECTORY "${SIRIUS_DUCKDB_SOURCE_DIR}"
                                COMPILE_DEFINITIONS)
set_target_properties(
  sirius_duckdb_dependency
  PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES
    "${_duckdb_headers};${SIRIUS_DUCKDB_SOURCE_DIR}/extension/core_functions/include;${SIRIUS_DUCKDB_SOURCE_DIR}/extension/parquet/include"
    INTERFACE_COMPILE_DEFINITIONS "${_duckdb_definitions}"
    INTERFACE_LINK_LIBRARIES
    "duckdb_static;core_functions_extension;parquet_extension")
