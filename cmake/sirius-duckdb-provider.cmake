# DuckDB remains a source dependency until its package exposes the required
# internal headers and extension libraries. Keep that contract in one place.
if(PROJECT_IS_TOP_LEVEL)
  set(_sirius_duckdb_default "${CMAKE_CURRENT_SOURCE_DIR}/duckdb")
else()
  set(_sirius_duckdb_default "${CMAKE_SOURCE_DIR}")
endif()
set(SIRIUS_DUCKDB_SOURCE_DIR
    "${_sirius_duckdb_default}"
    CACHE PATH "DuckDB source tree used to build Sirius")

if(NOT EXISTS "${SIRIUS_DUCKDB_SOURCE_DIR}/src/include/duckdb.hpp")
  message(
    FATAL_ERROR
      "Initialize the DuckDB submodule or set SIRIUS_DUCKDB_SOURCE_DIR")
endif()

function(sirius_add_duckdb_source)
  # Keep DuckDB options scoped to its dependency build.
  if(NOT DEFINED OVERRIDE_GIT_DESCRIBE)
    set(OVERRIDE_GIT_DESCRIBE "v1.5.6")
  endif()
  set(DUCKDB_EXTENSION_CONFIGS "")
  set(BUILD_EXTENSIONS "core_functions;parquet")
  set(BUILD_SHELL OFF)
  set(BUILD_UNITTESTS OFF)
  # Root sanitizer presets supply flags to both the library and its dependency.
  set(ENABLE_SANITIZER OFF)
  set(ENABLE_UBSAN OFF)
  set(ENABLE_THREAD_SANITIZER OFF)
  add_subdirectory("${SIRIUS_DUCKDB_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/duckdb"
                   EXCLUDE_FROM_ALL)
endfunction()
if(PROJECT_IS_TOP_LEVEL)
  sirius_add_duckdb_source()
endif()

add_library(sirius_duckdb_dependency INTERFACE IMPORTED)
add_library(sirius::duckdb_dependency ALIAS sirius_duckdb_dependency)
target_include_directories(
  sirius_duckdb_dependency
  INTERFACE "${SIRIUS_DUCKDB_SOURCE_DIR}/src/include"
            "${SIRIUS_DUCKDB_SOURCE_DIR}/third_party/fmt/include"
            "${SIRIUS_DUCKDB_SOURCE_DIR}/third_party/concurrentqueue"
            "${SIRIUS_DUCKDB_SOURCE_DIR}/third_party/fastpforlib"
            "${SIRIUS_DUCKDB_SOURCE_DIR}/third_party/yyjson/include"
            "${SIRIUS_DUCKDB_SOURCE_DIR}/third_party/utf8proc/include"
            "${SIRIUS_DUCKDB_SOURCE_DIR}/extension/core_functions/include"
            "${SIRIUS_DUCKDB_SOURCE_DIR}/extension/parquet/include"
            "${SIRIUS_DUCKDB_SOURCE_DIR}/third_party/parquet"
            "${SIRIUS_DUCKDB_SOURCE_DIR}/third_party/thrift")
target_compile_definitions(
  sirius_duckdb_dependency
  INTERFACE $<$<OR:$<CONFIG:Debug>,$<BOOL:${FORCE_DEBUG}>>:DEBUG>
            $<$<BOOL:${FORCE_ASSERT}>:DUCKDB_FORCE_ASSERT>
            $<$<BOOL:${DISABLE_STR_INLINE}>:DUCKDB_DEBUG_NO_INLINE>
            $<$<BOOL:${FORCE_ASYNC_SINK_SOURCE}>:DUCKDB_DEBUG_ASYNC_SINK_SOURCE>
            $<$<BOOL:${DISABLE_POINTER_SALT}>:DUCKDB_DISABLE_POINTER_SALT>
            $<$<BOOL:${HASH_ZERO}>:DUCKDB_HASH_ZERO>)
# These options change DuckDB's header layouts or data representation.
if(DEFINED STANDARD_VECTOR_SIZE AND NOT STANDARD_VECTOR_SIZE STREQUAL "")
  target_compile_definitions(
    sirius_duckdb_dependency
    INTERFACE STANDARD_VECTOR_SIZE=${STANDARD_VECTOR_SIZE})
endif()
target_link_libraries(
  sirius_duckdb_dependency INTERFACE duckdb_static core_functions_extension
                                     parquet_extension)
