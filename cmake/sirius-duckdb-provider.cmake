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

function(sirius_add_duckdb_source)
  # Keep DuckDB options scoped to its dependency build.
  set(DUCKDB_EXTENSION_CONFIGS "")
  set(BUILD_EXTENSIONS "core_functions;parquet")
  set(BUILD_SHELL OFF)
  set(BUILD_UNITTESTS OFF)
  if(NOT DEFINED DUCKDB_EXPLICIT_VERSION)
    execute_process(
      COMMAND git -C "${SIRIUS_DUCKDB_SOURCE_DIR}" rev-parse HEAD
      OUTPUT_VARIABLE sirius_duckdb_commit
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(sirius_duckdb_commit STREQUAL "d54cbbe3df17e8e5ce68312f483b4155836112aa")
      set(DUCKDB_EXPLICIT_VERSION "v2.0.0")
    endif()
  endif()
  # Root sanitizer presets supply flags to both the library and its dependency.
  set(ENABLE_SANITIZER OFF)
  set(ENABLE_UBSAN OFF)
  set(ENABLE_THREAD_SANITIZER OFF)
  add_subdirectory("${SIRIUS_DUCKDB_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/duckdb"
                   EXCLUDE_FROM_ALL)
endfunction()
sirius_add_duckdb_source()

add_library(sirius_duckdb_dependency INTERFACE IMPORTED)
add_library(sirius::duckdb_dependency ALIAS sirius_duckdb_dependency)
target_include_directories(
  sirius_duckdb_dependency
  INTERFACE "${SIRIUS_DUCKDB_SOURCE_DIR}/src/include"
            "${SIRIUS_DUCKDB_SOURCE_DIR}/third_party/fmt/include"
            "${SIRIUS_DUCKDB_SOURCE_DIR}/third_party/re2"
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

# duckdb's vendored moodycamel (namespace duckdb_moodycamel). cuCascade io links
# this so both sides of the boundary use one moodycamel ABI. IMPORTED (like
# sirius_duckdb_dependency) because DuckDB's build-tree export() keeps
# cucascade_io_thirdparty's $<BUILD_INTERFACE:> link to it, and a non-imported
# target would have to join an export set.
add_library(sirius_duckdb_moodycamel INTERFACE IMPORTED GLOBAL)
add_library(duckdb::moodycamel ALIAS sirius_duckdb_moodycamel)
target_include_directories(
  sirius_duckdb_moodycamel
  INTERFACE "${SIRIUS_DUCKDB_SOURCE_DIR}/third_party/concurrentqueue")

function(sirius_link_duckdb_extensions target)
  set(DUCKDB_MODULE_BASE_DIR "${SIRIUS_DUCKDB_SOURCE_DIR}")
  duckdb_link_extensions(${target} ${ARGN} httplib loadable_extensions
                         core_functions parquet)
endfunction()
