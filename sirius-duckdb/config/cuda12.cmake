if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR MUSL_ENABLED)
  return()
endif()

set(SIRIUS_DUCKDB_LINKAGE
    static
    CACHE STRING "Sirius library linkage")
duckdb_extension_load(sirius_cuda12 SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..
                      INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../src/include)
