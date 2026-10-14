if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR MUSL_ENABLED)
  return()
endif()

# Build the CUDA-free loader once per platform, alongside the CUDA 13 backend.
include("${CMAKE_CURRENT_LIST_DIR}/loader.cmake")

set(SIRIUS_DUCKDB_LINKAGE
    static
    CACHE STRING "Sirius library linkage")
duckdb_extension_load(sirius_cuda13 SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..
                      INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../src/include)
