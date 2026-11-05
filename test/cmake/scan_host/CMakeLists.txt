cmake_minimum_required(VERSION 3.30.4)
project(sirius_scan_host_test LANGUAGES CXX)
include(CTest)

set(SIRIUS_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../..")
set(SIRIUS_DUCKDB_SOURCE_DIR
    "${SIRIUS_SOURCE_DIR}/duckdb"
    CACHE PATH "DuckDB source tree")
set(DUCKDB_LIBRARY
    ""
    CACHE FILEPATH "DuckDB shared library")
if(NOT EXISTS "${DUCKDB_LIBRARY}")
  message(FATAL_ERROR "Set DUCKDB_LIBRARY to the built DuckDB shared library")
endif()
add_library(scan_host_duckdb SHARED IMPORTED)
set_target_properties(scan_host_duckdb PROPERTIES IMPORTED_LOCATION
                                                  "${DUCKDB_LIBRARY}")

foreach(mode pie pic nonpie)
  add_executable(scan_host_${mode} main.cpp)
  target_include_directories(
    scan_host_${mode} PRIVATE "${SIRIUS_SOURCE_DIR}/src"
                              "${SIRIUS_DUCKDB_SOURCE_DIR}/src/include")
  target_link_libraries(scan_host_${mode} PRIVATE scan_host_duckdb
                                                  ${CMAKE_DL_LIBS})
  target_compile_features(scan_host_${mode} PRIVATE cxx_std_20)
  add_test(NAME scan_host_${mode} COMMAND scan_host_${mode})
endforeach()
target_compile_options(scan_host_pie PRIVATE -fPIE)
target_compile_options(scan_host_pic PRIVATE -fPIC)
target_compile_options(scan_host_nonpie PRIVATE -fno-pic -fno-pie)
target_link_options(scan_host_pie PRIVATE -pie)
target_link_options(scan_host_pic PRIVATE -pie)
target_link_options(scan_host_nonpie PRIVATE -no-pie)
target_compile_definitions(scan_host_nonpie PRIVATE EXPECT_COPY_RELOCATION)
# GCC uses copy relocations for direct external data references on x86-64.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$" AND CMAKE_CXX_COMPILER_ID
                                                         STREQUAL "GNU")
  target_compile_definitions(scan_host_pie PRIVATE EXPECT_COPY_RELOCATION)
endif()
