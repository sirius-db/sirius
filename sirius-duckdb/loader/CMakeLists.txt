cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)
project(sirius-loader LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "The Sirius CUDA loader supports Linux")
endif()

# DuckDB's CI builds this directory without submodules; DuckDB is the host tree.
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/third-party-notices.cmake")
sirius_notices_add_duckdb(
  "${CMAKE_CURRENT_BINARY_DIR}/third_party_components/duckdb"
  "${DuckDB_SOURCE_DIR}" "${DUCKDB_VERSION}")
sirius_notices_generate(
  notices_source
  SYMBOL
  sirius
  TITLE
  "Third-party software notices for the sirius DuckDB extension"
  VERSION
  "${DUCKDB_EXTENSION_SIRIUS_EXT_VERSION}"
  SIRIUS_LICENSE
  "${CMAKE_CURRENT_LIST_DIR}/../../LICENSE"
  DIRECTORIES
  "${CMAKE_CURRENT_BINARY_DIR}/third_party_components")
set(loader_sources src/sirius_loader.cpp ../src/sirius/sirius_licenses.cpp
                   ${notices_source})

build_static_extension(sirius ${loader_sources})
build_loadable_extension(sirius " " ${loader_sources})
foreach(target IN ITEMS sirius_extension sirius_loadable_extension)
  target_link_libraries(${target} ${CMAKE_DL_LIBS})
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_include_directories(${target}
                             PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../src/include")
  target_compile_definitions(
    ${target} PRIVATE SIRIUS_NOTICES_NAMESPACE=${notices_source_NAMESPACE})
endforeach()

install(
  TARGETS sirius_extension
  EXPORT "${DUCKDB_EXPORT_SET}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")

option(SIRIUS_LOADER_TESTS "Build CPU-only loader test fixtures" OFF)
if(SIRIUS_LOADER_TESTS)
  add_subdirectory(test)
endif()
