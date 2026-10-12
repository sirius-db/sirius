cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

project(sirius-duckdb)

set(SIRIUS_DUCKDB_LINKAGE
    shared
    CACHE STRING "Sirius library linkage")
set_property(CACHE SIRIUS_DUCKDB_LINKAGE PROPERTY STRINGS static shared)
if(SIRIUS_DUCKDB_LINKAGE STREQUAL "static")
  find_package(sirius CONFIG REQUIRED COMPONENTS static)
  set(sirius_target sirius::sirius_static)
elseif(SIRIUS_DUCKDB_LINKAGE STREQUAL "shared")
  find_package(sirius CONFIG REQUIRED)
  if(NOT TARGET sirius::sirius)
    message(FATAL_ERROR "Shared linkage requires the Sirius shared library")
  endif()
  set(sirius_target sirius::sirius)
else()
  message(FATAL_ERROR "SIRIUS_DUCKDB_LINKAGE must be static or shared")
endif()

set(EXTENSION_SOURCES src/sirius/sirius_extension.cpp)

build_static_extension(sirius ${EXTENSION_SOURCES})
build_loadable_extension(sirius CPP ${EXTENSION_SOURCES})
# Preserve RMM's shared device-resource registry across library boundaries.
set_target_properties(sirius_loadable_extension PROPERTIES LINKER_TYPE BFD)

if(SIRIUS_DUCKDB_LINKAGE STREQUAL "static")
  get_target_property(sirius_link_libraries sirius_loadable_extension
                      LINK_LIBRARIES)
  list(REMOVE_ITEM sirius_link_libraries "-Wl,--exclude-libs,ALL" duckdb_static)
  set_target_properties(
    sirius_loadable_extension
    PROPERTIES LINK_LIBRARIES "${sirius_link_libraries}" SKIP_BUILD_RPATH ON)
  target_link_options(
    sirius_loadable_extension PRIVATE
    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map"
    -static-libgcc -static-libstdc++)
  set_property(
    TARGET sirius_loadable_extension
    APPEND
    PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
  add_custom_command(
    TARGET sirius_loadable_extension
    POST_BUILD
    COMMAND
      "${CMAKE_COMMAND}" -DREMOVE_RPATH=ON
      "-DEXTENSION=$<TARGET_FILE:sirius_loadable_extension>" -P
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/check-static-extension.cmake"
    VERBATIM)
endif()

foreach(target IN ITEMS sirius_extension sirius_loadable_extension)
  target_link_libraries(${target} ${sirius_target})
  set_target_properties(
    ${target}
    PROPERTIES CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CXX_SCAN_FOR_MODULES OFF)
  target_include_directories(${target}
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
endforeach()

install(
  TARGETS sirius_extension
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
