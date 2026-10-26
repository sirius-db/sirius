# Private provider ABI build boundary. This builds production connector code; no
# probes or fixtures are added here. Eligible local scans use normal GPU
# admission.
option(SIRIUS_BUILD_DUCKLAKE_BRIDGE
       "Build the pinned private DuckLake binding bridge" OFF)
if(NOT SIRIUS_BUILD_DUCKLAKE_BRIDGE)
  return()
endif()
set(SIRIUS_DUCKLAKE_SOURCE_DIR
    ""
    CACHE PATH "Pinned host-patched DuckLake source")
set(SIRIUS_DUCKLAKE_LIBRARY
    ""
    CACHE FILEPATH "Matching DuckLake extension library")
if(NOT EXISTS
   "${SIRIUS_DUCKLAKE_SOURCE_DIR}/src/include/storage/ducklake_scan.hpp"
   OR NOT EXISTS "${SIRIUS_DUCKLAKE_LIBRARY}")
  message(
    FATAL_ERROR "DuckLake bridge requires its qualified source and library")
endif()
find_package(Git REQUIRED)
execute_process(
  COMMAND "${GIT_EXECUTABLE}" -C "${SIRIUS_DUCKLAKE_SOURCE_DIR}" rev-parse HEAD
  OUTPUT_VARIABLE _ducklake_revision
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
execute_process(
  COMMAND "${GIT_EXECUTABLE}" -C "${SIRIUS_DUCKDB_SOURCE_DIR}" rev-parse HEAD
  OUTPUT_VARIABLE _ducklake_host_revision
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(NOT _ducklake_revision STREQUAL "417a9117f590e8141742ac0579d02f6a90162e37"
   OR NOT _ducklake_host_revision STREQUAL
      "d54cbbe3df17e8e5ce68312f483b4155836112aa")
  message(
    FATAL_ERROR "DuckLake private bridge source/host ABI is not qualified")
endif()
file(GLOB _ducklake_patches
     "${SIRIUS_DUCKDB_SOURCE_DIR}/.github/patches/extensions/ducklake/*.patch")
if(NOT _ducklake_patches)
  message(FATAL_ERROR "DuckLake host compatibility patches are missing")
endif()
execute_process(
  COMMAND "${GIT_EXECUTABLE}" -C "${SIRIUS_DUCKLAKE_SOURCE_DIR}" apply
          --reverse --check ${_ducklake_patches} COMMAND_ERROR_IS_FATAL ANY)
set_property(
  SOURCE "${PROJECT_SOURCE_DIR}/src/ducklake/binding_bridge.cpp"
         TARGET_DIRECTORY sirius_objects
  APPEND
  PROPERTY INCLUDE_DIRECTORIES "${SIRIUS_DUCKLAKE_SOURCE_DIR}/src/include")
set_property(
  SOURCE "${PROJECT_SOURCE_DIR}/src/ducklake/binding_bridge.cpp"
         TARGET_DIRECTORY sirius_objects
  APPEND
  PROPERTY COMPILE_DEFINITIONS SIRIUS_DUCKLAKE_PRIVATE_BRIDGE=1)
target_link_libraries(sirius_shared PRIVATE "${SIRIUS_DUCKLAKE_LIBRARY}")
target_link_libraries(
  sirius_core PRIVATE "$<BUILD_INTERFACE:${SIRIUS_DUCKLAKE_LIBRARY}>"
                      "$<INSTALL_INTERFACE:sirius::ducklake_provider>")

# The private bridge and provider must travel together. Use the provider's
# stable loadable-extension name so an installed package never depends on a
# build path.
install(
  FILES "${SIRIUS_DUCKLAKE_LIBRARY}"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/sirius"
  RENAME ducklake.duckdb_extension
  COMPONENT sirius_library)
install(
  FILES "${SIRIUS_DUCKLAKE_SOURCE_DIR}/LICENSE"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/sirius/ducklake"
  COMPONENT sirius_library)
set_property(
  TARGET sirius_shared
  APPEND
  PROPERTY INSTALL_RPATH "$ORIGIN/sirius")

# The .duckdb_extension suffix does not produce an automatic library RPATH.
# Build-tree consumers need the provider at startup; do not export this path.
get_filename_component(_ducklake_library_dir "${SIRIUS_DUCKLAKE_LIBRARY}"
                       DIRECTORY)
target_link_options(sirius_core INTERFACE
                    "$<BUILD_INTERFACE:LINKER:-rpath,${_ducklake_library_dir}>")
set_property(
  TARGET sirius_shared
  APPEND
  PROPERTY BUILD_RPATH "${_ducklake_library_dir}")
