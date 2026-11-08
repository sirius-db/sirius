# DuckDB configures the pinned provider and applies its host patches.
if(SIRIUS_DUCKLAKE_SOURCE_DIR)
  set(_ducklake_source "${SIRIUS_DUCKLAKE_SOURCE_DIR}")
else()
  get_target_property(_ducklake_source ducklake_loadable_extension SOURCE_DIR)
endif()
find_package(Git REQUIRED)
execute_process(
  COMMAND "${GIT_EXECUTABLE}" -C "${_ducklake_source}" rev-parse HEAD
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
  COMMAND "${GIT_EXECUTABLE}" -C "${_ducklake_source}" apply --reverse --check
          ${_ducklake_patches} COMMAND_ERROR_IS_FATAL ANY)
set_property(
  SOURCE "${PROJECT_SOURCE_DIR}/src/ducklake/binding_bridge.cpp"
         "${PROJECT_SOURCE_DIR}/src/ducklake/provider_api.cpp" TARGET_DIRECTORY
         sirius_objects
  APPEND
  PROPERTY INCLUDE_DIRECTORIES "${_ducklake_source}/src/include")
target_link_libraries(sirius_shared PRIVATE ${CMAKE_DL_LIBS})
target_link_libraries(sirius_core PRIVATE ${CMAKE_DL_LIBS})

install(
  FILES "${_ducklake_source}/LICENSE"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/sirius/ducklake"
  COMPONENT sirius_library)
