# cuCascade - GPU Memory Reservation Library (submodule)
#
# Options: CUCASCADE_GIT_TAG          - Specific git tag/commit hash to checkout
# (empty = use current) CUCASCADE_UPDATE_SUBMODULE - Update submodule to latest
# remote before building (OFF by default)

set(CUCASCADE_GIT_TAG
    ""
    CACHE STRING
          "Specific git tag/commit hash for cuCascade (empty = use current)")
option(CUCASCADE_UPDATE_SUBMODULE "Update cuCascade submodule to latest remote"
       OFF)

# #region agent log - Debug CMake directory variables (H1, H2, H3, H4)
message(STATUS "[DEBUG] CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "[DEBUG] CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "[DEBUG] PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message(STATUS "[DEBUG] sirius_SOURCE_DIR: ${sirius_SOURCE_DIR}")
# #endregion

set(CUCASCADE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cucascade")

message(STATUS "CUCASCADE_SOURCE_DIR: ${CUCASCADE_SOURCE_DIR}")

# Verify submodule exists
if(NOT EXISTS "${CUCASCADE_SOURCE_DIR}/CMakeLists.txt")
  message(
    FATAL_ERROR "cuCascade submodule not found at ${CUCASCADE_SOURCE_DIR}. "
                "Please run: git submodule update --init cucascade")
endif()

# Handle submodule update if requested
if(CUCASCADE_UPDATE_SUBMODULE)
  message(STATUS "Updating cuCascade submodule to latest remote...")
  execute_process(
    COMMAND git submodule update --remote cucascade
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RESULT_VARIABLE GIT_SUBMODULE_RESULT)
  if(NOT GIT_SUBMODULE_RESULT EQUAL 0)
    message(WARNING "Failed to update cuCascade submodule")
  endif()
endif()

# Handle specific git tag/commit checkout
if(CUCASCADE_GIT_TAG)
  message(STATUS "Checking out cuCascade at: ${CUCASCADE_GIT_TAG}")
  execute_process(
    COMMAND git checkout ${CUCASCADE_GIT_TAG}
    WORKING_DIRECTORY "${CUCASCADE_SOURCE_DIR}"
    RESULT_VARIABLE GIT_CHECKOUT_RESULT)
  if(NOT GIT_CHECKOUT_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to checkout cuCascade at ${CUCASCADE_GIT_TAG}")
  endif()
endif()

# Configure cuCascade build options
set(BUILD_TESTS
    OFF
    CACHE BOOL "Disable cuCascade tests" FORCE)
set(BUILD_SHARED_LIBS
    OFF
    CACHE BOOL "Disable cuCascade shared lib" FORCE)
set(BUILD_STATIC_LIBS
    ON
    CACHE BOOL "Enable cuCascade static lib" FORCE)
set(WARNINGS_AS_ERRORS
    OFF
    CACHE BOOL "Disable warnings as errors for cuCascade" FORCE)

# Add cuCascade subdirectory (EXCLUDE_FROM_ALL prevents cuCascade's own install
# rules)
add_subdirectory("${CUCASCADE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/cucascade"
                 EXCLUDE_FROM_ALL)

# Set include directory for use in main CMakeLists.txt
set(CUCASCADE_INCLUDE_DIR
    "${CUCASCADE_SOURCE_DIR}/include"
    CACHE PATH "cuCascade include directory")
