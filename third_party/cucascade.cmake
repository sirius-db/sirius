# cuCascade - GPU Memory Reservation Library (Submodule)

# Find Git for submodule operations
find_package(Git QUIET)

# Options for cuCascade configuration
option(CUCASCADE_UPDATE_SUBMODULE "Update cuCascade submodule to latest" OFF)
set(CUCASCADE_GIT_HASH "main" CACHE STRING "Git hash/tag/branch for cuCascade")

set(CUCASCADE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cucascade)

# Check if submodule exists
if(NOT EXISTS "${CUCASCADE_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR 
    "cuCascade submodule not found at ${CUCASCADE_SOURCE_DIR}. "
    "Please run: git submodule update --init --recursive")
endif()

# Update submodule if requested
if(CUCASCADE_UPDATE_SUBMODULE)
  if(NOT GIT_FOUND)
    message(FATAL_ERROR "Git not found. Cannot update cuCascade submodule.")
  endif()
  message(STATUS "Updating cuCascade submodule to latest...")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} submodule update --remote --recursive third_party/cucascade
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE GIT_SUBMODULE_RESULT
  )
  if(NOT GIT_SUBMODULE_RESULT EQUAL "0")
    message(FATAL_ERROR "Failed to update cuCascade submodule")
  endif()
endif()

# Checkout specific hash if not on main and not updating to latest
if(NOT CUCASCADE_UPDATE_SUBMODULE AND NOT CUCASCADE_GIT_HASH STREQUAL "main")
  if(NOT GIT_FOUND)
    message(FATAL_ERROR "Git not found. Cannot checkout cuCascade at specific hash.")
  endif()
  message(STATUS "Checking out cuCascade at ${CUCASCADE_GIT_HASH}...")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} fetch
    WORKING_DIRECTORY ${CUCASCADE_SOURCE_DIR}
    RESULT_VARIABLE GIT_FETCH_RESULT
  )
  execute_process(
    COMMAND ${GIT_EXECUTABLE} checkout ${CUCASCADE_GIT_HASH}
    WORKING_DIRECTORY ${CUCASCADE_SOURCE_DIR}
    RESULT_VARIABLE GIT_CHECKOUT_RESULT
  )
  if(NOT GIT_CHECKOUT_RESULT EQUAL "0")
    message(FATAL_ERROR "Failed to checkout cuCascade at ${CUCASCADE_GIT_HASH}")
  endif()
endif()

# Build cuCascade as a static library
set(CUCASCADE_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/cucascade_build)
set(CUCASCADE_INCLUDE_DIR ${CUCASCADE_SOURCE_DIR}/include)
set(CUCASCADE_STATIC_LIB
    ${CUCASCADE_BUILD_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}cucascade${CMAKE_STATIC_LIBRARY_SUFFIX}
)

# Configure and build cuCascade using its release preset
message(STATUS "Configuring cuCascade...")
execute_process(
  COMMAND ${CMAKE_COMMAND} --preset release 
    -S ${CUCASCADE_SOURCE_DIR} 
    -B ${CUCASCADE_BUILD_DIR}
  RESULT_VARIABLE CUCASCADE_CONFIGURE_RESULT
  OUTPUT_VARIABLE CUCASCADE_CONFIGURE_OUTPUT
  ERROR_VARIABLE CUCASCADE_CONFIGURE_ERROR
)

if(NOT CUCASCADE_CONFIGURE_RESULT EQUAL "0")
  message(STATUS "cuCascade configure output: ${CUCASCADE_CONFIGURE_OUTPUT}")
  message(STATUS "cuCascade configure error: ${CUCASCADE_CONFIGURE_ERROR}")
  message(FATAL_ERROR "Failed to configure cuCascade")
endif()

message(STATUS "Building cuCascade static library...")
execute_process(
  COMMAND ${CMAKE_COMMAND} --build ${CUCASCADE_BUILD_DIR}
  RESULT_VARIABLE CUCASCADE_BUILD_RESULT
  OUTPUT_VARIABLE CUCASCADE_BUILD_OUTPUT
  ERROR_VARIABLE CUCASCADE_BUILD_ERROR
)

if(NOT CUCASCADE_BUILD_RESULT EQUAL "0")
  message(STATUS "cuCascade build output: ${CUCASCADE_BUILD_OUTPUT}")
  message(STATUS "cuCascade build error: ${CUCASCADE_BUILD_ERROR}")
  message(FATAL_ERROR "Failed to build cuCascade")
endif()

# Verify the library was built
if(NOT EXISTS ${CUCASCADE_STATIC_LIB})
  message(FATAL_ERROR "cuCascade static library not found at ${CUCASCADE_STATIC_LIB}")
endif()

message(STATUS "cuCascade static library: ${CUCASCADE_STATIC_LIB}")

# Create imported target for cuCascade
add_library(cucascade::cucascade STATIC IMPORTED GLOBAL)
set_target_properties(cucascade::cucascade PROPERTIES 
  IMPORTED_LOCATION ${CUCASCADE_STATIC_LIB}
)
target_include_directories(cucascade::cucascade
  INTERFACE ${CUCASCADE_INCLUDE_DIR}
)

# cuCascade uses NUMA-aware memory allocation
set_property(TARGET cucascade::cucascade PROPERTY INTERFACE_LINK_LIBRARIES numa)

message(STATUS "cuCascade configured successfully")
message(STATUS "  Source: ${CUCASCADE_SOURCE_DIR}")
message(STATUS "  Build: ${CUCASCADE_BUILD_DIR}")
message(STATUS "  Include: ${CUCASCADE_INCLUDE_DIR}")
message(STATUS "  Library: ${CUCASCADE_STATIC_LIB}")
