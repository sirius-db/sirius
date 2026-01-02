# cuCascade - GPU Memory Reservation Library

set(CUCASCADE_VERSION "main")
set(CUCASCADE_GIT_URL "https://github.com/NVIDIA/cuCascade.git")

include(ExternalProject)

set(CUCASCADE_BASE cucascade_ep)
set(CUCASCADE_PREFIX ${DEPS_PREFIX}/${CUCASCADE_BASE})
set(CUCASCADE_BASE_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CUCASCADE_PREFIX})
set(CUCASCADE_SOURCE_DIR ${CUCASCADE_BASE_DIR}/src/${CUCASCADE_BASE})
set(CUCASCADE_BUILD_DIR ${CUCASCADE_SOURCE_DIR}/build/release)
set(CUCASCADE_INCLUDE_DIR ${CUCASCADE_SOURCE_DIR}/include)
set(CUCASCADE_LIB_DIR ${CUCASCADE_BUILD_DIR})
set(CUCASCADE_STATIC_LIB
    ${CUCASCADE_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}cucascade${CMAKE_STATIC_LIBRARY_SUFFIX}
)

ExternalProject_Add(
  ${CUCASCADE_BASE}
  PREFIX ${CUCASCADE_BASE_DIR}
  GIT_REPOSITORY ${CUCASCADE_GIT_URL}
  GIT_TAG ${CUCASCADE_VERSION}
  GIT_PROGRESS ON
  GIT_SHALLOW ON
  UPDATE_DISCONNECTED TRUE
  BUILD_BYPRODUCTS ${CUCASCADE_STATIC_LIB}
  CONFIGURE_COMMAND
    ${CMAKE_COMMAND} -E env "CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
    ${CMAKE_COMMAND} --preset release -S <SOURCE_DIR> -B
    <SOURCE_DIR>/build/release
  BUILD_COMMAND ${CMAKE_COMMAND} --build <SOURCE_DIR>/build/release
  INSTALL_COMMAND ""
  BUILD_IN_SOURCE FALSE)

file(MAKE_DIRECTORY ${CUCASCADE_INCLUDE_DIR}) # Include directory needs to exist
                                              # to run configure

add_library(cucascade::cucascade STATIC IMPORTED)
set_target_properties(cucascade::cucascade PROPERTIES IMPORTED_LOCATION
                                                      ${CUCASCADE_STATIC_LIB})
target_include_directories(cucascade::cucascade
                           INTERFACE ${CUCASCADE_INCLUDE_DIR})
# cuCascade uses NUMA-aware memory allocation
set_property(TARGET cucascade::cucascade PROPERTY INTERFACE_LINK_LIBRARIES numa)
add_dependencies(cucascade::cucascade ${CUCASCADE_BASE})
