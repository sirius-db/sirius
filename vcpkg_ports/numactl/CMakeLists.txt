cmake_minimum_required(VERSION 3.20)
project(numactl VERSION 2.0.19 LANGUAGES C)

include(CheckCSourceCompiles)
include(GNUInstallDirs)

# Check for symver attribute support
set(CMAKE_REQUIRED_FLAGS "-Werror")
check_c_source_compiles("
    __attribute__ ((symver (\"foo@foo_1\"))) void frob (void) { }
    int main() { return 0; }
" HAVE_ATTRIBUTE_SYMVER)
unset(CMAKE_REQUIRED_FLAGS)

# Generate config.h
configure_file(config.h.cmake.in config.h @ONLY)

# libnuma sources
add_library(numa
    libnuma.c
    syscall.c
    distance.c
    affinity.c
    sysfs.c
    rtnetlink.c
)

target_include_directories(numa
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}  # for config.h
    PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_definitions(numa PRIVATE
    _GNU_SOURCE
    _FILE_OFFSET_BITS=64
    TLS=__thread
)

set_target_properties(numa PROPERTIES
    OUTPUT_NAME numa
    POSITION_INDEPENDENT_CODE ON
    PUBLIC_HEADER "numa.h;numacompat1.h;numaif.h"
)

install(TARGETS numa
    EXPORT numactl-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# pkg-config file
set(prefix ${CMAKE_INSTALL_PREFIX})
set(exec_prefix "\${prefix}")
set(libdir "\${prefix}/${CMAKE_INSTALL_LIBDIR}")
set(includedir "\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
configure_file(numa.pc.in numa.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/numa.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

# CMake config
install(EXPORT numactl-targets
    FILE numactl-targets.cmake
    NAMESPACE numactl::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/numactl
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    numactl-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/numactl-config.cmake
"include(CMakeFindDependencyMacro)
include(\"\${CMAKE_CURRENT_LIST_DIR}/numactl-targets.cmake\")
")

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/numactl-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/numactl-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/numactl
)
