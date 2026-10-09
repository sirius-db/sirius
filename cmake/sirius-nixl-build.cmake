# Build the native NIXL SDK with its UCX backend embedded in the archive. UCX
# and Abseil come from the active development environment.
include(ExternalProject)
find_package(Git REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_program(SIRIUS_MESON_EXECUTABLE meson REQUIRED)
find_program(SIRIUS_NINJA_EXECUTABLE ninja REQUIRED)
pkg_check_modules(SIRIUS_UCX REQUIRED IMPORTED_TARGET "ucx>=1.20.1")
pkg_get_variable(_nixl_ucx_pkgconfig_dir ucx pcfiledir)

set(_nixl_port "${CMAKE_CURRENT_LIST_DIR}/../sirius-duckdb/vcpkg_ports/nixl")
include("${_nixl_port}/nixl-source.cmake")
set(_nixl_patches
    "${_nixl_port}/native-cpp-only.patch"
    "${_nixl_port}/system-tomlplusplus.patch" "${_nixl_port}/static-sdk.patch")
set(SIRIUS_NIXL_BUILD_JOBS
    "$ENV{CMAKE_BUILD_PARALLEL_LEVEL}"
    CACHE STRING "Parallel NIXL jobs (empty uses Ninja's default)")
set(_nixl_parallel)
if(SIRIUS_NIXL_BUILD_JOBS)
  set(_nixl_parallel -j "${SIRIUS_NIXL_BUILD_JOBS}")
endif()
set(_nixl_prefix "${CMAKE_BINARY_DIR}/_deps/nixl-install")
file(MAKE_DIRECTORY "${_nixl_prefix}/include")
list(GET CUDAToolkit_INCLUDE_DIRS 0 _nixl_cuda_include)
get_filename_component(_nixl_cuda_libdir "${CUDA_cudart_LIBRARY}" DIRECTORY)
get_filename_component(_nixl_cuda_driver_libdir "${CUDA_cuda_driver_LIBRARY}"
                       DIRECTORY)

# An explicit native file keeps Meson on the same compiler as Sirius, including
# when CMake was configured with a non-default compiler.
file(
  WRITE "${CMAKE_BINARY_DIR}/_deps/nixl-native.ini"
  "[binaries]\ncpp = '${CMAKE_CXX_COMPILER}'\ncuda = '${CMAKE_CUDA_COMPILER}'\npkg-config = '${PKG_CONFIG_EXECUTABLE}'\n"
)
set(_nixl_archives)
foreach(_library nixl plugin_UCX nixl_build stream serdes nixl_common)
  list(APPEND _nixl_archives "${_nixl_prefix}/lib/lib${_library}.a")
endforeach()
ExternalProject_Add(
  sirius_nixl_build
  SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/nixl-src"
  BINARY_DIR "${CMAKE_BINARY_DIR}/_deps/nixl-build"
  URL https://github.com/ai-dynamo/nixl/archive/${NIXL_SOURCE_REF}.tar.gz
  URL_HASH SHA512=${NIXL_SOURCE_SHA512}
  PATCH_COMMAND "${GIT_EXECUTABLE}" apply --whitespace=nowarn ${_nixl_patches}
  CONFIGURE_COMMAND
    "${SIRIUS_MESON_EXECUTABLE}" setup --reconfigure <BINARY_DIR> <SOURCE_DIR>
    --native-file "${CMAKE_BINARY_DIR}/_deps/nixl-native.ini" --prefix
    "${_nixl_prefix}" --libdir lib --wrap-mode nofallback -Dnative_only=true
    -Ddefault_library=static -Db_staticpic=true -Dstatic_plugins=UCX
    -Denable_plugins=UCX -Dbuild_tests=false -Dbuild_examples=false
    -Dbuild_nixl_ep=false -Dwith_trace=false -Dwerror=false
    "-Dpkg_config_path=${_nixl_ucx_pkgconfig_dir}"
    "-Dcudapath_inc=${_nixl_cuda_include}" "-Dcudapath_lib=${_nixl_cuda_libdir}"
    "-Dcudapath_stub=${_nixl_cuda_driver_libdir}"
  BUILD_COMMAND "${SIRIUS_NINJA_EXECUTABLE}" -C <BINARY_DIR> ${_nixl_parallel}
  INSTALL_COMMAND "${SIRIUS_MESON_EXECUTABLE}" install -C <BINARY_DIR>
                  --no-rebuild INSTALL_BYPRODUCTS ${_nixl_archives})

# Re-extract pristine sources before applying changed patches.
ExternalProject_Add_Step(
  sirius_nixl_build patch_inputs
  DEPENDEES mkdir
  DEPENDERS download
  DEPENDS ${_nixl_patches} INDEPENDENT TRUE)

include("${_nixl_port}/nixl-targets.cmake")
nixl_import_targets("${_nixl_prefix}" PkgConfig::SIRIUS_UCX CUDA::cudart)
add_dependencies(nixl::nixl sirius_nixl_build)
