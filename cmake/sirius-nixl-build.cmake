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

set(_nixl_port "${CMAKE_CURRENT_LIST_DIR}/../vcpkg_ports/nixl")
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
# Keep the source pin in sync with vcpkg_ports/nixl/portfile.cmake.
ExternalProject_Add(
  sirius_nixl_build
  SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/nixl-src"
  BINARY_DIR "${CMAKE_BINARY_DIR}/_deps/nixl-build"
  URL https://github.com/ai-dynamo/nixl/archive/1683cf3b7f3d11674c03c5e861cea22339876c96.tar.gz
  URL_HASH
    SHA512=bd27d3ab6e5a9e4bd14781731a738a3668befe15fad2b7aca848c237d9b808d30f5e5a8c067c52d6ebb6744ec096f1a12d6f539f0604962764c984d07f4acdfb
  PATCH_COMMAND
    "${GIT_EXECUTABLE}" apply --whitespace=nowarn
    "${_nixl_port}/native-cpp-only.patch"
    "${_nixl_port}/system-tomlplusplus.patch" "${_nixl_port}/static-sdk.patch"
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
  BUILD_COMMAND "${SIRIUS_NINJA_EXECUTABLE}" -C <BINARY_DIR> -j4
  INSTALL_COMMAND "${SIRIUS_MESON_EXECUTABLE}" install -C <BINARY_DIR>
                  --no-rebuild INSTALL_BYPRODUCTS ${_nixl_archives})

include("${_nixl_port}/nixl-targets.cmake")
nixl_import_targets("${_nixl_prefix}" PkgConfig::SIRIUS_UCX CUDA::cudart)
add_dependencies(nixl::nixl sirius_nixl_build)
