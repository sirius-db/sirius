# Dependencies
find_package(cudf REQUIRED CONFIG)
if(NOT VCPKG_BUILD)
  # TODO: Remove with ast_compat.hpp and the cudf port patch after cudf#24173
  # ships. Cover transitive AST includes regardless of source header order.
  target_compile_options(
    cudf::cudf
    INTERFACE "$<$<COMPILE_LANGUAGE:CXX,CUDA>:SHELL:-include variant>")
endif()
find_package(raft REQUIRED CONFIG) # raft must be found before cuvs
find_package(cuvs REQUIRED CONFIG)
find_package(spdlog REQUIRED CONFIG)
find_package(yaml-cpp REQUIRED CONFIG)
# CRoaring decodes Iceberg Puffin deletion vectors (portable Roaring). Host-side
# only; the same library duckdb-iceberg uses for the same blob, so the two
# readers agree by construction.
find_package(roaring REQUIRED CONFIG)
find_package(absl REQUIRED CONFIG)
find_package(PkgConfig REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(ZLIB REQUIRED)

# The static vcpkg build only exports cuvs::cuvs_static; shared cuvs packages
# already provide cuvs::cuvs. Provide the canonical name when it is missing so
# both build paths link cuvs::cuvs the same way below.
if(NOT TARGET cuvs::cuvs)
  add_library(cuvs::cuvs INTERFACE IMPORTED)
  set_target_properties(cuvs::cuvs PROPERTIES INTERFACE_LINK_LIBRARIES
                                              cuvs::cuvs_static)
endif()

# Static NVRTC (vcpkg overlay port). Statically linking the runtime JIT compiler
# keeps libnvrtc.so out of the distributed extension's runtime dependencies.
# Like cuco/nvcomp, this is gated on the vcpkg build: the overlay port provides
# the nvrtc::nvrtc_static target, which simpatico links instead of CUDA::nvrtc
# (see src/compression/simpatico_codegen/CMakeLists.txt). Other builds use
# CUDA::nvrtc for the toolkit's shared library.
if(VCPKG_BUILD)
  find_package(nvrtc CONFIG REQUIRED)
  find_package(nvjitlink CONFIG REQUIRED)
  # FindCUDAToolkit also adds nvJitLink through cuSPARSE's link interface.
  foreach(target CUDA::cusparse CUDA::cusparse_static)
    if(TARGET ${target})
      get_target_property(_cusparse_links ${target} INTERFACE_LINK_LIBRARIES)
      if(_cusparse_links)
        list(TRANSFORM _cusparse_links REPLACE "^CUDA::nvJitLink(_static)?$"
                                               "nvjitlink::nvjitlink_static")
        set_target_properties(${target} PROPERTIES INTERFACE_LINK_LIBRARIES
                                                   "${_cusparse_links}")
      endif()
    endif()
  endforeach()
endif()

# --- cuCollections (cuco) --- #

# libcudf no longer ships its bundled copy. Use an installed cuco package when
# one is found (the vcpkg overlay port, or one the caller provides); otherwise
# fetch the commit cudf is built against. Either way cuco is header-only and
# exposed as cuco::cuco (CCCL comes from cudf).
if(VCPKG_BUILD)
  find_package(cuco CONFIG REQUIRED)
else()
  include(FetchContent)
  FetchContent_Declare(
    cuco
    URL https://github.com/NVIDIA/cuCollections/archive/4b26118c99866221f99f35f4e3bc74afdbe063bc.tar.gz
    URL_HASH
      SHA256=cfff0dfe8552ca2a8e3c53d04c26ab4d95364d14c159aaf8a37b3971b78b609d
    SOURCE_SUBDIR do-not-build FIND_PACKAGE_ARGS CONFIG)
  FetchContent_MakeAvailable(cuco)
  # The fetched sources are not configured, so define the header-only target.
  if(NOT TARGET cuco::cuco)
    add_library(cuco::cuco INTERFACE IMPORTED)
    set_target_properties(cuco::cuco PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                                "${cuco_SOURCE_DIR}/include")
  endif()
endif()

# CTrack is retained as an opt-in profiling dependency.
if(BUILD_WITH_CTRACK)
  # --- ctrack (profiling instrumentation) --- #
  include(FetchContent)
  FetchContent_Declare(
    ctrack
    GIT_REPOSITORY https://github.com/Compaile/ctrack.git
    GIT_TAG 6dfa9b0eef26507cfa9bc17ae4d817ab011a28f5 # v1.1.0
    SOURCE_SUBDIR do-not-build)
  FetchContent_MakeAvailable(ctrack)
  if(NOT TARGET ctrack::ctrack)
    add_library(ctrack::ctrack INTERFACE IMPORTED)
    set_target_properties(
      ctrack::ctrack PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                "${ctrack_SOURCE_DIR}/include")
  endif()

endif()

if(VCPKG_BUILD)
  set(_sirius_pkg_config_path "$ENV{PKG_CONFIG_PATH}")
  set(ENV{PKG_CONFIG_PATH}
      "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib/pkgconfig:${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share/pkgconfig:$ENV{PKG_CONFIG_PATH}"
  )
endif()
pkg_check_modules(NUMA REQUIRED IMPORTED_TARGET numa)
pkg_check_modules(LIBURING REQUIRED IMPORTED_TARGET liburing)
if(VCPKG_BUILD)
  # The CMake target includes curl's private static dependencies.
  find_package(CURL CONFIG REQUIRED)
  set(SIRIUS_CURL_TARGET CURL::libcurl)
else()
  pkg_check_modules(CURL REQUIRED IMPORTED_TARGET libcurl)
  set(SIRIUS_CURL_TARGET PkgConfig::CURL)
endif()

if(VCPKG_BUILD)
  set(ENV{PKG_CONFIG_PATH} "${_sirius_pkg_config_path}")
endif()

# Scope dependency options without changing the caller's cache.
block()
set(CUCASCADE_BUILD_TESTS OFF)
set(CUCASCADE_BUILD_BENCHMARKS OFF)
set(CUCASCADE_BUILD_SHARED_LIBS OFF)
set(CUCASCADE_BUILD_STATIC_LIBS ON)
set(CUCASCADE_BUILD_CUDF ON)
set(CUCASCADE_WARNINGS_AS_ERRORS OFF)
# cuCascade io's swappable third-party backends are pointed at the copies Sirius
# already links so both sides of the boundary share one ABI: exec::invocable is
# absl::AnyInvocable (Sirius's type) and moodycamel is duckdb's vendored fork
# (namespace duckdb_moodycamel, target from sirius-duckdb-provider.cmake).
set(CUCASCADE_BUILD_IO ON)
set(CUCASCADE_USE_ABSEIL_INVOCABLE ON)
set(CUCASCADE_MOODYCAMEL_TARGET duckdb::moodycamel)
set(CUCASCADE_MOODYCAMEL_NAMESPACE duckdb_moodycamel)
add_subdirectory(cucascade "${CMAKE_BINARY_DIR}/cucascade" EXCLUDE_FROM_ALL)
endblock()
# cuCascade has no C++ modules; scanning breaks older compiler caches.
# cucascade_io_objects exists only when CUCASCADE_BUILD_IO is ON.
foreach(target cucascade_objects cucascade_cudf_objects
               cucascade_topology_discovery_objects cucascade_io_objects)
  if(TARGET ${target})
    set_target_properties(${target} PROPERTIES CXX_SCAN_FOR_MODULES OFF)
  endif()
endforeach()
foreach(target cucascade_objects cucascade_cudf_objects cucascade_io_objects)
  if(TARGET ${target})
    target_compile_definitions(${target} PRIVATE CCCL_DISABLE_WARPSPEED_SCAN)
  endif()
endforeach()
# Topology discovery gets NVTX only from CUDA::nvtx3, i.e. the toolkit include
# directory. Split toolkits (e.g. conda) install the headers elsewhere. Remove
# once cuCascade looks them up itself.
find_path(SIRIUS_NVTX3_INCLUDE_DIR nvtx3/nvtx3.hpp
          HINTS ${CUDAToolkit_INCLUDE_DIRS})
if(SIRIUS_NVTX3_INCLUDE_DIR)
  target_include_directories(cucascade_topology_discovery_objects SYSTEM
                             PRIVATE "${SIRIUS_NVTX3_INCLUDE_DIR}")
endif()

# Name of the NVTX domain every Sirius range is published into. Derived from the
# project name so it has a single authoritative source; simpatico turns it into
# the SIRIUS_NVTX_DOMAIN_NAME compile definition both include trees consume.
set(SIRIUS_NVTX_DOMAIN_NAME "${PROJECT_NAME}")

# Simpatico GPU compression engine Tests disabled here — run them via the
# standalone simpatico_codegen build. Benchmark harness
# (compress_with_plan_benchmark) is built by default.
add_subdirectory(src/compression/simpatico_codegen
                 "${CMAKE_BINARY_DIR}/simpatico_codegen" EXCLUDE_FROM_ALL)

if(VCPKG_BUILD)
  # cucascade's topology discovery gained rmm includes (NUMA capacity
  # detection), so it needs the same vcpkg-over-toolkit include priority as
  # sirius's own rapids consumers: without it GCC drops the duplicated vcpkg -I
  # in favor of the later -isystem, the CUDA toolkit's CCCL wins the search
  # order, and CUDA 12's bundled CCCL (< 3.3) trips rmm's version check. Only
  # the object target compiles TUs; the static wrapper just links the objects.
  if(TARGET cucascade_topology_discovery_objects)
    set_target_properties(cucascade_topology_discovery_objects
                          PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)
    target_include_directories(cucascade_topology_discovery_objects BEFORE
                               PRIVATE ${_VCPKG_INC})
  endif()
endif()

# Rust telemetry instrumentation (C++ FFI via Corrosion)
add_subdirectory(rust/crates/telemetry/bridge)

if(SIRIUS_WITH_NIXL)
  if(NOT SIRIUS_NIXL_PREFIX)
    # $NIXL_PREFIX, else the script's default prefix (it names the pinned NIXL
    # commit). Cached, so that a later automatic reconfigure (ninja, or make
    # after a CMake file changed) keeps this NIXL whatever the shell's
    # NIXL_PREFIX is then; -DSIRIUS_NIXL_PREFIX= resolves it again.
    execute_process(
      COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/scripts/install_nixl.sh"
              --print-prefix
      OUTPUT_VARIABLE _nixl_prefix
      OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    set(SIRIUS_NIXL_PREFIX
        "${_nixl_prefix}"
        CACHE PATH
              "NIXL install prefix (empty: $NIXL_PREFIX, else the default)"
              FORCE)
  endif()
  string(
    CONCAT
      _nixl_hint
      "Install it with: pixi run scripts/install_nixl.sh, or choose an install "
      "with -DSIRIUS_NIXL_PREFIX=<prefix> (an empty value takes $NIXL_PREFIX, "
      "else the default, again; pixi run build-distributed passes it empty)")
  # The found paths are cached: look again when the prefix changes.
  if(NOT "${SIRIUS_NIXL_PREFIX}" STREQUAL "${_SIRIUS_NIXL_FOUND_IN}")
    unset(NIXL_INCLUDE_DIR CACHE)
    unset(NIXL_LIBRARY CACHE)
    unset(NIXL_BUILD_LIBRARY CACHE)
    set(_SIRIUS_NIXL_FOUND_IN
        "${SIRIUS_NIXL_PREFIX}"
        CACHE INTERNAL "")
  endif()
  find_path(
    NIXL_INCLUDE_DIR nixl.h
    PATHS ${SIRIUS_NIXL_PREFIX}/include
    NO_DEFAULT_PATH)
  find_library(
    NIXL_LIBRARY nixl
    PATHS ${SIRIUS_NIXL_PREFIX}/lib
    PATH_SUFFIXES aarch64-linux-gnu x86_64-linux-gnu
    NO_DEFAULT_PATH)
  # The descriptor lists live in libnixl_build.
  find_library(
    NIXL_BUILD_LIBRARY nixl_build
    PATHS ${SIRIUS_NIXL_PREFIX}/lib
    PATH_SUFFIXES aarch64-linux-gnu x86_64-linux-gnu
    NO_DEFAULT_PATH)
  if(NOT NIXL_INCLUDE_DIR
     OR NOT NIXL_LIBRARY
     OR NOT NIXL_BUILD_LIBRARY)
    message(FATAL_ERROR "SIRIUS_WITH_NIXL: no NIXL in ${SIRIUS_NIXL_PREFIX}. "
                        "${_nixl_hint}")
  endif()
  # NIXL dlopens its UCX plugin from plugins/ next to libnixl; only the plugin's
  # own RPATH finds libucp there.
  get_filename_component(NIXL_LIBRARY_DIR "${NIXL_LIBRARY}" DIRECTORY)
  execute_process(
    COMMAND ldd "${NIXL_LIBRARY_DIR}/plugins/libplugin_UCX.so"
    OUTPUT_VARIABLE _nixl_ucx_ldd
    ERROR_VARIABLE _nixl_ucx_ldd
    RESULT_VARIABLE _nixl_ucx_ldd_result)
  if(NOT _nixl_ucx_ldd_result EQUAL 0 OR _nixl_ucx_ldd MATCHES "not found")
    message(
      FATAL_ERROR
        "SIRIUS_WITH_NIXL: ${NIXL_LIBRARY_DIR}/plugins/libplugin_UCX.so "
        "does not resolve its libraries (was the UCX it was built against "
        "removed?):\n${_nixl_ucx_ldd}${_nixl_hint}")
  endif()
  message(STATUS "Exchange NIXL fabric: ${NIXL_LIBRARY}")
endif()
