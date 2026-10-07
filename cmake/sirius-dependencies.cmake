# Dependencies
find_package(cudf REQUIRED CONFIG)
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

# The static vcpkg build only exports cuvs::cuvs_static; conda's shared build
# already provides cuvs::cuvs. Provide the canonical name when it is missing so
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
# (see src/compression/simpatico_codegen/CMakeLists.txt). The pixi build uses
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

# libcudf no longer ships its bundled copy. In the vcpkg build cuco comes from
# the overlay port (vcpkg_ports/cuco); configure-time downloads are disabled
# there. Otherwise (pixi build) fetch the same commit cudf is built against
# (populate sources without add_subdirectory; CCCL comes from cudf). Either way
# cuco is header-only and exposed as the cuco::cuco target, so both paths
# consume it identically below.
if(VCPKG_BUILD)
  find_package(cuco CONFIG REQUIRED)
else()
  include(FetchContent)
  FetchContent_Declare(
    cuco
    URL https://github.com/NVIDIA/cuCollections/archive/4b26118c99866221f99f35f4e3bc74afdbe063bc.tar.gz
    URL_HASH
      SHA256=cfff0dfe8552ca2a8e3c53d04c26ab4d95364d14c159aaf8a37b3971b78b609d
    SOURCE_SUBDIR do-not-build)
  FetchContent_MakeAvailable(cuco)
  # SOURCE_SUBDIR do-not-build populates headers without running cuco's CMake,
  # so no target is created. Define the same header-only cuco::cuco the vcpkg
  # port exports (CCCL still comes from cudf) so both build paths consume cuco
  # identically below.
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

# Scope dependency options without changing the caller's cache.
block()
set(CUCASCADE_BUILD_TESTS OFF)
set(CUCASCADE_BUILD_BENCHMARKS OFF)
set(CUCASCADE_BUILD_SHARED_LIBS OFF)
set(CUCASCADE_BUILD_STATIC_LIBS ON)
set(CUCASCADE_BUILD_CUDF ON)
set(CUCASCADE_WARNINGS_AS_ERRORS OFF)
set(CUCASCADE_BUILD_IO OFF)
add_subdirectory(cucascade "${CMAKE_BINARY_DIR}/cucascade" EXCLUDE_FROM_ALL)
endblock()
foreach(target cucascade_objects cucascade_cudf_objects)
  target_compile_definitions(${target} PRIVATE CCCL_DISABLE_WARPSPEED_SCAN)
endforeach()

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

find_package(kvikio REQUIRED CONFIG)

# The legacy DuckDB parent enables only C/CXX.
if(NOT PROJECT_IS_TOP_LEVEL)
  if(TARGET BS::thread_pool)
    get_target_property(_bs_thread_pool_features BS::thread_pool
                        INTERFACE_COMPILE_FEATURES)
    if(_bs_thread_pool_features)
      list(REMOVE_ITEM _bs_thread_pool_features cuda_std_17)
      set_target_properties(
        BS::thread_pool PROPERTIES INTERFACE_COMPILE_FEATURES
                                   "${_bs_thread_pool_features}")
    endif()
  endif()
endif()
