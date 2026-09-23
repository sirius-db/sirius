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
  get_target_property(_cusparse_links CUDA::cusparse INTERFACE_LINK_LIBRARIES)
  if(_cusparse_links)
    list(TRANSFORM _cusparse_links REPLACE "^CUDA::nvJitLink$"
                                           "nvjitlink::nvjitlink_static")
    set_target_properties(CUDA::cusparse PROPERTIES INTERFACE_LINK_LIBRARIES
                                                    "${_cusparse_links}")
  endif()
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
    URL https://github.com/NVIDIA/cuCollections/archive/0883368d39296f3bef3a058033141bcc642c5c54.tar.gz
    URL_HASH
      SHA256=4ec8320a0372839b991f0b431c7f8bf0e770006cb3c8631c6e373c434471fd45
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

# cuCascade - GPU Memory Reservation Library (submodule)
set(BUILD_TESTS
    OFF
    CACHE BOOL "" FORCE)
set(CUCASCADE_BUILD_TESTS
    OFF
    CACHE BOOL "" FORCE)
set(CUCASCADE_BUILD_BENCHMARKS
    OFF
    CACHE BOOL "" FORCE)
set(CUCASCADE_BUILD_SHARED_LIBS
    OFF
    CACHE BOOL "" FORCE)
set(CUCASCADE_BUILD_STATIC_LIBS
    ON
    CACHE BOOL "" FORCE)
# Sirius consumes cucascade's cudf-coupled representations and converters (PR
# #150 split these into the optional cucascade_cudf target), so build it.
set(CUCASCADE_BUILD_CUDF
    ON
    CACHE BOOL "" FORCE)
set(CUCASCADE_WARNINGS_AS_ERRORS
    OFF
    CACHE BOOL "" FORCE)
set(CUCASCADE_BUILD_IO
    OFF
    CACHE BOOL "" FORCE)
add_subdirectory(cucascade "${CMAKE_BINARY_DIR}/cucascade" EXCLUDE_FROM_ALL)

# Name of the NVTX domain every Sirius range is published into. Derived from the
# project name so it has a single authoritative source; simpatico turns it into
# the SIRIUS_NVTX_DOMAIN_NAME compile definition both include trees consume.
set(SIRIUS_NVTX_DOMAIN_NAME "${PROJECT_NAME}")

# Simpatico GPU compression engine Tests disabled here — run them via the
# standalone simpatico_codegen build. Benchmark harness
# (compress_with_plan_benchmark) is built by default.
add_subdirectory(src/compression/simpatico_codegen
                 "${CMAKE_BINARY_DIR}/simpatico_codegen" EXCLUDE_FROM_ALL)

if(VCPKG_BUILD AND TARGET CUDA::cudart_static)
  function(sirius_prefer_static_cudart target_name)
    foreach(_prop LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
      get_target_property(_libs "${target_name}" "${_prop}")
      if(NOT _libs OR _libs STREQUAL "_libs-NOTFOUND")
        continue()
      endif()

      set(_patched_libs "${_libs}")
      list(TRANSFORM _patched_libs REPLACE "^CUDA::cudart$"
                                           "CUDA::cudart_static")
      list(TRANSFORM _patched_libs REPLACE "^\\$<LINK_ONLY:CUDA::cudart>$"
                                           "$<LINK_ONLY:CUDA::cudart_static>")

      if(NOT _patched_libs STREQUAL _libs)
        set_target_properties("${target_name}" PROPERTIES "${_prop}"
                                                          "${_patched_libs}")
      endif()
    endforeach()

    set_target_properties("${target_name}" PROPERTIES CUDA_RUNTIME_LIBRARY
                                                      Static)
  endfunction()

  foreach(_target
          cucascade_objects cucascade_static cucascade_shared
          cucascade_cudf_objects cucascade_cudf_static cucascade_cudf_shared)
    if(TARGET "${_target}")
      sirius_prefer_static_cudart("${_target}")
    endif()
  endforeach()

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

# Upstream testcontainers-native, fetched + patched at configure time (see
# cmake/testcontainers_native.cmake), used by the S3 integration test harness to
# start MinIO containers from the test binary. Builds a Go c-archive, so a Go
# toolchain (provided by pixi) and network access on the first configure/build
# are required — hence gated behind SIRIUS_BUILD_S3_TESTS.
if(SIRIUS_BUILD_S3_TESTS)
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/testcontainers_native.cmake")
endif()
