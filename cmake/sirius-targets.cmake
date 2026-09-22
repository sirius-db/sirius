add_library(sirius_objects OBJECT ${EXTENSION_SOURCES} ${CUDA_SOURCES})
set_target_properties(sirius_objects PROPERTIES POSITION_INDEPENDENT_CODE ON
                                                CXX_VISIBILITY_PRESET hidden)
add_library(sirius_core STATIC $<TARGET_OBJECTS:sirius_objects>)

add_library(sirius_shared SHARED src/sirius_library_anchor.cpp
                                 $<TARGET_OBJECTS:sirius_objects>)
build_static_extension(sirius src/sirius_extension_entry.cpp
                       $<TARGET_OBJECTS:sirius_objects>)
build_loadable_extension(sirius CPP src/sirius_extension_entry.cpp
                         $<TARGET_OBJECTS:sirius_objects>)

# The standalone FFI constructs an embedded DuckDB, which needs the no-op static
# extension loader retained regardless of archive ordering.
set_property(
  TARGET sirius_loadable_extension
  PROPERTY LINK_LIBRARY_OVERRIDE_dummy_static_extension_loader WHOLE_ARCHIVE)

# rapidsai/rmm#826: DuckDB links the loadable extension with
# -Wl,--exclude-libs,ALL, under which mold (but not bfd) hides RMM's GNU_UNIQUE
# current-device-resource registry symbols, so cuDF's internal allocations
# bypass the cuCascade reservation system. Force bfd to keep them exported.
# Harmless for the single-DSO static vcpkg build.
set_target_properties(sirius_loadable_extension PROPERTIES LINKER_TYPE BFD)

# Shared configuration for both extension targets
set(SIRIUS_CLANG_CXX_WARNING_OPTIONS -Wunreachable-code -Wimplicit-fallthrough
                                     -Wrange-loop-analysis -Wnull-dereference)

foreach(_target sirius_objects sirius_core sirius_extension
                sirius_loadable_extension sirius_shared)
  set(_link_scope "")
  if(_target STREQUAL "sirius_shared")
    set(_link_scope PRIVATE)
  endif()
  set_target_properties(
    ${_target}
    PROPERTIES CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
               CUDA_SEPARABLE_COMPILATION ON)
  if(NOT _target STREQUAL "sirius_objects")
    set_target_properties(${_target} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  endif()

  # cuco's device APIs need nvcc's extended device lambda; cuco compiles its own
  # consumers (tests/benchmarks) with --expt-extended-lambda.
  target_compile_options(
    ${_target}
    PRIVATE
      $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-extended-lambda>
      "$<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:${SIRIUS_CLANG_CXX_WARNING_OPTIONS}>"
  )
  target_compile_definitions(${_target}
                             PRIVATE CCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER)

  if(VCPKG_BUILD)
    set_target_properties(${_target} PROPERTIES CUDA_RUNTIME_LIBRARY Static)
  endif()

  # In the vcpkg build, vcpkg include dir must be searched before DuckDB's
  # bundled fmt (which uses duckdb_fmt namespace, incompatible with spdlog).
  # NO_SYSTEM_FROM_IMPORTED prevents -isystem/-I collapse by GCC; BEFORE ensures
  # vcpkg includes precede DuckDB's in the search order.
  if(VCPKG_BUILD)
    set_target_properties(${_target} PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)
    target_include_directories(${_target} BEFORE PRIVATE ${_VCPKG_INC})
  endif()

  target_include_directories(
    ${_target}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/compression/simpatico_codegen/src>
      $<$<BOOL:${SIRIUS_LEGACY_INCLUDE_DIR}>:$<BUILD_INTERFACE:${SIRIUS_LEGACY_INCLUDE_DIR}>>
  )

  # Substrait->DuckDB reader headers (from_substrait.hpp) and its bundled
  # protobuf.
  target_include_directories(
    ${_target}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/extension/core_functions/include
            ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/extension/parquet/include
            ${SIRIUS_SUBSTRAIT_DIR}/src/include
            ${SIRIUS_SUBSTRAIT_DIR}/third_party
            ${SIRIUS_SUBSTRAIT_DIR}/third_party/substrait)

  if(SIRIUS_LEGACY_COMPILE_DEFINITIONS)
    target_compile_definitions(${_target}
                               PRIVATE ${SIRIUS_LEGACY_COMPILE_DEFINITIONS})
  endif()

  # cuCascade::cucascade_cudf holds the cudf-coupled representations and
  # converters Sirius uses; it transitively links the cudf-free core
  # (cuCascade::cucascade) and cudf::cudf.

  target_link_libraries(
    ${_target}
    ${_link_scope}
    cudf::cudf
    cuvs::cuvs
    raft::raft
    cuco::cuco
    rmm::rmm
    spdlog::spdlog
    cuCascade::cucascade
    cuCascade::cucascade_cudf
    yaml-cpp::yaml-cpp
    roaring::roaring
    telemetry_bridge
    core_functions_extension
    parquet_extension
    simpatico)
  if(BUILD_WITH_CTRACK)
    target_link_libraries(${_target} ${_link_scope} ctrack::ctrack)
  endif()

  # Corrosion exposes telemetry_bridge as an INTERFACE target whose concrete
  # Rust archive is telemetry_bridge-static. Apply WHOLE_ARCHIVE to that real
  # archive so the unreferenced NVTX static-injection shim reaches the final
  # loadable extension instead of being dropped at the nested archive boundary.
  set_property(
    TARGET ${_target} PROPERTY "LINK_LIBRARY_OVERRIDE_telemetry_bridge-static"
                               WHOLE_ARCHIVE)

  target_link_libraries(
    ${_target}
    ${_link_scope}
    PkgConfig::NUMA
    PkgConfig::LIBURING
    ${SIRIUS_CURL_TARGET}
    OpenSSL::Crypto
    absl::any_invocable)
  add_dependencies(${_target} duckdb_static)
endforeach()

# `sirius_core` is itself an archive, so its LINK_LIBRARY_OVERRIDE does not
# perform a final link. Carry the concrete Rust archive as a transitive
# WHOLE_ARCHIVE item instead; DuckDB and every other final consumer then retain
# the static NVTX pointer shim as well.
foreach(_target sirius_core sirius_extension)
  target_link_libraries(${_target}
                        "$<LINK_LIBRARY:WHOLE_ARCHIVE,telemetry_bridge-static>")
endforeach()

# A statically embedded Sirius cannot give NVTX a DSO path. Its private dlopen
# interposer maps one sentinel path to the running executable instead. Carry
# both symbols into the final executable's dynamic symbol table so dependency
# images such as libcudf can resolve the Quent initializer from that handle.
foreach(_target sirius_core sirius_extension)
  target_link_options(
    ${_target} INTERFACE
    "LINKER:--export-dynamic-symbol=InitializeInjectionNvtx2"
    "LINKER:--export-dynamic-symbol=dlopen")
endforeach()

target_link_libraries(sirius_loadable_extension PkgConfig::LIBURING
                      ${SIRIUS_CURL_TARGET} OpenSSL::Crypto)

# NVTX's runtime injection lookup dlopens the path named by
# NVTX_INJECTION64_PATH and resolves InitializeInjectionNvtx2 from it. Export
# the statically embedded Quent entry point from the loadable extension so
# Sirius can point NVTX at its own already-loaded DSO without deploying a second
# injection library.
target_link_options(sirius_loadable_extension PRIVATE
                    "LINKER:--export-dynamic-symbol=InitializeInjectionNvtx2")

add_library(sirius::sirius ALIAS sirius_shared)
set_target_properties(
  sirius_shared
  PROPERTIES OUTPUT_NAME sirius
             EXPORT_NAME sirius
             VERSION "${PROJECT_VERSION}"
             SOVERSION 0
             INSTALL_RPATH "$ORIGIN"
             INSTALL_REMOVE_ENVIRONMENT_RPATH ON
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(
  sirius_shared PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                       $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(sirius_shared PUBLIC cxx_std_20)
target_link_libraries(
  sirius_shared
  PRIVATE duckdb_static
          "$<LINK_LIBRARY:WHOLE_ARCHIVE,dummy_static_extension_loader>")
set_target_properties(sirius_shared PROPERTIES LINKER_TYPE LLD)

# Discard unused sections pulled in by whole archives.
target_link_options(sirius_shared PRIVATE "LINKER:--gc-sections")

# The sirius-sys + sirius Rust crates are built by cargo, not CMake (unlike the
# telemetry bridge above, which CMake drives via Corrosion). Their build.rs
# discovers the Sirius headers (repo + conda) and links the libsirius artifact
# this build produces under build/<preset>/; see rust/crates/sirius-sys.

# cucascade upstream PRs #126/#128/#130 moved cucascade to static-CUDA linkage
# and pulled the NVML *static stub* (libnvidia-ml.a) into libcucascade.a via the
# `CUDA::nvml_static` imported target. Upstream's
# `LINKER:--exclude-libs,libnvidia-ml` workaround (PR #130) keeps cucascade's
# own tests happy by hiding the bundled stub symbols from the dynamic export
# table (NVIDIA bug 6174166: libnvidia-ml.so dlsyms into the host process and
# recurses into the stub). For sirius's larger transitive graph, hiding the
# symbols segfaults at runtime — DuckDB/RMM call nvml directly before nvmlInit's
# jump-table patch fires — while leaving them visible deadlocks in stubSpinLock
# when the real driver loads.
#
# The robust fix for sirius: override CUDA::nvml_static's IMPORTED location to
# point at the *shared* stub (libnvidia-ml.so) in the same pixi env. The shared
# stub uses normal dynamic linking via ld.so, which resolves to the real
# libnvidia-ml.so.1 at runtime, matching the working Phase-24 configuration.
# This bypasses both failure modes (no stub symbols bundled, no dynamic-export
# conflict).
if(TARGET CUDA::nvml_static)
  get_target_property(_nvml_static_loc CUDA::nvml_static IMPORTED_LOCATION)
  if(_nvml_static_loc AND _nvml_static_loc MATCHES "libnvidia-ml\\.a$")
    string(REGEX REPLACE "libnvidia-ml\\.a$" "libnvidia-ml.so" _nvml_shared_loc
                         "${_nvml_static_loc}")
    if(EXISTS "${_nvml_shared_loc}")
      set_target_properties(
        CUDA::nvml_static
        PROPERTIES IMPORTED_LOCATION "${_nvml_shared_loc}"
                   IMPORTED_LOCATION_RELEASE "${_nvml_shared_loc}")
      message(
        STATUS
          "Sirius: redirected CUDA::nvml_static from ${_nvml_static_loc} to ${_nvml_shared_loc}"
      )
    endif()
  endif()
endif()
