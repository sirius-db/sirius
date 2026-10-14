# Keep the public configuration implementation's C++23 requirement local.
add_library(sirius_context_config OBJECT src/context_config.cpp)
target_compile_features(sirius_context_config PRIVATE cxx_std_23)
set_target_properties(
  sirius_context_config PROPERTIES POSITION_INDEPENDENT_CODE ON
                                   CXX_VISIBILITY_PRESET hidden)

set_target_properties(sirius_objects PROPERTIES POSITION_INDEPENDENT_CODE ON
                                                CXX_VISIBILITY_PRESET hidden)
add_library(sirius_core STATIC $<TARGET_OBJECTS:sirius_objects>)
set_target_properties(sirius_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(sirius_shared SHARED src/sirius_library_anchor.cpp
                                 $<TARGET_OBJECTS:sirius_objects>)
# Link the public configuration implementation into both Sirius libraries.
foreach(_target sirius_core sirius_shared)
  target_sources(${_target} PRIVATE $<TARGET_OBJECTS:sirius_context_config>)
endforeach()

if(VCPKG_BUILD AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(_sirius_cuda_link_script
      "${CMAKE_CURRENT_LIST_DIR}/sirius-cuda-fatbin.ld")
  set(_sirius_cuda_link_interface
      "$<BUILD_INTERFACE:${_sirius_cuda_link_script}>$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_DATADIR}/sirius/sirius-cuda-fatbin.ld>"
  )
  target_link_options(sirius_core INTERFACE
                      "$<HOST_LINK:LINKER:-T,${_sirius_cuda_link_interface}>")
  set_property(
    TARGET sirius_core
    APPEND
    PROPERTY INTERFACE_LINK_DEPENDS "${_sirius_cuda_link_interface}")

  target_link_options(sirius_shared PRIVATE
                      "$<HOST_LINK:LINKER:-T,${_sirius_cuda_link_script}>")
  set_property(
    TARGET sirius_shared
    APPEND
    PROPERTY LINK_DEPENDS "${_sirius_cuda_link_script}")

endif()

# Shared configuration for the engine targets
set(SIRIUS_CLANG_CXX_WARNING_OPTIONS -Wunreachable-code -Wimplicit-fallthrough
                                     -Wrange-loop-analysis -Wnull-dereference)

set(SIRIUS_LINK_LIBRARIES
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
    $<BUILD_INTERFACE:sirius::duckdb_dependency>
    simpatico
    PkgConfig::NUMA
    PkgConfig::LIBURING
    ${SIRIUS_CURL_TARGET}
    OpenSSL::Crypto
    absl::any_invocable
    kvikio::kvikio)
if(BUILD_WITH_CTRACK)
  list(APPEND SIRIUS_LINK_LIBRARIES $<BUILD_INTERFACE:ctrack::ctrack>)
endif()

foreach(_target sirius_objects sirius_context_config sirius_core sirius_shared)
  set_target_properties(${_target} PROPERTIES CXX_SCAN_FOR_MODULES OFF)
  if(NOT _target STREQUAL "sirius_context_config")
    set_target_properties(
      ${_target}
      PROPERTIES CXX_STANDARD 20
                 CXX_STANDARD_REQUIRED ON
                 CUDA_STANDARD 20
                 CUDA_STANDARD_REQUIRED ON
                 CUDA_SEPARABLE_COMPILATION ON)
    if(NOT _target STREQUAL "sirius_objects")
      set_target_properties(${_target} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS
                                                  ON)
    endif()
  endif()

  # cuco's device APIs need nvcc's extended device lambda; cuco compiles its own
  # consumers (tests/benchmarks) with --expt-extended-lambda.
  target_compile_options(
    ${_target}
    PRIVATE
      $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-extended-lambda>
      "$<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:${SIRIUS_CLANG_CXX_WARNING_OPTIONS}>"
  )
  target_compile_definitions(
    ${_target} PRIVATE CCCL_DISABLE_WARPSPEED_SCAN
                       CCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER)

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
  )

  # Substrait->DuckDB reader headers (from_substrait.hpp) and its bundled
  # protobuf.
  target_include_directories(
    ${_target}
    PRIVATE ${SIRIUS_SUBSTRAIT_DIR}/src/include
            ${SIRIUS_SUBSTRAIT_DIR}/third_party
            ${SIRIUS_SUBSTRAIT_DIR}/third_party/substrait)

  # cuCascade::cucascade_cudf holds the cudf-coupled representations and
  # converters Sirius uses; it transitively links the cudf-free core
  # (cuCascade::cucascade) and cudf::cudf.

  target_link_libraries(${_target} PRIVATE ${SIRIUS_LINK_LIBRARIES})

  # Corrosion exposes telemetry_bridge as an INTERFACE target whose concrete
  # Rust archive is telemetry_bridge-static. Apply WHOLE_ARCHIVE to that real
  # archive so the unreferenced NVTX static-injection shim reaches the final
  # loadable extension instead of being dropped at the nested archive boundary.
  set_property(
    TARGET ${_target} PROPERTY "LINK_LIBRARY_OVERRIDE_telemetry_bridge-static"
                               WHOLE_ARCHIVE)

endforeach()

# Internal tests include engine headers; installed consumers use the public API.
target_link_libraries(sirius_core
                      PUBLIC "$<BUILD_INTERFACE:${SIRIUS_LINK_LIBRARIES}>")

# `sirius_core` is itself an archive, so its LINK_LIBRARY_OVERRIDE does not
# perform a final link. Carry the concrete Rust archive as a transitive
# WHOLE_ARCHIVE item instead; DuckDB and every other final consumer then retain
# the static NVTX pointer shim as well.
target_link_libraries(
  sirius_core PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,telemetry_bridge-static>")

# A statically embedded Sirius cannot give NVTX a DSO path. Its private dlopen
# interposer maps one sentinel path to the running executable instead. Carry
# both symbols into the final executable's dynamic symbol table so dependency
# images such as libcudf can resolve the Quent initializer from that handle.
target_link_options(
  sirius_core INTERFACE
  "LINKER:--export-dynamic-symbol=InitializeInjectionNvtx2"
  "LINKER:--export-dynamic-symbol=dlopen")

add_library(sirius::sirius ALIAS sirius_shared)
set_target_properties(
  sirius_shared
  PROPERTIES OUTPUT_NAME sirius
             EXPORT_NAME sirius
             VERSION "${PROJECT_VERSION}"
             SOVERSION 0
             INSTALL_RPATH "$ORIGIN"
             INSTALL_REMOVE_ENVIRONMENT_RPATH ON)
target_compile_features(sirius_shared INTERFACE cxx_std_23)
# Installed static consumers use the public headers too. Keep this requirement
# out of DuckDB's in-tree build graph.
target_compile_features(sirius_core INTERFACE "$<INSTALL_INTERFACE:cxx_std_23>")
set_target_properties(sirius_shared PROPERTIES LINKER_TYPE LLD)

# Keep the embedded DuckDB private when the host exposes another DuckDB
# globally.
foreach(target duckdb_static core_functions_extension parquet_extension)
  target_link_options(sirius_shared PRIVATE
                      "LINKER:--exclude-libs,$<TARGET_FILE_NAME:${target}>")
endforeach()

# Discard unused sections pulled in by whole archives.
target_link_options(sirius_shared PRIVATE "LINKER:--gc-sections"
                    "LINKER:--allow-multiple-definition")

# The sirius-sys + sirius Rust crates are built by cargo, not CMake (unlike the
# telemetry bridge above, which CMake drives via Corrosion). Their build.rs
# discovers the Sirius headers (repo + conda) and links the libsirius artifact
# this build produces under build/<preset>/; see rust/crates/sirius-sys.

# NVML's static stub cannot be embedded safely; the real library is supplied by
# the NVIDIA driver. Keep this requirement in the exported support target.
foreach(property LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
  get_target_property(nvml_links cucascade_topology_discovery_static
                      ${property})
  if(nvml_links)
    string(REPLACE "CUDA::nvml_static" "CUDA::nvml" nvml_links "${nvml_links}")
    set_property(TARGET cucascade_topology_discovery_static
                 PROPERTY ${property} "${nvml_links}")
  endif()
endforeach()

if(NOT SIRIUS_BUILD_SHARED)
  set_target_properties(sirius_shared PROPERTIES EXCLUDE_FROM_ALL ON)
endif()
add_custom_target(sirius_library ALL)
if(SIRIUS_BUILD_SHARED)
  add_dependencies(sirius_library sirius_shared)
endif()
if(SIRIUS_BUILD_STATIC)
  add_dependencies(sirius_library sirius_core)
endif()

if(SIRIUS_BUILD_STATIC)
  add_library(sirius::sirius_static ALIAS sirius_core)
  set_target_properties(sirius_core PROPERTIES OUTPUT_NAME sirius EXPORT_NAME
                                                                  sirius_static)
  target_link_libraries(
    sirius_core PRIVATE duckdb_static core_functions_extension
                        parquet_extension)
  target_link_options(
    sirius_core INTERFACE "LINKER:--undefined=InitializeInjectionNvtx2"
    "LINKER:--allow-multiple-definition")
endif()
