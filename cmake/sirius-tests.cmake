find_package(Catch2 3 REQUIRED CONFIG)

if(PROJECT_IS_TOP_LEVEL)
  set(sirius_test_library sirius_core)
  add_executable(sirius_unittest ${TEST_SOURCES} src/sirius_extension_entry.cpp
                                 test/cpp/utils/sirius_extension_loader.cpp)
  target_compile_definitions(sirius_unittest PRIVATE SIRIUS_STANDALONE_TESTS)
else()
  set(sirius_test_library sirius_extension)
  add_executable(sirius_unittest ${TEST_SOURCES})
  add_dependencies(sirius_unittest sirius_loadable_extension)
  link_extension_libraries(sirius_unittest "")
endif()

if(VCPKG_BUILD)
  set_target_properties(sirius_unittest PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)
  target_include_directories(sirius_unittest BEFORE PRIVATE ${_VCPKG_INC})
endif()

# Prefer our Catch2 compatibility header over package-provided shims.
target_include_directories(
  sirius_unittest BEFORE
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/test/cpp>)

target_include_directories(
  sirius_unittest
  PRIVATE
    ${SIRIUS_DUCKDB_SOURCE_DIR}/test/include
    ${SIRIUS_DUCKDB_SOURCE_DIR}/third_party/catch
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/compression/simpatico_codegen/src>
)

target_link_libraries(sirius_unittest ${sirius_test_library} duckdb_static
                      ZLIB::ZLIB Catch2::Catch2 ${SIRIUS_CURL_TARGET})

target_include_directories(
  sirius_unittest BEFORE PRIVATE ${SIRIUS_SUBSTRAIT_DIR}/third_party
                                 ${SIRIUS_SUBSTRAIT_DIR}/third_party/substrait)

# DuckDB v1.5.0 unity builds emit strong symbols for static constexpr members
# that conflict with inline definitions from headers included in test files.
target_link_options(sirius_unittest PRIVATE
                    "LINKER:--allow-multiple-definition")

set_target_properties(
  sirius_unittest
  PROPERTIES CXX_SCAN_FOR_MODULES OFF
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test/cpp")

target_compile_definitions(
  sirius_unittest
  PRIVATE
    CCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER
    $<BUILD_INTERFACE:SIRIUS_DEFAULT_LOG_DIR="${CMAKE_BINARY_DIR}/log">
    $<BUILD_INTERFACE:SIRIUS_UNITTEST_LOG_DIR="${CMAKE_CURRENT_BINARY_DIR}/test/cpp/log">
    $<BUILD_INTERFACE:SIRIUS_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}">)

# -----------------------------------------------------------------------------
# test/io/parquet_benchmark — standalone benchmark binary for sirius_datasource
# -----------------------------------------------------------------------------
add_executable(parquet_benchmark test/io/parquet_benchmark.cpp)

target_compile_definitions(parquet_benchmark
                           PRIVATE CCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER)

if(VCPKG_BUILD)
  set_target_properties(parquet_benchmark PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)
  target_include_directories(parquet_benchmark BEFORE PRIVATE ${_VCPKG_INC})
endif()

target_include_directories(
  parquet_benchmark
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)

target_link_libraries(
  parquet_benchmark
  ${sirius_test_library}
  duckdb_static
  cudf::cudf
  rmm::rmm
  spdlog::spdlog
  cuCascade::cucascade
  cuCascade::cucascade_cudf
  PkgConfig::LIBURING
  PkgConfig::NUMA)
if(PROJECT_IS_TOP_LEVEL)
  target_link_libraries(parquet_benchmark duckdb_generated_extension_loader)
else()
  link_extension_libraries(parquet_benchmark "")
endif()

target_link_options(parquet_benchmark PRIVATE
                    "LINKER:--allow-multiple-definition")

set_target_properties(
  parquet_benchmark
  PROPERTIES CXX_SCAN_FOR_MODULES OFF
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test/io")

# -----------------------------------------------------------------------------
# test/io/prefetch_benchmark — plain read_parquet vs. prefetch-then-read
# -----------------------------------------------------------------------------
add_executable(prefetch_benchmark test/io/prefetch_benchmark.cpp)

if(VCPKG_BUILD)
  set_target_properties(prefetch_benchmark PROPERTIES NO_SYSTEM_FROM_IMPORTED
                                                      ON)
  target_include_directories(prefetch_benchmark BEFORE PRIVATE ${_VCPKG_INC})
endif()

target_include_directories(
  prefetch_benchmark
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)

target_link_libraries(
  prefetch_benchmark
  ${sirius_test_library}
  duckdb_static
  cudf::cudf
  rmm::rmm
  spdlog::spdlog
  cuCascade::cucascade
  cuCascade::cucascade_cudf
  PkgConfig::LIBURING
  PkgConfig::NUMA)
if(PROJECT_IS_TOP_LEVEL)
  target_link_libraries(prefetch_benchmark duckdb_generated_extension_loader)
else()
  link_extension_libraries(prefetch_benchmark "")
endif()

target_link_options(prefetch_benchmark PRIVATE
                    "LINKER:--allow-multiple-definition")

set_target_properties(
  prefetch_benchmark
  PROPERTIES CXX_SCAN_FOR_MODULES OFF
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test/io")

# -----------------------------------------------------------------------------
# test/io/prefetch_hybrid_scan_benchmark — read_parquet vs. direct-to-device
# hybrid scan
# -----------------------------------------------------------------------------
add_executable(prefetch_hybrid_scan_benchmark
               test/io/prefetch_hybrid_scan_benchmark.cpp)

if(VCPKG_BUILD)
  set_target_properties(prefetch_hybrid_scan_benchmark
                        PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)
  target_include_directories(prefetch_hybrid_scan_benchmark BEFORE
                             PRIVATE ${_VCPKG_INC})
endif()

target_include_directories(
  prefetch_hybrid_scan_benchmark
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)

target_link_libraries(
  prefetch_hybrid_scan_benchmark
  ${sirius_test_library}
  duckdb_static
  cudf::cudf
  rmm::rmm
  spdlog::spdlog
  cuCascade::cucascade
  cuCascade::cucascade_cudf
  PkgConfig::LIBURING
  PkgConfig::NUMA)
if(PROJECT_IS_TOP_LEVEL)
  target_link_libraries(prefetch_hybrid_scan_benchmark
                        duckdb_generated_extension_loader)
else()
  link_extension_libraries(prefetch_hybrid_scan_benchmark "")
endif()

target_link_options(prefetch_hybrid_scan_benchmark PRIVATE
                    "LINKER:--allow-multiple-definition")

set_target_properties(
  prefetch_hybrid_scan_benchmark
  PROPERTIES CXX_SCAN_FOR_MODULES OFF
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test/io")

# -----------------------------------------------------------------------------
# test/io/columnar_parquet_poc — per-column IO/decode pipelining
# -----------------------------------------------------------------------------
add_executable(columnar_parquet_poc test/io/columnar_parquet_poc.cpp)

if(VCPKG_BUILD)
  set_target_properties(columnar_parquet_poc PROPERTIES NO_SYSTEM_FROM_IMPORTED
                                                        ON)
  target_include_directories(columnar_parquet_poc BEFORE PRIVATE ${_VCPKG_INC})
endif()

target_include_directories(
  columnar_parquet_poc
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)

target_link_libraries(
  columnar_parquet_poc
  ${sirius_test_library}
  duckdb_static
  cudf::cudf
  rmm::rmm
  spdlog::spdlog
  cuCascade::cucascade
  cuCascade::cucascade_cudf
  PkgConfig::LIBURING
  PkgConfig::NUMA)
if(PROJECT_IS_TOP_LEVEL)
  target_link_libraries(columnar_parquet_poc duckdb_generated_extension_loader)
else()
  link_extension_libraries(columnar_parquet_poc "")
endif()

target_link_options(columnar_parquet_poc PRIVATE
                    "LINKER:--allow-multiple-definition")

set_target_properties(
  columnar_parquet_poc
  PROPERTIES CXX_SCAN_FOR_MODULES OFF
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test/io")

# -----------------------------------------------------------------------------
# test/io/retirer_benchmark — cuda_event_completion_poll vs. event + retire
# threads
# -----------------------------------------------------------------------------
add_executable(retirer_benchmark test/io/retirer_benchmark.cpp)

if(VCPKG_BUILD)
  set_target_properties(retirer_benchmark PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)
  target_include_directories(retirer_benchmark BEFORE PRIVATE ${_VCPKG_INC})
endif()

target_include_directories(
  retirer_benchmark
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)

target_link_libraries(
  retirer_benchmark
  ${sirius_test_library}
  duckdb_static
  cudf::cudf
  rmm::rmm
  spdlog::spdlog
  cuCascade::cucascade
  cuCascade::cucascade_cudf
  PkgConfig::LIBURING
  PkgConfig::NUMA)
if(PROJECT_IS_TOP_LEVEL)
  target_link_libraries(retirer_benchmark duckdb_generated_extension_loader)
else()
  link_extension_libraries(retirer_benchmark "")
endif()

target_link_options(retirer_benchmark PRIVATE
                    "LINKER:--allow-multiple-definition")

set_target_properties(
  retirer_benchmark
  PROPERTIES CXX_SCAN_FOR_MODULES OFF
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test/io")

# -----------------------------------------------------------------------------
# test/io/s3_throughput_test — raw S3 read throughput via REST reactor, no
# parquet decoding
# -----------------------------------------------------------------------------
add_executable(s3_throughput_test test/io/s3_throughput_test.cpp)

if(VCPKG_BUILD)
  set_target_properties(s3_throughput_test PROPERTIES NO_SYSTEM_FROM_IMPORTED
                                                      ON)
  target_include_directories(s3_throughput_test BEFORE PRIVATE ${_VCPKG_INC})
endif()

target_include_directories(
  s3_throughput_test
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)

target_link_libraries(
  s3_throughput_test
  ${sirius_test_library}
  duckdb_static
  cudf::cudf
  rmm::rmm
  spdlog::spdlog
  cuCascade::cucascade
  cuCascade::cucascade_cudf
  PkgConfig::LIBURING
  PkgConfig::NUMA)
if(PROJECT_IS_TOP_LEVEL)
  target_link_libraries(s3_throughput_test duckdb_generated_extension_loader)
else()
  link_extension_libraries(s3_throughput_test "")
endif()

target_link_options(s3_throughput_test PRIVATE
                    "LINKER:--allow-multiple-definition")

set_target_properties(
  s3_throughput_test
  PROPERTIES CXX_SCAN_FOR_MODULES OFF
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test/io")

# -----------------------------------------------------------------------------
# test/io/s3_autotune_throughput_bench — connection / GET sizing model vs.
# measured S3 throughput
# -----------------------------------------------------------------------------
add_executable(s3_autotune_throughput_bench
               test/io/s3_autotune_throughput_bench.cpp)

if(VCPKG_BUILD)
  set_target_properties(s3_autotune_throughput_bench
                        PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)
  target_include_directories(s3_autotune_throughput_bench BEFORE
                             PRIVATE ${_VCPKG_INC})
endif()

target_include_directories(
  s3_autotune_throughput_bench
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)

target_link_libraries(
  s3_autotune_throughput_bench
  ${sirius_test_library}
  duckdb_static
  cudf::cudf
  rmm::rmm
  spdlog::spdlog
  cuCascade::cucascade
  cuCascade::cucascade_cudf
  PkgConfig::LIBURING
  PkgConfig::NUMA)
if(PROJECT_IS_TOP_LEVEL)
  target_link_libraries(s3_autotune_throughput_bench
                        duckdb_generated_extension_loader)
else()
  link_extension_libraries(s3_autotune_throughput_bench "")
endif()

target_link_options(s3_autotune_throughput_bench PRIVATE
                    "LINKER:--allow-multiple-definition")

set_target_properties(
  s3_autotune_throughput_bench
  PROPERTIES CXX_SCAN_FOR_MODULES OFF
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test/io")

# -----------------------------------------------------------------------------
# test/io/range_prefetch_benchmark — raw byte-range reads, no parquet decoding
# -----------------------------------------------------------------------------
add_executable(range_prefetch_benchmark test/io/range_prefetch_benchmark.cpp)

if(VCPKG_BUILD)
  set_target_properties(range_prefetch_benchmark
                        PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)
  target_include_directories(range_prefetch_benchmark BEFORE
                             PRIVATE ${_VCPKG_INC})
endif()

target_include_directories(
  range_prefetch_benchmark
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)

target_link_libraries(
  range_prefetch_benchmark
  ${sirius_test_library}
  duckdb_static
  cudf::cudf
  rmm::rmm
  spdlog::spdlog
  cuCascade::cucascade
  cuCascade::cucascade_cudf
  PkgConfig::LIBURING
  PkgConfig::NUMA)
if(PROJECT_IS_TOP_LEVEL)
  target_link_libraries(range_prefetch_benchmark
                        duckdb_generated_extension_loader)
else()
  link_extension_libraries(range_prefetch_benchmark "")
endif()

target_link_options(range_prefetch_benchmark PRIVATE
                    "LINKER:--allow-multiple-definition")

set_target_properties(
  range_prefetch_benchmark
  PROPERTIES CXX_SCAN_FOR_MODULES OFF
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test/io")
