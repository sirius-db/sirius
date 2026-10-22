cmake_minimum_required(VERSION 4.0)
project(sirius_udf_benchmark LANGUAGES CXX CUDA)

find_package(sirius CONFIG REQUIRED COMPONENTS udf)
find_package(CUDAToolkit REQUIRED)
set(SIRIUS_UDF_DUCKDB_LIBRARY
    ""
    CACHE FILEPATH "DuckDB shared library from the Sirius provider build")
if(NOT EXISTS "${SIRIUS_UDF_DUCKDB_LIBRARY}")
  message(
    FATAL_ERROR "Set SIRIUS_UDF_DUCKDB_LIBRARY to the matching libduckdb.so")
endif()
add_library(provider_duckdb SHARED IMPORTED)
set_target_properties(provider_duckdb PROPERTIES IMPORTED_LOCATION
                                                 "${SIRIUS_UDF_DUCKDB_LIBRARY}")

add_library(benchmark_cudf MODULE udfs/cudf/udf.cpp)
add_library(benchmark_cuda MODULE udfs/cuda/udf.cpp udfs/cuda/udf.cu)
add_library(benchmark_q1 MODULE udfs/q1/udf.cpp)
foreach(target benchmark_cudf benchmark_cuda benchmark_q1)
  target_link_libraries(${target} PRIVATE sirius::udf_sdk)
  target_compile_features(${target} PRIVATE cxx_std_20)
  set_target_properties(${target} PROPERTIES PREFIX "" CXX_SCAN_FOR_MODULES OFF)
endforeach()
target_link_libraries(benchmark_cuda PRIVATE CUDA::cudart)
set_target_properties(benchmark_cuda PROPERTIES CUDA_STANDARD 20
                                                CUDA_STANDARD_REQUIRED ON)

add_executable(bench_host bench_host.cpp)
target_link_libraries(bench_host PRIVATE sirius::udf_sdk provider_duckdb
                                         CUDA::cudart ${CMAKE_DL_LIBS})
target_include_directories(
  bench_host
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../.agents/skills/sirius-udf/assets/native")
target_compile_features(bench_host PRIVATE cxx_std_20)
set_target_properties(bench_host PROPERTIES ENABLE_EXPORTS ON
                                            CXX_SCAN_FOR_MODULES OFF)

file(
  GENERATE
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/benchmark-build.json"
  CONTENT
    "{\n  \"cxx_compiler\": \"${CMAKE_CXX_COMPILER_ID}\",\n  \"compiler_version\": \"${CMAKE_CXX_COMPILER_VERSION}\",\n  \"cuda_compiler_version\": \"${CMAKE_CUDA_COMPILER_VERSION}\",\n  \"cudf_version\": \"${SIRIUS_UDF_CUDF_VERSION}\",\n  \"rmm_version\": \"${SIRIUS_UDF_RMM_VERSION}\",\n  \"duckdb_revision\": \"${SIRIUS_UDF_DUCKDB_REVISION}\",\n  \"build_type\": \"$<CONFIG>\"\n}\n"
)
