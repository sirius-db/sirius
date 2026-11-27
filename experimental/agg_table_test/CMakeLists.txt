# Standalone differential test of the aggregation-table library (src/cuda/agg_table) against
# cudf::groupby. NOT part of the Sirius build: configure this directory directly
# (scripts/build_local.sh), like experimental/agg_bench.
cmake_minimum_required(VERSION 3.26)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Build type" FORCE)
endif()

project(agg_table_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# libcudf brings rmm and CCCL (thrust/cub/libcudacxx) transitively.
find_package(cudf REQUIRED CONFIG)
find_package(Threads REQUIRED)

set(AGG_TABLE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../src/cuda/agg_table")

add_executable(
  agg_table_test
  src/agg_table_test.cpp "${AGG_TABLE_DIR}/agg_table.cu" "${AGG_TABLE_DIR}/detail/seg_table.cu"
  "${AGG_TABLE_DIR}/detail/seg_hot.cu" "${AGG_TABLE_DIR}/detail/seg_keys.cu"
  "${AGG_TABLE_DIR}/detail/seg_gen.cu")
target_include_directories(agg_table_test PRIVATE "${AGG_TABLE_DIR}")
target_link_libraries(agg_table_test PRIVATE cudf::cudf Threads::Threads)
set_target_properties(
  agg_table_test
  PROPERTIES CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS OFF
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
             CXX_SCAN_FOR_MODULES OFF)
target_compile_options(
  agg_table_test PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-extended-lambda>)
# Same definitions Sirius uses when compiling against this CCCL/cudf.
target_compile_definitions(
  agg_table_test PRIVATE CCCL_DISABLE_WARPSPEED_SCAN
                         CCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER)
