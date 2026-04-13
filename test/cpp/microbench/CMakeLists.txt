# =============================================================================
# sirius_gpu_microbench — libcudf GPU microbenchmarks (Google Benchmark). Enable
# with -DSIRIUS_BUILD_GPU_MICROBENCH=ON (release preset sets this ON).
#
# Reuses Google Benchmark from cuCascade (FetchContent in cucascade/benchmark);
# do not FetchContent benchmark again here — target names would collide.
# =============================================================================

if(NOT TARGET benchmark::benchmark_main)
  message(FATAL_ERROR "sirius_gpu_microbench needs Google Benchmark targets. "
                      "Ensure CUCASCADE_BUILD_BENCHMARKS is ON (default).")
endif()

add_executable(sirius_gpu_microbench sirius_gpu_microbench.cpp
                                     microbench_data.cpp)

target_include_directories(
  sirius_gpu_microbench PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

target_link_libraries(
  sirius_gpu_microbench PRIVATE cudf::cudf rmm::rmm benchmark::benchmark_main
                                CUDA::cudart)

get_filename_component(_SIRIUS_MICROBENCH_BIN_DIR
                       "${CMAKE_CURRENT_BINARY_DIR}/.." ABSOLUTE)
set_target_properties(
  sirius_gpu_microbench
  PROPERTIES CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             RUNTIME_OUTPUT_DIRECTORY "${_SIRIUS_MICROBENCH_BIN_DIR}")

if(VCPKG_BUILD)
  set_target_properties(sirius_gpu_microbench PROPERTIES NO_SYSTEM_FROM_IMPORTED
                                                         ON)
  target_include_directories(sirius_gpu_microbench BEFORE PRIVATE ${_VCPKG_INC})
endif()
