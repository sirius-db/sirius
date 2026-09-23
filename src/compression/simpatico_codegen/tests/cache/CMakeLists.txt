cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(simpatico_cache_tests LANGUAGES CXX)
  enable_testing()
endif()

include(../../cmake/cache_identity.cmake)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(Threads REQUIRED)
add_executable(test_cache_identity test_cache_identity.cpp)
target_link_libraries(test_cache_identity PRIVATE simpatico_cache_identity)
add_test(NAME simpatico_cache_identity COMMAND test_cache_identity)
add_executable(test_disk_cache test_disk_cache.cpp ../../src/jit/disk_cache.cpp)
target_link_libraries(test_disk_cache PRIVATE simpatico_cache_identity
                                              Threads::Threads)
target_link_options(test_disk_cache PRIVATE "-Wl,--wrap=write"
                    "-Wl,--wrap=close" "-Wl,--wrap=rename")
add_test(NAME simpatico_disk_cache COMMAND test_disk_cache)
add_test(
  NAME simpatico_header_manifests
  COMMAND
    "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_header_manifests.py" --cmake
    "${CMAKE_COMMAND}")
set_tests_properties(simpatico_cache_identity simpatico_header_manifests
                     simpatico_disk_cache PROPERTIES LABELS cache_host)

if(TARGET simpatico)
  set(_source "${CMAKE_CURRENT_SOURCE_DIR}/../..")
  set(_generated "${CMAKE_CURRENT_BINARY_DIR}/../../generated")
  add_executable(test_cache_worker cache_worker.cpp)
  target_include_directories(test_cache_worker PRIVATE "${_source}/src")
  target_link_libraries(test_cache_worker PRIVATE simpatico Threads::Threads)
  add_executable(simpatico_cache_benchmark cache_benchmark.cpp)
  target_include_directories(simpatico_cache_benchmark PRIVATE "${_source}/src")
  target_link_libraries(simpatico_cache_benchmark PRIVATE simpatico)

  foreach(variant project cccl)
    set(_fixture "${CMAKE_CURRENT_BINARY_DIR}/fixtures/${variant}")
    add_custom_command(
      OUTPUT "${_fixture}/codegen/jit/embedded_headers.h"
             "${_fixture}/codegen/jit/cccl_embedded_headers.cpp"
      COMMAND
        ${CMAKE_COMMAND} "-DVARIANT=${variant}" "-DSOURCE=${_source}"
        "-DOUT_DIR=${_fixture}" "-DINCLUDE_DIRS_FILE=${EMBED_CCCL_INCLUDE_DIRS}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/generate_variant.cmake"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/generate_variant.cmake"
              "${EMBED_JIT_HEADERS_OUT}" "${EMBED_CCCL_HEADERS_OUT}"
      VERBATIM)
    add_executable(
      test_cache_worker_${variant}
      cache_worker.cpp
      "${_source}/src/jit/kernel_cache.cpp"
      "${_source}/src/jit/nvrtc_compiler.cpp"
      "${_source}/src/jit/compiler_identity.cpp"
      "${_source}/src/jit/disk_cache.cpp"
      "${_fixture}/codegen/jit/embedded_headers.h"
      "${_fixture}/codegen/jit/cccl_embedded_headers.cpp")
    add_dependencies(test_cache_worker_${variant} simpatico_compiler_identity)
    target_include_directories(
      test_cache_worker_${variant} PRIVATE "${_fixture}" "${_generated}"
                                           "${_source}/src")
    target_link_libraries(
      test_cache_worker_${variant}
      PRIVATE simpatico_cache_identity ${SIMPATICO_NVRTC_LIB}
              ${SIMPATICO_CUDART_LIB} CUDA::cuda_driver ${CMAKE_DL_LIBS}
              Threads::Threads)
  endforeach()
  add_custom_target(
    simpatico_cache_gpu_tests
    DEPENDS test_cache_worker test_cache_worker_project test_cache_worker_cccl)
  add_test(
    NAME simpatico_cache_process
    COMMAND
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/test_cache_process.py" --worker
      $<TARGET_FILE:test_cache_worker> --project-worker
      $<TARGET_FILE:test_cache_worker_project> --cccl-worker
      $<TARGET_FILE:test_cache_worker_cccl>)
  set_tests_properties(simpatico_cache_process PROPERTIES LABELS cache_gpu
                                                          TIMEOUT 300)
endif()
