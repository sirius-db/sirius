cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(simpatico_cache_tests LANGUAGES CXX)
  enable_testing()
endif()

include(../../cmake/cache_identity.cmake)
include(../../cmake/jit_header_manifest.cmake)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_executable(test_cache_identity test_cache_identity.cpp)
target_link_libraries(test_cache_identity PRIVATE simpatico_cache_identity)
add_test(NAME simpatico_cache_identity COMMAND test_cache_identity)
add_test(
  NAME simpatico_header_manifests
  COMMAND
    "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_header_manifests.py" --cmake
    "${CMAKE_COMMAND}" --xxhsum "${XXHSUM_EXECUTABLE}" --identity-tool
    "$<TARGET_FILE:test_cache_identity>")
set_tests_properties(simpatico_cache_identity simpatico_header_manifests
                     PROPERTIES LABELS cache_host)

if(TARGET simpatico)
  set(_source "${CMAKE_CURRENT_SOURCE_DIR}/../..")
  set(_generated "${CMAKE_CURRENT_BINARY_DIR}/../../generated")
  add_executable(test_cache_worker cache_worker.cpp)
  target_include_directories(test_cache_worker PRIVATE "${_source}/src")
  target_link_libraries(test_cache_worker PRIVATE simpatico ${CMAKE_DL_LIBS})
  set(_version_worker_args)
  set(_version_worker_target)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # The compiler replacement test uses LD_LIBRARY_PATH. Emit RUNPATH, not
    # RPATH, so it can select the relocated NVRTC library.
    target_link_options(test_cache_worker PRIVATE "LINKER:--enable-new-dtags")
    # Exercise the production version query, including failures, without adding
    # overrides or test hooks to the production cache/compiler API.
    add_executable(test_cache_worker_version cache_worker.cpp
                                             nvrtc_version_stub.cpp)
    target_include_directories(test_cache_worker_version
                               PRIVATE "${_source}/src")
    target_link_libraries(test_cache_worker_version PRIVATE simpatico
                                                            ${CMAKE_DL_LIBS})
    target_link_options(test_cache_worker_version PRIVATE
                        "LINKER:--wrap=nvrtcVersion")
    set(_version_worker_target test_cache_worker_version)
    set(_version_worker_args --version-worker
                             $<TARGET_FILE:test_cache_worker_version>)
    add_executable(test_cache_environment test_cache_environment.cpp)
    add_dependencies(test_cache_environment simpatico_embedded_headers)
    target_include_directories(test_cache_environment PRIVATE "${_source}/src"
                                                              "${_generated}")
    target_link_libraries(test_cache_environment PRIVATE simpatico)
    target_link_options(test_cache_environment PRIVATE
                        "LINKER:--wrap=nvrtcVersion")
    add_test(NAME simpatico_cache_environment COMMAND test_cache_environment)
    # Runs without a GPU, but requires the full NVRTC-enabled build.
    set_tests_properties(simpatico_cache_environment PROPERTIES LABELS
                                                                cache_host)
    list(APPEND _version_worker_target test_cache_environment)
  endif()
  foreach(variant project cccl)
    set(_fixture "${CMAKE_CURRENT_BINARY_DIR}/fixtures/${variant}")
    add_custom_command(
      OUTPUT "${_fixture}/codegen/jit/embedded_headers.h"
             "${_fixture}/codegen/jit/cccl_embedded_headers.cpp"
      COMMAND
        ${CMAKE_COMMAND} "-DVARIANT=${variant}" "-DSOURCE=${_source}"
        "-DOUT_DIR=${_fixture}" "-DINCLUDE_DIRS_FILE=${EMBED_CCCL_INCLUDE_DIRS}"
        "-DXXHSUM_EXECUTABLE=${XXHSUM_EXECUTABLE}" -P
        "${CMAKE_CURRENT_SOURCE_DIR}/generate_variant.cmake"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/generate_variant.cmake"
              "${XXHSUM_EXECUTABLE}" "${EMBED_JIT_HEADERS_OUT}"
              "${EMBED_CCCL_HEADERS_OUT}"
      VERBATIM)
    add_executable(
      test_cache_worker_${variant}
      cache_worker.cpp
      "${_source}/src/jit/kernel_cache.cpp"
      "${_source}/src/jit/nvrtc_compiler.cpp"
      "${_fixture}/codegen/jit/embedded_headers.h"
      "${_fixture}/codegen/jit/cccl_embedded_headers.cpp")
    target_include_directories(
      test_cache_worker_${variant} PRIVATE "${_fixture}" "${_generated}"
                                           "${_source}/src")
    target_link_libraries(
      test_cache_worker_${variant}
      PRIVATE simpatico_cache_identity ${SIMPATICO_NVRTC_LIB}
              ${SIMPATICO_CUDART_LIB} CUDA::cuda_driver ${CMAKE_DL_LIBS})
  endforeach()
  add_custom_target(
    simpatico_cache_gpu_tests
    DEPENDS test_cache_worker test_cache_worker_project test_cache_worker_cccl
            ${_version_worker_target})
  add_test(
    NAME simpatico_cache_process
    COMMAND
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/test_cache_process.py" --worker
      $<TARGET_FILE:test_cache_worker> --project-worker
      $<TARGET_FILE:test_cache_worker_project> --cccl-worker
      $<TARGET_FILE:test_cache_worker_cccl> ${_version_worker_args})
  set_tests_properties(simpatico_cache_process PROPERTIES LABELS cache_gpu
                                                          TIMEOUT 300)
endif()
