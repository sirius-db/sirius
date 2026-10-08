cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(simpatico_cache_tests LANGUAGES CXX)
  enable_testing()
endif()

include(../../cmake/cache_identity.cmake)
add_executable(test_cache_identity test_cache_identity.cpp)
target_link_libraries(test_cache_identity PRIVATE simpatico_cache_identity)
add_test(NAME simpatico_cache_identity COMMAND test_cache_identity)
add_test(
  NAME simpatico_jit_epoch
  COMMAND
    "${CMAKE_COMMAND}"
    "-DJIT_EPOCH_SCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/jit_epoch.cmake"
    "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/jit_epoch_test" -P
    "${CMAKE_CURRENT_SOURCE_DIR}/test_jit_epoch.cmake")
set_tests_properties(simpatico_cache_identity simpatico_jit_epoch
                     PROPERTIES LABELS cache_host)
