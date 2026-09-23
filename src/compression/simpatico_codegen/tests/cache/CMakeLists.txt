cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(simpatico_cache_tests LANGUAGES CXX)
  enable_testing()
endif()

include(../../cmake/cache_identity.cmake)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_executable(test_cache_identity test_cache_identity.cpp)
target_link_libraries(test_cache_identity PRIVATE simpatico_cache_identity)
add_test(NAME simpatico_cache_identity COMMAND test_cache_identity)
add_test(
  NAME simpatico_header_manifests
  COMMAND
    "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_header_manifests.py" --cmake
    "${CMAKE_COMMAND}")
set_tests_properties(simpatico_cache_identity simpatico_header_manifests
                     PROPERTIES LABELS cache_host)
