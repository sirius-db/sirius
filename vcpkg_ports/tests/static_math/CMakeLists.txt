cmake_minimum_required(VERSION 3.30)
project(static_cuda_math_probe LANGUAGES CXX)

find_package(cusolver CONFIG REQUIRED)

add_library(static_cuda_math_probe SHARED link_probe.cpp)
target_compile_features(static_cuda_math_probe PRIVATE cxx_std_17)
target_link_libraries(static_cuda_math_probe PRIVATE cusolver::cusolver_static)
target_link_options(static_cuda_math_probe PRIVATE "LINKER:--no-undefined")

enable_testing()
add_test(
  NAME static_cuda_math_needed
  COMMAND
    ${CMAKE_COMMAND} "-DREADELF=${CMAKE_READELF}"
    "-DLIBRARY=$<TARGET_FILE:static_cuda_math_probe>" -P
    "${CMAKE_CURRENT_SOURCE_DIR}/check_needed.cmake")
