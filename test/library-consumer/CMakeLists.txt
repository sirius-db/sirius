cmake_minimum_required(VERSION 3.30)
project(libsirius_consumer LANGUAGES CXX)

find_package(Sirius CONFIG REQUIRED)

add_executable(libsirius_consumer main.cpp)
target_link_libraries(libsirius_consumer PRIVATE Sirius::sirius)
target_compile_features(libsirius_consumer PRIVATE cxx_std_20)

enable_testing()
add_test(NAME libsirius_consumer COMMAND libsirius_consumer)
