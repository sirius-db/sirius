cmake_minimum_required(VERSION 3.30.4)
project(sirius_installed_consumer LANGUAGES CXX)
set(SIRIUS_TARGET
    sirius::sirius
    CACHE STRING "Installed Sirius target to link")
if(SIRIUS_TARGET STREQUAL "sirius::sirius_static")
  find_package(sirius REQUIRED CONFIG COMPONENTS static udf)
else()
  find_package(sirius REQUIRED CONFIG COMPONENTS udf)
endif()
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE ${SIRIUS_TARGET} sirius::udf_sdk)

if(SIRIUS_TARGET STREQUAL "sirius::sirius_static")
  target_link_options(consumer PRIVATE -static-libgcc -static-libstdc++)
endif()
