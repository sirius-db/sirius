cmake_minimum_required(VERSION 3.30.4)
project(sirius_installed_consumer LANGUAGES C CXX)
set(SIRIUS_TARGET
    sirius::sirius
    CACHE STRING "Installed Sirius target to link")
if(SIRIUS_TARGET STREQUAL "sirius::sirius_static")
  find_package(sirius REQUIRED CONFIG COMPONENTS static)
else()
  find_package(sirius REQUIRED CONFIG)
endif()
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE ${SIRIUS_TARGET})

if(SIRIUS_TARGET STREQUAL "sirius::sirius_static")
  target_link_options(consumer PRIVATE -static-libgcc -static-libstdc++)
endif()

add_executable(c_consumer config.c)
set_target_properties(c_consumer PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED
                                                          ON)
target_link_libraries(c_consumer PRIVATE ${SIRIUS_TARGET})
if(SIRIUS_TARGET STREQUAL "sirius::sirius_static")
  set_target_properties(c_consumer PROPERTIES LINKER_LANGUAGE CXX)
  target_link_options(c_consumer PRIVATE -static-libgcc -static-libstdc++)
endif()
