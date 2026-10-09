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
add_executable(consumer main.cpp context.cpp)
target_link_libraries(consumer PRIVATE ${SIRIUS_TARGET})
target_compile_features(consumer PRIVATE cxx_std_23)

if(SIRIUS_TARGET STREQUAL "sirius::sirius_static")
  target_link_options(consumer PRIVATE -static-libgcc -static-libstdc++)
endif()

add_executable(c_consumer config.c)
set_target_properties(c_consumer PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED
                                                          ON)
target_link_libraries(c_consumer PRIVATE ${SIRIUS_TARGET})
add_executable(allocation_consumer allocation.cpp)
target_link_libraries(allocation_consumer PRIVATE ${SIRIUS_TARGET})
target_compile_features(allocation_consumer PRIVATE cxx_std_23)
if(SIRIUS_TARGET STREQUAL "sirius::sirius_static")
  set_target_properties(c_consumer PROPERTIES LINKER_LANGUAGE CXX)
  target_link_options(c_consumer PRIVATE -static-libgcc -static-libstdc++)
  target_link_options(allocation_consumer PRIVATE -static-libgcc
                      -static-libstdc++)
endif()
enable_testing()
add_test(NAME c_configuration COMMAND c_consumer)
add_test(NAME allocation_failure COMMAND allocation_consumer)

add_executable(cpp_configuration_consumer config_cpp.cpp)
target_compile_features(cpp_configuration_consumer PRIVATE cxx_std_23)
target_link_libraries(cpp_configuration_consumer PRIVATE ${SIRIUS_TARGET})
if(SIRIUS_TARGET STREQUAL "sirius::sirius_static")
  target_link_options(cpp_configuration_consumer PRIVATE -static-libgcc
                      -static-libstdc++)
endif()
add_test(NAME cpp_configuration COMMAND cpp_configuration_consumer)
