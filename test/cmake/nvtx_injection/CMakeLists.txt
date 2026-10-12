cmake_minimum_required(VERSION 3.30.4)
project(sirius_nvtx_injection_test LANGUAGES CXX)
include(CTest)

set(SIRIUS_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../..")
add_library(injection_objects OBJECT
            "${SIRIUS_SOURCE_DIR}/src/telemetry/nvtx_injection.cpp" fixture.cpp)
target_include_directories(injection_objects PUBLIC "${SIRIUS_SOURCE_DIR}/src")
set_target_properties(
  injection_objects
  PROPERTIES CXX_STANDARD 20
             POSITION_INDEPENDENT_CODE ON
             CXX_VISIBILITY_PRESET hidden)

add_library(injection_shared SHARED $<TARGET_OBJECTS:injection_objects>)
target_link_libraries(injection_shared PRIVATE ${CMAKE_DL_LIBS})
# Discovery must not rely on a shared-library suffix.
set_target_properties(injection_shared PROPERTIES SUFFIX ".injector")

foreach(mode pie nonpie)
  add_executable(injection_${mode} driver.cpp
                                   $<TARGET_OBJECTS:injection_objects>)
  target_include_directories(injection_${mode}
                             PRIVATE "${SIRIUS_SOURCE_DIR}/src")
  target_link_libraries(injection_${mode} PRIVATE ${CMAKE_DL_LIBS})
  target_link_options(
    injection_${mode} PRIVATE
    "LINKER:--export-dynamic-symbol=InitializeInjectionNvtx2"
    "LINKER:--export-dynamic-symbol=dlopen")
  set_target_properties(injection_${mode} PROPERTIES CXX_STANDARD 20)
  add_test(NAME nvtx_${mode} COMMAND injection_${mode})
endforeach()
target_compile_options(injection_pie PRIVATE -fPIE)
target_link_options(injection_pie PRIVATE -pie)
target_compile_options(injection_nonpie PRIVATE -fno-pie)
target_link_options(injection_nonpie PRIVATE -no-pie)

add_executable(injection_shared_driver driver.cpp)
target_include_directories(injection_shared_driver
                           PRIVATE "${SIRIUS_SOURCE_DIR}/src")
target_link_libraries(injection_shared_driver PRIVATE injection_shared
                                                      ${CMAKE_DL_LIBS})
set_target_properties(injection_shared_driver PROPERTIES CXX_STANDARD 20)
add_test(NAME nvtx_shared COMMAND injection_shared_driver
                                  $<TARGET_FILE:injection_shared>)
