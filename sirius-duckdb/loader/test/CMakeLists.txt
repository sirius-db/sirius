add_library(sirius_test_cuda_driver SHARED driver.cpp)
set_target_properties(
  sirius_test_cuda_driver
  PROPERTIES OUTPUT_NAME cuda
             SUFFIX ".so.1"
             LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/driver")
add_library(sirius_test_missing_driver_symbol SHARED driver.cpp)
target_compile_definitions(sirius_test_missing_driver_symbol
                           PRIVATE MISSING_VERSION_SYMBOL)
set_target_properties(
  sirius_test_missing_driver_symbol
  PROPERTIES OUTPUT_NAME cuda
             SUFFIX ".so.1"
             LIBRARY_OUTPUT_DIRECTORY
             "${CMAKE_CURRENT_BINARY_DIR}/missing-symbol")
add_library(sirius_test_hide_driver SHARED hide_driver.cpp)
target_link_libraries(sirius_test_hide_driver PRIVATE ${CMAKE_DL_LIBS})

foreach(version IN ITEMS 12 13)
  set(name sirius_cuda${version})
  build_loadable_extension(${name} " " backend.cpp)
  target_compile_definitions(
    ${name}_loadable_extension PRIVATE SIRIUS_TEST_BACKEND=${name}
                                       SIRIUS_TEST_CUDA_VERSION=${version})
endforeach()

add_custom_target(
  sirius_loader_test_fixtures
  DEPENDS sirius_test_cuda_driver sirius_test_missing_driver_symbol
          sirius_test_hide_driver sirius_cuda12_loadable_extension
          sirius_cuda13_loadable_extension)
