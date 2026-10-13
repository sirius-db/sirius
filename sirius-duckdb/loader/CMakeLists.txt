cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)
project(sirius-loader LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "The Sirius CUDA loader supports Linux")
endif()

build_static_extension(sirius src/sirius_loader.cpp)
build_loadable_extension(sirius " " src/sirius_loader.cpp)
foreach(target IN ITEMS sirius_extension sirius_loadable_extension)
  target_link_libraries(${target} ${CMAKE_DL_LIBS})
  target_compile_features(${target} PRIVATE cxx_std_17)
endforeach()

install(
  TARGETS sirius_extension
  EXPORT "${DUCKDB_EXPORT_SET}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")

option(SIRIUS_LOADER_TESTS "Build CPU-only loader test fixtures" OFF)
if(SIRIUS_LOADER_TESTS)
  add_subdirectory(test)
endif()
