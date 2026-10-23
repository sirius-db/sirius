cmake_minimum_required(VERSION 4.0)
project(sirius_udf_example LANGUAGES CXX)

find_package(sirius CONFIG REQUIRED COMPONENTS udf)
set(SIRIUS_UDF_DUCKDB_LIBRARY
    ""
    CACHE FILEPATH
          "Matching DuckDB shared library used by the embedding validation host"
)
if(NOT EXISTS "${SIRIUS_UDF_DUCKDB_LIBRARY}")
  message(
    FATAL_ERROR "Set SIRIUS_UDF_DUCKDB_LIBRARY to the SDK-matching libduckdb.so"
  )
endif()
add_library(duckdb_udf_host SHARED IMPORTED)
set_target_properties(duckdb_udf_host PROPERTIES IMPORTED_LOCATION
                                                 "${SIRIUS_UDF_DUCKDB_LIBRARY}")

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/udf.cu")
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(sirius_udf_example MODULE udf.cpp udf.cu)
  target_link_libraries(sirius_udf_example PRIVATE CUDA::cudart)
  set_target_properties(sirius_udf_example PROPERTIES CUDA_STANDARD 20)
else()
  add_library(sirius_udf_example MODULE udf.cpp)
endif()
target_link_libraries(sirius_udf_example PRIVATE sirius::udf_sdk)
target_compile_features(sirius_udf_example PRIVATE cxx_std_20)
set_target_properties(sirius_udf_example PROPERTIES PREFIX ""
                                                    CXX_SCAN_FOR_MODULES OFF)

add_executable(udf_host host.cpp)
target_link_libraries(udf_host PRIVATE sirius::udf_sdk duckdb_udf_host
                                       ${CMAKE_DL_LIBS})
target_compile_features(udf_host PRIVATE cxx_std_20)
set_target_properties(udf_host PROPERTIES ENABLE_EXPORTS ON CXX_SCAN_FOR_MODULES
                                                            OFF)
