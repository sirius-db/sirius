cmake_minimum_required(VERSION 3.30.4)

# Enable compile_commands.json generation for clang lsp
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Check for USE_CUDF environment variable
if(DEFINED ENV{USE_CUDF})
    message(STATUS "USE_CUDF environment variable is set")
    set(USE_CUDF ON)
    add_compile_definitions(ENABLE_CUDF)
else()
    message(STATUS "USE_CUDF environment variable is not set")
    set(USE_CUDF OFF)
endif()

# Set extension name here
set(TARGET_NAME sirius)

project(${TARGET_NAME} LANGUAGES CXX CUDA)
if (USE_CUDF)
  find_package(cudf REQUIRED CONFIG)
  if(DEFINED ENV{LIBCUDF_ENV_PREFIX})
    set(CONDA_PREFIX "$ENV{LIBCUDF_ENV_PREFIX}")
  else()
    message(FATAL_ERROR "LIBCUDF_ENV_PREFIX environment variable is not set.")
  endif()
else()
  # find_package(CUDA REQUIRED)
  enable_language(CUDA)
endif()

# Set CUDA architecture and flags
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr -rdc=true")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -lcudart -lcupti -lcuda")
set(ENV{SKIP_SUBSTRAIT_C_TESTS} ON)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

include_directories(
    src/include
    src/include/operator
)
include_directories(duckdb/extension_external/substrait/src/include)
include_directories(${TARGET_NAME} duckdb/extension_external/substrait/third_party/substrait)
include_directories(${TARGET_NAME} duckdb/extension_external/substrait/third_party)

set(EXTENSION_SOURCES 
  src/sirius_extension.cpp
  src/gpu_context.cpp
  src/gpu_executor.cpp
  src/gpu_physical_operator.cpp
  src/gpu_pipeline.cpp
  src/gpu_meta_pipeline.cpp
  src/gpu_physical_plan_generator.cpp
  src/gpu_buffer_manager.cpp
  src/gpu_columns.cpp
  src/gpu_expression_executor.cpp
)
add_subdirectory(src/operator)
add_subdirectory(src/plan)
add_subdirectory(src/query_specific)

set(CUDA_SOURCES 
  src/cuda/sirius_extension_cuda.cu 
  src/cuda/communication.cu 
  src/cuda/allocator.cu
  src/cuda/print_column.cu
  src/cuda/utils.cu
)
add_subdirectory(src/cuda/operator)
add_subdirectory(src/cuda/query_specific)
if (USE_CUDF)
  add_subdirectory(src/cuda/cudf)
  add_subdirectory(src/cuda/expression_executor)
  add_subdirectory(src/expression_executor)
endif()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Manually specify CUDA include and library directories
# Include directories

if (USE_CUDF)
  include_directories(${CUDA_INCLUDE_DIRS})
  include_directories(src/include/cudf)
  include_directories(${CONDA_PREFIX}/include)
  include_directories(${CONDA_PREFIX}/include/cudf)
  include_directories(${CONDA_PREFIX}/include/rmm)
else()
  include_directories(/usr/local/cuda/include)
endif()

# add_library(${EXTENSION_NAME} STATIC ${EXTENSION_SOURCES} ${CUDA_SOURCES})

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES} ${CUDA_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES} ${CUDA_SOURCES})

add_dependencies(${EXTENSION_NAME} duckdb_static)
add_dependencies(${LOADABLE_EXTENSION_NAME} duckdb_static)

set_target_properties(${EXTENSION_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(${LOADABLE_EXTENSION_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(${EXTENSION_NAME} PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(${EXTENSION_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${LOADABLE_EXTENSION_NAME} PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(${LOADABLE_EXTENSION_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if (USE_CUDF)
  target_link_libraries(${EXTENSION_NAME}
    substrait_extension
    cudf::cudf 
    rmm::rmm
  )
  target_link_libraries(${LOADABLE_EXTENSION_NAME} 
    substrait_extension
    cudf::cudf 
    rmm::rmm
  )
else()
  find_library(CUDA_RUNTIME_LIBRARY cudart HINTS /usr/local/cuda/lib64)
  find_library(CUDA_CUPTI_LIBRARY cupti HINTS /usr/local/cuda/lib64)
  find_library(CUDA_CURAND_LIBRARY curand HINTS /usr/local/cuda/lib64)
  find_library(CUDA_CUDA_LIBRARY cuda HINTS /usr/local/cuda/lib64)
  find_library(CUDA_DEVRT_LIBRARY cudadevrt HINTS /usr/local/cuda/lib64)

  target_link_libraries(${EXTENSION_NAME} 
    substrait_extension
    ${CUDA_RUNTIME_LIBRARY}
    ${CUDA_CUPTI_LIBRARY}
    ${CUDA_CURAND_LIBRARY}
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_DEVRT_LIBRARY}
  )

  target_link_libraries(${LOADABLE_EXTENSION_NAME} 
    substrait_extension
    ${CUDA_RUNTIME_LIBRARY}
    ${CUDA_CUPTI_LIBRARY}
    ${CUDA_CURAND_LIBRARY}
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_DEVRT_LIBRARY}
  )
endif()

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")