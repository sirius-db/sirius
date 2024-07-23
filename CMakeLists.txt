cmake_minimum_required(VERSION 3.20)

# Set extension name here
set(TARGET_NAME komodo)

project(${TARGET_NAME} LANGUAGES CXX CUDA)
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    enable_language(CUDA)
endif()
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lcuda -lcudart -lcupti -lcurand -lcudadevrt --expt-extended-lambda --expt-relaxed-constexpr -rdc=true")
# SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -lcudart -lcupti -lcuda")

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

include_directories(src/include)
include_directories(duckdb/extension_external/substrait/src/include)
include_directories(${TARGET_NAME} duckdb/extension_external/substrait/third_party/substrait)
include_directories(${TARGET_NAME} duckdb/extension_external/substrait/third_party)

set(EXTENSION_SOURCES src/komodo_extension.cpp)
set(CUDA_SOURCES src/cuda/komodo_extension_cuda.cu src/cuda/communication.cu)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Manually specify CUDA include and library directories
# Include directories
include_directories(${CUDA_INCLUDE_DIRS})

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

target_link_libraries(${EXTENSION_NAME} substrait_extension)
target_link_libraries(${LOADABLE_EXTENSION_NAME} substrait_extension)
target_link_libraries(${EXTENSION_NAME} ${CUDA_LIBRARIES} ${CUDA_cudart_LIBRARY} ${CUDA_cudadevrt_LIBRARY})
target_link_libraries(${LOADABLE_EXTENSION_NAME} ${CUDA_LIBRARIES} ${CUDA_cudart_LIBRARY} ${CUDA_cudadevrt_LIBRARY})

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
