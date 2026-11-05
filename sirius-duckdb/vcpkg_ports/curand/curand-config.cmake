include(CMakeFindDependencyMacro)
find_dependency(CUDAToolkit)
find_dependency(culibos CONFIG)

if(NOT TARGET curand::curand_static)
  add_library(curand::curand_static STATIC IMPORTED)
  set_target_properties(
    curand::curand_static
    PROPERTIES IMPORTED_LOCATION
               "${CMAKE_CURRENT_LIST_DIR}/../../lib/libcurand_static.a"
               INTERFACE_INCLUDE_DIRECTORIES
               "${CMAKE_CURRENT_LIST_DIR}/../../include"
               INTERFACE_LINK_LIBRARIES "culibos::culibos;CUDA::cudart_static")
endif()
