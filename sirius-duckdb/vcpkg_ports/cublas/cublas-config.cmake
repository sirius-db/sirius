include(CMakeFindDependencyMacro)
find_dependency(CUDAToolkit)
find_dependency(culibos CONFIG)

foreach(library cublasLt_static cublas_static)
  if(NOT TARGET cublas::${library})
    add_library(cublas::${library} STATIC IMPORTED)
    set_target_properties(
      cublas::${library}
      PROPERTIES IMPORTED_LOCATION
                 "${CMAKE_CURRENT_LIST_DIR}/../../lib/lib${library}.a"
                 INTERFACE_INCLUDE_DIRECTORIES
                 "${CMAKE_CURRENT_LIST_DIR}/../../include")
  endif()
endforeach()

set_target_properties(
  cublas::cublasLt_static PROPERTIES INTERFACE_LINK_LIBRARIES
                                     "culibos::culibos;CUDA::cudart_static")
set_target_properties(
  cublas::cublas_static
  PROPERTIES INTERFACE_LINK_LIBRARIES
             "cublas::cublasLt_static;culibos::culibos;CUDA::cudart_static")
