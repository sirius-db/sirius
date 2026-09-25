include(CMakeFindDependencyMacro)
find_dependency(cublas CONFIG)
find_dependency(cusparse CONFIG)

foreach(library cusolver_lapack_static cusolver_metis_static cusolver_static)
  if(NOT TARGET cusolver::${library})
    add_library(cusolver::${library} STATIC IMPORTED)
    set_target_properties(
      cusolver::${library}
      PROPERTIES IMPORTED_LOCATION
                 "${CMAKE_CURRENT_LIST_DIR}/../../lib/lib${library}.a"
                 INTERFACE_INCLUDE_DIRECTORIES
                 "${CMAKE_CURRENT_LIST_DIR}/../../include")
  endif()
endforeach()

set_target_properties(
  cusolver::cusolver_static
  PROPERTIES
    INTERFACE_LINK_LIBRARIES
    "cusolver::cusolver_lapack_static;cusolver::cusolver_metis_static;cublas::cublas_static;cusparse::cusparse_static;CUDA::cudart_static"
)
