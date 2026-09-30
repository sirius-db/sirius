include(CMakeFindDependencyMacro)
find_dependency(culibos CONFIG)
find_dependency(nvjitlink CONFIG)

if(NOT TARGET cusparse::cusparse_static)
  add_library(cusparse::cusparse_static STATIC IMPORTED)
  set_target_properties(
    cusparse::cusparse_static
    PROPERTIES
      IMPORTED_LOCATION
      "${CMAKE_CURRENT_LIST_DIR}/../../lib/libcusparse_static.a"
      INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/../../include"
      INTERFACE_LINK_LIBRARIES
      "nvjitlink::nvjitlink_static;culibos::culibos;CUDA::cudart_static")
endif()
