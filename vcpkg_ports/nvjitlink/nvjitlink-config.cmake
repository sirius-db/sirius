include(CMakeFindDependencyMacro)
find_dependency(Threads)

if(NOT TARGET nvjitlink::nvjitlink_static)
  add_library(nvjitlink::nvjitlink_static STATIC IMPORTED)
  set_target_properties(
    nvjitlink::nvjitlink_static
    PROPERTIES IMPORTED_LOCATION
               "${CMAKE_CURRENT_LIST_DIR}/../../lib/libnvJitLink_static.a"
               INTERFACE_INCLUDE_DIRECTORIES
               "${CMAKE_CURRENT_LIST_DIR}/../../include"
               INTERFACE_LINK_LIBRARIES "${CMAKE_DL_LIBS};Threads::Threads")
endif()
