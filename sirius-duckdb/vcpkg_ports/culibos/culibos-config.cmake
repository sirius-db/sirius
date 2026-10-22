include(CMakeFindDependencyMacro)
find_dependency(Threads)

if(NOT TARGET culibos::culibos)
  add_library(culibos::culibos STATIC IMPORTED)
  set_target_properties(
    culibos::culibos
    PROPERTIES IMPORTED_LOCATION
               "${CMAKE_CURRENT_LIST_DIR}/../../lib/libculibos.a"
               INTERFACE_LINK_LIBRARIES "Threads::Threads;${CMAKE_DL_LIBS}")
endif()
