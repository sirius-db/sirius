include(CMakeFindDependencyMacro)
find_dependency(Threads)
find_dependency(CUDAToolkit)

if(TARGET ucx::ucx)
  return()
endif()

get_filename_component(_ucx_prefix "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

foreach(_ucx_name IN ITEMS ucp uct ucs ucm)
  add_library(ucx::${_ucx_name} STATIC IMPORTED)
  set_target_properties(
    ucx::${_ucx_name}
    PROPERTIES IMPORTED_LOCATION "${_ucx_prefix}/lib/lib${_ucx_name}.a"
               INTERFACE_INCLUDE_DIRECTORIES "${_ucx_prefix}/include")
  if(EXISTS "${_ucx_prefix}/debug/lib/lib${_ucx_name}.a")
    set_target_properties(
      ucx::${_ucx_name} PROPERTIES IMPORTED_LOCATION_DEBUG
                                   "${_ucx_prefix}/debug/lib/lib${_ucx_name}.a")
  endif()
endforeach()

foreach(_ucx_name IN ITEMS uct_cma uct_cuda ucm_cuda)
  add_library(ucx::${_ucx_name} STATIC IMPORTED)
  set_target_properties(
    ucx::${_ucx_name} PROPERTIES IMPORTED_LOCATION
                                 "${_ucx_prefix}/lib/ucx/lib${_ucx_name}.a")
  if(EXISTS "${_ucx_prefix}/debug/lib/ucx/lib${_ucx_name}.a")
    set_target_properties(
      ucx::${_ucx_name}
      PROPERTIES IMPORTED_LOCATION_DEBUG
                 "${_ucx_prefix}/debug/lib/ucx/lib${_ucx_name}.a")
  endif()
endforeach()

# Transport registration uses constructors in otherwise unreferenced objects.
add_library(ucx::ucx INTERFACE IMPORTED)
set_target_properties(
  ucx::ucx
  PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_ucx_prefix}/include"
    INTERFACE_LINK_LIBRARIES
    "-Wl,--start-group;-Wl,--whole-archive;ucx::uct_cma;ucx::uct_cuda;ucx::ucm_cuda;-Wl,--no-whole-archive;ucx::ucp;ucx::uct;ucx::ucs;ucx::ucm;-Wl,--end-group;CUDA::cuda_driver;CUDA::cudart_static;CUDA::nvml;Threads::Threads;${CMAKE_DL_LIBS};rt;m"
    INTERFACE_LINK_OPTIONS
    "LINKER:--undefined=ucp_global_init;LINKER:--undefined=uct_init;LINKER:--undefined=ucs_init"
)

set(UCX_LIBRARIES ucx::ucx)
set(UCX_INCLUDE_DIRS "${_ucx_prefix}/include")
unset(_ucx_name)
unset(_ucx_prefix)
