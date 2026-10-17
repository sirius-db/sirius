include_guard(GLOBAL)

# Shared by the packaged dependency and Sirius's development source build.
function(nixl_import_targets prefix ucx_target cuda_runtime_target)
  if(TARGET nixl::nixl)
    return()
  endif()

  set(_archives nixl plugin_UCX nixl_build stream serdes nixl_common)
  set(_targets)
  foreach(_archive IN LISTS _archives)
    set(_target "nixl::${_archive}_archive")
    add_library("${_target}" STATIC IMPORTED GLOBAL)
    set_target_properties(
      "${_target}" PROPERTIES IMPORTED_LOCATION
                              "${prefix}/lib/lib${_archive}.a")
    list(APPEND _targets "${_target}")
  endforeach()
  list(JOIN _targets "," _archive_group)

  add_library(nixl::nixl INTERFACE IMPORTED GLOBAL)
  set_target_properties(
    nixl::nixl
    PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${prefix}/include"
      INTERFACE_COMPILE_FEATURES cxx_std_20
      INTERFACE_LINK_LIBRARIES
      "$<LINK_GROUP:RESCAN,${_archive_group}>;${ucx_target};absl::log;absl::check;absl::log_initialize;absl::flat_hash_map;absl::status;absl::statusor;absl::strings;absl::str_format;absl::synchronization;Threads::Threads;CUDA::cuda_driver;${cuda_runtime_target};${CMAKE_DL_LIBS};rt"
  )
endfunction()
