# Header-only cuco. Sirius consumes only the include dir; CCCL (libcudacxx/CUB/
# Thrust) comes from cudf, so this target intentionally has no CCCL dependency.
if(NOT TARGET cuco::cuco)
  add_library(cuco::cuco INTERFACE IMPORTED)
  set_target_properties(
    cuco::cuco PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                          "${CMAKE_CURRENT_LIST_DIR}/../../include")
endif()
