target_sources(
  sirius_objects
  PRIVATE all_gather.cpp
          batch_io.cpp
          exchange_runtime.cpp
          exchange_service.cpp
          fabric.cpp
          messages.cpp
          transport.cpp
          wire.cpp)

if(SIRIUS_WITH_NIXL)
  target_sources(sirius_objects PRIVATE nixl_fabric.cpp)
  # NIXL's include directory also ships asio and other headers: keep it to this
  # one file.
  set_source_files_properties(
    nixl_fabric.cpp TARGET_DIRECTORY sirius_objects
    PROPERTIES INCLUDE_DIRECTORIES ${NIXL_INCLUDE_DIR})
  # exchange_runtime.cpp builds the NIXL exchange only when nixl_fabric.cpp is
  # compiled in.
  set_source_files_properties(
    exchange_runtime.cpp TARGET_DIRECTORY sirius_objects
    PROPERTIES COMPILE_DEFINITIONS SIRIUS_WITH_NIXL)
endif()
