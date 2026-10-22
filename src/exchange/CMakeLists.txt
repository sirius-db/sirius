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
  # The NIXL exchange is built only with the fabric it needs.
  set_source_files_properties(
    exchange_runtime.cpp TARGET_DIRECTORY sirius_objects
    PROPERTIES COMPILE_DEFINITIONS SIRIUS_WITH_NIXL)
endif()
