target_sources(
  sirius_objects PRIVATE batch_io.cpp exchange_service.cpp fabric.cpp
                         messages.cpp transport.cpp wire.cpp)

if(SIRIUS_WITH_NIXL)
  target_sources(sirius_objects PRIVATE nixl_fabric.cpp)
  # NIXL's include directory also ships asio and other headers: keep it to this
  # one file.
  set_source_files_properties(
    nixl_fabric.cpp TARGET_DIRECTORY sirius_objects
    PROPERTIES INCLUDE_DIRECTORIES ${NIXL_INCLUDE_DIR})
endif()
