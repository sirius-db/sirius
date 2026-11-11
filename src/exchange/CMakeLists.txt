target_sources(
  sirius_objects
  PRIVATE all_gather.cpp
          batch_io.cpp
          exchange_runtime.cpp
          exchange_service.cpp
          fabric.cpp
          messages.cpp
          nixl_fabric.cpp
          transport.cpp
          wire.cpp)
