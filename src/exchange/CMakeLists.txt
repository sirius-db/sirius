target_sources(
  sirius_objects
  PRIVATE all_gather.cpp
          batch_io.cpp
          distributed_session.cpp
          exchange_runtime.cpp
          exchange_service.cpp
          fabric.cpp
          messages.cpp
          nixl_fabric.cpp
          rank_control.cpp
          transport.cpp
          wire.cpp)
