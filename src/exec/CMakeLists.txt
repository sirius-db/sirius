target_sources(
  sirius_objects
  PRIVATE admission_control.cpp
          arrow_result_stream.cpp
          batch_stream.cpp
          exchange_ring.cpp
          exchange_ring_wire.cpp
          stream_session.cpp
          stream_bind_catalog.cpp
          stream_plan_bindings.cpp
          streaming_fragment.cpp
          thread_util.cpp)
