target_sources(
  sirius_objects
  PRIVATE compressed_disk_representation.cpp
          compressed_representation.cpp
          compressed_scan.cpp
          compression_alloc_stats.cpp
          compression_converters.cpp
          compression_device_pool.cpp
          device_compressed_blob.cpp
          output_compression.cpp
          plan_register.cpp
          simpatico_bridge.cpp
          spill_context.cpp)
