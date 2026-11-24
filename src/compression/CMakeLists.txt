target_sources(
  sirius_objects
  PRIVATE compressed_disk_representation.cpp
          compressed_payload_sources.cpp
          compressed_representation.cpp
          compressed_row_range_readers.cpp
          compressed_scan.cpp
          compression_converters.cpp
          plan_utils.cpp
          spill_compression.cpp)
