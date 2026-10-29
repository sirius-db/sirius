target_sources(
  sirius_objects
  PRIVATE binding_bridge.cpp
          dv_visibility.cpp
          field_mapping.cpp
          inline_batch_producer.cpp
          inline_varchar.cpp
          parquet_unit_ingestible.cpp
          snapshot_visibility_test_scope.cpp
          scan_ingestible.cpp
          source_registration.cpp
          unit_descriptor.cpp)
