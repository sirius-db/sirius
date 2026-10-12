target_sources(
  sirius_objects
  PRIVATE prepared_selection.cpp
          materialize.cpp
          defer_policy.cpp
          defer_directive.cpp
          port_materialize.cpp
          pin_uniqueness.cpp
          multi_source_gather.cu)
