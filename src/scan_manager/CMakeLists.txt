target_sources(
  sirius_objects
  PRIVATE late_mat_resolver.cpp
          load_balancing_scan_batch_coalescer.cpp
          insert_delta_job.cpp
          mvcc_mask_job.cpp
          pinned_chunk_stats.cpp
          round_robin_strategy.cpp
          memory_prefetcher.cpp
          sirius_scan_manager.cpp
          split_connector.cpp
          split_provider.cpp)
