target_sources(
  sirius_objects
  PRIVATE preparation_coordinator.cpp
          preparation.cpp
          preparation_ledger.cpp
          late_mat_resolver.cpp
          load_balancing_scan_batch_coalescer.cpp
          insert_delta_job.cpp
          mvcc_mask_job.cpp
          prefetching_scheduler.cpp
          pinned_chunk_stats.cpp
          readahead_scan_manager.cpp
          round_robin_strategy.cpp
          memory_prefetcher.cpp
          sirius_scan_manager.cpp
          split_connector.cpp
          split_provider.cpp)
