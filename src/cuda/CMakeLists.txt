target_sources(
  sirius_objects
  PRIVATE dense_count_join_impl.cu
          device_copy_batch.cpp
          dynamic_filter_replica_transfer.cu
          sirius_dynamic_bloom_filter.cu
          sirius_dynamic_in_list_filter.cu
          sirius_dynamic_small_in_list_filter.cu
          sirius_like_multiliteral.cu
          scan/gpu_decode_alp.cu
          scan/gpu_decode_bitpacking.cu
          scan/gpu_decode_rle.cu
          scan/gpu_decode_strings.cu
          scan/gpu_native_decode.cu
          scan/strings/dict_fsst.cu
          scan/strings/dictionary.cu
          scan/strings/fsst.cu
          scan/strings/uncompressed.cu
          vss/cudf_raft_interop.cu
          vss/brute_force_search.cu
          vss/brute_force_threshold.cu
          vss/cluster_fold.cu
          vss/device_rates.cu
          vss/gemm_topk.cu
          vss/knn_merge.cu
          vss/vector_clustering.cu
          vss/ivf_flat_index.cu)


# The bounded tile kernel sits at 128 registers for two blocks per SM, and separable compilation
# makes it spill (a standalone copy on an A5000: 83 TOPS whole-program, 74 with -rdc). The file
# neither defines nor calls device code across files, so it is its own object library compiled
# whole-program, with sirius_objects' includes, definitions and options; the targets that take
# sirius_objects' objects take these too (cmake/sirius-targets.cmake).
add_library(sirius_whole_program_objects OBJECT vss/bound_gemm.cu)
set_target_properties(
  sirius_whole_program_objects
  PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
             POSITION_INDEPENDENT_CODE ON
             CXX_VISIBILITY_PRESET hidden
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON)
target_include_directories(sirius_whole_program_objects
                           PRIVATE $<TARGET_PROPERTY:sirius_objects,INCLUDE_DIRECTORIES>)
target_compile_definitions(sirius_whole_program_objects
                           PRIVATE $<TARGET_PROPERTY:sirius_objects,COMPILE_DEFINITIONS>)
target_compile_options(sirius_whole_program_objects
                       PRIVATE $<TARGET_PROPERTY:sirius_objects,COMPILE_OPTIONS>)
