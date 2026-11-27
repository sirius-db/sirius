# GPU aggregation table (sirius::agg_table): the segmented hash table and its
# kernels, behind agg_table.hpp. Engine-independent (cuDF / RMM / CUDA only);
# its differential test builds these same sources standalone
# (experimental/agg_table_test).
target_sources(
  sirius_objects PRIVATE agg_table.cu detail/seg_table.cu detail/seg_hot.cu
                         detail/seg_keys.cu detail/seg_gen.cu)
