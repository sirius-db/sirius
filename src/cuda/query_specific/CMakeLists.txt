set(CUDA_SOURCES
        ${CUDA_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/query_specific_filter.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/query_specific_projection.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/test_cudf.cu
        PARENT_SCOPE
)