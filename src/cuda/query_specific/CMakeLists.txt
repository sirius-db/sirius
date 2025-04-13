set(CUDA_SOURCES
        ${CUDA_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/query_specific_filter.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/query_specific_projection.cu
        PARENT_SCOPE
)