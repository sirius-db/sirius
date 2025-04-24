set(CUDA_SOURCES
        ${CUDA_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_cudf.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/cudf_join.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/cudf_utils.cu
        PARENT_SCOPE
)