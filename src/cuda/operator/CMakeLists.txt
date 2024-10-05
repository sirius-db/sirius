set(CUDA_SOURCES
        ${CUDA_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/comparison_expression.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/arithmetic_expression.cu
        PARENT_SCOPE
)