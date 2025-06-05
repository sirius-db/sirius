set(CUDA_SOURCES
        ${CUDA_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_dispatch_string.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_dispatch_materialize.cu
        PARENT_SCOPE
)