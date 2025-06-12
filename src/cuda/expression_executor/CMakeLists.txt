set(CUDA_SOURCES
        ${CUDA_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_dispatch_string.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_dispatch_materialize.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_dispatch_select.cu
        PARENT_SCOPE
)