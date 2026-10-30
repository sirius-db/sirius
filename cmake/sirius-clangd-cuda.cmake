# Clang looks for include/cuda.h and nvvm/libdevice in one toolkit root. In the
# pixi layout that root is targets/<triple> (nvcc.profile, which clangd does not
# read, also adds the prefix include that holds libcudf). Publish a stable
# symlink for .clangd, whose flags are relative to build/release.
set(_sirius_clangd_cuda "${CMAKE_SOURCE_DIR}/tools/clangd/cuda-toolkit")
file(GLOB _sirius_cuda_headers
     "${CMAKE_SOURCE_DIR}/.pixi/envs/default/targets/*/include/cuda.h")
if(_sirius_cuda_headers)
  list(GET _sirius_cuda_headers 0 _sirius_cuda_header)
  cmake_path(GET _sirius_cuda_header PARENT_PATH _sirius_cuda_include)
  cmake_path(GET _sirius_cuda_include PARENT_PATH _sirius_cuda_target)
  if(EXISTS "${_sirius_clangd_cuda}")
    file(REMOVE "${_sirius_clangd_cuda}")
  endif()
  file(CREATE_LINK "${_sirius_cuda_target}" "${_sirius_clangd_cuda}" SYMBOLIC)
endif()
