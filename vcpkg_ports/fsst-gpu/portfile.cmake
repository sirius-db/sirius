# FSST-GPU (CompactionV5T) string codec. Source-only port: installs the
# library's headers and .cu sources preserving the repo layout (no build here —
# simpatico compiles them itself so CUDA archs, flags and GTSST_SPLIT_COUNT stay
# uniform with the rest of the build; see
# src/compression/simpatico_codegen/CMakeLists.txt). The config exports
# FSST_GPU_ROOT pointing at the installed tree.
vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  joosthooz/fsst-gpu
  REF
  c86c0426a7fd2e3568ffcf6e6122d885867aaad4
  SHA512
  48dc8bd75814063666fa1ba6a18de1d465282c3053a45af28d68e12adea1898a9e809db5f176d49662729d807bb689c35bd1f0806779ac3ca59a76708e1de818
  HEAD_REF
  gpu-decompressor)

# Library subset only: skip the bench harness and the gtsst executable's main.
file(COPY "${SOURCE_PATH}/include"
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/fsst-gpu")
file(COPY "${SOURCE_PATH}/src"
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/fsst-gpu")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/share/fsst-gpu/src/bench")
file(REMOVE "${CURRENT_PACKAGES_DIR}/share/fsst-gpu/src/main.cu")

file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/fsst-gpu-config.cmake"
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/fsst-gpu")

set(VCPKG_POLICY_EMPTY_INCLUDE_FOLDER enabled)

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
