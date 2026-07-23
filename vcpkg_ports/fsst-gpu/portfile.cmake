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
  1958667a5709d15b4b50c0592fb7c05fc6cbc572
  SHA512
  3e7e6e946520fe6d9ccf776e93a5de63e11a108a4636544329dbcbef4159fc18782abfa9f1089da68d36e995716c3b519a13d62da913e0f9112e6fa8edf3f864
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
