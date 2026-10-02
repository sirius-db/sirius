vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  NVIDIA/cuCollections
  REF
  4b26118c99866221f99f35f4e3bc74afdbe063bc
  SHA512
  8fb63a413dc7591b4b305ac27bdeb58f6f12655ceb73990a26ca3264a501637e64214e7b30a6475acbab1eef45715ad4f7aec393eb803113c8e63f484a0cab00
  HEAD_REF
  dev)

# cuco is header-only. Install just its include tree (headers live under
# include/cuco/...) plus a minimal config that exports an include-dir-only
# cuco::cuco target. We skip cuco's own CMake export, which would add a
# find_dependency(CCCL) chain -- Sirius gets CCCL from cudf.
file(COPY "${SOURCE_PATH}/include/cuco"
     DESTINATION "${CURRENT_PACKAGES_DIR}/include")

file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/cuco-config.cmake"
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/cuco")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
