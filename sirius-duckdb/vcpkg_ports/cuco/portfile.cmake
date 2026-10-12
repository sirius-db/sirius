vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  NVIDIA/cuCollections
  REF
  8ed532dc0ba3ba664d1d41027269908ddcde04c8
  SHA512
  dade9af000bb490b053288312ab60e24ac1e1ba07bd4bfdde84beca98d19e20538d5b3dde285bc5feb7c8d208791aff018b8428d2b3e493721c352b6b83f127b
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
