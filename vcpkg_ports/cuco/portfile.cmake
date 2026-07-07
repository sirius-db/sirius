vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  NVIDIA/cuCollections
  REF
  f517bbb1277753b1852dfd388993383e401eaa38
  SHA512
  53f6185db57eba7391fd9b93b342d9da3c4cdf906dc47fad02cc1eb34867bb11ce798f0e94fee71a3cfb6423e6cad907ebe21e46997b4f4ef2b6e1e3b0a45d82
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
