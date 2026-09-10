vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  NVIDIA/cuCollections
  REF
  0883368d39296f3bef3a058033141bcc642c5c54
  SHA512
  be8040f9ad46a2f8cc9d9c53a2bdbd0c9ff1e83c3f93788ae5508bc9f4610028e2d326e6b357a41a65a93064960db8d18963a540f25609fe2f254eb231484aee
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
