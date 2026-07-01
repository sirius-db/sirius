vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  NVIDIA/NVTX
  REF
  v${VERSION}
  SHA512
  b65eb392f2fcf4a96fef84932cf61ceb85ae8a424e1128e77adde1e0452291d5e3eacd8bc0354f8878551ede0070dc0881406a4ffac1f3b13d2f7e56f4e0c41a
  HEAD_REF
  release-v3)

# nvtx3 is header-only; drive its own install rules (NVTX3_INSTALL) so the
# exported nvtx3::nvtx3-c / nvtx3::nvtx3-cpp targets and the version file come
# from upstream instead of a hand-written config.
#
# NVTX defines the targets' $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
# include dir before including GNUInstallDirs, and vcpkg does not predefine
# CMAKE_INSTALL_INCLUDEDIR, so it must be set here or the exported targets carry
# no INTERFACE_INCLUDE_DIRECTORIES.
vcpkg_cmake_configure(SOURCE_PATH "${SOURCE_PATH}/c" OPTIONS -DNVTX3_INSTALL=ON
                      -DCMAKE_INSTALL_INCLUDEDIR=include)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(PACKAGE_NAME nvtx3 CONFIG_PATH lib/cmake/nvtx3)

# Header-only: config_fixup moves the CMake package to share/, leaving lib/
# empty.
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug"
     "${CURRENT_PACKAGES_DIR}/lib")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.txt")
