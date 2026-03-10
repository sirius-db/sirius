vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  numactl/numactl
  REF
  v2.0.19
  SHA512
  a9aa93bdc6333b620c10ff3573d6ff645ab54beece75e67be8cdddb27d062cc56cea34db342005a171877f85f05eb1d24e43f8466be907ba3b7c8b1f897cd954
  HEAD_REF
  master)

# Copy our CMakeLists.txt and support files into the source tree (numactl uses
# autotools which doesn't work on NixOS due to libtoolize issues)
file(COPY "${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt"
     DESTINATION "${SOURCE_PATH}")
file(COPY "${CMAKE_CURRENT_LIST_DIR}/config.h.cmake.in"
     DESTINATION "${SOURCE_PATH}")
file(COPY "${CMAKE_CURRENT_LIST_DIR}/numa.pc.in" DESTINATION "${SOURCE_PATH}")

# Disable symbol versioning — .symver directives in static archives cause
# "undefined version" errors with linkers like mold.
vcpkg_replace_string(
  "${SOURCE_PATH}/util.h"
  [[#if HAVE_ATTRIBUTE_SYMVER
#define SYMVER(a,b) __attribute__ ((symver (b)))
#else
#define SYMVER(a,b) __asm__ (".symver " a "," b);
#endif]]
  [[#define SYMVER(a,b) /* disabled for static linking */]])

vcpkg_cmake_configure(SOURCE_PATH "${SOURCE_PATH}")

vcpkg_cmake_install()
vcpkg_fixup_pkgconfig()
vcpkg_cmake_config_fixup(PACKAGE_NAME numactl CONFIG_PATH lib/cmake/numactl)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.LGPL2.1")
