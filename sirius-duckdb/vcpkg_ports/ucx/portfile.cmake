set(VCPKG_BUILD_TYPE release)
vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  openucx/ucx
  REF
  v${VERSION}
  SHA512
  7d2b02736a19a948dd8ee83386ce21304304ba4c981fabcef8e7668c847a57ab280a66dbd4762e5282f6858950843f585b9e30cba50f36bdf62c9048bbb49580
  PATCHES
  static-pic.patch
  static-cudart.patch
  library-only.patch)

find_program(
  UCX_NVCC nvcc HINTS "$ENV{CUDAToolkit_ROOT}/bin" "$ENV{CUDA_HOME}/bin"
                      "$ENV{CUDA_PATH}/bin" REQUIRED)
get_filename_component(UCX_CUDA_ROOT "${UCX_NVCC}" DIRECTORY)
get_filename_component(UCX_CUDA_ROOT "${UCX_CUDA_ROOT}" DIRECTORY)
if(VCPKG_TARGET_ARCHITECTURE STREQUAL "arm64")
  set(UCX_CUDA_ARCH aarch64)
else()
  set(UCX_CUDA_ARCH x86_64)
endif()
if(EXISTS "${UCX_CUDA_ROOT}/targets/${UCX_CUDA_ARCH}-linux/include/cuda.h")
  set(UCX_CUDA_ROOT "${UCX_CUDA_ROOT}/targets/${UCX_CUDA_ARCH}-linux")
endif()
if(NOT EXISTS "${UCX_CUDA_ROOT}/lib64/libcudart_static.a"
   AND NOT EXISTS "${UCX_CUDA_ROOT}/lib/libcudart_static.a")
  message(
    FATAL_ERROR "UCX requires a CUDA toolkit containing libcudart_static.a")
endif()

vcpkg_configure_make(
  SOURCE_PATH
  "${SOURCE_PATH}"
  AUTOCONFIG
  OPTIONS
  --with-pic
  --enable-mt
  --enable-cma
  --disable-ucg
  --disable-examples
  --disable-test-apps
  --without-verbs
  --without-rdmacm
  --without-mlx5
  --without-gdrcopy
  --without-rocm
  --without-ze
  --without-xpmem
  --without-knem
  --without-ugni
  --without-fuse3
  --without-mad
  --without-java
  --without-go
  --without-bfd
  --without-mpi
  --with-cuda=${UCX_CUDA_ROOT}
  NVCC=${UCX_NVCC})
vcpkg_install_make()

vcpkg_fixup_pkgconfig()

file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/ucx-config.cmake"
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/ucx")
file(INSTALL "${CURRENT_PACKAGES_DIR}/lib/cmake/ucx/ucx-config-version.cmake"
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/ucx")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/lib/cmake"
     "${CURRENT_PACKAGES_DIR}/tools" "${CURRENT_PACKAGES_DIR}/etc")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
