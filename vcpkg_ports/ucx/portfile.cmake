vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  openucx/ucx
  REF
  v${VERSION}
  SHA512
  1e177fb3162d63e8172ca81951e3465673cfaed57bd28ab42fa29e5b868637a55d04c91eaac72cc3e1e57c41e5a84942adb3daca4bbfd9e43a9570597d485e9d
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

# Upstream's CUDA pkg-config entry is empty and omits static registration.
foreach(UCX_CONFIG_DIR IN ITEMS "${CURRENT_PACKAGES_DIR}"
                                "${CURRENT_PACKAGES_DIR}/debug")
  if(EXISTS "${UCX_CONFIG_DIR}/lib/pkgconfig/ucx.pc")
    set(UCX_PC_INCLUDE include)
    if(UCX_CONFIG_DIR MATCHES "/debug$")
      set(UCX_PC_INCLUDE ../include)
    endif()
    configure_file("${CMAKE_CURRENT_LIST_DIR}/ucx.pc.in"
                   "${UCX_CONFIG_DIR}/lib/pkgconfig/ucx.pc" @ONLY)
  endif()
endforeach()
vcpkg_fixup_pkgconfig()

file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/ucx-config.cmake"
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/ucx")
file(INSTALL "${CURRENT_PACKAGES_DIR}/lib/cmake/ucx/ucx-config-version.cmake"
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/ucx")
file(
  REMOVE_RECURSE
  "${CURRENT_PACKAGES_DIR}/lib/cmake"
  "${CURRENT_PACKAGES_DIR}/debug/lib/cmake"
  "${CURRENT_PACKAGES_DIR}/debug/include"
  "${CURRENT_PACKAGES_DIR}/debug/share"
  "${CURRENT_PACKAGES_DIR}/tools"
  "${CURRENT_PACKAGES_DIR}/etc"
  "${CURRENT_PACKAGES_DIR}/debug/etc")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
