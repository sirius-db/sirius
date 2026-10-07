set(VCPKG_BUILD_TYPE release)
vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  ai-dynamo/nixl
  REF
  1683cf3b7f3d11674c03c5e861cea22339876c96
  SHA512
  bd27d3ab6e5a9e4bd14781731a738a3668befe15fad2b7aca848c237d9b808d30f5e5a8c067c52d6ebb6744ec096f1a12d6f539f0604962764c984d07f4acdfb
  HEAD_REF
  main
  PATCHES
  native-cpp-only.patch
  system-tomlplusplus.patch
  static-sdk.patch)

find_program(
  NIXL_NVCC nvcc HINTS "$ENV{CUDAToolkit_ROOT}/bin" "$ENV{CUDA_HOME}/bin"
                       "$ENV{CUDA_PATH}/bin" REQUIRED)
get_filename_component(NIXL_CUDA_ROOT "${NIXL_NVCC}" DIRECTORY)
get_filename_component(NIXL_CUDA_ROOT "${NIXL_CUDA_ROOT}" DIRECTORY)
if(VCPKG_TARGET_ARCHITECTURE STREQUAL "arm64")
  set(NIXL_CUDA_ARCH aarch64)
else()
  set(NIXL_CUDA_ARCH x86_64)
endif()
if(EXISTS "${NIXL_CUDA_ROOT}/targets/${NIXL_CUDA_ARCH}-linux/include/cuda.h")
  set(NIXL_CUDA_ROOT "${NIXL_CUDA_ROOT}/targets/${NIXL_CUDA_ARCH}-linux")
endif()
if(EXISTS "${NIXL_CUDA_ROOT}/lib64/libcudart_static.a")
  set(NIXL_CUDA_LIBDIR "${NIXL_CUDA_ROOT}/lib64")
elseif(EXISTS "${NIXL_CUDA_ROOT}/lib/libcudart_static.a")
  set(NIXL_CUDA_LIBDIR "${NIXL_CUDA_ROOT}/lib")
else()
  message(
    FATAL_ERROR "NIXL requires a CUDA toolkit containing libcudart_static.a")
endif()
set(NIXL_CUDA_STUBDIR "${NIXL_CUDA_LIBDIR}/stubs")
if(NOT EXISTS "${NIXL_CUDA_STUBDIR}/libcuda.so")
  message(FATAL_ERROR "NIXL requires the CUDA toolkit driver stub library")
endif()
vcpkg_configure_meson(
  SOURCE_PATH
  "${SOURCE_PATH}"
  LANGUAGES
  CXX
  ADDITIONAL_BINARIES
  "cuda = ['${NIXL_NVCC}']"
  OPTIONS
  -Dnative_only=true
  -Ddefault_library=static
  -Dprefer_static=true
  -Db_staticpic=true
  -Dstatic_plugins=UCX
  -Denable_plugins=UCX
  -Dbuild_tests=false
  -Dbuild_examples=false
  -Dbuild_nixl_ep=false
  -Dwith_trace=false
  -Dwerror=false
  "-Dcudapath_inc=${NIXL_CUDA_ROOT}/include"
  "-Dcudapath_lib=${NIXL_CUDA_LIBDIR}"
  "-Dcudapath_stub=${NIXL_CUDA_STUBDIR}"
  # Override vcpkg's wrap mode after its per-configuration defaults.
  OPTIONS_RELEASE
  --wrap-mode=nofallback)
vcpkg_install_meson()

file(
  INSTALL
  "${CMAKE_CURRENT_LIST_DIR}/nixl-config.cmake"
  "${CMAKE_CURRENT_LIST_DIR}/nixl-config-version.cmake"
  "${CMAKE_CURRENT_LIST_DIR}/nixl-targets.cmake"
  "${CMAKE_CURRENT_LIST_DIR}/usage"
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/nixl")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
