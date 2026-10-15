vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  rapidsai/cuvs
  REF
  v${VERSION}
  SHA512
  73c98310f8f2723545eba183ef4e40f531e60f62976c0306a281a8dd4da9927262982f687ec5d2ead1bf61eb224b16a991499c32474667b4785d6b2e87f2a124
  PATCHES
  # NVCC 13.2 miscompiles kernel pointers with std::optional parameters.
  fix-pq-kernel-optional.patch
  static-cuda-math.patch
  disable-openmp.patch
  fix-host-refine-template-arguments.patch
  HEAD_REF
  main)

vcpkg_from_github(
  OUT_SOURCE_PATH
  RAPIDS_CMAKE_PATH
  REPO
  rapidsai/rapids-cmake
  REF
  v26.10.00
  SHA512
  c5c2be8672faae3990713a990b8e0d4e1082307c9dfd5857805bd543243daf0ec2c1fe41bbdb1736d5ea4e4e9fa261de44b2e8b885d812ce822814251d9f65d2
  HEAD_REF
  main)

# rapids-cmake otherwise downloads the CPM release asset during configure.
vcpkg_download_distfile(
  CPM_CMAKE_PATH
  URLS
  "https://api.github.com/repos/cpm-cmake/CPM.cmake/contents/cmake/CPM.cmake?ref=v0.42.0"
  FILENAME
  "CPM-v0.42.0.cmake"
  SHA512
  f9f671cc3fe5c07eea853cff23400d7d47e8e539b875c973917ac41ce2cbd7cc5b385cde17b21eccd8fc6b8b27f1e4c1c19b37f12f9a5730dee2d245f6bdaf9b
  HEADERS
  "Accept: application/vnd.github.raw+json")

vcpkg_from_github(
  OUT_SOURCE_PATH
  RAPIDS_LOGGER_PATH
  REPO
  rapidsai/rapids-logger
  REF
  v0.3.0
  SHA512
  0952bcd7c6579b0cc613e848c1d695466942d3a09d3f98af4c8fed97d23f5e7ad86d73603bfc58b55a4f33461b4e567716d0a70957935fefb896907bdf1074d1
  HEAD_REF
  main)

# cuvs CPM-clones raft (header-only, RAFT_COMPILE_LIBRARY OFF) via git; vcpkg
# builds offline (FETCHCONTENT_FULLY_DISCONNECTED=ON), so provide it locally.
vcpkg_from_github(
  OUT_SOURCE_PATH
  RAFT_PATH
  REPO
  rapidsai/raft
  REF
  v26.10.00
  SHA512
  337e03c28fe351bb6108cf0409b661781db6db44b40b957f86e50cdc92b2dd33ed92379ed4d3e1789bc7230b2c7112623312b90d5fdd18562fc06f279384a106
  PATCHES
  "${CMAKE_CURRENT_LIST_DIR}/../raft/static-cuda-math.patch"
  HEAD_REF
  main)

# cutlass 4.1.0 pinned via cpp/cmake/patches/cutlass_override.json
# (header-only).
vcpkg_from_github(
  OUT_SOURCE_PATH
  CUTLASS_PATH
  REPO
  NVIDIA/cutlass
  REF
  v4.1.0
  SHA512
  a8c2cdf772ea3b1a35bfc948ca70240477d6e8ee004ae9e487275a7b35e40424b2820396cbc827482ddb75172fcdf56372ea0d4d96ae6f3253369bd315de3ce6
  HEAD_REF
  main)

# cuco: rapids-cmake always_download forces a clone. Same pin as
# vcpkg_ports/cudf.
vcpkg_from_github(
  OUT_SOURCE_PATH
  CUCO_PATH
  REPO
  NVIDIA/cuCollections
  REF
  8ed532dc0ba3ba664d1d41027269908ddcde04c8
  SHA512
  dade9af000bb490b053288312ab60e24ac1e1ba07bd4bfdde84beca98d19e20538d5b3dde285bc5feb7c8d208791aff018b8428d2b3e493721c352b6b83f127b
  HEAD_REF
  dev)

# cuco's bootstrap otherwise downloads RAPIDS.cmake from a moving branch.
vcpkg_replace_string("${CUCO_PATH}/CMakeLists.txt" "file(DOWNLOAD"
                     "file(COPY_FILE")
vcpkg_replace_string(
  "${CUCO_PATH}/CMakeLists.txt"
  "https://raw.githubusercontent.com/rapidsai/rapids-cmake/\${rapids-cmake-branch}/RAPIDS.cmake"
  "\"${RAPIDS_CMAKE_PATH}/RAPIDS.cmake\"")

# Patch rapids_logger to use vcpkg's spdlog::spdlog target instead of spdlog
vcpkg_replace_string(
  "${RAPIDS_LOGGER_PATH}/CMakeLists.txt"
  "set_target_properties(spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)"
  "set_target_properties(spdlog::spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)"
)

# Sirius uses the existing brute-force and IVF-Flat kernels. Skip cuVS's
# optional cuTile generation so distribution builds need no cuda.tile package.
vcpkg_replace_string(
  "${SOURCE_PATH}/cpp/cmake/modules/generate_cutile_kernels.cmake"
  "  find_package(CUDAToolkit REQUIRED)"
  "  if(CUVS_DISABLE_CUTILE)\n    message(STATUS \"cuTile kernels disabled for the Sirius vcpkg port\")\n    set(_CUTILE_SETUP_OK FALSE PARENT_SCOPE)\n    return()\n  endif()\n\n  find_package(CUDAToolkit REQUIRED)"
)

# cuVS consumes RTCX, which is installed by the cuDF port and declared as a
# direct manifest dependency below. Static link forces BUILD_TESTS/
# BUILD_C_LIBRARY/BUILD_CAGRA_HNSWLIB OFF (so no gtest/hnswlib source is
# needed). BUILD_MG_ALGOS OFF drops the multi-GPU/NCCL path. Sirius uses GPU
# brute-force and IVF-Flat; disable CPU OpenMP support.
vcpkg_cmake_configure(
  SOURCE_PATH
  "${SOURCE_PATH}/cpp"
  OPTIONS
  -DFETCHCONTENT_SOURCE_DIR_RAPIDS-CMAKE=${RAPIDS_CMAKE_PATH}
  -DCPM_DOWNLOAD_LOCATION=${CPM_CMAKE_PATH}
  -DEXTRACTED_CPM_VERSION=0.42.0
  -DCPM_LOCAL_PACKAGES_ONLY=ON
  -DCPM_rapids_logger_SOURCE=${RAPIDS_LOGGER_PATH}
  -DCPM_raft_SOURCE=${RAFT_PATH}
  -DCPM_NvidiaCutlass_SOURCE=${CUTLASS_PATH}
  -DCPM_cuco_SOURCE=${CUCO_PATH}
  -DBUILD_SHARED_LIBS=OFF
  -DCUDA_STATIC_MATH_LIBRARIES=ON
  -DBUILD_TESTS=OFF
  -DBUILD_C_LIBRARY=OFF
  -DBUILD_CAGRA_HNSWLIB=OFF
  -DBUILD_MG_ALGOS=OFF
  -DDISABLE_OPENMP=ON
  -DCUVS_NVTX=OFF
  -DCUVS_DISABLE_CUTILE=ON
  "-DCMAKE_CUDA_ARCHITECTURES=${VCPKG_CUDA_ARCHITECTURES}"
  -DCMAKE_CUDA_RUNTIME_LIBRARY=Static
  # GCC 15 diagnoses RAFT's error.hpp snprintf buffer sizing under cuVS -Werror.
  "-DCMAKE_CXX_FLAGS=-I${CURRENT_INSTALLED_DIR}/include -DCCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER -Wno-error=format-truncation"
  "-DCMAKE_CUDA_FLAGS=-I${CURRENT_INSTALLED_DIR}/include -DCCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER -Xcompiler=-Wno-error=format-truncation"
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(PACKAGE_NAME cuvs CONFIG_PATH lib/cmake/cuvs)

# Static cuVS exports shared JIT dependencies; use the overlay redistributables.
vcpkg_replace_string(
  "${CURRENT_PACKAGES_DIR}/share/cuvs/cuvs-cuvs_static-static-targets.cmake"
  "CUDA::nvJitLink" "nvjitlink::nvjitlink_static")
vcpkg_replace_string(
  "${CURRENT_PACKAGES_DIR}/share/cuvs/cuvs-cuvs_static-static-targets.cmake"
  "CUDA::nvrtc" "nvrtc::nvrtc_static")
file(READ "${CURRENT_PACKAGES_DIR}/share/cuvs/cuvs-config.cmake" CUVS_CONFIG)
file(
  WRITE "${CURRENT_PACKAGES_DIR}/share/cuvs/cuvs-config.cmake"
  "include(CMakeFindDependencyMacro)\nfind_dependency(cusolver CONFIG)\nfind_dependency(curand CONFIG)\nfind_dependency(nvjitlink CONFIG)\nfind_dependency(nvrtc CONFIG)\n${CUVS_CONFIG}"
)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
