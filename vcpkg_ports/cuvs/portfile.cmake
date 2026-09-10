vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  rapidsai/cuvs
  REF
  v${VERSION}
  SHA512
  20ab7b08c47f27ccc1d43f880c350d9f20e48b2c7bb9e0b1c6c9bf2995fa49ec7b542eaaded32b32aa53bb7294f680761ef515ba4fc9e89fa6b1384eb6b4f930
  # NVCC 13.2 miscompiles kernel pointers with std::optional parameters.
  PATCHES
  fix-pq-kernel-optional.patch
  HEAD_REF
  main)

vcpkg_from_github(
  OUT_SOURCE_PATH
  RAPIDS_CMAKE_PATH
  REPO
  rapidsai/rapids-cmake
  REF
  v26.08.00
  SHA512
  472e3bbc0aeedce6632c339f5a383a25524df8462a58891474523cd558b7d8d8bc09b24e8f75da6f2fbfcb83b302caa911fb5a38bc6c04871a570390e7b4a5b8
  HEAD_REF
  main)

vcpkg_from_github(
  OUT_SOURCE_PATH
  RAPIDS_LOGGER_PATH
  REPO
  rapidsai/rapids-logger
  REF
  v0.2.3
  SHA512
  eb7b5ebf6289d10307b8a34d9d1469ffcb63e9371e9dd5ccbda0351923b920ebae8220ceaa8d1d52c9bed57200f35921a6365a5f9a25a209a98314f75195310c
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
  v26.08.00
  SHA512
  cbfe6c618bac35b16f5b9313f1f0315c8c9e331dcb1bdc5038be772951e45798d644f4c602a4370552cd22778676f860b54ed489662d8ac2ac775ba5efe52cf5
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
  0883368d39296f3bef3a058033141bcc642c5c54
  SHA512
  be8040f9ad46a2f8cc9d9c53a2bdbd0c9ff1e83c3f93788ae5508bc9f4610028e2d326e6b357a41a65a93064960db8d18963a540f25609fe2f254eb231484aee
  HEAD_REF
  dev)

# Patch rapids_logger to use vcpkg's spdlog::spdlog target instead of spdlog
vcpkg_replace_string(
  "${RAPIDS_LOGGER_PATH}/CMakeLists.txt"
  "set_target_properties(spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)"
  "set_target_properties(spdlog::spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)"
)

# Static link forces BUILD_TESTS/BUILD_C_LIBRARY/BUILD_CAGRA_HNSWLIB OFF (so no
# gtest/hnswlib source is needed). BUILD_MG_ALGOS OFF drops the multi-GPU/NCCL
# path. Only cuvs::neighbors::brute_force + cuvs::distance are consumed by
# Sirius, but libcuvs still compiles the full kernel set.
vcpkg_cmake_configure(
  SOURCE_PATH
  "${SOURCE_PATH}/cpp"
  OPTIONS
  -DFETCHCONTENT_SOURCE_DIR_RAPIDS-CMAKE=${RAPIDS_CMAKE_PATH}
  -DCPM_rapids_logger_SOURCE=${RAPIDS_LOGGER_PATH}
  -DCPM_raft_SOURCE=${RAFT_PATH}
  -DCPM_NvidiaCutlass_SOURCE=${CUTLASS_PATH}
  -DCPM_cuco_SOURCE=${CUCO_PATH}
  -DBUILD_SHARED_LIBS=OFF
  -DBUILD_TESTS=OFF
  -DBUILD_C_LIBRARY=OFF
  -DBUILD_CAGRA_HNSWLIB=OFF
  -DBUILD_MG_ALGOS=OFF
  -DCUVS_NVTX=OFF
  -DCMAKE_CUDA_ARCHITECTURES=RAPIDS
  -DCMAKE_CUDA_RUNTIME_LIBRARY=Static
  "-DCMAKE_CXX_FLAGS=-I${CURRENT_INSTALLED_DIR}/include -DCCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER"
  "-DCMAKE_CUDA_FLAGS=-I${CURRENT_INSTALLED_DIR}/include -DCCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER"
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
  "include(CMakeFindDependencyMacro)\nfind_dependency(nvjitlink CONFIG)\nfind_dependency(nvrtc CONFIG)\n${CUVS_CONFIG}"
)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
