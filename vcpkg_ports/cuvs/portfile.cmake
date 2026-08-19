vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  rapidsai/cuvs
  REF
  v${VERSION}
  SHA512
  73cdf0e16e701063528c71ac48f0d3d9072dce11d1ab2ac768173c676cb8d704fcb2d3eb5b0ae8892b3b657eefc8bc03e3d83fb204befaa81cb9538328410cf1
  HEAD_REF
  main)

vcpkg_from_github(
  OUT_SOURCE_PATH
  RAPIDS_CMAKE_PATH
  REPO
  rapidsai/rapids-cmake
  REF
  v${VERSION}
  SHA512
  d3d7a1f807a9b71ed15c972742a4dbee0746cc65b1bfa7eef9a8e036a992a37fcfdfbff79fc27cf053dc5a37978abf86b93b56bc6f605f04244e8f6776595bdd
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
  v${VERSION}
  SHA512
  b5c25d369f7e69941118b342ac581d0908a0f0c7763f4e42c6bc7af0afee4ab85306dfed057b28b115096bdf2799d8da5ce7eb3d2eb796468b210ea5f7724d41
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
  f517bbb1277753b1852dfd388993383e401eaa38
  SHA512
  53f6185db57eba7391fd9b93b342d9da3c4cdf906dc47fad02cc1eb34867bb11ce798f0e94fee71a3cfb6423e6cad907ebe21e46997b4f4ef2b6e1e3b0a45d82
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
  "-DCMAKE_CXX_FLAGS=-I${CURRENT_INSTALLED_DIR}/include"
  "-DCMAKE_CUDA_FLAGS=-I${CURRENT_INSTALLED_DIR}/include")

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(PACKAGE_NAME cuvs CONFIG_PATH lib/cmake/cuvs)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
