vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

# raft is exported header-only here (RAFT_COMPILE_LIBRARY OFF), so this port
# produces no compiled library. Its only job is to install raft's headers and
# the raft-config.cmake that exports the raft::raft interface target. cuvs's
# installed config does find_dependency(raft), and Sirius includes raft headers
# directly, so both need raft to be a findable package in the vcpkg tree.
vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  rapidsai/raft
  REF
  v26.10.00
  SHA512
  337e03c28fe351bb6108cf0409b661781db6db44b40b957f86e50cdc92b2dd33ed92379ed4d3e1789bc7230b2c7112623312b90d5fdd18562fc06f279384a106
  PATCHES
  static-cuda-math.patch
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

# Patch rapids_logger to use vcpkg's spdlog::spdlog target instead of spdlog.
# Same fix the rmm and cuvs ports apply.
vcpkg_replace_string(
  "${RAPIDS_LOGGER_PATH}/CMakeLists.txt"
  "set_target_properties(spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)"
  "set_target_properties(spdlog::spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)"
)

# Header-only export: RAFT_COMPILE_LIBRARY OFF means no libraft is built, so no
# gtest/benchmark sources are needed either. rmm is already an installed vcpkg
# port, so raft finds it from CMAKE_PREFIX_PATH.
vcpkg_cmake_configure(
  SOURCE_PATH
  "${SOURCE_PATH}/cpp"
  OPTIONS
  -DFETCHCONTENT_SOURCE_DIR_RAPIDS-CMAKE=${RAPIDS_CMAKE_PATH}
  -DCPM_rapids_logger_SOURCE=${RAPIDS_LOGGER_PATH}
  -DRAFT_COMPILE_LIBRARY=OFF
  -DCUDA_STATIC_MATH_LIBRARIES=ON
  -DBUILD_TESTS=OFF
  -DDISABLE_OPENMP=ON
  -DRAFT_NVTX=OFF
  -DCMAKE_CUDA_ARCHITECTURES=RAPIDS
  -DCMAKE_CUDA_RUNTIME_LIBRARY=Static
  "-DCMAKE_CXX_FLAGS=-I${CURRENT_INSTALLED_DIR}/include"
  "-DCMAKE_CUDA_FLAGS=-I${CURRENT_INSTALLED_DIR}/include")

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(PACKAGE_NAME raft CONFIG_PATH lib/cmake/raft)

file(READ "${CURRENT_PACKAGES_DIR}/share/raft/raft-config.cmake" RAFT_CONFIG)
file(
  WRITE "${CURRENT_PACKAGES_DIR}/share/raft/raft-config.cmake"
  "include(CMakeFindDependencyMacro)\nfind_dependency(cusolver CONFIG)\nfind_dependency(curand CONFIG)\n${RAFT_CONFIG}"
)

# rmm is the port that owns rapids_logger in this vcpkg layout. raft's build
# produces its own copy of the headers and static lib; drop them so they do not
# conflict with the files rmm already installed. raft-config still resolves
# rapids_logger through find_dependency against rmm's install (same version).
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/include/rapids_logger"
     "${CURRENT_PACKAGES_DIR}/share/rapids_logger")
# raft is header-only and librapids_logger.a was the only file in lib/, so once
# it is gone lib/ is empty. Remove it (both release and debug) to avoid vcpkg's
# empty-directory post-build warning.
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/lib"
     "${CURRENT_PACKAGES_DIR}/debug/lib")

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
