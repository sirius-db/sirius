vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  rapidsai/kvikio
  REF
  v26.02.00
  SHA512
  36405e2cb907b84061789206e9c8dbea95bd0f68dd1762a4e8ca274933018b5c20637fb67be7b6f33109a2666edb2b5359a6b7c9e36bff08d63e5ba8f2a4e57f
  HEAD_REF
  main)

vcpkg_from_github(
  OUT_SOURCE_PATH
  RAPIDS_CMAKE_PATH
  REPO
  rapidsai/rapids-cmake
  REF
  v26.02.00
  SHA512
  00d2bb2c005f9e2c4e525af4350c8e8b7d6e67369da9c664c6c3d6080c33d6359b7142a311f3563479f7ac8e1bb0a2a520e8e926719dfeb8e27e49f8cd3e65ca
  HEAD_REF
  main)

vcpkg_from_github(
  OUT_SOURCE_PATH
  BS_THREAD_POOL_PATH
  REPO
  bshoshany/thread-pool
  REF
  097aa718f25d44315cadb80b407144ad455ee4f9
  SHA512
  94177c61c5161c3cb5d088058d999239fb8bc446100e948bb9bbae44b73d0a020240c39d5232b13a628b56e233cb55a29e70baa69e511c73a6ba6a2505de1250
  HEAD_REF
  master)

# Patch kvikio to not require cuFile Batch/Stream API (may not be available in
# the CUDA toolkit). KvikIO still works without cuFile, just disables GPUDirect
# Storage (GDS).
vcpkg_replace_string(
  "${SOURCE_PATH}/cpp/CMakeLists.txt" "if(NOT TARGET CUDA::cuFile)"
  "if(TRUE) # Disable cuFile/GDS - batch/stream API requires newer cuFile SDK")

# Add stub declarations for cuFile batch/stream API functions so the shim
# compiles without cufile.h. These are never called - they only provide type
# info for decltype().
vcpkg_replace_string(
  "${SOURCE_PATH}/cpp/include/kvikio/shim/cufile_h_wrapper.hpp"
  "CUfileError_t cuFileDriverSetMaxPinnedMemSize(...);"
  "CUfileError_t cuFileDriverSetMaxPinnedMemSize(...);
using CUfileBatchHandle_t = void*;
enum CUfileOpcode_t { CUFILE_READ = 0, CUFILE_WRITE = 1 };
enum CUfileBatchMode_t { CUFILE_BATCH = 0 };
struct CUfileIOEvents_t { int dummy; };
struct CUfileIOParams_t { CUfileBatchMode_t mode; union { struct { void* devPtr_base; off_t file_offset; off_t devPtr_offset; size_t size; } batch; } u; CUfileHandle_t fh; CUfileOpcode_t opcode; void* cookie; };
CUfileError_t cuFileBatchIOSetUp(...);
CUfileError_t cuFileBatchIOSubmit(...);
CUfileError_t cuFileBatchIOGetStatus(...);
CUfileError_t cuFileBatchIOCancel(...);
CUfileError_t cuFileBatchIODestroy(...);
CUfileError_t cuFileReadAsync(...);
CUfileError_t cuFileWriteAsync(...);
CUfileError_t cuFileStreamRegister(...);
CUfileError_t cuFileStreamDeregister(...);")

vcpkg_cmake_configure(
  SOURCE_PATH
  "${SOURCE_PATH}/cpp"
  OPTIONS
  -DFETCHCONTENT_SOURCE_DIR_RAPIDS-CMAKE=${RAPIDS_CMAKE_PATH}
  -DCPM_bs_thread_pool_SOURCE=${BS_THREAD_POOL_PATH}
  -DKvikIO_BUILD_EXAMPLES=OFF
  -DKvikIO_BUILD_TESTS=OFF
  -DKvikIO_BUILD_BENCHMARKS=OFF
  -DKvikIO_REMOTE_SUPPORT=OFF
  -DCMAKE_CUDA_ARCHITECTURES=RAPIDS)

vcpkg_cmake_install()

# bs_thread_pool cmake config is generated but not installed. We need to
# manually install it for consumers to find it.
file(
  GLOB
  BS_THREAD_POOL_CMAKE_FILES
  "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-rel/bs_thread_pool-*.cmake"
  "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-rel/CMakeFiles/Export/*/bs_thread_pool-targets*.cmake"
)
file(INSTALL ${BS_THREAD_POOL_CMAKE_FILES}
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/bs_thread_pool")

# Fix bs_thread_pool-targets.cmake path computation (4 dirs -> 3 dirs for
# share/bs_thread_pool/ layout)
execute_process(
  COMMAND
    sed -i "52d"
    "${CURRENT_PACKAGES_DIR}/share/bs_thread_pool/bs_thread_pool-targets.cmake")

vcpkg_cmake_config_fixup(PACKAGE_NAME kvikio CONFIG_PATH lib/cmake/kvikio)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
