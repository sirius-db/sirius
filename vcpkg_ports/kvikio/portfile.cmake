vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  rapidsai/kvikio
  REF
  v${VERSION}
  SHA512
  8a40f874ec38c423ef7bd2a8399985e4b2dda6f4346674a262e572b45bfdd50d368af49ebdd54bf63b7d8816d84942699c0f42ff1b03c331602319ae236c0f8e
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
  de9234549a96b2a73caa4bbbaf10500419c5f16069a430d172432c23e6d404b3ab9e595f6c0958d15f4f190fb50e1cd7dd02945b93cf428180df8feb8822ed94
  HEAD_REF
  main)

vcpkg_from_github(
  OUT_SOURCE_PATH
  BS_THREAD_POOL_PATH
  REPO
  bshoshany/thread-pool
  REF
  v4.1.0
  SHA512
  4908f00def23082e7ddc0b24a710e53b3fde51b02188e79cfcd9dabb22627ebd1b6e5b3c4bf1b366eae79660c26878cc034c171747c3d0b7ef8a98c85a77033b
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
