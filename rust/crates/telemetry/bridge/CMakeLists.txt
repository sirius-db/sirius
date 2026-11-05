include(FetchContent)
FetchContent_Declare(
  Corrosion
  GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
  GIT_TAG v0.6.1
  FIND_PACKAGE_ARGS 0.6 CONFIG)
FetchContent_MakeAvailable(Corrosion)

# --- Import the Rust bridge crate (CXX bridges) ---
corrosion_import_crate(MANIFEST_PATH Cargo.toml CRATES telemetry-bridge)

# build.rs stages rust/cxx.h and telemetry-bridge/gen/*.h into this build-tree
# directory, so each CMake build directory gets its own copy and plain `cargo`
# runs never touch it.
set(TELEMETRY_BRIDGE_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
corrosion_set_env_vars(
  telemetry_bridge
  "SIRIUS_TELEMETRY_BRIDGE_INCLUDE_DIR=${TELEMETRY_BRIDGE_INCLUDE_DIR}")
# BUILD_INTERFACE: telemetry_bridge is in DuckDB's install export set, which may
# not reference build-tree paths. The headers are not installed, so there is no
# INSTALL_INTERFACE counterpart.
target_include_directories(
  telemetry_bridge INTERFACE $<BUILD_INTERFACE:${TELEMETRY_BRIDGE_INCLUDE_DIR}>)

# bindgen (nvtx-sys) loads libclang and asks clang for its include paths. Unless
# set in the environment, use the ones on CMake's search path; clang-sys
# searches the system paths itself.
if(NOT DEFINED ENV{LIBCLANG_PATH})
  find_library(SIRIUS_LIBCLANG NAMES clang libclang.so.13)
  if(SIRIUS_LIBCLANG)
    cmake_path(GET SIRIUS_LIBCLANG PARENT_PATH libclang_dir)
    corrosion_set_env_vars(telemetry_bridge "LIBCLANG_PATH=${libclang_dir}")
  endif()
endif()
if(NOT DEFINED ENV{CLANG_PATH})
  find_program(SIRIUS_CLANG NAMES clang)
  if(SIRIUS_CLANG)
    corrosion_set_env_vars(telemetry_bridge "CLANG_PATH=${SIRIUS_CLANG}")
  endif()
endif()

# Corrosion's installer exports the archive location, but not its native links.
get_target_property(telemetry_native_libraries telemetry_bridge-static
                    INTERFACE_LINK_LIBRARIES)
if(VCPKG_BUILD)
  list(REMOVE_ITEM telemetry_native_libraries gcc_s)
endif()
if(telemetry_native_libraries)
  target_link_libraries(telemetry_bridge
                        INTERFACE ${telemetry_native_libraries})
endif()
