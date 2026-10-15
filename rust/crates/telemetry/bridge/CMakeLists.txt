include(FetchContent)
FetchContent_Declare(
  Corrosion
  GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
  GIT_TAG v0.6.1)
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
