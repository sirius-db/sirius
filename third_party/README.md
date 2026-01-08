# Third-Party Dependencies

## cuCascade

cuCascade is included as a git submodule and built as a static library during CMake configuration.

### Initial Setup

To initialize the cuCascade submodule:

```bash
git submodule update --init --recursive
```

### CMake Configuration Options

#### CUCASCADE_UPDATE_SUBMODULE

Update the cuCascade submodule to the latest version from the remote repository.

```bash
cmake -DCUCASCADE_UPDATE_SUBMODULE=ON ..
```

Default: `OFF`

#### CUCASCADE_GIT_HASH

Specify a specific git hash, tag, or branch to checkout for cuCascade.

```bash
# Use a specific commit hash
cmake -DCUCASCADE_GIT_HASH=abc123def456 ..

# Use a specific tag
cmake -DCUCASCADE_GIT_HASH=v1.0.0 ..

# Use a specific branch
cmake -DCUCASCADE_GIT_HASH=develop ..
```

Default: `main`

### Examples

```bash
# Standard build with default (main branch)
cmake -B build -S .
cmake --build build

# Update to latest and build
cmake -B build -S . -DCUCASCADE_UPDATE_SUBMODULE=ON
cmake --build build

# Use a specific commit
cmake -B build -S . -DCUCASCADE_GIT_HASH=abc123def456
cmake --build build

# Update to latest, then lock to a specific version
cmake -B build -S . -DCUCASCADE_UPDATE_SUBMODULE=ON
# After verifying it works, find the commit hash and use it
cmake -B build -S . -DCUCASCADE_GIT_HASH=<commit-hash-from-update>
```

### How It Works

1. The cuCascade submodule is located at `third_party/cucascade/`
2. During CMake configuration, the `third_party/cucascade.cmake` script:
   - Verifies the submodule is initialized
   - Optionally updates to the latest version (if `CUCASCADE_UPDATE_SUBMODULE=ON`)
   - Optionally checks out a specific hash/tag/branch (if `CUCASCADE_GIT_HASH` is set)
   - Configures cuCascade using its release preset
   - Builds the cuCascade static library
   - Creates an imported CMake target `cucascade::cucascade`
3. The Sirius extension links against the static library

### Troubleshooting

If you encounter issues with the cuCascade submodule:

```bash
# Reinitialize the submodule
git submodule deinit -f third_party/cucascade
git submodule update --init --recursive

# Or manually clone it
cd third_party
git clone https://github.com/NVIDIA/cuCascade.git cucascade
```

## spdlog

spdlog is managed via ExternalProject and automatically downloaded during the build process.
