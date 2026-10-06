# RAPIDS logger relocation probe

Build the RMM overlay port into a dedicated vcpkg installation, then move that
installation so its original paths no longer exist. Configure this probe against
the moved triplet prefix without the vcpkg toolchain:

```sh
cmake -S vcpkg_ports/tests/rapids_logger -B build/rapids-logger-probe \
  -DCMAKE_PREFIX_PATH=/path/to/relocated/x64-linux
cmake --build build/rapids-logger-probe
ctest --test-dir build/rapids-logger-probe --output-on-failure
```

Move the complete triplet prefix, including its fmt and spdlog dependencies, and
place it first in `CMAKE_PREFIX_PATH` so those matching packages are selected.

The probe imports the package in a child directory and links from its parent to
check dependency target visibility. It writes and checks a formatted log message.
It needs no GPU, driver, or CPM download.
