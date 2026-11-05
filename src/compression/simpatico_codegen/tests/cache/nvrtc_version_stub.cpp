// Linked only into the version-policy worker. Compilation still uses real NVRTC.
#include <nvrtc.h>

#include <cstdlib>
#include <string_view>

extern "C" nvrtcResult __real_nvrtcVersion(int*, int*);

extern "C" nvrtcResult __wrap_nvrtcVersion(int* major, int* minor)
{
  const char* setting         = std::getenv("SIMPATICO_TEST_NVRTC_VERSION");
  const std::string_view mode = setting ? setting : "";
  if (mode == "error") return NVRTC_ERROR_INTERNAL_ERROR;
  const auto result = __real_nvrtcVersion(major, minor);
  if (result == NVRTC_SUCCESS) {
    if (mode == "major") ++*major;
    if (mode == "minor") ++*minor;
  }
  return result;
}
