# Use the versioned release asset; the repository copy reports a development
# version.
vcpkg_download_distfile(
  CPM_CMAKE_PATH
  URLS
  "https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.42.0/CPM.cmake"
  FILENAME
  "CPM-0.42.0-release.cmake"
  SHA512
  5c3415601e6e894601df19d1371c69bcd7dbff8512c0a55be7588f7b5e51e3be594a005df8056c1042c85e84b4f88df6992a10db138de5646c4231f20d31205b
)
