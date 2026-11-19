# SPDX-License-Identifier: Apache-2.0
#
# Patch for onpair_cpp: replaces boost::unordered_flat_map with
# std::unordered_map so Simpatico needs no Boost. Only the dictionary trainer
# and its longest-prefix matcher are compiled, and only they use the Boost
# container. Idempotent, and rewrites a file only when its content changes so a
# reconfigure does not rebuild the trainer. Included with SRC_DIR set to the
# onpair_cpp source dir, which vcpkg builds provide pre-fetched.

if(NOT DEFINED SRC_DIR)
  message(FATAL_ERROR "onpair_std_unordered.cmake: SRC_DIR not set")
endif()

set(_pair_hash
    "#include <functional>\n#include <utility>\nnamespace std {\ntemplate <> struct hash<std::pair<uint64_t, uint8_t>> {\n  size_t operator()(const std::pair<uint64_t, uint8_t>& p) const noexcept {\n    return std::hash<uint64_t>{}(p.first) ^ (std::hash<uint8_t>{}(p.second) << 1);\n  }\n};\n}  // namespace std\n"
)

foreach(_file include/onpair/encoding/lpm.h
              src/onpair/encoding/training/trainer.cpp)
  file(READ "${SRC_DIR}/${_file}" _content)
  set(_original "${_content}")
  string(REPLACE "#include <boost/unordered/unordered_flat_map.hpp>"
                 "#include <unordered_map>" _content "${_content}")
  string(REPLACE "boost::unordered_flat_map" "std::unordered_map" _content
                 "${_content}")
  if(_file MATCHES "lpm.h$" AND NOT _content MATCHES "struct hash<std::pair")
    string(REPLACE "#include <unordered_map>"
                   "#include <unordered_map>\n${_pair_hash}" _content
                   "${_content}")
  endif()
  if(NOT _content STREQUAL _original)
    file(WRITE "${SRC_DIR}/${_file}" "${_content}")
  endif()
endforeach()
