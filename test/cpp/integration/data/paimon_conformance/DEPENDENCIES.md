# Pinned reader dependency downloads

This inventory covers the enabled Linux x86_64 local-filesystem recipe in
[source-build.json](source-build.json). It is a map of checksum enforcement in
fixed upstream build sources, not a cold-build result, endpoint availability
promise, or bit-for-bit reproducibility claim. Update this audit when the source
pins or enabled build options change.

## Top-level sources

The recipe records full commits, archive sizes and SHA256 values for DuckDB
`88ecd0a49d7d3d838e002b2fbc0ce91ccf74110e`, duckdb-paimon
`307fc34c04623fe18f0ac39125b6319ea0013c0f`, and paimon-cpp
`53f9c86d45aabb0a6f1a379271da07d7a9f27a3d`. The extension comes from maintainer mrdrivingduck's v2 adaptation; the native
Paimon pin and enabled native dependencies remain unchanged. The Python builder checks size and SHA256
before extraction. DuckDB receives the extension through a local `SOURCE_DIR`,
so its remote-extension fetch path is unused. The native source is supplied in
the extension's otherwise empty source-archive gitlink directory.

## Enabled bundled downloads

The table lists 19 CMake ExternalProject fetches, representing 18 distinct
archives. RapidJSON appears twice because Paimon and Arrow each build it.
Each linked rule uses `URL_HASH SHA256`; checksum values come from the fixed
[native versions file][native-versions], [Arrow versions file][arrow-versions],
or the linked ORC rule. The rules also identify their source URLs and mirrors.

| Parent / dependency | Version or source commit | SHA256 | Enforcement |
| --- | --- | --- | --- |
| Paimon / FMT | `11.2.0` | `bc23066d87ab3168f27cef3e97d545fa63314f5c79df5ea444d41d56f962c6af` | [URL_HASH](https://github.com/apache/paimon-cpp/blob/53f9c86d45aabb0a6f1a379271da07d7a9f27a3d/cmake_modules/ThirdpartyToolchain.cmake#L1010) |
| Paimon / RAPIDJSON | `232389d4f1012dddec4ef84861face2d2ba85709` | `b9290a9a6d444c8e049bd589ab804e0ccf2b05dc5984a19ed5ae75d090064806` | [URL_HASH](https://github.com/apache/paimon-cpp/blob/53f9c86d45aabb0a6f1a379271da07d7a9f27a3d/cmake_modules/ThirdpartyToolchain.cmake#L957) |
| Paimon / DATASKETCHES | `5.2.0` | `63e6bda660ee9730cc39746de87fd983118c28149e32ff9389fac98b27010a53` | [URL_HASH](https://github.com/apache/paimon-cpp/blob/53f9c86d45aabb0a6f1a379271da07d7a9f27a3d/cmake_modules/ThirdpartyToolchain.cmake#L980) |
| Paimon / RE2 | `2022-06-01` | `f89c61410a072e5cbcf8c27e3a778da7d6fd2f2b5b1445cd4f4508bee946ab0f` | [URL_HASH](https://github.com/apache/paimon-cpp/blob/53f9c86d45aabb0a6f1a379271da07d7a9f27a3d/cmake_modules/ThirdpartyToolchain.cmake#L1144) |
| Paimon / SNAPPY | `1.1.10` | `49d831bffcc5f3d01482340fe5af59852ca2fe76c3e05df0e67203ebbe0f1d90` | [URL_HASH](https://github.com/apache/paimon-cpp/blob/53f9c86d45aabb0a6f1a379271da07d7a9f27a3d/cmake_modules/ThirdpartyToolchain.cmake#L1170) |
| Paimon / ZSTD | `1.5.7` | `eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3` | [URL_HASH](https://github.com/apache/paimon-cpp/blob/53f9c86d45aabb0a6f1a379271da07d7a9f27a3d/cmake_modules/ThirdpartyToolchain.cmake#L1225) |
| Paimon / ZLIB | `1.3.1` | `9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23` | [URL_HASH](https://github.com/apache/paimon-cpp/blob/53f9c86d45aabb0a6f1a379271da07d7a9f27a3d/cmake_modules/ThirdpartyToolchain.cmake#L1196) |
| Paimon / LZ4 | `v1.9.4` | `0b0e3aa07c8c063ddf40b082bdf7e37a1562bda40a0ff5272957f3e987e0e54b` | [URL_HASH](https://github.com/apache/paimon-cpp/blob/53f9c86d45aabb0a6f1a379271da07d7a9f27a3d/cmake_modules/ThirdpartyToolchain.cmake#L1252) |
| Paimon / ARROW | `17.0.0` | `9d280d8042e7cf526f8c28d170d93bfab65e50f94569f6a790982a878d8d898d` | [URL_HASH](https://github.com/apache/paimon-cpp/blob/53f9c86d45aabb0a6f1a379271da07d7a9f27a3d/cmake_modules/ThirdpartyToolchain.cmake#L1849) |
| Paimon / TBB | `v2021.13.0` | `3ad5dd08954b39d113dc5b3f8a8dc6dc1fd5250032b7c491eb07aed5c94133e1` | [URL_HASH](https://github.com/apache/paimon-cpp/blob/53f9c86d45aabb0a6f1a379271da07d7a9f27a3d/cmake_modules/ThirdpartyToolchain.cmake#L2018) |
| Paimon / GLOG | `v0.7.1` | `00e4a87e87b7e7612f519a41e491f16623b12423620006f59f5688bfd8d13b08` | [URL_HASH](https://github.com/apache/paimon-cpp/blob/53f9c86d45aabb0a6f1a379271da07d7a9f27a3d/cmake_modules/ThirdpartyToolchain.cmake#L2094) |
| Paimon / AVRO | `c499eefb48aa2db906c7bca14a047223806f36db` | `9771f1dcfe3c01aff7ff670e873e66d3406362f71941821d482de65f3d32d780` | [URL_HASH](https://github.com/apache/paimon-cpp/blob/53f9c86d45aabb0a6f1a379271da07d7a9f27a3d/cmake_modules/ThirdpartyToolchain.cmake#L1650) |
| Paimon / PROTOBUF | `3.8.0` | `b7220b41481011305bf9100847cf294393973e869973a9661046601959b2960b` | [URL_HASH](https://github.com/apache/paimon-cpp/blob/53f9c86d45aabb0a6f1a379271da07d7a9f27a3d/cmake_modules/ThirdpartyToolchain.cmake#L1594) |
| Paimon / ORC | `v2.1.1` | `1f8eef537814fdcd003de13e49c6edb35427b45eb40bafd3355f775d99a0ff99` | [URL_HASH](https://github.com/apache/paimon-cpp/blob/53f9c86d45aabb0a6f1a379271da07d7a9f27a3d/cmake_modules/ThirdpartyToolchain.cmake#L1723) |
| Arrow / BOOST | `1.81.0` | `9e0ffae35528c35f90468997bc8d99500bf179cbae355415a89a600c38e13574` | [URL_HASH](https://github.com/apache/arrow/blob/6a2e19a852b367c72d7b12da4d104456491ed8b7/cpp/cmake_modules/ThirdpartyToolchain.cmake#L1144) |
| Arrow / BROTLI | `v1.0.9` | `f9e8d81d0405ba66d181529af42a3354f838c939095ff99930da6aa9cdf6fe46` | [URL_HASH](https://github.com/apache/arrow/blob/6a2e19a852b367c72d7b12da4d104456491ed8b7/cpp/cmake_modules/ThirdpartyToolchain.cmake#L1465) |
| Arrow / THRIFT | `0.16.0` | `f460b5c1ca30d8918ff95ea3eb6291b3951cf518553566088f3f2be8981f6209` | [URL_HASH](https://github.com/apache/arrow/blob/6a2e19a852b367c72d7b12da4d104456491ed8b7/cpp/cmake_modules/ThirdpartyToolchain.cmake#L1758) |
| Arrow / RAPIDJSON | `232389d4f1012dddec4ef84861face2d2ba85709` | `b9290a9a6d444c8e049bd589ab804e0ccf2b05dc5984a19ed5ae75d090064806` | [URL_HASH](https://github.com/apache/arrow/blob/6a2e19a852b367c72d7b12da4d104456491ed8b7/cpp/cmake_modules/ThirdpartyToolchain.cmake#L2443) |
| ORC / ORC_FORMAT | `1.0.0` | `739fae5ff94b1f812b413077280361045bf92e510ef04b34a610e23a945d8cd5` | [URL_HASH](https://github.com/apache/orc/blob/5225b51824c529cec0aab525765dfa8367da394a/cmake_modules/ThirdpartyToolchain.cmake#L148) |

The Arrow 17.0.0 and ORC 2.1.1 source links resolve their tags to full commits.
The downloaded release archives are governed by their separate pinned SHA256
values; these links are navigational evidence, not a claim that release tarballs
and Git-generated archives have identical bytes.

Arrow reuses Paimon's zstd, Snappy, LZ4, zlib and RE2 through explicit installation
roots. ORC similarly reuses the compression libraries and Protobuf. Avro reuses
fmt, Snappy, zstd and zlib. Thrift uses Arrow's Boost headers with tests, language
generators, libevent and zlib disabled. Roaring bitmap and xxhash are vendored in
the hash-pinned native source archive rather than separately downloaded.

The [native Arrow patch][arrow-patch] adds an Apache archive mirror for Thrift.
The [native ORC patch][orc-patch] switches ORC-format to the Apache archive
endpoint. Both retain the upstream checksum checks.

Arrow's Boost 1.81.0 is a trimmed Arrow-maintained bundle, with preferred URL
`https://apache.jfrog.io/artifactory/arrow/thirdparty/7.0.0/boost_1_81_0.tar.gz`.
A full Boost release archive is not interchangeable with that digest. An upstream
fallback mirror returning unavailable or differently packaged bytes must fail
rather than cause a checksum change. Source caching saves downloads only; it
does not establish how long a fresh compilation will take.

## Disabled branches and environment boundary

- OSS, S3 and REST are OFF. Their SDK/authentication and network-catalog branches
  are not covered as enabled downloads by this recipe.
- Native tests, benchmarks, Jindo, Mosaic, Lumina, Lucene and Tantivy are OFF.
  Native `versions.txt`'s Boost 1.66 belongs to the disabled Lucene branch; it is
  not the enabled Arrow Boost dependency. Rust/Corrosion and those optional
  binary downloads are not needed by this reader build.
- Arrow cloud backends, Flight, Substrait, tests, benchmarks, ORC, jemalloc, BZ2
  and UTF8PROC are OFF. ORC Java, tests, tools and HDFS client are OFF; its
  Windows-only tzdata download is outside this Linux recipe.
- [Avro contains a Git-tag fmt fallback][avro-fmt] if `find_package(fmt)` fails.
  It is inactive here: [native CMake passes the pinned fmt installation root and
  orders Avro after fmt][avro-config]. Do not interpret this inventory as a claim
  that all possible upstream configurations have archive checksums.
- Python, CMake, Ninja, the compiler and its runtime are supplied by the locked
  Pixi environment. OS facilities, system libraries and ambient discovery are
  not part of this CMake archive inventory. The receipt records build identities;
  it does not make an arbitrary host hermetic or provide a signed attestation.

## Python download policy

Both top-level source fetching and explicit historical artifact provisioning use
the same downloader. It requires HTTPS without URL credentials or fragments,
checks every redirect target before contacting it, permits at most five redirects,
and detects redirect loops. HTTPS cross-host redirects and signed query strings
are allowed. Redirect bodies are closed without reading. Final responses must be
HTTP 200; a declared Content-Length must match the pinned size, and unexpected
Content-Encoding is rejected. Streaming accepts only the exact pinned number of
bytes, with at most one additional byte read to detect overflow. The final SHA256
check remains mandatory.

The socket timeout is 30 seconds. A ten-minute elapsed-time budget is checked
between redirect requests and before/after body `read1` operations. This is not a
hard process deadline: blocking DNS, response-header parsing or a current socket
operation (including HTTP chunk framing/trailer parsing inside `read1`) cannot
be preempted by those checks. Manual runs have no hard outer deadline unless
the caller supplies one. Transport/body-length failures retain bounded retries;
URL-policy violations, invalid headers and completed hash mismatches do not retry.

These Python protections apply to the three top-level archives and the explicit
historical artifact downloader. Nested CMake downloads use the upstream rules
above; they are not routed through this Python downloader. Their checksums do not
imply the same redirect or response-size policy.

The source-build receipt binds both `build_extension.py` and its shared downloader
`qualified_extension.py`, along with the recipe, Pixi lock, target CLI and output.
Missing/stale downloader hashes are rejected. Rebuild with the current scripts;
never edit an old receipt to bypass that requirement.

[native-versions]: https://github.com/apache/paimon-cpp/blob/53f9c86d45aabb0a6f1a379271da07d7a9f27a3d/third_party/versions.txt
[arrow-versions]: https://github.com/apache/arrow/blob/6a2e19a852b367c72d7b12da4d104456491ed8b7/cpp/thirdparty/versions.txt
[arrow-patch]: https://github.com/apache/paimon-cpp/blob/53f9c86d45aabb0a6f1a379271da07d7a9f27a3d/cmake_modules/arrow.diff
[orc-patch]: https://github.com/apache/paimon-cpp/blob/53f9c86d45aabb0a6f1a379271da07d7a9f27a3d/cmake_modules/orc.diff
[avro-fmt]: https://github.com/apache/avro/blob/c499eefb48aa2db906c7bca14a047223806f36db/lang/c++/CMakeLists.txt
[avro-config]: https://github.com/apache/paimon-cpp/blob/53f9c86d45aabb0a6f1a379271da07d7a9f27a3d/cmake_modules/ThirdpartyToolchain.cmake#L1614
