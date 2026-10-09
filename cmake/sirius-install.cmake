if(VCPKG_BUILD AND NOT PROJECT_IS_TOP_LEVEL)
  install(FILES "${CMAKE_CURRENT_LIST_DIR}/sirius-cuda-fatbin.ld"
          DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/sirius")
endif()

if(NOT PROJECT_IS_TOP_LEVEL)
  install(
    TARGETS sirius_extension
            sirius_core
            cucascade_static
            cucascade_cudf_static
            cucascade_topology_discovery_static
            telemetry_bridge
            simpatico
    EXPORT "${DUCKDB_EXPORT_SET}"
    LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
endif()

include(CMakePackageConfigHelpers)
configure_package_config_file(
  cmake/sirius-config.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/sirius-config.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/sirius")
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/sirius-config-version.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMinorVersion)

install(
  DIRECTORY include/sirius
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT sirius_library)
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/sirius-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/sirius-config-version.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/sirius"
  COMPONENT sirius_library)

if(SIRIUS_BUILD_SHARED)
  install(
    TARGETS sirius_shared
    EXPORT sirius-targets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT sirius_library)
  install(
    EXPORT sirius-targets
    NAMESPACE sirius::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/sirius"
    COMPONENT sirius_library)
endif()

if(SIRIUS_BUILD_STATIC)
  # Source-built support libraries remain separate archives in the package.
  install(
    TARGETS sirius_core
            simpatico
            duckdb_static
            dummy_static_extension_loader
            core_functions_extension
            parquet_extension
            cucascade_static
            cucascade_cudf_static
            cucascade_topology_discovery_static
    EXPORT sirius-static-targets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT sirius_library)
  set(sirius_install_component "${CMAKE_INSTALL_DEFAULT_COMPONENT_NAME}")
  set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME sirius_library)
  corrosion_install(
    TARGETS
    telemetry_bridge
    EXPORT
    sirius-static-targets
    ARCHIVE
    DESTINATION
    "${CMAKE_INSTALL_LIBDIR}")
  set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "${sirius_install_component}")
  install(
    FILES "${CMAKE_BINARY_DIR}/corrosion/sirius-static-targetsCorrosion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/sirius"
    COMPONENT sirius_library)
  install(
    EXPORT sirius-static-targets
    NAMESPACE sirius::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/sirius"
    COMPONENT sirius_library)
  install(
    FILES "${CMAKE_CURRENT_LIST_DIR}/sirius-static-dependencies.cmake"
          "${CMAKE_CURRENT_LIST_DIR}/sirius-cuda-fatbin.ld"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/sirius"
    COMPONENT sirius_library)
endif()
