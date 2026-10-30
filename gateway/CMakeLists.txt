cmake_minimum_required(VERSION 3.5)

project(SiriusGatewayExtension)

include_directories(src/include)

set(EXTENSION_SOURCES src/sirius_gateway_extension.cpp src/gateway_state.cpp
                      src/rank_supervisor.cpp src/statement_router.cpp)

build_static_extension(sirius_gateway ${EXTENSION_SOURCES})
build_loadable_extension(sirius_gateway " " ${EXTENSION_SOURCES})
