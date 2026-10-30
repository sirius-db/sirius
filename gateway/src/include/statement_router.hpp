#pragma once

#include <string>

namespace sirius_gateway {

/// Rewrites a Quack client's SQL so each statement runs on every rank through
/// `sirius_gateway_run`, except reads of the gateway itself (`sirius_gateway_status`,
/// `sirius_gateway_log`), which run on the gateway.
/// @throws duckdb::InvalidInputException for settings the gateway owns, and for the other
/// `sirius_gateway_*` and `quack_*` functions, which only the gateway's console may call.
std::string route(std::string const& sql);

}  // namespace sirius_gateway
