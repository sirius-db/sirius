/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "catch.hpp"
#include "io/rest/s3/static_credentials.hpp"

#include <chrono>

using sirius::io::rest::s3::static_credentials;

TEST_CASE("static_credentials default constructs to empty inert values",
          "[s3][authorizer][static_credentials]")
{
  static_credentials creds;

  CHECK(creds.access_key_id.empty());
  CHECK(creds.secret_access_key.empty());
  CHECK(creds.session_token.empty());
  CHECK_FALSE(creds.expires_at.has_value());
}

TEST_CASE("static_credentials preserves session token and expiration across copies",
          "[s3][authorizer][static_credentials]")
{
  static_credentials creds;
  creds.access_key_id     = "access";
  creds.secret_access_key = "secret";
  creds.session_token     = "session";
  creds.expires_at        = std::chrono::system_clock::time_point{std::chrono::seconds{12345}};

  auto copy = creds;

  CHECK(copy.access_key_id == "access");
  CHECK(copy.secret_access_key == "secret");
  CHECK(copy.session_token == "session");
  REQUIRE(copy.expires_at.has_value());
  CHECK(*copy.expires_at == *creds.expires_at);
}
