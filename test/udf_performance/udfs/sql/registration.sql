-- Copyright 2026, Sirius Contributors.
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
-- http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- Nullable BIGINT -> nullable BIGINT. Clip before adding to prevent signed64
-- overflow. The dividend is then 0..42, so SQL truncation equals Python floor.
CREATE MACRO bounded_band(x BIGINT) AS (
    (
        (
            (
                CASE WHEN x < -13::BIGINT THEN -13::BIGINT
                     WHEN x > 29::BIGINT THEN 29::BIGINT
                     ELSE x END
                + 13::BIGINT
            ) // 4::BIGINT
        ) * 3::BIGINT
    ) - 7::BIGINT
)::BIGINT;
