-- Copyright 2025 Sirius Contributors
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

CREATE TABLE T (A BIGINT);
INSERT INTO T VALUES (1), (NULL), (3), (NULL), (5);
call gpu_buffer_init("1 GB", "1 GB");
call gpu_processing("select * from T");