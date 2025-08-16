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

DROP TABLE IF EXISTS T;
DROP TABLE IF EXISTS S;
CREATE TABLE T (A BIGINT, B BIGINT);
CREATE TABLE S (C BIGINT, D BIGINT);
INSERT INTO T VALUES (3, 1), (NULL, 2), (3, 3), (NULL, 4), (2, 5);
INSERT INTO S VALUES (NULL, 1), (NULL, 2), (NULL, 3), (NULL, 4), (NULL, 5);
call gpu_buffer_init("1 GB", "1 GB");
call gpu_processing("select A from T where A = 3");
call gpu_processing("select A from T where A = NULL");
call gpu_processing("select A from T where A IS NOT NULL");
call gpu_processing("select A from T where A IS NULL");
call gpu_processing("select count(A) from T");
call gpu_processing("select count(*) from T");
call gpu_processing("select sum(A) from T");
call gpu_processing("select A, sum(B) from T group by A");
call gpu_processing("select A, count(*) from T group by A");
call gpu_processing("select A, count(A) from T group by A");
call gpu_processing("select sum(C) from S");