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

CREATE TABLE crossfit (exercise TEXT, difficulty_level INTEGER);
INSERT INTO crossfit VALUES ('Push Ups', 3), ('Pull Ups', 5), ('Push Jerk', 5), ('Bar Muscle Up', 10), ('a', 10), ('b', 13), ('c', 15), ('d', 16);

CREATE TABLE equipment (equip TEXT, difficulty_level INTEGER);
INSERT INTO equipment VALUES ('Board', 3), ('Weight', 5), ('Trampoline', 3), ('Horse', 12);

select * from crossfit left outer join equipment on crossfit.difficulty_level = equipment.difficulty_level;

select * from crossfit right outer join equipment on crossfit.difficulty_level = equipment.difficulty_level;

select * from crossfit semi join equipment on crossfit.difficulty_level = equipment.difficulty_level;

select * from equipment semi join crossfit on crossfit.difficulty_level = equipment.difficulty_level;

select * from crossfit anti join equipment on crossfit.difficulty_level = equipment.difficulty_level;

select * from equipment anti join crossfit on crossfit.difficulty_level = equipment.difficulty_level;