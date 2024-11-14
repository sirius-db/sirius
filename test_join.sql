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