-- test1.sql

-- Delete all data from the tables
DELETE FROM Person;
DELETE FROM Knows;

-- Or drop and recreate the tables
DROP TABLE IF EXISTS Person;
DROP TABLE IF EXISTS Knows;

CREATE TABLE Person (id BIGINT, name VARCHAR);
INSERT INTO Person VALUES (14, 'Alice'), (25, 'Bob'), (37, 'Charlie'), (42, 'Dave');

CREATE TABLE Knows (src BIGINT, dst BIGINT);
INSERT INTO Knows VALUES
                      (14, 25),
                      (14, 37),
                      (25, 37),
                      (37, 42),
                      (14, 42);